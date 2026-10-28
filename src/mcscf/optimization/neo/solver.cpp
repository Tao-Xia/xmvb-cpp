#include "core/index.hpp"
#include "mcscf/optimization/neo/solver.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "mcscf/optimization/trust_region/spectral.hpp"
#include "linalg/symmetry.hpp"
#include "mcscf/optimization/neo/globalization.hpp"

namespace xmvb::mcscf {
namespace {

struct Basis {
    std::vector<Eigen::VectorXd> vectors;
    std::vector<Eigen::VectorXd> metric_images;
    std::vector<Eigen::VectorXd> hessian_images;
};

Eigen::VectorXd residual_direction(const QuadraticProblem& problem,
                                   const Eigen::VectorXd& residual) {
    return problem.has_preconditioner() ? problem.apply_preconditioner(residual) : residual;
}

bool append_direction(const QuadraticProblem& problem,
                      Eigen::VectorXd direction,
                      Basis* basis,
                      int* hessian_actions) {
    Eigen::VectorXd metric_direction = problem.apply_metric(direction);
    const double initial_norm = direction.norm();
    if (!(initial_norm > 0.0) || !std::isfinite(initial_norm))
        return false;

    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t j = 0; j < basis->vectors.size(); ++j) {
            const double overlap = basis->vectors[j].dot(direction);
            direction.noalias() -= overlap * basis->vectors[j];
            metric_direction.noalias() -= overlap * basis->metric_images[j];
        }
    }
    const double norm = direction.norm();
    const double threshold = 64.0 * std::numeric_limits<double>::epsilon() * initial_norm;
    if (!(norm > threshold) || !std::isfinite(norm))
        return false;

    const double inverse_norm = 1.0 / norm;
    direction *= inverse_norm;
    metric_direction *= inverse_norm;
    basis->vectors.push_back(direction);
    basis->metric_images.push_back(metric_direction);
    basis->hessian_images.push_back(problem.apply_hessian(direction));
    ++*hessian_actions;
    return true;
}

Eigen::MatrixXd columns(const std::vector<Eigen::VectorXd>& vectors) {
    Eigen::MatrixXd matrix(vectors.front().size(), vectors.size());
    for (std::size_t j = 0; j < vectors.size(); ++j) {
        matrix.col(static_cast<Eigen::Index>(j)) = vectors[j];
    }
    return matrix;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

Eigen::VectorXd generic_probe(Eigen::Index size) {
    Eigen::VectorXd probe(size);
    for (Eigen::Index j = 0; j < size; ++j) {
        const std::uint64_t bits = splitmix64(static_cast<std::uint64_t>(j));
        const double unit = static_cast<double>(bits >> 11U) * 0x1.0p-53;
        probe[j] = 2.0 * unit - 1.0;
    }
    return probe;
}

void set_augmented_certificate(const QuadraticProblem& problem,
                               const Eigen::VectorXd& minimum_ritz_vector,
                               NeoResult* result) {
    result->augmented_eigenvalue = -result->shift;
    result->augmented_certificate_valid = false;
    result->augmented_eigenvector = Eigen::VectorXd::Zero(problem.size() + 1);
    if (result->hard_case) {
        result->augmented_eigenvector.tail(problem.size()) = minimum_ritz_vector;
        return;
    }
    const double descent = problem.gradient().dot(result->step);
    if (result->shift > 0.0 && descent < 0.0) {
        result->gradient_scale = std::sqrt(result->shift / -descent);
        result->augmented_eigenvector[0] = 1.0;
        result->augmented_eigenvector.tail(problem.size()) = result->gradient_scale * result->step;
        result->augmented_certificate_valid = true;
        return;
    }
    // An interior Newton step is the alpha -> 0 limit of NEO.
    result->augmented_eigenvector[0] = 1.0;
}

} // namespace

NeoResult
solve_neo(const QuadraticProblem& problem, const NeoOptions& options, bool allow_early_trial) {
    if (!(options.trust_radius > 0.0) || !std::isfinite(options.trust_radius) ||
        !(options.relative_residual_tolerance > 0.0) ||
        !std::isfinite(options.relative_residual_tolerance) ||
        options.absolute_residual_tolerance < 0.0 ||
        !std::isfinite(options.absolute_residual_tolerance) ||
        options.maximum_subspace_dimension < 0) {
        throw std::invalid_argument("invalid NEO solver options");
    }

    const Eigen::Index dimension = problem.size();
    const Eigen::Index maximum_dimension =
        options.maximum_subspace_dimension == 0
            ? dimension
            : std::min<Eigen::Index>(dimension, options.maximum_subspace_dimension);
    if (options.stop_on_negative_curvature && !problem.gradient().isZero(0.0)) {
        throw std::invalid_argument(
            "NEO negative-curvature probe requires an exactly zero gradient");
    }
    const double gradient_norm = problem.dual_norm(problem.gradient());

    NeoResult result;
    result.step = Eigen::VectorXd::Zero(dimension);
    result.hessian_step = Eigen::VectorXd::Zero(dimension);
    result.metric_step = Eigen::VectorXd::Zero(dimension);
    result.kkt_residual = problem.gradient();
    result.minimum_curvature_direction = Eigen::VectorXd::Zero(dimension);
    result.minimum_curvature_hessian_image = Eigen::VectorXd::Zero(dimension);
    result.minimum_curvature_metric_image = Eigen::VectorXd::Zero(dimension);
    result.residual_target =
        options.absolute_residual_tolerance + options.relative_residual_tolerance * gradient_norm;

    Basis basis;
    Eigen::VectorXd first = residual_direction(problem, -problem.gradient());
    if (!append_direction(problem, std::move(first), &basis, &result.hessian_actions)) {
        append_direction(problem, generic_probe(problem.size()), &basis, &result.hessian_actions);
    }
    if (basis.vectors.empty()) {
        throw std::runtime_error("NEO could not construct a metric basis");
    }

    // A reproducible generic seed supplies the nonzero-overlap assumption used
    // by matrix-free lowest-root Davidson/Lanczos methods.
    if ((options.require_curvature_certificate || options.stop_on_negative_curvature) &&
        maximum_dimension > 1) {
        append_direction(problem, generic_probe(problem.size()), &basis, &result.hessian_actions);
    }

    Eigen::Index canonical_index = 0;
    while (true) {
        ++result.iterations;
        const Eigen::MatrixXd q = columns(basis.vectors);
        const Eigen::MatrixXd mq = columns(basis.metric_images);
        const Eigen::MatrixXd hq = columns(basis.hessian_images);
        const Eigen::MatrixXd projected_metric_action = q.transpose() * mq;
        const Eigen::MatrixXd metric_skew =
            projected_metric_action - projected_metric_action.transpose();
        const double metric_scale = std::max(1.0, projected_metric_action.norm());
        const double symmetry_factor = 256.0 * std::numeric_limits<double>::epsilon() *
                                       static_cast<double>(projected_metric_action.rows());
        const double metric_threshold = symmetry_factor * metric_scale;
        if (metric_skew.norm() > metric_threshold &&
            metric_skew.norm() > metric_threshold + linalg::projection_skew_roundoff(
                                                        q, mq, projected_metric_action)) {
            throw std::runtime_error("NEO metric action is not symmetric");
        }
        const Eigen::MatrixXd projected_metric =
            0.5 * (projected_metric_action + projected_metric_action.transpose());
        Eigen::LLT<Eigen::MatrixXd> metric_factor(projected_metric);
        if (metric_factor.info() != Eigen::Success) {
            throw std::runtime_error("NEO metric action is not positive definite");
        }
        const Eigen::MatrixXd whitening = metric_factor.matrixU().solve(
            Eigen::MatrixXd::Identity(projected_metric.rows(), projected_metric.cols()));

        const Eigen::MatrixXd projected_action = q.transpose() * hq;
        const Eigen::MatrixXd projected_skew = projected_action - projected_action.transpose();
        const double hessian_scale = std::max(1.0, projected_action.norm());
        const double base_threshold = symmetry_factor * hessian_scale;
        const double projection_roundoff =
            projected_skew.norm() > base_threshold
                ? linalg::projection_skew_roundoff(q, hq, projected_action)
                : 0.0;
        if (projected_skew.norm() > base_threshold + projection_roundoff) {
            // Re-evaluate the worst pair only on failure. This distinguishes a
            // reproducible bilinear defect from a mutable/nonrepeatable action,
            // without weakening the symmetry certificate or changing valid steps.
            Eigen::Index i, j;
            projected_skew.cwiseAbs().maxCoeff(&i, &j);
            const Eigen::VectorXd hi = problem.apply_hessian(q.col(i));
            const Eigen::VectorXd hj = problem.apply_hessian(q.col(j));
            std::ostringstream message;
            message.precision(17);
            message << "NEO Hessian action is not symmetric"
                    << "; dimension=" << dimension << "; subspace=" << q.cols()
                    << "; skew_norm=" << projected_skew.norm()
                    << "; threshold=" << base_threshold + projection_roundoff
                    << "; projection_roundoff=" << projection_roundoff
                    << "; action_scale=" << hessian_scale << "; worst_pair=" << i << ',' << j
                    << "; bilinear_skew=" << projected_skew(i, j)
                    << "; repeated_bilinear_skew=" << q.col(i).dot(hj) - q.col(j).dot(hi)
                    << "; repeat_error_i=" << (hi - hq.col(i)).norm()
                    << "; repeat_error_j=" << (hj - hq.col(j)).norm()
                    << "; image_norm_i=" << hi.norm() << "; image_norm_j=" << hj.norm()
                    << "; basis_orthogonality_error="
                    << (q.transpose() * q - Eigen::MatrixXd::Identity(q.cols(), q.cols())).norm();
            throw std::runtime_error(message.str());
        }
        const Eigen::MatrixXd projected_hessian =
            whitening.transpose() * (0.5 * (projected_action + projected_action.transpose())) *
            whitening;
        const Eigen::VectorXd projected_gradient = q.transpose() * problem.gradient();
        const Eigen::VectorXd whitened_gradient = whitening.transpose() * projected_gradient;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected_hessian);
        if (spectrum.info() != Eigen::Success) {
            result.stop_reason = NeoStopReason::NumericalFailure;
            return result;
        }
        const SpectralTrustRegionSolution projected_step =
            solve_spectral_trust_region(spectrum.eigenvalues(),
                                        spectrum.eigenvectors().transpose() * whitened_gradient,
                                        options.trust_radius);
        const Eigen::VectorXd white_step = spectrum.eigenvectors() * projected_step.step;
        const Eigen::VectorXd coefficients = whitening * white_step;

        result.step.noalias() = q * coefficients;
        result.hessian_step.noalias() = hq * coefficients;
        result.metric_step.noalias() = mq * coefficients;
        result.shift = projected_step.shift;
        result.boundary = projected_step.boundary;
        result.hard_case = projected_step.hard_case;
        result.step_norm = std::sqrt(std::max(0.0, result.step.dot(result.metric_step)));
        result.kkt_residual =
            problem.gradient() + result.hessian_step + result.shift * result.metric_step;
        result.residual_norm = problem.dual_norm(result.kkt_residual);
        result.residual_target = options.absolute_residual_tolerance +
                                 options.relative_residual_tolerance * gradient_norm;
        result.predicted_reduction =
            -problem.gradient().dot(result.step) - 0.5 * result.step.dot(result.hessian_step);

        const Eigen::VectorXd minimum_coordinates = whitening * spectrum.eigenvectors().col(0);
        const Eigen::VectorXd minimum_ritz_vector = q * minimum_coordinates;
        const Eigen::VectorXd minimum_hessian_image = hq * minimum_coordinates;
        const Eigen::VectorXd minimum_metric_image = mq * minimum_coordinates;
        const double metric_length_squared = minimum_ritz_vector.dot(minimum_metric_image);
        result.minimum_curvature_direction = minimum_ritz_vector;
        result.minimum_curvature_hessian_image = minimum_hessian_image;
        result.minimum_curvature_metric_image = minimum_metric_image;
        result.minimum_curvature_value =
            minimum_ritz_vector.dot(minimum_hessian_image) / metric_length_squared;
        const Eigen::VectorXd curvature_residual =
            minimum_hessian_image - spectrum.eigenvalues()[0] * minimum_metric_image;
        result.curvature_residual_norm = curvature_residual.norm();
        result.curvature_residual_target =
            options.relative_residual_tolerance *
            std::max({1.0,
                      minimum_hessian_image.norm(),
                      std::abs(spectrum.eigenvalues()[0]) * minimum_metric_image.norm()});
        const double positivity_tolerance =
            options.relative_residual_tolerance *
            std::max(1.0, std::max(result.shift, std::abs(spectrum.eigenvalues()[0])));

        const bool stationary = result.residual_norm <= result.residual_target;
        const bool curvature_converged =
            result.curvature_residual_norm <= result.curvature_residual_target;
        const bool shifted_positive =
            spectrum.eigenvalues()[0] + result.shift >= -positivity_tolerance;
        const bool complete_basis = static_cast<int>(basis.vectors.size()) == dimension;
        const bool lower_bound_certifies =
            problem.hessian_lower_bound() &&
            *problem.hessian_lower_bound() + result.shift >= -positivity_tolerance;
        result.global_curvature_certified = complete_basis || lower_bound_certifies;
        // A directly negative Rayleigh quotient witnesses indefiniteness even
        // before its Ritz residual has converged to final accuracy. The margin
        // follows local operator accuracy and roundoff. This is only an inertia
        // witness: its Ritz residual need not converge, and no nonlinear trial or
        // shifted Newton KKT certificate is implied by the return.
        if (options.stop_on_negative_curvature) {
            const double local_action_scale =
                minimum_ritz_vector.norm() * minimum_hessian_image.norm() / metric_length_squared;
            const double negative_curvature_margin =
                std::max(options.relative_residual_tolerance * local_action_scale,
                         symmetry_factor * projected_hessian.norm());
            if (std::isfinite(result.minimum_curvature_value) &&
                result.minimum_curvature_value < -negative_curvature_margin) {
                result.stop_reason = NeoStopReason::NegativeCurvature;
                set_augmented_certificate(problem, minimum_ritz_vector, &result);
                return result;
            }
        }
        // A regular boundary step is certified by the exact nonlinear keyframe
        // and its actual/predicted decrease.  Resolving an unrelated lowest Ritz
        // vector after the shifted Newton equation has converged only oversolves
        // the stale quadratic model.  A hard case remains different because the
        // step itself depends on the minimum-curvature direction.
        const bool need_curvature_certificate = options.require_curvature_certificate ||
                                                options.stop_on_negative_curvature ||
                                                result.hard_case;
        const bool required_curvature_converged =
            !need_curvature_certificate || curvature_converged;
        if (stationary && required_curvature_converged && shifted_positive &&
            (result.global_curvature_certified || !problem.hessian_lower_bound())) {
            result.stop_reason = NeoStopReason::Converged;
            set_augmented_certificate(problem, minimum_ritz_vector, &result);
            return result;
        }

        // Match coupled NEO's historical candidate policy, not its physical
        // stationarity certificate. The caller must validate a real nonlinear
        // energy/gradient keyframe and disable early trials near convergence.
        const double coordinate_gradient = problem.gradient().stableNorm();
        const double coordinate_target = neo_trial_residual_target(
            coordinate_gradient,
            gradient_norm,
            options.absolute_residual_tolerance +
                coordinate_gradient * coordinate_gradient / (1.0 + coordinate_gradient),
            options.absolute_residual_tolerance);
        if (allow_early_trial && !need_curvature_certificate && shifted_positive &&
            -problem.gradient().dot(result.step) > 0.0 &&
            result.kkt_residual.stableNorm() <= coordinate_target) {
            result.stop_reason = NeoStopReason::TrialReady;
            set_augmented_certificate(problem, minimum_ritz_vector, &result);
            return result;
        }

        if (static_cast<int>(basis.vectors.size()) >= maximum_dimension) {
            result.stop_reason = NeoStopReason::SubspaceLimit;
            set_augmented_certificate(problem, minimum_ritz_vector, &result);
            return result;
        }

        bool expanded = false;
        if (!stationary) {
            expanded = append_direction(problem,
                                        residual_direction(problem, -result.kkt_residual),
                                        &basis,
                                        &result.hessian_actions);
        }
        if (!expanded && need_curvature_certificate && !curvature_converged) {
            expanded = append_direction(problem,
                                        residual_direction(problem, -curvature_residual),
                                        &basis,
                                        &result.hessian_actions);
        }
        while (!expanded && canonical_index < problem.size()) {
            Eigen::VectorXd canonical = Eigen::VectorXd::Zero(problem.size());
            canonical[canonical_index++] = 1.0;
            expanded =
                append_direction(problem, std::move(canonical), &basis, &result.hessian_actions);
        }
        if (!expanded) {
            result.stop_reason =
                stationary && required_curvature_converged && shifted_positive &&
                        (result.global_curvature_certified || !problem.hessian_lower_bound())
                    ? NeoStopReason::Converged
                    : NeoStopReason::NumericalFailure;
            set_augmented_certificate(problem, minimum_ritz_vector, &result);
            return result;
        }
    }
}

} // namespace xmvb::mcscf
