#include "vbscf/optimization/neo/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

#include "vbscf/optimization/trust_region/spectral.hpp"

namespace xmvb::vb {
namespace {

struct Basis {
  std::vector<Eigen::VectorXd> vectors;
  std::vector<Eigen::VectorXd> metric_images;
  std::vector<Eigen::VectorXd> hessian_images;
};

double dual_norm(const NeoProblem& problem, const Eigen::VectorXd& covector) {
  if (covector.size() == 0) return 0.0;
  const Eigen::VectorXd mapped = problem.apply_inverse_metric(covector);
  const double square = covector.dot(mapped);
  const double scale = covector.norm() * mapped.norm();
  const double roundoff = 64.0 * std::numeric_limits<double>::epsilon() * scale;
  if (!std::isfinite(square) || !std::isfinite(scale) ||
      (covector.squaredNorm() > 0.0 && square <= roundoff)) {
    throw std::runtime_error("NEO inverse metric is not positive definite");
  }
  return std::sqrt(square);
}

bool append_direction(
    const NeoProblem& problem,
    Eigen::VectorXd direction,
    Basis* basis,
    int* hessian_actions) {
  Eigen::VectorXd metric_direction = problem.apply_metric(direction);
  const double initial_square = direction.dot(metric_direction);
  if (!(initial_square > 0.0) || !std::isfinite(initial_square)) return false;

  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t j = 0; j < basis->vectors.size(); ++j) {
      const double overlap = basis->vectors[j].dot(metric_direction);
      direction.noalias() -= overlap * basis->vectors[j];
      metric_direction.noalias() -= overlap * basis->metric_images[j];
    }
  }
  const double square = direction.dot(metric_direction);
  const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
      initial_square;
  if (!(square > threshold) || !std::isfinite(square)) return false;

  const double inverse_norm = 1.0 / std::sqrt(square);
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

Eigen::VectorXd deterministic_probe(Eigen::Index size) {
  Eigen::VectorXd probe(size);
  for (Eigen::Index j = 0; j < size; ++j) {
    probe[j] = (j % 2 == 0 ? 1.0 : -1.0) *
        (1.0 + static_cast<double>(j) / static_cast<double>(size));
  }
  return probe;
}

void set_augmented_certificate(
    const NeoProblem& problem,
    const Eigen::VectorXd& minimum_ritz_vector,
    NeoResult* result) {
  result->augmented_eigenvalue = -result->shift;
  result->augmented_certificate_valid = false;
  result->augmented_eigenvector =
      Eigen::VectorXd::Zero(problem.size() + 1);
  if (result->hard_case) {
    result->augmented_eigenvector.tail(problem.size()) = minimum_ritz_vector;
    return;
  }
  const double descent = problem.gradient().dot(result->step);
  if (result->shift > 0.0 && descent < 0.0) {
    result->gradient_scale = std::sqrt(result->shift / -descent);
    result->augmented_eigenvector[0] = 1.0;
    result->augmented_eigenvector.tail(problem.size()) =
        result->gradient_scale * result->step;
    result->augmented_certificate_valid = true;
    return;
  }
  // An interior Newton step is the alpha -> 0 limit of NEO.
  result->augmented_eigenvector[0] = 1.0;
}

}  // namespace

NeoResult solve_neo(const NeoProblem& problem, const NeoOptions& options) {
  if (!(options.trust_radius > 0.0) ||
      !std::isfinite(options.trust_radius) ||
      !(options.relative_residual_tolerance > 0.0) ||
      !std::isfinite(options.relative_residual_tolerance) ||
      options.absolute_residual_tolerance < 0.0 ||
      !std::isfinite(options.absolute_residual_tolerance) ||
      options.maximum_subspace_dimension < 0) {
    throw std::invalid_argument("invalid NEO solver options");
  }

  const int dimension = static_cast<int>(problem.size());
  const int maximum_dimension = options.maximum_subspace_dimension == 0
      ? dimension
      : std::min(dimension, options.maximum_subspace_dimension);
  const double gradient_norm = dual_norm(problem, problem.gradient());

  NeoResult result;
  result.step = Eigen::VectorXd::Zero(dimension);
  result.hessian_step = Eigen::VectorXd::Zero(dimension);
  result.metric_step = Eigen::VectorXd::Zero(dimension);
  result.kkt_residual = problem.gradient();
  result.residual_target = options.absolute_residual_tolerance +
      options.relative_residual_tolerance * gradient_norm;

  Basis basis;
  Eigen::VectorXd first = problem.apply_inverse_metric(-problem.gradient());
  if (!append_direction(
          problem, std::move(first), &basis, &result.hessian_actions)) {
    append_direction(
        problem, deterministic_probe(problem.size()), &basis,
        &result.hessian_actions);
  }
  if (basis.vectors.empty()) {
    throw std::runtime_error("NEO could not construct a metric basis");
  }

  // A second, gradient-independent seed exposes negative curvature in exact
  // hard cases where the gradient Krylov sequence cannot reach that subspace.
  if (maximum_dimension > 1) {
    append_direction(
        problem, deterministic_probe(problem.size()), &basis,
        &result.hessian_actions);
  }

  Eigen::Index canonical_index = 0;
  while (true) {
    ++result.iterations;
    const Eigen::MatrixXd q = columns(basis.vectors);
    const Eigen::MatrixXd mq = columns(basis.metric_images);
    const Eigen::MatrixXd hq = columns(basis.hessian_images);
    const Eigen::MatrixXd projected_action = q.transpose() * hq;
    const Eigen::MatrixXd projected_skew =
        projected_action - projected_action.transpose();
    const double symmetry_scale = std::max(1.0, projected_action.norm());
    const double symmetry_tolerance = 256.0 *
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(projected_action.rows()) * symmetry_scale;
    if (projected_skew.norm() > symmetry_tolerance) {
      throw std::runtime_error("NEO Hessian action is not symmetric");
    }
    const Eigen::MatrixXd projected_hessian =
        0.5 * (projected_action + projected_action.transpose());
    const Eigen::VectorXd projected_gradient = q.transpose() * problem.gradient();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected_hessian);
    if (spectrum.info() != Eigen::Success) {
      result.stop_reason = NeoStopReason::NumericalFailure;
      return result;
    }
    const SpectralTrustRegionSolution projected_step =
        solve_spectral_trust_region(
            spectrum.eigenvalues(),
            spectrum.eigenvectors().transpose() * projected_gradient,
            options.trust_radius);
    const Eigen::VectorXd coefficients =
        spectrum.eigenvectors() * projected_step.step;

    result.step.noalias() = q * coefficients;
    result.hessian_step.noalias() = hq * coefficients;
    result.metric_step.noalias() = mq * coefficients;
    result.shift = projected_step.shift;
    result.boundary = projected_step.boundary;
    result.hard_case = projected_step.hard_case;
    result.step_norm = std::sqrt(std::max(
        0.0, result.step.dot(result.metric_step)));
    result.kkt_residual = problem.gradient() + result.hessian_step +
        result.shift * result.metric_step;
    result.residual_norm = dual_norm(problem, result.kkt_residual);
    result.residual_target = options.absolute_residual_tolerance +
        options.relative_residual_tolerance * std::max(
            gradient_norm, result.shift * options.trust_radius);
    result.predicted_reduction = -problem.gradient().dot(result.step) -
        0.5 * result.step.dot(result.hessian_step);

    const Eigen::VectorXd minimum_coefficients = spectrum.eigenvectors().col(0);
    const Eigen::VectorXd minimum_ritz_vector = q * minimum_coefficients;
    const Eigen::VectorXd curvature_residual =
        hq * minimum_coefficients -
        spectrum.eigenvalues()[0] * (mq * minimum_coefficients);
    const double curvature_residual_norm =
        dual_norm(problem, curvature_residual);
    const double curvature_target =
        options.relative_residual_tolerance *
        std::max(1.0, std::abs(spectrum.eigenvalues()[0]));
    const double positivity_tolerance =
        options.relative_residual_tolerance * std::max(
            1.0, std::max(result.shift,
                          std::abs(spectrum.eigenvalues()[0])));

    const bool stationary = result.residual_norm <= result.residual_target;
    const bool curvature_converged =
        curvature_residual_norm <= curvature_target;
    const bool shifted_positive =
        spectrum.eigenvalues()[0] + result.shift >= -positivity_tolerance;
    const bool complete_basis =
        static_cast<int>(basis.vectors.size()) == dimension;
    const bool lower_bound_certifies = problem.hessian_lower_bound() &&
        *problem.hessian_lower_bound() + result.shift >=
            -positivity_tolerance;
    const bool global_curvature_certified =
        complete_basis || lower_bound_certifies;
    if (stationary && curvature_converged && shifted_positive &&
        global_curvature_certified) {
      result.stop_reason = NeoStopReason::Converged;
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
      expanded = append_direction(
          problem,
          problem.apply_inverse_metric(-result.kkt_residual),
          &basis,
          &result.hessian_actions);
    }
    if (!expanded && !curvature_converged) {
      expanded = append_direction(
          problem,
          problem.apply_inverse_metric(-curvature_residual),
          &basis,
          &result.hessian_actions);
    }
    while (!expanded && canonical_index < problem.size()) {
      Eigen::VectorXd canonical = Eigen::VectorXd::Zero(problem.size());
      canonical[canonical_index++] = 1.0;
      expanded = append_direction(
          problem, std::move(canonical), &basis, &result.hessian_actions);
    }
    if (!expanded) {
      result.stop_reason = stationary && curvature_converged &&
              shifted_positive && global_curvature_certified
          ? NeoStopReason::Converged
          : NeoStopReason::NumericalFailure;
      set_augmented_certificate(problem, minimum_ritz_vector, &result);
      return result;
    }
  }
}

}  // namespace xmvb::vb
