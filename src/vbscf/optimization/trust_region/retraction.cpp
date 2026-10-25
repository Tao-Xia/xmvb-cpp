#include "vbscf/optimization/trust_region/retraction.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include "core/timing.hpp"

namespace xmvb::vb {

Eigen::VectorXd
gather_nonredundant_retract_tangent(const OrbitalPreparationInput& orbital_preparation_input,
                                    const OrbitalChart& space,
                                    const SparseParameterLayout& parameter_view,
                                    const Eigen::VectorXd& reduced_step) {
    const Eigen::VectorXd full_tangent =
        space.expand_retract_input_tangent(orbital_preparation_input, reduced_step);
    Eigen::VectorXd packed_tangent =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(parameter_view.size()));
    const auto& differentiable_indices = parameter_view.differentiable_parameter_indices();
    for (Eigen::Index packed_index = 0; packed_index < packed_tangent.size(); ++packed_index) {
        packed_tangent[packed_index] = full_tangent[differentiable_indices[packed_index]];
    }
    return packed_tangent;
}

NonredundantRetractionMetric::NonredundantRetractionMetric(
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const OrbitalPreparationInput& input)
    : space_(space), parameter_view_(parameter_view),
      physical_metric_(input, space.uses_active_subspace_gauge()) {}

Eigen::VectorXd NonredundantRetractionMetric::tangent(const Eigen::VectorXd& reduced_step) const {
    return reduced_step;
}

Eigen::VectorXd NonredundantRetractionMetric::apply(const Eigen::VectorXd& reduced_step) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::MetricAction);
    return space_.project_reduced_gradient(
        physical_metric_.apply(parameter_view_, space_.expand_step(reduced_step)));
}

Eigen::VectorXd NonredundantRetractionMetric::solve(const Eigen::VectorXd& covector) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::MetricSolve);
    const Eigen::Index dimension = space_.reduced_size();
    if (covector.size() != dimension) {
        throw std::runtime_error("retraction metric solve received an incompatible covector");
    }
    if (!covector.allFinite()) {
        throw std::runtime_error("retraction metric solve received a non-finite covector");
    }
    if (dimension == 0 || covector.isZero()) {
        return Eigen::VectorXd::Zero(dimension);
    }
    std::call_once(block_inverse_once_,
                   [this] { block_inverse_ = space_.metric_block_inverse(physical_metric_); });

    const double right_hand_side_norm = covector.norm();
    const double relative_tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
    const double residual_tolerance = relative_tolerance * right_hand_side_norm;

    Eigen::VectorXd solution = Eigen::VectorXd::Zero(dimension);
    Eigen::VectorXd residual = covector;
    Eigen::VectorXd preconditioned = block_inverse_(residual);
    Eigen::VectorXd direction = preconditioned;
    double residual_product = residual.dot(preconditioned);
    double cycle_initial_residual_norm = right_hand_side_norm;
    Eigen::Index iteration_in_cycle = 0;
    while (true) {
        Eigen::VectorXd metric_direction = apply(direction);
        double curvature = direction.dot(metric_direction);
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            // Finite-precision CG can lose conjugacy long before an SPD operator
            // loses positivity.  Restart from the explicitly recomputed residual;
            // only failure of that preconditioned steepest direction is evidence
            // of a singular reduced metric.
            residual = covector - apply(solution);
            const double restarted_residual_norm = residual.norm();
            if (restarted_residual_norm <= residual_tolerance) {
                return solution;
            }
            preconditioned = block_inverse_(residual);
            direction = preconditioned;
            metric_direction = apply(direction);
            curvature = direction.dot(metric_direction);
            if (!(curvature > 0.0) || !std::isfinite(curvature)) {
                throw std::runtime_error("retraction metric is singular in reduced space");
            }
            residual_product = residual.dot(preconditioned);
            cycle_initial_residual_norm = restarted_residual_norm;
            iteration_in_cycle = 0;
        }

        const double step = residual_product / curvature;
        if (!std::isfinite(step)) {
            throw std::runtime_error("retraction metric solve produced a non-finite CG step");
        }
        solution.noalias() += step * direction;

        // Preserve the CG recurrence between certifications. Replacing its
        // residual by b-Gx on every iteration introduces fresh rounding errors
        // into the conjugacy recurrence and also doubles metric applications.
        residual.noalias() -= step * metric_direction;
        const double next_residual_norm = residual.norm();
        if (!std::isfinite(next_residual_norm)) {
            throw std::runtime_error("retraction metric solve produced a non-finite residual");
        }
        ++iteration_in_cycle;
        if (next_residual_norm <= residual_tolerance || iteration_in_cycle == dimension) {
            // Recursive residuals never certify acceptance. Check the same true
            // residual tolerance as before, and restart if a residual gap remains.
            residual = covector - apply(solution);
            const double true_residual_norm = residual.norm();
            if (!std::isfinite(true_residual_norm)) {
                throw std::runtime_error("retraction metric CG has a non-finite true residual");
            }
            if (true_residual_norm <= residual_tolerance)
                return solution;
            const double resolved_decrease = cycle_initial_residual_norm - true_residual_norm;
            const double roundoff_decrease =
                std::numeric_limits<double>::epsilon() *
                std::max(cycle_initial_residual_norm, right_hand_side_norm);
            if (!(resolved_decrease > roundoff_decrease)) {
                throw std::runtime_error(
                    "retraction metric CG stagnated above its residual tolerance");
            }
            preconditioned = block_inverse_(residual);
            direction = preconditioned;
            residual_product = residual.dot(preconditioned);
            cycle_initial_residual_norm = true_residual_norm;
            iteration_in_cycle = 0;
            continue;
        }
        preconditioned = block_inverse_(residual);
        const double next_residual_product = residual.dot(preconditioned);
        const double beta = next_residual_product / residual_product;
        if (!std::isfinite(beta)) {
            throw std::runtime_error("retraction metric solve produced a non-finite CG recurrence");
        }
        direction = preconditioned + beta * direction;
        residual_product = next_residual_product;
    }
}

double NonredundantRetractionMetric::dual_norm(const Eigen::VectorXd& covector) const {
    if (covector.size() == 0)
        return 0.0;
    if (space_.uses_active_subspace_gauge() && physical_metric_.has_full_support(parameter_view_)) {
        return physical_metric_.subspace_dual_norm(parameter_view_,
                                                   space_.expand_gradient(covector));
    }
    const Eigen::VectorXd riesz_gradient = solve(covector);
    const double squared_norm = covector.dot(riesz_gradient);
    const double roundoff =
        64.0 * std::numeric_limits<double>::epsilon() * covector.norm() * riesz_gradient.norm();
    if (!std::isfinite(squared_norm) || squared_norm < -roundoff) {
        throw std::runtime_error("retraction metric produced an invalid gradient dual norm");
    }
    return std::sqrt(std::max(0.0, squared_norm));
}

double NonredundantRetractionMetric::norm(const Eigen::VectorXd& reduced_step) const {
    const double squared_norm =
        physical_metric_.squared_norm(parameter_view_, space_.expand_step(reduced_step));
    const double tangent_norm = std::sqrt(squared_norm);
    return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

Eigen::VectorXd NonredundantRetractionMetric::clip_to_radius(const Eigen::VectorXd& reduced_step,
                                                             double trust_radius) const {
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
        return Eigen::VectorXd::Zero(reduced_step.size());
    }
    const double tangent_norm = norm(reduced_step);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
        return Eigen::VectorXd::Zero(reduced_step.size());
    }
    if (tangent_norm <= trust_radius) {
        return reduced_step;
    }
    return (trust_radius / tangent_norm) * reduced_step;
}

Eigen::VectorXd
shrink_reduced_step_inside_trust_radius(const Eigen::VectorXd& reduced_step,
                                        double trust_radius,
                                        const NonredundantRetractionMetric& metric) {
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
        return Eigen::VectorXd::Zero(reduced_step.size());
    }
    constexpr double kInitialStepSafetyFraction = 0.95;
    const double target_radius = kInitialStepSafetyFraction * trust_radius;
    if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
        return Eigen::VectorXd::Zero(reduced_step.size());
    }
    const double tangent_norm = metric.norm(reduced_step);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
        return Eigen::VectorXd::Zero(reduced_step.size());
    }
    if (tangent_norm < target_radius) {
        return reduced_step;
    }
    return (target_radius / tangent_norm) * reduced_step;
}

} // namespace xmvb::vb
