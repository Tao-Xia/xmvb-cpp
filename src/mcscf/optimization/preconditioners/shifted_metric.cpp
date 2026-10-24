#include "mcscf/optimization/preconditioners/shifted_metric.hpp"
#include "mcscf/optimization/trace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::mcscf {
namespace {

Eigen::VectorXd checked_action(const linalg::VectorAction& action,
                               const Eigen::VectorXd& vector,
                               const char* message) {
    Eigen::VectorXd image = action(vector);
    if (image.size() != vector.size() || !image.allFinite()) {
        throw std::runtime_error(message);
    }
    return image;
}

} // namespace

Eigen::VectorXd
apply_shifted_metric_preconditioner(const Eigen::VectorXd& covector,
                                    double shift,
                                    double relative_residual_target,
                                    const linalg::VectorAction& apply_model,
                                    const linalg::VectorAction& apply_metric,
                                    const linalg::VectorAction& apply_block_inverse) {
    if (covector.size() == 0 || !covector.allFinite() || !(shift >= 0.0) || !std::isfinite(shift) ||
        !(relative_residual_target >= 0.0) || !std::isfinite(relative_residual_target) ||
        !apply_model || !apply_metric || !apply_block_inverse) {
        throw std::invalid_argument("invalid shifted-metric preconditioner data");
    }
    if (covector.isZero())
        return Eigen::VectorXd::Zero(covector.size());
    if (shift == 0.0 || relative_residual_target >= 1.0) {
        diagnostics::ciah_trace("precondition",
                                "iterations=0,shift=%.17g,budget=%.17g,reason=block",
                                shift,
                                relative_residual_target);
        return checked_action(
            apply_block_inverse, covector, "shifted block inverse returned an invalid vector");
    }

    const auto apply_shifted = [&](const Eigen::VectorXd& vector) {
        Eigen::VectorXd image =
            checked_action(apply_model, vector, "local model returned an invalid vector");
        image.noalias() +=
            shift *
            checked_action(apply_metric, vector, "orbital metric returned an invalid vector");
        return image;
    };

    // This solve only supplies a direction to the outer coupled Krylov space.
    // Measure the inner residual in the dual norm of the same positive block
    // inverse used by PCG. A relative budget is independent of the coordinate
    // scale of the outer physical KKT target. Its square root preserves the
    // geometric-mean accuracy rule for a basis direction.
    const Eigen::VectorXd rhs_preconditioned = checked_action(
        apply_block_inverse, covector, "shifted block inverse returned an invalid vector");
    const double rhs_pairing = covector.dot(rhs_preconditioned);
    if (!(rhs_pairing > 0.0) || !std::isfinite(rhs_pairing)) {
        throw std::runtime_error("shifted-metric block inverse is not positive definite");
    }
    const double rhs_norm = std::sqrt(rhs_pairing);
    const double target = std::max(std::sqrt(relative_residual_target),
                                   std::sqrt(std::numeric_limits<double>::epsilon())) *
                          rhs_norm;
    const auto residual_norm = [&](const Eigen::VectorXd& residual) {
        const Eigen::VectorXd inverse_residual = checked_action(
            apply_block_inverse, residual, "shifted block inverse returned an invalid vector");
        const double pairing = residual.dot(inverse_residual);
        if (!(pairing >= 0.0) || !std::isfinite(pairing)) {
            throw std::runtime_error("shifted-metric block inverse is not positive definite");
        }
        return std::sqrt(pairing);
    };
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(covector.size());
    Eigen::VectorXd residual = covector;
    Eigen::VectorXd preconditioned = rhs_preconditioned;
    Eigen::VectorXd direction = preconditioned;
    double residual_pairing = rhs_pairing;

    Eigen::VectorXd best_solution = preconditioned;
    double best_residual = residual_norm(covector - apply_shifted(best_solution));
    if (best_residual <= target) {
        diagnostics::ciah_trace(
            "precondition",
            "iterations=0,shift=%.17g,budget=%.17g,residual_ratio=%.17g,reason=initial",
            shift,
            relative_residual_target,
            best_residual / rhs_norm);
        return best_solution;
    }

    for (Eigen::Index iteration = 0; iteration < covector.size(); ++iteration) {
        const Eigen::VectorXd image = apply_shifted(direction);
        const double curvature = direction.dot(image);
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            throw std::runtime_error("shifted local metric model is not positive definite");
        }
        const double step = residual_pairing / curvature;
        if (!std::isfinite(step)) {
            throw std::runtime_error("shifted-metric preconditioner produced a non-finite step");
        }
        solution.noalias() += step * direction;
        residual = covector - apply_shifted(solution);
        preconditioned = checked_action(
            apply_block_inverse, residual, "shifted block inverse returned an invalid vector");
        const double next_pairing = residual.dot(preconditioned);
        if (!(next_pairing >= 0.0) || !std::isfinite(next_pairing)) {
            throw std::runtime_error("shifted-metric preconditioner lost positive definiteness");
        }
        const double current_residual_norm = std::sqrt(next_pairing);
        if (current_residual_norm < best_residual) {
            best_residual = current_residual_norm;
            best_solution = solution;
        }
        if (current_residual_norm <= target) {
            diagnostics::ciah_trace(
                "precondition",
                "iterations=%ld,shift=%.17g,budget=%.17g,residual_ratio=%.17g,reason=converged",
                static_cast<long>(iteration + 1),
                shift,
                relative_residual_target,
                current_residual_norm / rhs_norm);
            return solution;
        }
        if (!(next_pairing > 0.0)) {
            throw std::runtime_error("shifted-metric preconditioner lost positive definiteness");
        }
        direction = preconditioned + (next_pairing / residual_pairing) * direction;
        residual_pairing = next_pairing;
    }
    // Preconditioning is not an acceptance test. Return the best positive
    // model direction even if the outer target was unreachable here.
    diagnostics::ciah_trace(
        "precondition",
        "iterations=%ld,shift=%.17g,budget=%.17g,residual_ratio=%.17g,reason=dimension",
        static_cast<long>(covector.size()),
        shift,
        relative_residual_target,
        best_residual / rhs_norm);
    return best_solution;
}

} // namespace xmvb::mcscf
