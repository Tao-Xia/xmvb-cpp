#include "mcscf/optimization/problem.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace xmvb::mcscf {

QuadraticProblem::QuadraticProblem(Eigen::VectorXd gradient,
                                   linalg::VectorAction apply_hessian,
                                   linalg::VectorAction apply_metric,
                                   DualNorm dual_norm,
                                   linalg::VectorAction apply_preconditioner,
                                   std::optional<double> hessian_lower_bound)
    : gradient_(std::move(gradient)), apply_hessian_(std::move(apply_hessian)),
      apply_metric_(std::move(apply_metric)), dual_norm_(std::move(dual_norm)),
      apply_preconditioner_(std::move(apply_preconditioner)),
      hessian_lower_bound_(hessian_lower_bound) {
    if (gradient_.size() == 0 || !gradient_.allFinite()) {
        throw std::invalid_argument("quadratic model requires a finite nonempty gradient");
    }
    if (!apply_hessian_ || !apply_metric_ || !dual_norm_) {
        throw std::invalid_argument(
            "quadratic model requires Hessian, metric, and dual-norm actions");
    }
    if (hessian_lower_bound_ && !std::isfinite(*hessian_lower_bound_)) {
        throw std::invalid_argument("quadratic model Hessian lower bound must be finite");
    }
}

Eigen::VectorXd QuadraticProblem::apply_checked(const linalg::VectorAction& action,
                                                const Eigen::VectorXd& vector,
                                                const char* name) const {
    if (vector.size() != size() || !vector.allFinite()) {
        throw std::invalid_argument("invalid vector passed to quadratic model operator");
    }
    Eigen::VectorXd result = action(vector);
    if (result.size() != size() || !result.allFinite()) {
        throw std::runtime_error(name);
    }
    return result;
}

Eigen::VectorXd QuadraticProblem::apply_hessian(const Eigen::VectorXd& direction) const {
    return apply_checked(
        apply_hessian_, direction, "quadratic model Hessian action returned an invalid vector");
}

Eigen::VectorXd QuadraticProblem::apply_metric(const Eigen::VectorXd& direction) const {
    return apply_checked(
        apply_metric_, direction, "quadratic model metric action returned an invalid vector");
}

double QuadraticProblem::dual_norm(const Eigen::VectorXd& covector) const {
    if (covector.size() != size() || !covector.allFinite()) {
        throw std::invalid_argument("invalid covector passed to quadratic model dual norm");
    }
    const double norm = dual_norm_(covector);
    if (!std::isfinite(norm) || norm < 0.0) {
        throw std::runtime_error("quadratic model dual norm returned an invalid value");
    }
    return norm;
}

Eigen::VectorXd QuadraticProblem::apply_preconditioner(const Eigen::VectorXd& covector) const {
    if (!apply_preconditioner_) {
        throw std::logic_error("quadratic model problem has no residual preconditioner");
    }
    return apply_checked(apply_preconditioner_,
                         covector,
                         "quadratic model preconditioner returned an invalid vector");
}

} // namespace xmvb::mcscf
