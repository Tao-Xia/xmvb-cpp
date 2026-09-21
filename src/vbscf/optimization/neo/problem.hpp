#pragma once

#include <functional>
#include <optional>

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Matrix-free action of a linear operator in NEO coordinates. */
using NeoAction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/**
 * @brief Immutable quadratic model used by norm-extended optimization.
 *
 * The Hessian maps a coordinate increment to a gradient covector. The metric
 * and inverse metric map between coordinate vectors and covectors. Both metric
 * actions must represent the same symmetric positive-definite operator.
 */
class NeoProblem {
public:
  /**
   * @brief Constructs a matrix-free generalized-metric NEO problem.
   *
   * @param gradient Gradient covector at the accepted point.
   * @param apply_hessian Symmetric Hessian action.
   * @param apply_metric Symmetric positive-definite metric action.
   * @param apply_inverse_metric Inverse of `apply_metric`.
   * @param hessian_lower_bound Optional certified lower bound on every
   * generalized Hessian eigenvalue. An estimate is not sufficient.
   */
  NeoProblem(
      Eigen::VectorXd gradient,
      NeoAction apply_hessian,
      NeoAction apply_metric,
      NeoAction apply_inverse_metric,
      std::optional<double> hessian_lower_bound = std::nullopt);

  /** @brief Number of optimization coordinates. */
  Eigen::Index size() const noexcept { return gradient_.size(); }

  /** @brief Gradient covector at the accepted point. */
  const Eigen::VectorXd& gradient() const noexcept { return gradient_; }

  /** @brief Applies the Hessian and validates its result. */
  Eigen::VectorXd apply_hessian(const Eigen::VectorXd& direction) const;

  /** @brief Applies the coordinate metric and validates its result. */
  Eigen::VectorXd apply_metric(const Eigen::VectorXd& direction) const;

  /** @brief Applies the inverse coordinate metric and validates its result. */
  Eigen::VectorXd apply_inverse_metric(const Eigen::VectorXd& covector) const;

  /** @brief Certified lower bound on the generalized Hessian spectrum. */
  const std::optional<double>& hessian_lower_bound() const noexcept {
    return hessian_lower_bound_;
  }

private:
  Eigen::VectorXd apply_checked(
      const NeoAction& action,
      const Eigen::VectorXd& vector,
      const char* name) const;

  const Eigen::VectorXd gradient_;
  const NeoAction apply_hessian_;
  const NeoAction apply_metric_;
  const NeoAction apply_inverse_metric_;
  const std::optional<double> hessian_lower_bound_;
};

}  // namespace xmvb::vb
