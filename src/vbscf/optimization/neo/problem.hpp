#pragma once

#include <functional>
#include <optional>

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Matrix-free action of a linear operator in NEO coordinates. */
using NeoAction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/** @brief Hessian and metric images evaluated in one shared operator pass. */
struct NeoOperatorImages {
  Eigen::VectorXd hessian;
  Eigen::VectorXd metric;
};

using NeoCombinedAction =
    std::function<NeoOperatorImages(const Eigen::VectorXd&)>;

/** @brief Eigenvalue-aware augmented-Hessian residual preconditioner. */
using NeoPreconditioner =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;

/**
 * @brief Immutable quadratic model used by norm-extended optimization.
 *
 * The Hessian maps a coordinate increment to a gradient covector. The metric
 * defines the physical trust-region norm. An optional signed diagonal
 * preconditioner may accelerate augmented-Hessian subspace growth, but it is
 * not part of the NEO equations.
 */
class NeoProblem {
public:
  /**
   * @brief Constructs a matrix-free generalized-metric NEO problem.
   *
   * @param gradient Gradient covector at the accepted point.
   * @param apply_hessian Symmetric Hessian action.
   * @param apply_metric Symmetric positive-definite metric action.
   * @param apply_preconditioner Optional augmented-Hessian residual
   * preconditioner. Its scalar argument is the current AH eigenvalue.
   * @param hessian_lower_bound Optional certified lower bound on every
   * generalized Hessian eigenvalue. An estimate is not sufficient.
   */
  NeoProblem(
      Eigen::VectorXd gradient,
      NeoAction apply_hessian,
      NeoAction apply_metric,
      NeoPreconditioner apply_preconditioner = {},
      std::optional<double> hessian_lower_bound = std::nullopt,
      Eigen::VectorXd initial_guess = {},
      NeoCombinedAction apply_hessian_metric = {});

  /** @brief Number of optimization coordinates. */
  Eigen::Index size() const noexcept { return gradient_.size(); }

  /** @brief Gradient covector at the accepted point. */
  const Eigen::VectorXd& gradient() const noexcept { return gradient_; }

  /** @brief Applies the Hessian and validates its result. */
  Eigen::VectorXd apply_hessian(const Eigen::VectorXd& direction) const;

  /** @brief Applies the coordinate metric and validates its result. */
  Eigen::VectorXd apply_metric(const Eigen::VectorXd& direction) const;

  /** @brief Applies both operators, sharing intermediates when available. */
  NeoOperatorImages apply_hessian_metric(
      const Eigen::VectorXd& direction) const;

  /** @brief Whether a residual preconditioner was supplied. */
  bool has_preconditioner() const noexcept {
    return static_cast<bool>(apply_preconditioner_);
  }

  /** @brief Applies the optional residual preconditioner. */
  Eigen::VectorXd apply_preconditioner(
      const Eigen::VectorXd& covector,
      double shift) const;

  /** @brief Optional recycled direction used to seed the action subspace. */
  const Eigen::VectorXd& initial_guess() const noexcept {
    return initial_guess_;
  }

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
  const NeoPreconditioner apply_preconditioner_;
  const std::optional<double> hessian_lower_bound_;
  const Eigen::VectorXd initial_guess_;
  const NeoCombinedAction apply_hessian_metric_;
};

}  // namespace xmvb::vb
