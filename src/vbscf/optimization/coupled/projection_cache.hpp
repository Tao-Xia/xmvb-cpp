#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"

namespace xmvb::vb {

/** @brief Number of cached matrix-free block evaluations. */
struct CoupledActionCounts {
  int orbital_hessian = 0;
  int orbital_to_response = 0;
  int response_to_orbital = 0;
  int response_hessian = 0;
  int orbital_metric = 0;

  bool operator==(const CoupledActionCounts& other) const noexcept {
    return orbital_hessian == other.orbital_hessian &&
        orbital_to_response == other.orbital_to_response &&
        response_to_orbital == other.response_to_orbital &&
        response_hessian == other.response_hessian &&
        orbital_metric == other.orbital_metric;
  }
};

/** @brief Orbital and response blocks reconstructed from projection caches. */
struct CoupledCachedBlocks {
  Eigen::VectorXd orbital;
  Eigen::VectorXd response;

  /** @brief Concatenates the orbital and response blocks. */
  Eigen::VectorXd packed() const;
};

/**
 * @brief Two-space matrix-free projection cache for coupled Newton models.
 *
 * The orbital basis @f$V@f$ is @f$G@f$-orthonormal and stores @f$AV@f$,
 * @f$BV@f$, and @f$GV@f$. The response basis @f$W@f$ is Euclidean-
 * orthonormal and stores @f$B^TW@f$ and @f$CW@f$. Candidate blocks are filtered
 * only at their floating-point numerical rank; there are no physical or
 * chemistry-specific thresholds.
 *
 * The referenced operator must outlive this cache.
 */
class CoupledProjectionCache {
public:
  explicit CoupledProjectionCache(
      const CoupledNewtonOperator& coupled_operator);

  int orbital_subspace_size() const noexcept;
  int response_subspace_size() const noexcept;

  const Eigen::MatrixXd& orbital_basis() const noexcept;
  const Eigen::MatrixXd& response_basis() const noexcept;
  const Eigen::MatrixXd& orbital_hessian_images() const noexcept;
  const Eigen::MatrixXd& orbital_to_response_images() const noexcept;
  const Eigen::MatrixXd& orbital_metric_images() const noexcept;
  const Eigen::MatrixXd& response_to_orbital_images() const noexcept;
  const Eigen::MatrixXd& response_hessian_images() const noexcept;
  const CoupledActionCounts& action_counts() const noexcept;

  /**
   * @brief Appends the numerically independent part of an orbital block.
   *
   * For a block with at least one accepted column, @f$G@f$, @f$A@f$, and
   * @f$B@f$ are each evaluated exactly once as block actions.
   *
   * @return Number of accepted columns.
   */
  int append_orbital_block(
      const Eigen::Ref<const Eigen::MatrixXd>& candidates);

  /**
   * @brief Appends the numerically independent part of a response block.
   *
   * For a block with at least one accepted column, @f$B^T@f$ and @f$C@f$ are
   * each evaluated exactly once as block actions.
   *
   * @return Number of accepted columns.
   */
  int append_response_block(
      const Eigen::Ref<const Eigen::MatrixXd>& candidates);

  /** @brief Symmetric projected orbital block @f$V^TAV@f$. */
  Eigen::MatrixXd projected_orbital_hessian() const;

  /** @brief Symmetric projected orbital metric @f$V^TGV@f$. */
  Eigen::MatrixXd projected_orbital_metric() const;

  /** @brief Symmetric projected response block @f$W^TCW@f$. */
  Eigen::MatrixXd projected_response_hessian() const;

  /** @brief Projected coupling @f$D=W^TBV@f$. */
  Eigen::MatrixXd projected_coupling() const;

  /** @brief Relative mismatch between @f$W^TBV@f$ and @f$(V^TB^TW)^T@f$. */
  double projected_coupling_adjoint_error() const;

  /**
   * @brief Reconstructs the full coupled image from cached block actions.
   *
   * The optional shift contributes @f$\lambda GVx@f$ only to the orbital
   * block. No matrix-free operator action is performed.
   */
  CoupledCachedBlocks reconstruct_image(
      const Eigen::Ref<const Eigen::VectorXd>& orbital_coordinates,
      const Eigen::Ref<const Eigen::VectorXd>& response_coordinates,
      double orbital_shift = 0.0) const;

  /**
   * @brief Reconstructs a generalized Ritz residual from cached actions.
   *
   * This returns
   * @f[(AVx+B^TWy-\theta GVx,\;BVx+CWy)]@f$.
   */
  CoupledCachedBlocks reconstruct_ritz_residual(
      const Eigen::Ref<const Eigen::VectorXd>& orbital_coordinates,
      const Eigen::Ref<const Eigen::VectorXd>& response_coordinates,
      double ritz_value) const;

private:
  const CoupledNewtonOperator* coupled_operator_ = nullptr;
  Eigen::MatrixXd orbital_basis_;
  Eigen::MatrixXd response_basis_;
  Eigen::MatrixXd orbital_hessian_images_;
  Eigen::MatrixXd orbital_to_response_images_;
  Eigen::MatrixXd orbital_metric_images_;
  Eigen::MatrixXd response_to_orbital_images_;
  Eigen::MatrixXd response_hessian_images_;
  CoupledActionCounts action_counts_;
};

}  // namespace xmvb::vb
