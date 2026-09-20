#pragma once

#include <functional>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Matrix-free positive-curvature block inverse-BFGS update.
 *
 * For tangent directions `S`, exact Hessian covectors `Y = H S`, and a base
 * inverse action `M`, this class applies
 *
 * @f[
 * M^+=(I-SR Y^T)M(I-YR S^T)+SRS^T,
 * \qquad R=(S^TY)^{-1}.
 * @f]
 *
 * No ambient inverse Hessian is assembled. The update is valid only when the
 * symmetrized block curvature `S.transpose() * Y` is numerically positive
 * definite. Directions with nonpositive curvature belong to the explicit
 * trust-region model instead.
 */
class BlockInverseBfgs {
public:
  using VectorInverseAction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using BlockInverseAction =
      std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

  BlockInverseBfgs(
      Eigen::MatrixXd directions,
      Eigen::MatrixXd hessian_images);

  int dimension() const noexcept;
  int rank() const noexcept;

  Eigen::VectorXd apply(
      const Eigen::VectorXd& covector,
      const VectorInverseAction& base_inverse) const;

  Eigen::MatrixXd apply_block(
      const Eigen::MatrixXd& covectors,
      const BlockInverseAction& base_inverse) const;

private:
  Eigen::MatrixXd solve_curvature(const Eigen::MatrixXd& right_hand_side) const;

  Eigen::MatrixXd directions_;
  Eigen::MatrixXd hessian_images_;
  Eigen::MatrixXd curvature_;
  Eigen::LLT<Eigen::MatrixXd> curvature_factor_;
};

}  // namespace xmvb::vb
