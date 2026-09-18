#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief One equal-weight selected-state cluster in the response coordinates.
 *
 * States with unequal weights must be placed in different clusters. The
 * positive weight is the weight of each state, not the total cluster weight.
 */
struct SelectedStateCluster {
  int n_states = 0;
  double state_weight = 0.0;
};

/** @brief Coefficient and multiplier matrices for one selected-state cluster. */
struct ClusterResponse {
  Eigen::MatrixXd coefficients;
  Eigen::MatrixXd multipliers;
};

/**
 * @brief Packs horizontal selected-subspace responses without fixing a root gauge.
 *
 * For a cluster with @f$m@f$ selected states and @f$n_s@f$ structure
 * coefficients, the block contains a coefficient response
 * @f$Z\in\mathbb{R}^{n_s\times m}@f$ followed by a full multiplier matrix
 * @f$M\in\mathbb{R}^{m\times m}@f$. The full multiplier is required by the
 * horizontal constraint
 *
 * @f[
 * C^T S Z=-\frac{1}{2}C^T(\delta S)C,
 * @f]
 *
 * and avoids treating rotations inside an equal-weight selected subspace as
 * physical response degrees of freedom. A one-state cluster reduces to the
 * usual bordered eigenvector response with one scalar multiplier.
 *
 * Matrices are packed in Eigen's column-major order. `pack` maps the physical
 * response matrices to symmetric coupled Newton coordinates
 * @f$\sqrt{2w}[Z;M]@f$, where @f$w@f$ is the per-state weight stored in
 * SelectedStateCluster; `unpack` applies the inverse scaling.
 */
class SelectedSubspaceResponseLayout {
public:
  SelectedSubspaceResponseLayout(
      int n_structures,
      std::vector<SelectedStateCluster> clusters);

  int n_structures() const noexcept;
  int n_clusters() const noexcept;
  int response_size() const noexcept;
  const SelectedStateCluster& cluster(int cluster_index) const;
  int coefficient_offset(int cluster_index) const;
  int multiplier_offset(int cluster_index) const;
  int cluster_size(int cluster_index) const;
  double coordinate_scale(int cluster_index) const;

  Eigen::VectorXd pack(
      const std::vector<ClusterResponse>& cluster_responses) const;
  std::vector<ClusterResponse> unpack(
      const Eigen::Ref<const Eigen::VectorXd>& response) const;

private:
  int n_structures_ = 0;
  int response_size_ = 0;
  std::vector<SelectedStateCluster> clusters_;
  std::vector<int> coefficient_offsets_;
  std::vector<int> multiplier_offsets_;
};

/**
 * @brief Batched linear action used by the coupled Newton block operator.
 *
 * Every column is an independent direction. Implementations must return a
 * matrix with the advertised output row count and the same column count.
 */
using CoupledBlockAction = std::function<Eigen::MatrixXd(
    const Eigen::Ref<const Eigen::MatrixXd>&)>;

/**
 * @brief Four matrix-free blocks of the orbital--structure Newton system.
 *
 * The callbacks represent @f$A@f$, @f$B@f$, @f$B^T@f$, and @f$C@f$ in
 *
 * @f[
 * \begin{pmatrix}A+\lambda G&B^T\\B&C\end{pmatrix}
 * \begin{pmatrix}p\\q\end{pmatrix}.
 * @f]
 *
 * Here @f$p@f$ is an orbital step and @f$q@f$ is a packed, weight-scaled
 * selected-subspace response. `orbital_metric` is @f$G@f$; an empty callback
 * means the identity. `orbital_to_response` must include both the directional
 * generalized-eigen forcing and the directional horizontal-normalization
 * constraint. `response_to_orbital` must be its exact adjoint in these scaled
 * coordinates.
 */
struct CoupledNewtonActions {
  CoupledBlockAction orbital_hessian;
  CoupledBlockAction orbital_to_response;
  CoupledBlockAction response_to_orbital;
  CoupledBlockAction response_hessian;
  CoupledBlockAction orbital_metric;
};

/**
 * @brief Matrix-free coupled orbital--selected-subspace Newton operator.
 *
 * This class owns no Hessian or structure matrix. It validates and composes
 * the four independently supplied matrix-free block actions. Trust-region
 * regularization acts only on the orbital block; response coordinates are
 * induced first-order variables and are never included in the trust norm.
 */
class CoupledNewtonOperator {
public:
  CoupledNewtonOperator(
      int n_orbital_coordinates,
      SelectedSubspaceResponseLayout response_layout,
      CoupledNewtonActions actions);

  int n_orbital_coordinates() const noexcept;
  int n_response_coordinates() const noexcept;
  int size() const noexcept;
  const SelectedSubspaceResponseLayout& response_layout() const noexcept;

  Eigen::VectorXd apply(
      const Eigen::Ref<const Eigen::VectorXd>& direction,
      double orbital_shift = 0.0) const;
  Eigen::MatrixXd apply_block(
      const Eigen::Ref<const Eigen::MatrixXd>& directions,
      double orbital_shift = 0.0) const;

  Eigen::MatrixXd apply_orbital_hessian(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;
  Eigen::MatrixXd apply_orbital_to_response(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;
  Eigen::MatrixXd apply_response_to_orbital(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;
  Eigen::MatrixXd apply_response_hessian(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;
  Eigen::MatrixXd apply_orbital_metric(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;

  /** @brief Relative bilinear mismatch between @f$B@f$ and @f$B^T@f$. */
  double coupling_adjoint_error(
      const Eigen::Ref<const Eigen::VectorXd>& orbital_direction,
      const Eigen::Ref<const Eigen::VectorXd>& response_direction) const;

private:
  Eigen::MatrixXd apply_checked(
      const CoupledBlockAction& action,
      const Eigen::Ref<const Eigen::MatrixXd>& directions,
      int input_rows,
      int output_rows,
      const char* label) const;

  int n_orbital_coordinates_ = 0;
  SelectedSubspaceResponseLayout response_layout_;
  CoupledNewtonActions actions_;
};

}  // namespace xmvb::vb
