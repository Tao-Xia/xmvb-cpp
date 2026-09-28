#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Batched RI specialization of the contracted-density hierarchy.
 *
 * For every auxiliary channel `A_Q = K M_Q`, the state stores
 * `B_Q = A_Q K` and `C_Q = A_Q^2 K`.  The second exterior contraction and
 * its overlap adjoint then follow from `e_2(A_Q)` and
 * `J_Q = tr(A_Q) B_Q - C_Q`, without a dense cofactor reconstruction.
 * Low-rank graph edges update all tables with dense-thin products only.
 */
class RiContractedMoments {
 public:
  using Table = Eigen::Matrix<
      double,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

  void initialize(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
      const Eigen::Ref<const Table>& channels);

  /**
   * @brief Applies a shared inverse update and per-channel low-rank edges.
   *
   * Each factor-table row is one flattened column-major `n x r` matrix:
   * `K' = K + L_K R_K^T` and `A_Q' = A_Q + L_Q R_Q^T`.
   */
  void update(
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
      const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
      const Eigen::Ref<const Table>& channel_left,
      const Eigen::Ref<const Table>& channel_right);

  int dimension() const noexcept { return inverse_overlap_.rows(); }
  Eigen::Index channel_count() const noexcept { return channels_.rows(); }
  const Eigen::MatrixXd& inverse_overlap() const noexcept {
    return inverse_overlap_;
  }
  const Table& channels() const noexcept { return channels_; }
  Eigen::Map<const Eigen::MatrixXd> channel(Eigen::Index auxiliary) const;

  double second_coefficient_sum() const noexcept {
    return second_coefficient_sum_;
  }
  const Eigen::MatrixXd& second_response_moment() const noexcept {
    return second_response_moment_;
  }

 private:
  Eigen::Map<const Eigen::MatrixXd> first_moment(
      Eigen::Index auxiliary) const;
  Eigen::Map<const Eigen::MatrixXd> second_moment(
      Eigen::Index auxiliary) const;
  void rebuild_aggregates();

  Eigen::MatrixXd inverse_overlap_;
  Table channels_;
  Table first_moments_;
  Table second_moments_;
  Eigen::MatrixXd second_response_moment_;
  double second_coefficient_sum_ = 0.0;
};

}  // namespace xmvb::vb
