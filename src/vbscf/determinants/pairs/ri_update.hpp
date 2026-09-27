#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/contracts/types.hpp"

namespace xmvb::vb {

/**
 * @brief Row-local RI channel state for a low-rank right-string traversal.
 *
 * The contiguous workspace stores every contracted channel
 * `A^Q = X^{-1} M^Q`.  It is bounded by one traversal row and never scales as
 * the square of the number of unique strings.
 */
class RiPairUpdateState {
 public:
  bool initialize(
      const std::vector<int>& occupied_left,
      const std::vector<int>& occupied_right,
      const DeterminantOverlapResult& overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
      bool track_response = false);

  bool update_right(
      const std::vector<int>& occupied_left,
      const std::vector<int>& occupied_right_old,
      const std::vector<int>& occupied_right_new,
      const DeterminantOverlapResult& overlap_old,
      const DeterminantOverlapResult& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  /**
   * @brief Propagates regular RI channels after one left-string substitution.
   *
   * A left substitution changes one column of both the occupied overlap and
   * each occupied RI transition block.  The update is a sum of two rank-one
   * channel corrections.  Response tracking is deliberately rejected until
   * its adjoint aggregate has the same residual certificate as the scalar
   * channels.
   */
  bool update_left(
      const std::vector<int>& occupied_left_old,
      const std::vector<int>& occupied_left_new,
      const std::vector<int>& occupied_right,
      const DeterminantOverlapResult& overlap_old,
      const DeterminantOverlapResult& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  void reset();
  bool valid() const noexcept { return valid_; }
  bool tracks_response() const noexcept { return track_response_; }
  double two_electron_phi() const;
  /**
   * @brief Builds the RI image of the first cofactor without reprojecting it.
   *
   * For a regular pair, each stored channel is
   * `A^Q = X^{-1} M^Q`.  The requested auxiliary vector is therefore
   * `g_Q = det(X) tr(A^Q) = (B x)_Q`, where `x` is the packed first-cofactor
   * projection.  The method returns `false` if the propagated channel error
   * does not certify the trace contraction.
   */
  bool first_order_cofactor_auxiliary(
      double overlap_determinant,
      Eigen::Ref<Eigen::VectorXd> auxiliary) const;
  Eigen::MatrixXd two_electron_inverse_overlap_gradient(
      const DeterminantOverlapResult& overlap) const;

 private:
  using ChannelTable = Eigen::Matrix<
      double,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

  std::vector<int> occupied_left_;
  std::vector<int> occupied_right_;
  std::vector<double> channel_error_bounds_;
  ChannelTable channels_;
  Eigen::MatrixXd response_aggregate_;
  double response_roundoff_bound_ = 0.0;
  int n_electrons_ = 0;
  bool track_response_ = false;
  bool valid_ = false;
};

}  // namespace xmvb::vb
