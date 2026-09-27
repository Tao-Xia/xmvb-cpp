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
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  bool update_right(
      const std::vector<int>& occupied_left,
      const std::vector<int>& occupied_right_old,
      const std::vector<int>& occupied_right_new,
      const DeterminantOverlapResult& overlap_old,
      const DeterminantOverlapResult& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  void reset();
  bool valid() const noexcept { return valid_; }
  double two_electron_phi() const;

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
  int n_electrons_ = 0;
  bool valid_ = false;
};

}  // namespace xmvb::vb
