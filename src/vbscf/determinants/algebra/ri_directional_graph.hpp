#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

class ContractedDensityJet;

/** Result of one regular RI auxiliary tile at a string-pair vertex. */
struct RiDirectionalTileValue {
  double two_electron = 0.0;
  double two_electron_direction = 0.0;
  Eigen::MatrixXd overlap_gradient_direction;
  Eigen::VectorXd accepted_first_contractions;
  Eigen::VectorXd directional_first_contractions;
};

/**
 * @brief Exact directional contracted-density traversal for one RI tile.
 *
 * One anchor performs the dense occupied-space products.  Every subsequent
 * right-string single-site substitution differentiates the same rank-one
 * Woodbury edge and updates all contracted-density jets in quadratic work.
 */
class RiDirectionalGraph {
 public:
  RiDirectionalGraph();
  ~RiDirectionalGraph();
  RiDirectionalGraph(RiDirectionalGraph&&) noexcept;
  RiDirectionalGraph& operator=(RiDirectionalGraph&&) noexcept;
  RiDirectionalGraph(const RiDirectionalGraph&) = delete;
  RiDirectionalGraph& operator=(const RiDirectionalGraph&) = delete;

  bool initialize(
      const std::vector<int>& occupied_left,
      const std::vector<int>& occupied_right,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction);

  bool update_right(
      const std::vector<int>& occupied_right_new,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction);

  bool valid() const noexcept { return !jets_.empty(); }
  RiDirectionalTileValue value() const;

 private:
  std::vector<int> occupied_left_;
  std::vector<int> occupied_right_;
  Eigen::MatrixXd overlap_;
  Eigen::MatrixXd overlap_direction_;
  Eigen::MatrixXd inverse_;
  Eigen::MatrixXd inverse_direction_;
  std::vector<std::unique_ptr<ContractedDensityJet>> jets_;
  double determinant_ = 0.0;
  double determinant_direction_ = 0.0;
};

}  // namespace xmvb::vb
