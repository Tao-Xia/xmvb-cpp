#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

class WoodburyCore;

struct WoodburyRiDirection {
  double overlap_determinant = 0.0;
  double hamiltonian = 0.0;
  Eigen::MatrixXd first_cofactor;
  Eigen::MatrixXd hamiltonian_overlap_gradient;
  Eigen::VectorXd accepted_auxiliary;
  Eigen::VectorXd directional_auxiliary;
};

/**
 * @brief RI channels propagated with a stable Woodbury base and singular core.
 *
 * The state remains valid when the represented occupied overlap is singular.
 * It stores `A^Q = K_base M^Q` and contracts the retained dangerous core
 * without forming or inverting the physical overlap inverse.
 */
class WoodburyRiState {
 public:
  WoodburyRiState();
  ~WoodburyRiState();

  bool initialize(
      const std::vector<int>& occupied_left,
      const std::vector<int>& occupied_right,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  bool update_right(
      const std::vector<int>& occupied_right_new,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  bool update_left(
      const std::vector<int>& occupied_left_new,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors);

  bool valid() const noexcept { return core_ != nullptr; }
  int core_rank() const noexcept;
  int overlap_nullity() const noexcept;
  const Eigen::MatrixXd& regular_inverse() const;
  double overlap_determinant() const;
  Eigen::MatrixXd first_cofactor() const;
  Eigen::VectorXd first_cofactor_auxiliary() const;
  double one_electron_contraction(
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron) const;
  double two_electron_contraction() const;
  Eigen::MatrixXd hamiltonian_overlap_gradient(
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) const;
  WoodburyRiDirection hamiltonian_direction(
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron,
      const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
      const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction,
      bool auxiliary_projection) const;

 private:
  using ChannelTable = Eigen::Matrix<
      double,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

  void update_base_channels(
      const Eigen::Ref<const Eigen::MatrixXd>& left,
      const Eigen::Ref<const Eigen::MatrixXd>& right);
  Eigen::Map<Eigen::MatrixXd> channel(Eigen::Index auxiliary);
  Eigen::Map<const Eigen::MatrixXd> channel(Eigen::Index auxiliary) const;

  std::vector<int> occupied_left_;
  std::vector<int> occupied_right_;
  ChannelTable channels_;
  std::unique_ptr<WoodburyCore> core_;
  int n_electrons_ = 0;
};

}  // namespace xmvb::vb
