#pragma once

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"

namespace xmvb::vb::detail {

/** Accumulates the complete same-spin local HVP from one directional stream. */
class LocalSameSpinTileAccumulator {
 public:
  LocalSameSpinTileAccumulator(
      const SameSpinPairCacheContext& accepted_pair_cache,
      const SelectedStateDeterminantMatrices& selected_states,
      const std::vector<double>& selected_state_energies,
      int n_active_orbitals,
      const Eigen::MatrixXd& active_one_electron,
      const ActiveSpaceTwoElectronResult& active_two_electron,
      const ActiveSpaceIntegralDirectionView& direction);

  void consume(
      bool alpha_channel,
      bool beta_channel,
      const SameSpinDirectionalPairTile& tile);

  SameSpinMatrixBackwardContribution finish();

 private:
  void consume_alpha_primary(const SameSpinDirectionalPairTile& tile);
  void consume_beta_primary(const SameSpinDirectionalPairTile& tile);
  void accumulate_beta_weight_response(
      const SameSpinDirectionalPairTile& alpha_tile);
  void accumulate_alpha_weight_response(
      const SameSpinDirectionalPairTile& beta_tile);

  const SameSpinPairCacheContext& accepted_pair_cache_;
  const SelectedStateDeterminantMatrices& selected_states_;
  const std::vector<double>& selected_state_energies_;
  int n_active_orbitals_ = 0;
  const Eigen::MatrixXd& active_one_electron_;
  const ActiveSpaceTwoElectronResult& active_two_electron_;
  const ActiveSpaceIntegralDirectionView& direction_;
  bool close_shell_same_spin_ = false;
  SameSpinMatrixBackwardContribution result_;
  Eigen::MatrixXd one_electron_gradient_;
};

}  // namespace xmvb::vb::detail
