#pragma once

#include <cstddef>
#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"

namespace xmvb::vb::detail {

struct SameSpinPartnerActionPanel {
  int begin = -1;
  int size = 0;
  AcceptedSpinPairActionResult action;
};

/** Accumulates the complete same-spin local HVP from one directional stream. */
class LocalSameSpinTileAccumulator {
 public:
  LocalSameSpinTileAccumulator(
      const SameSpinPairCacheContext& accepted_pair_cache,
      const SelectedStateDeterminantMatrices& selected_states,
      const std::vector<double>& selected_state_energies,
      int n_active_orbitals,
      const std::vector<double>& active_overlap,
      const Eigen::MatrixXd& active_one_electron,
      const ActiveSpaceTwoElectronResult& active_two_electron,
      const ActiveSpaceIntegralDirectionView& direction,
      std::size_t workspace_bytes = kPairTileWorkspaceBytes);

  void consume(
      bool alpha_channel,
      bool beta_channel,
      const AcceptedSpinPairTile& accepted,
      const SameSpinDirectionalPairTileView& tile);

  SameSpinMatrixBackwardContribution finish();

  /** Number of bounded partner panels built by the streamed path. */
  std::size_t partner_panel_build_count() const noexcept {
    return alpha_partner_panel_build_count_ + beta_partner_panel_build_count_;
  }

 private:
  void consume_alpha_primary(
      const AcceptedSpinPairTile& accepted,
      const SameSpinDirectionalPairTileView& tile);
  void consume_beta_primary(
      const AcceptedSpinPairTile& accepted,
      const SameSpinDirectionalPairTileView& tile);
  void accumulate_beta_weight_response(
      const SameSpinDirectionalPairTileView& alpha_tile);
  void accumulate_alpha_weight_response(
      const SameSpinDirectionalPairTileView& beta_tile);

  const SameSpinPairCacheContext& accepted_pair_cache_;
  const SelectedStateDeterminantMatrices& selected_states_;
  const std::vector<double>& selected_state_energies_;
  int n_active_orbitals_ = 0;
  const std::vector<double>& active_overlap_;
  const Eigen::MatrixXd& active_one_electron_;
  const ActiveSpaceTwoElectronResult& active_two_electron_;
  const ActiveSpaceIntegralDirectionView& direction_;
  bool close_shell_same_spin_ = false;
  LocalResponseTilePlan tile_plan_;
  SameSpinPartnerActionPanel alpha_partner_panel_;
  SameSpinPartnerActionPanel beta_partner_panel_;
  std::size_t alpha_partner_panel_build_count_ = 0;
  std::size_t beta_partner_panel_build_count_ = 0;
  SameSpinMatrixBackwardContribution result_;
  Eigen::MatrixXd one_electron_gradient_;
};

}  // namespace xmvb::vb::detail
