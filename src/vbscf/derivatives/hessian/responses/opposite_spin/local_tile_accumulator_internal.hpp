#pragma once

#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph_internal.hpp"

namespace xmvb::vb::detail {

/** Accumulates the complete local opposite-spin adjoint from pair tiles. */
class LocalOppositeSpinTileAccumulator {
 public:
  LocalOppositeSpinTileAccumulator(
      const SameSpinPairCacheContext& accepted_pair_cache,
      const SelectedStateDeterminantMatrices& selected_states,
      int n_active_orbitals);

  void consume(
      bool alpha_channel,
      bool beta_channel,
      const DirectionalOppositeSpinPairTileView& tile);

  OppositeSpinBackwardContribution finish();

 private:
  void accumulate_primary(
      PrimarySpin spin,
      const DirectionalOppositeSpinPairTileView& tile,
      bool accumulate_packed_gradient);
  void accumulate_cross_response(
      PrimarySpin target_spin,
      const DirectionalOppositeSpinPairTileView& partner_tile);

  const SameSpinPairCacheContext& accepted_pair_cache_;
  const SelectedStateDeterminantMatrices& selected_states_;
  int n_active_orbitals_ = 0;
  int n_packed_pairs_ = 0;
  SelectedStatePairGraph alpha_graph_;
  SelectedStatePairGraph beta_graph_;
  OppositeSpinBackwardContribution result_;
};

}  // namespace xmvb::vb::detail
