#pragma once

#include "vbscf/derivatives/hessian/responses/active_space/outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/local_tile_accumulator_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/local_tile_accumulator_internal.hpp"

namespace xmvb::vb::detail {

/** Fuses same-spin and opposite-spin local adjoints in one pair-tile sweep. */
class LocalActiveSpaceTileAccumulator {
 public:
  LocalActiveSpaceTileAccumulator(
      const VbScfInput& input,
      const AcceptedPointContext& accepted_point,
      const ActiveSpaceIntegralDirectionView& direction);

  void consume(
      bool alpha_channel,
      bool beta_channel,
      const SameSpinDirectionalPairTile& same_spin,
      const DirectionalOppositeSpinPairTile* opposite_spin);

  ActiveSpaceGradientDirection finish();

 private:
  LocalSameSpinTileAccumulator same_spin_;
  LocalOppositeSpinTileAccumulator opposite_spin_;
};

}  // namespace xmvb::vb::detail
