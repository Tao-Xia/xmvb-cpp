#include "vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator_internal.hpp"

#include <stdexcept>
#include <utility>

namespace xmvb::vb::detail {
namespace {

void add_vector(
    const std::vector<double>& source,
    std::vector<double>* target) {
  if (source.size() != target->size()) {
    throw std::logic_error(
        "local pair-tile adjoint channels have inconsistent dimensions");
  }
  for (std::size_t entry = 0; entry < source.size(); ++entry) {
    (*target)[entry] += source[entry];
  }
}

}  // namespace

LocalActiveSpaceTileAccumulator::LocalActiveSpaceTileAccumulator(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point,
    const ActiveSpaceIntegralDirectionView& direction)
    : same_spin_(
          accepted_point.same_spin_pair_cache,
          accepted_point.selected_state_matrices,
          accepted_point.selected_state_energies,
          input.orbital_preparation_input.n_active_orbitals,
          accepted_point.prepared_active_space
              .active_space_one_electron_result.h1e_act,
          accepted_point.prepared_active_space
              .active_space_two_electron_result,
          direction),
      opposite_spin_(
          accepted_point.same_spin_pair_cache,
          accepted_point.selected_state_matrices,
          input.orbital_preparation_input.n_active_orbitals) {
  if (!accepted_point.use_pair_graph_opposite_spin_adjoint) {
    throw std::runtime_error(
        "local pair-tile adjoint requires selected-state pair graphs");
  }
}

void LocalActiveSpaceTileAccumulator::consume(
    bool alpha_channel,
    bool beta_channel,
    const SameSpinDirectionalPairTileView& same_spin,
    const DirectionalOppositeSpinPairTileView* opposite_spin) {
  if (opposite_spin == nullptr) {
    throw std::logic_error(
        "local pair-tile adjoint requires opposite-spin channels");
  }
  same_spin_.consume(alpha_channel, beta_channel, same_spin);
  opposite_spin_.consume(alpha_channel, beta_channel, *opposite_spin);
}

ActiveSpaceGradientDirection LocalActiveSpaceTileAccumulator::finish() {
  SameSpinMatrixBackwardContribution same = same_spin_.finish();
  OppositeSpinBackwardContribution opposite = opposite_spin_.finish();
  ActiveSpaceGradientDirection result;
  result.active_orbital_overlap_gradient =
      std::move(same.active_orbital_overlap_gradient);
  result.active_one_electron_gradient =
      std::move(same.active_one_electron_gradient);
  result.packed_active_two_electron_gradient =
      std::move(same.packed_active_two_electron_gradient);
  add_vector(
      opposite.active_orbital_overlap_gradient,
      &result.active_orbital_overlap_gradient);
  add_vector(
      opposite.packed_active_two_electron_gradient,
      &result.packed_active_two_electron_gradient);
  return result;
}

}  // namespace xmvb::vb::detail
