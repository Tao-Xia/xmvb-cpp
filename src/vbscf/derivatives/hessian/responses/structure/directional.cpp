#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"

namespace xmvb::vb {
namespace {

void require_finite(const Eigen::MatrixXd& values, const char* label) {
  if (!values.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite values");
  }
}

struct ChannelEntry {
  int row = 0;
  int column = 0;
  double value = 0.0;
};

/** Channel-major exact nonzeros used to stream one packed-pair channel. */
using SparseChannels = std::vector<std::vector<ChannelEntry>>;

SparseChannels tile_channels(
    const AcceptedSpinPairTile& tile,
    int n_pairs,
    bool local_indices) {
  SparseChannels channels(n_pairs);
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const auto& projection = tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      if (projection.packed_pair_indices.size() !=
          projection.packed_pair_values.size()) {
        throw std::invalid_argument(
            "packed-pair projection indices and values differ in size");
      }
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        const int packed_pair = projection.packed_pair_indices[entry];
        if (packed_pair < 0 || packed_pair >= n_pairs) {
          throw std::out_of_range(
              "packed-pair projection index is out of range");
        }
        channels[packed_pair].push_back(ChannelEntry{
            local_indices ? left : tile.left_begin + left,
            local_indices ? right : tile.right_begin + right,
            projection.packed_pair_values[entry]});
      }
    }
  }
  return channels;
}

Eigen::MatrixXd pack_selected_coefficients(
    const SelectedStateDeterminantMatrices& selected_states,
    bool alpha_action) {
  const int columns_per_state = alpha_action
      ? selected_states.n_unique_beta
      : selected_states.n_unique_alpha;
  const int rows = alpha_action
      ? selected_states.n_unique_alpha
      : selected_states.n_unique_beta;
  Eigen::MatrixXd packed(
      rows,
      columns_per_state *
          static_cast<int>(selected_states.states.size()));
  for (std::size_t state = 0;
       state < selected_states.states.size();
       ++state) {
    packed.middleCols(
        static_cast<int>(state) * columns_per_state,
        columns_per_state) = alpha_action
        ? selected_states.states[state].coefficient_matrix
        : selected_states.states[state].coefficient_matrix.transpose();
  }
  return packed;
}

}  // namespace

struct AcceptedStructureResponseFactors {
  SelectedStateDeterminantMatrices selected_states;
  AcceptedSpinPairActionResult alpha_action;
  AcceptedSpinPairActionResult beta_action;
};

std::shared_ptr<const AcceptedStructureResponseFactors>
build_accepted_structure_response_factors(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point,
    const Eigen::MatrixXd& selected_columns) {
  const auto& same_spin = accepted_point.same_spin_pair_cache;
  if (!same_spin.enabled()) {
    throw std::invalid_argument(
        "accepted structure response requires the same-spin cache");
  }
  auto factors = std::make_shared<AcceptedStructureResponseFactors>();
  factors->selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          input.structure_data,
          selected_columns,
          accepted_point.selected_state_indices,
          accepted_point.normalized_state_weights,
          same_spin);
  const auto& prepared = accepted_point.prepared_active_space;
  const auto& overlap = prepared.orbital_result.active_orbital_overlap_matrix;
  const auto& h1e = prepared.active_space_one_electron_result.h1e_act;
  const auto& two_electron = prepared.active_space_two_electron_result;
  factors->alpha_action = apply_accepted_spin_pair_action(
      same_spin.alpha_provider(),
      overlap,
      h1e,
      two_electron,
      pack_selected_coefficients(factors->selected_states, true));
  factors->beta_action = apply_accepted_spin_pair_action(
      same_spin.beta_provider(),
      overlap,
      h1e,
      two_electron,
      pack_selected_coefficients(factors->selected_states, false));
  return factors;
}

SelectedStateDirectionalStructureImages
build_selected_structure_direction(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  if (accepted.input == nullptr ||
      accepted.accepted_point_context == nullptr) {
    throw std::invalid_argument(
        "factorized structure direction requires a complete accepted context");
  }
  const auto& input = *accepted.input;
  const auto& accepted_point = *accepted.accepted_point_context;
  const auto& same_spin = accepted_point.same_spin_pair_cache;

  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  const int n_alpha = static_cast<int>(
      same_spin.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      same_spin.beta_reuse_table.unique_determinants.size());
  if ((accepted_ri_active_pair_factors == nullptr) !=
      (directional_ri_active_pair_factors == nullptr)) {
    throw std::invalid_argument(
        "structure RI direction requires both accepted and directional factors");
  }
  if (accepted_ri_active_pair_factors != nullptr &&
      (accepted_ri_active_pair_factors->cols() != n_pairs ||
       directional_ri_active_pair_factors->rows() !=
           accepted_ri_active_pair_factors->rows() ||
       directional_ri_active_pair_factors->cols() != n_pairs)) {
    throw std::invalid_argument(
        "structure RI direction factor dimensions are inconsistent");
  }
  const StructureAction* structure_action = accepted
      .selected_state_eigen_response_operator.structure_action;
  if (structure_action == nullptr) {
    throw std::invalid_argument(
        "factorized structure direction requires the structure action");
  }
  if (!structure_action->supports_integral_direction()) {
    throw std::logic_error(
        "factorized structure directions require the pair-tile sweep");
  }
  if (!accepted_point.structure_adjoint_state.has_value()) {
    accepted_point.structure_adjoint_state =
        structure_action->prepare_active_adjoint(
            accepted_point.selected_state_matrices,
            accepted_point.selected_state_energies);
  }
  StructureIntegralDirection direct_ci_direction;
  const StructureActionResult direct_direction =
      structure_action->apply_integral_direction(
          *accepted_point.structure_adjoint_state,
          direction.overlap,
          direction.one_electron,
          direction.packed_two_electron,
          &direct_ci_direction);
  require_finite(
      direct_direction.hamiltonian,
      "orthogonal direct-CI directional Hamiltonian images");
  require_finite(
      direct_direction.overlap,
      "orthogonal direct-CI directional overlap images");
  return SelectedStateDirectionalStructureImages{
      direct_direction.hamiltonian,
      direct_direction.overlap,
      std::move(direct_ci_direction)};
}

SelectedStateDirectionalStructureImages
build_selected_structure_direction_from_pair_tiles(
    const AcceptedOuterResponseContext& accepted,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    const detail::DirectionalPairTileConsumer& additional_consumer) {
  if (accepted.input == nullptr ||
      accepted.accepted_point_context == nullptr) {
    throw std::invalid_argument(
        "tiled structure direction requires a complete accepted context");
  }
  const auto& input = *accepted.input;
  const auto& accepted_point = *accepted.accepted_point_context;
  const StructureAction* structure_action =
      accepted.selected_state_eigen_response_operator.structure_action;
  if (structure_action == nullptr) {
    throw std::invalid_argument(
        "tiled structure direction requires the structure action");
  }
  if (structure_action->supports_integral_direction()) {
    return build_selected_structure_direction(
        accepted,
        direction,
        accepted_ri_active_pair_factors,
        directional_ri_active_pair_factors);
  }
  if (!accepted_point.same_spin_pair_cache.enabled() ||
      accepted.structure_factors == nullptr) {
    throw std::invalid_argument(
        "tiled factorized structure direction requires accepted pair factors");
  }

  const auto& same_spin = accepted_point.same_spin_pair_cache;
  const auto& factors = *accepted.structure_factors;
  const auto& selected_states = factors.selected_states;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  const int n_alpha = selected_states.n_unique_alpha;
  const int n_beta = selected_states.n_unique_beta;
  const int n_states = static_cast<int>(selected_states.states.size());
  const auto& prepared = accepted_point.prepared_active_space;
  const auto& accepted_overlap =
      prepared.orbital_result.active_orbital_overlap_matrix;
  const auto& accepted_h1e =
      prepared.active_space_one_electron_result.h1e_act;
  const auto& accepted_two_electron =
      prepared.active_space_two_electron_result;
  const detail::PairTileExtents tile_extents =
      detail::plan_pair_tile_extents(
          same_spin,
          n_active_orbitals,
          accepted_two_electron,
          true);

  Eigen::MatrixXd delta_hamiltonian_selected =
      Eigen::MatrixXd::Zero(structure_action->n_structures(), n_states);
  Eigen::MatrixXd delta_overlap_selected =
      Eigen::MatrixXd::Zero(structure_action->n_structures(), n_states);
  detail::stream_directional_pair_tiles(
      same_spin,
      n_active_orbitals,
      accepted_point.prepared_active_space.orbital_result
          .active_orbital_overlap_matrix,
      accepted_point.prepared_active_space.active_space_one_electron_result
          .h1e_act,
      accepted_point.prepared_active_space.active_space_two_electron_result,
      direction,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      true,
      [&](bool alpha_channel,
          bool beta_channel,
          const AcceptedSpinPairTile& accepted_tile,
          const detail::SameSpinDirectionalPairTileView& same_tile,
          const detail::DirectionalOppositeSpinPairTileView* opposite_tile) {
        if (opposite_tile == nullptr) {
          throw std::logic_error(
              "tiled structure direction is missing opposite-spin data");
        }
        const Eigen::MatrixXd delta_hamiltonian =
            same_tile.delta_regular_hamiltonian() +
            same_tile.delta_singular_hamiltonian();

        if (alpha_channel) {
          Eigen::MatrixXd tile_hamiltonian = Eigen::MatrixXd::Zero(
              same_tile.left_size(), n_states * n_beta);
          Eigen::MatrixXd tile_overlap = Eigen::MatrixXd::Zero(
              same_tile.left_size(), n_states * n_beta);
          for (int state = 0; state < n_states; ++state) {
            const Eigen::MatrixXd& coefficients =
                selected_states.states[state].coefficient_matrix;
            const int action_begin = state * n_alpha;
            const Eigen::MatrixXd beta_overlap_image =
                factors.beta_action.overlap
                    .middleCols(action_begin, n_alpha).transpose();
            const Eigen::MatrixXd beta_hamiltonian_image =
                factors.beta_action.hamiltonian
                    .middleCols(action_begin, n_alpha).transpose();
            auto overlap_image = tile_overlap.block(
                0,
                state * n_beta,
                same_tile.left_size(),
                n_beta);
            overlap_image.noalias() += same_tile.delta_overlap() *
                beta_overlap_image.middleRows(
                    same_tile.right_begin(), same_tile.right_size());
            auto hamiltonian_image = tile_hamiltonian.block(
                0,
                state * n_beta,
                same_tile.left_size(),
                n_beta);
            hamiltonian_image.noalias() += delta_hamiltonian *
                beta_overlap_image.middleRows(
                    same_tile.right_begin(), same_tile.right_size());
            hamiltonian_image.noalias() += same_tile.delta_overlap() *
                beta_hamiltonian_image.middleRows(
                    same_tile.right_begin(), same_tile.right_size());

            for (int beta_left = 0;
                 beta_left < n_beta;
                 beta_left += tile_extents.beta) {
              const int beta_left_end = std::min(
                  n_beta, beta_left + tile_extents.beta);
              for (int beta_right = 0;
                   beta_right < n_beta;
                   beta_right += tile_extents.beta) {
                const int beta_right_end = std::min(
                    n_beta, beta_right + tile_extents.beta);
                const AcceptedSpinPairTile beta_tile =
                    same_spin.beta_provider().build(
                        beta_left,
                        beta_left_end,
                        beta_right,
                        beta_right_end,
                        accepted_overlap,
                        accepted_h1e,
                        accepted_two_electron,
                        AcceptedPairTileBuildOptions{
                            .materialize_projected_pair_values = false,
                            .populate_response_payload = false});
                const SparseChannels channels = tile_channels(
                    beta_tile, n_pairs, true);
                const auto coefficients_right = coefficients.block(
                    same_tile.right_begin(),
                    beta_right,
                    same_tile.right_size(),
                    beta_tile.right_size);
                for (int target = 0; target < n_pairs; ++target) {
                  const auto projected =
                      opposite_tile->projected_channel(target);
                  if (channels[target].empty() || projected.isZero(0.0)) {
                    continue;
                  }
                  const Eigen::MatrixXd projected_coefficients =
                      projected * coefficients_right;
                  for (const ChannelEntry& entry : channels[target]) {
                    hamiltonian_image.col(beta_left + entry.row).noalias() +=
                        entry.value *
                        projected_coefficients.col(entry.column);
                  }
                }
              }
            }
          }
          structure_action->add_spin_product_tile(
              tile_hamiltonian,
              same_tile.left_begin(),
              0,
              &delta_hamiltonian_selected);
          structure_action->add_spin_product_tile(
              tile_overlap,
              same_tile.left_begin(),
              0,
              &delta_overlap_selected);
        }

        if (beta_channel) {
          Eigen::MatrixXd tile_hamiltonian = Eigen::MatrixXd::Zero(
              n_alpha, n_states * same_tile.left_size());
          Eigen::MatrixXd tile_overlap = Eigen::MatrixXd::Zero(
              n_alpha, n_states * same_tile.left_size());
          for (int state = 0; state < n_states; ++state) {
            const Eigen::MatrixXd& coefficients =
                selected_states.states[state].coefficient_matrix;
            const int action_begin = state * n_beta;
            const auto alpha_overlap_image =
                factors.alpha_action.overlap.middleCols(
                    action_begin, n_beta);
            const auto alpha_hamiltonian_image =
                factors.alpha_action.hamiltonian.middleCols(
                    action_begin, n_beta);
            auto overlap_image = tile_overlap.middleCols(
                state * same_tile.left_size(),
                same_tile.left_size());
            overlap_image.noalias() += alpha_overlap_image.middleCols(
                same_tile.right_begin(), same_tile.right_size()) *
                same_tile.delta_overlap().transpose();
            auto hamiltonian_image = tile_hamiltonian.middleCols(
                state * same_tile.left_size(),
                same_tile.left_size());
            hamiltonian_image.noalias() +=
                alpha_hamiltonian_image.middleCols(
                    same_tile.right_begin(), same_tile.right_size()) *
                same_tile.delta_overlap().transpose();
            hamiltonian_image.noalias() +=
                alpha_overlap_image.middleCols(
                    same_tile.right_begin(), same_tile.right_size()) *
                delta_hamiltonian.transpose();

            for (int alpha_left = 0;
                 alpha_left < n_alpha;
                 alpha_left += tile_extents.alpha) {
              const int alpha_left_end = std::min(
                  n_alpha, alpha_left + tile_extents.alpha);
              for (int alpha_right = 0;
                   alpha_right < n_alpha;
                   alpha_right += tile_extents.alpha) {
                const int alpha_right_end = std::min(
                    n_alpha, alpha_right + tile_extents.alpha);
                const AcceptedSpinPairTile alpha_tile =
                    same_spin.alpha_provider().build(
                        alpha_left,
                        alpha_left_end,
                        alpha_right,
                        alpha_right_end,
                        accepted_overlap,
                        accepted_h1e,
                        accepted_two_electron,
                        AcceptedPairTileBuildOptions{
                            .materialize_projected_pair_values = false,
                            .populate_response_payload = false});
                const SparseChannels channels = tile_channels(
                    alpha_tile, n_pairs, true);
                const auto coefficients_right = coefficients.block(
                    alpha_right,
                    same_tile.right_begin(),
                    alpha_tile.right_size,
                    same_tile.right_size());
                for (int target = 0; target < n_pairs; ++target) {
                  const auto directional =
                      opposite_tile->projected_channel(target);
                  if (channels[target].empty() || directional.isZero(0.0)) {
                    continue;
                  }
                  const Eigen::MatrixXd directional_coefficients =
                      coefficients_right * directional.transpose();
                  for (const ChannelEntry& entry : channels[target]) {
                    hamiltonian_image.row(
                        alpha_left + entry.row).noalias() +=
                        entry.value *
                        directional_coefficients.row(entry.column);
                  }
                }
              }
            }
          }
          structure_action->add_spin_product_tile(
              tile_hamiltonian,
              0,
              same_tile.left_begin(),
              &delta_hamiltonian_selected);
          structure_action->add_spin_product_tile(
              tile_overlap,
              0,
              same_tile.left_begin(),
              &delta_overlap_selected);
        }
        if (additional_consumer) {
          additional_consumer(
              alpha_channel,
              beta_channel,
              accepted_tile,
              same_tile,
              opposite_tile);
        }
      });

  SelectedStateDirectionalStructureImages result;
  result.delta_hamiltonian_selected = std::move(delta_hamiltonian_selected);
  result.delta_overlap_selected = std::move(delta_overlap_selected);
  require_finite(
      result.delta_hamiltonian_selected,
      "tiled directional Hamiltonian images");
  require_finite(
      result.delta_overlap_selected,
      "tiled directional overlap images");
  return result;
}

}  // namespace xmvb::vb
