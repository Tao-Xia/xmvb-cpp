#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream_internal.hpp"

namespace xmvb::vb {
namespace {

void require_finite(const Eigen::MatrixXd& values, const char* label) {
  if (!values.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite values");
  }
}

Eigen::MatrixXd pair_scalar_matrix(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    bool hamiltonian) {
  if (pair_cache.size() !=
      static_cast<std::size_t>(n_unique) * n_unique) {
    throw std::invalid_argument(
        "same-spin pair cache has inconsistent dimensions");
  }
  Eigen::MatrixXd result(n_unique, n_unique);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& pair = pair_cache[ordered_spin_pair_storage_index(
          left, right, n_unique)];
      result(left, right) = hamiltonian
          ? pair.total_hamiltonian
          : pair.overlap_result.overlap_determinant;
    }
  }
  return result;
}

struct ChannelEntry {
  int row = 0;
  int column = 0;
  double value = 0.0;
};

/** Channel-major exact nonzeros used to stream one packed-pair channel. */
using SparseChannels = std::vector<std::vector<ChannelEntry>>;

template <typename ProjectionProvider>
SparseChannels index_channels(
    int n_unique,
    int n_pairs,
    ProjectionProvider&& projection_at) {
  SparseChannels channels(n_pairs);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& projection = projection_at(left, right);
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
            left,
            right,
            projection.packed_pair_values[entry]});
      }
    }
  }
  return channels;
}

SparseChannels accepted_channels(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    int n_pairs) {
  return index_channels(
      n_unique,
      n_pairs,
      [&](int left, int right) -> const OppositeSpinPackedPairProjection& {
        return pair_cache[ordered_spin_pair_storage_index(
            left, right, n_unique)]
            .opposite_spin_pair_cache.first_order_cofactor_projection;
      });
}

void add_sparse_right_transpose(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const std::vector<ChannelEntry>& right,
    Eigen::Ref<Eigen::MatrixXd> output) {
  for (const ChannelEntry& entry : right) {
    output.col(entry.row).noalias() +=
        entry.value * left.col(entry.column);
  }
}

void add_projected_left_product(
    const std::vector<SpinDeterminantPairEvaluation>& accepted_pairs,
    int n_unique,
    int target_channel,
    const Eigen::Ref<const Eigen::MatrixXd>& right,
    Eigen::MatrixXd* output) {
  if (output == nullptr || right.rows() != n_unique) {
    throw std::invalid_argument(
        "accepted projected channel product has inconsistent dimensions");
  }
  output->setZero(n_unique, right.cols());
  for (int left = 0; left < n_unique; ++left) {
    for (int contracted = 0; contracted < n_unique; ++contracted) {
      const auto& projected = accepted_pairs[
          ordered_spin_pair_storage_index(
              std::min(left, contracted),
              std::max(left, contracted),
              n_unique)]
                                  .opposite_spin_pair_cache
                                  .first_order_cofactor_projection
                                  .projected_pair_values;
      if (target_channel >= static_cast<int>(projected.size())) {
        throw std::logic_error(
            "accepted opposite-spin channel is missing its kernel image");
      }
      const double value = projected[target_channel];
      if (value != 0.0) {
        output->row(left).noalias() += value * right.row(contracted);
      }
    }
  }
}

}  // namespace

struct AcceptedStructureResponseFactors {
  SelectedStateDeterminantMatrices selected_states;
  Eigen::MatrixXd alpha_overlap;
  Eigen::MatrixXd alpha_hamiltonian;
  Eigen::MatrixXd beta_overlap;
  Eigen::MatrixXd beta_hamiltonian;
  SparseChannels alpha_channels;
  SparseChannels beta_channels;
  bool beta_channels_reuse_alpha = false;
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
  const int n_alpha = static_cast<int>(
      same_spin.alpha_reuse_table.unique_determinants.size());
  const int n_beta = static_cast<int>(
      same_spin.beta_reuse_table.unique_determinants.size());
  const int n_pairs = packed_active_pair_count(
      input.orbital_preparation_input.n_active_orbitals);

  auto factors = std::make_shared<AcceptedStructureResponseFactors>();
  factors->selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          input.structure_data,
          selected_columns,
          accepted_point.selected_state_indices,
          accepted_point.normalized_state_weights,
          same_spin);
  const auto& alpha_cache = same_spin.alpha_pair_cache_ref();
  const auto& beta_cache = same_spin.beta_pair_cache_ref();
  factors->alpha_overlap = pair_scalar_matrix(alpha_cache, n_alpha, false);
  factors->alpha_hamiltonian = pair_scalar_matrix(alpha_cache, n_alpha, true);
  factors->beta_overlap = pair_scalar_matrix(beta_cache, n_beta, false);
  factors->beta_hamiltonian = pair_scalar_matrix(beta_cache, n_beta, true);
  factors->alpha_channels = accepted_channels(alpha_cache, n_alpha, n_pairs);
  factors->beta_channels_reuse_alpha =
      same_spin.shares_same_spin_pair_cache_between_spins();
  if (!factors->beta_channels_reuse_alpha) {
    factors->beta_channels = accepted_channels(beta_cache, n_beta, n_pairs);
  }
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
  const auto& beta_channels = factors.beta_channels_reuse_alpha
      ? factors.alpha_channels
      : factors.beta_channels;

  Eigen::MatrixXd delta_hamiltonian_images =
      Eigen::MatrixXd::Zero(n_alpha, n_states * n_beta);
  Eigen::MatrixXd delta_overlap_images =
      Eigen::MatrixXd::Zero(n_alpha, n_states * n_beta);
  detail::stream_directional_pair_tiles(
      same_spin,
      n_active_orbitals,
      accepted_point.prepared_active_space.active_space_two_electron_result,
      direction,
      accepted_ri_active_pair_factors != nullptr
          ? &accepted_point.prepared_active_space
                 .active_space_one_electron_result.h1e_act
          : nullptr,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      true,
      [&](bool alpha_channel,
          bool beta_channel,
          const detail::SameSpinDirectionalPairTile& same_tile,
          const detail::DirectionalOppositeSpinPairTile* opposite_tile) {
        if (opposite_tile == nullptr) {
          throw std::logic_error(
              "tiled structure direction is missing opposite-spin data");
        }
        const Eigen::MatrixXd delta_hamiltonian =
            same_tile.delta_regular_hamiltonian +
            same_tile.delta_singular_hamiltonian;

        if (alpha_channel) {
          for (int state = 0; state < n_states; ++state) {
            const Eigen::MatrixXd& coefficients =
                selected_states.states[state].coefficient_matrix;
            const auto coefficients_right = coefficients.middleRows(
                same_tile.right_begin, same_tile.right_size());
            const Eigen::MatrixXd delta_overlap_coefficients =
                same_tile.delta_overlap * coefficients_right;
            const Eigen::MatrixXd delta_hamiltonian_coefficients =
                delta_hamiltonian * coefficients_right;
            auto overlap_image = delta_overlap_images.block(
                same_tile.left_begin,
                state * n_beta,
                same_tile.left_size(),
                n_beta);
            overlap_image.noalias() +=
                delta_overlap_coefficients * factors.beta_overlap.transpose();
            auto hamiltonian_image = delta_hamiltonian_images.block(
                same_tile.left_begin,
                state * n_beta,
                same_tile.left_size(),
                n_beta);
            hamiltonian_image.noalias() +=
                delta_hamiltonian_coefficients *
                factors.beta_overlap.transpose();
            hamiltonian_image.noalias() +=
                delta_overlap_coefficients *
                factors.beta_hamiltonian.transpose();

            for (int target = 0; target < n_pairs; ++target) {
              const Eigen::MatrixXd& projected =
                  opposite_tile->projected_channel(target);
              if (projected.isZero(0.0)) {
                continue;
              }
              const Eigen::MatrixXd projected_coefficients =
                  projected * coefficients_right;
              add_sparse_right_transpose(
                  projected_coefficients,
                  beta_channels[target],
                  hamiltonian_image);
            }
          }
        }

        if (beta_channel) {
          Eigen::MatrixXd accepted_alpha_times_coefficients;
          for (int state = 0; state < n_states; ++state) {
            const Eigen::MatrixXd& coefficients =
                selected_states.states[state].coefficient_matrix;
            const auto coefficients_right = coefficients.middleCols(
                same_tile.right_begin, same_tile.right_size());
            auto overlap_image = delta_overlap_images.middleCols(
                state * n_beta + same_tile.left_begin,
                same_tile.left_size());
            overlap_image.noalias() +=
                factors.alpha_overlap * coefficients_right *
                same_tile.delta_overlap.transpose();
            auto hamiltonian_image = delta_hamiltonian_images.middleCols(
                state * n_beta + same_tile.left_begin,
                same_tile.left_size());
            hamiltonian_image.noalias() +=
                factors.alpha_hamiltonian * coefficients_right *
                same_tile.delta_overlap.transpose();
            hamiltonian_image.noalias() +=
                factors.alpha_overlap * coefficients_right *
                delta_hamiltonian.transpose();

            accepted_alpha_times_coefficients.resize(
                n_alpha, same_tile.right_size());
            for (int target = 0; target < n_pairs; ++target) {
              const Eigen::MatrixXd& directional =
                  opposite_tile->raw_channel(target);
              if (directional.isZero(0.0)) {
                continue;
              }
              add_projected_left_product(
                  same_spin.alpha_pair_cache_ref(),
                  n_alpha,
                  target,
                  coefficients_right,
                  &accepted_alpha_times_coefficients);
              hamiltonian_image.noalias() +=
                  accepted_alpha_times_coefficients *
                  directional.transpose();
            }
          }
        }
        if (additional_consumer) {
          additional_consumer(
              alpha_channel, beta_channel, same_tile, opposite_tile);
        }
      });

  SelectedStateDirectionalStructureImages result;
  result.delta_hamiltonian_selected =
      structure_action->contract_spin_product_block(
          delta_hamiltonian_images);
  result.delta_overlap_selected =
      structure_action->contract_spin_product_block(delta_overlap_images);
  require_finite(
      result.delta_hamiltonian_selected,
      "tiled directional Hamiltonian images");
  require_finite(
      result.delta_overlap_selected,
      "tiled directional overlap images");
  return result;
}

}  // namespace xmvb::vb
