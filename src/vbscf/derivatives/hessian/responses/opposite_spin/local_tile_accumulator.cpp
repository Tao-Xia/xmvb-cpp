#include "vbscf/derivatives/hessian/responses/opposite_spin/local_tile_accumulator_internal.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb::detail {
namespace {

int packed_pair_count(
    const SameSpinPairCacheContext& cache,
    int n_active_orbitals) {
  const int expected = packed_active_pair_count(n_active_orbitals);
  const auto require_count = [&](const auto& pairs) {
    for (const auto& pair : pairs) {
      const auto& projected = pair.opposite_spin_pair_cache
          .first_order_cofactor_projection.projected_pair_values;
      if (!projected.empty() &&
          static_cast<int>(projected.size()) != expected) {
        throw std::invalid_argument(
            "opposite-spin accepted projection has inconsistent dimensions");
      }
    }
  };
  require_count(cache.alpha_pair_cache_ref());
  require_count(cache.beta_pair_cache_ref());
  return expected;
}

void add_packed_outer_product(
    const OppositeSpinPackedPairProjection& primary,
    const std::vector<double>& partner_image,
    const std::vector<int>& touched_partner_channels,
    std::vector<double>* packed_gradient) {
  for (std::size_t entry = 0;
       entry < primary.packed_pair_indices.size();
       ++entry) {
    const int primary_channel = primary.packed_pair_indices[entry];
    const double primary_value = primary.packed_pair_values[entry];
    for (const int partner_channel : touched_partner_channels) {
      (*packed_gradient)[TwoElectronIndexer::packed_pair_of_pairs_index(
          partner_channel, primary_channel)] +=
          primary_value * partner_image[partner_channel];
    }
  }
}

void add_accepted_projection_channel(
    const OppositeSpinPackedPairProjection& accepted,
    int directional_channel,
    double weight,
    std::vector<double>* packed_gradient) {
  for (std::size_t entry = 0;
       entry < accepted.packed_pair_indices.size();
       ++entry) {
    (*packed_gradient)[TwoElectronIndexer::packed_pair_of_pairs_index(
        directional_channel,
        accepted.packed_pair_indices[entry])] +=
        weight * accepted.packed_pair_values[entry];
  }
}

}  // namespace

LocalOppositeSpinTileAccumulator::LocalOppositeSpinTileAccumulator(
    const SameSpinPairCacheContext& accepted_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals)
    : accepted_pair_cache_(accepted_pair_cache),
      selected_states_(selected_states),
      n_active_orbitals_(n_active_orbitals),
      n_packed_pairs_(packed_pair_count(
          accepted_pair_cache, n_active_orbitals)),
      alpha_graph_(selected_states, PrimarySpin::Alpha),
      beta_graph_(selected_states, PrimarySpin::Beta) {
  result_.active_orbital_overlap_gradient.assign(
      static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals,
      0.0);
  result_.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);
}

void LocalOppositeSpinTileAccumulator::consume(
    bool alpha_channel,
    bool beta_channel,
    const DirectionalOppositeSpinPairTile& tile) {
  if (alpha_channel) {
    accumulate_primary(PrimarySpin::Alpha, tile, true);
    accumulate_cross_response(PrimarySpin::Beta, tile);
  }
  if (beta_channel) {
    accumulate_primary(PrimarySpin::Beta, tile, false);
    accumulate_cross_response(PrimarySpin::Alpha, tile);
  }
}

void LocalOppositeSpinTileAccumulator::accumulate_primary(
    PrimarySpin spin,
    const DirectionalOppositeSpinPairTile& tile,
    bool accumulate_packed_gradient) {
  const bool alpha = spin == PrimarySpin::Alpha;
  const auto& graph = alpha ? alpha_graph_ : beta_graph_;
  const auto& partner_pairs = alpha
      ? accepted_pair_cache_.beta_pair_cache_ref()
      : accepted_pair_cache_.alpha_pair_cache_ref();
  const int n_partner = alpha
      ? selected_states_.n_unique_beta
      : selected_states_.n_unique_alpha;
  const auto& primary_pairs = alpha
      ? accepted_pair_cache_.alpha_pair_cache_ref()
      : accepted_pair_cache_.beta_pair_cache_ref();
  const auto& primary_determinants = alpha
      ? accepted_pair_cache_.alpha_reuse_table.unique_determinants
      : accepted_pair_cache_.beta_reuse_table.unique_determinants;
  const int n_primary = alpha
      ? selected_states_.n_unique_alpha
      : selected_states_.n_unique_beta;
  const int n_electrons = primary_determinants.empty()
      ? 0
      : static_cast<int>(primary_determinants.front().size());

  std::vector<double> partner_image(n_packed_pairs_, 0.0);
  std::vector<unsigned char> touched_flags(n_packed_pairs_, 0u);
  std::vector<int> touched_channels;
  std::vector<int> target_channels;
  std::vector<double> accepted_values;
  Eigen::MatrixXd accepted_weight(n_electrons, n_electrons);
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(
      n_electrons, n_electrons);
  for (int left_local = 0; left_local < tile.left_size; ++left_local) {
    const int left = tile.left_begin + left_local;
    for (int right_local = 0; right_local < tile.right_size; ++right_local) {
      const int right = tile.right_begin + right_local;
      const auto& directional_pair = tile.pair(left_local, right_local);
      if (accumulate_packed_gradient) {
        graph.accumulate_partner_projection(
            left,
            right,
            partner_pairs,
            n_partner,
            &partner_image,
            &touched_flags,
            &touched_channels);
        add_packed_outer_product(
            directional_pair.delta_first_order_cofactor_projection,
            partner_image,
            touched_channels,
            &result_.packed_active_two_electron_gradient);
      }

      const auto& occupied_left = primary_determinants[left];
      const auto& occupied_right = primary_determinants[right];
      target_channels.clear();
      target_channels.reserve(
          static_cast<std::size_t>(n_electrons) * n_electrons);
      for (const int orbital_left : occupied_left) {
        for (const int orbital_right : occupied_right) {
          target_channels.push_back(TwoElectronIndexer::packed_pair_index(
              orbital_right, orbital_left));
        }
      }
      accepted_values.assign(target_channels.size(), 0.0);
      graph.accumulate_partner_projected_values(
          left,
          right,
          partner_pairs,
          n_partner,
          target_channels,
          &accepted_values);
      std::size_t target = 0;
      for (int left_electron = 0;
           left_electron < n_electrons;
           ++left_electron) {
        for (int right_electron = 0;
             right_electron < n_electrons;
             ++right_electron, ++target) {
          accepted_weight(left_electron, right_electron) =
              accepted_values[target];
        }
      }
      const auto& accepted_pair = primary_pairs[
          ordered_spin_pair_storage_index(left, right, n_primary)];
      accumulate_pair_overlap_gradient_direction(
          occupied_left,
          occupied_right,
          cached_cofactor_differential(accepted_pair),
          directional_pair.delta_overlap_submatrix,
          accepted_weight,
          zero,
          n_active_orbitals_,
          &result_.active_orbital_overlap_gradient);

      for (const int channel : touched_channels) {
        partner_image[channel] = 0.0;
        touched_flags[channel] = 0u;
      }
      touched_channels.clear();
    }
  }
}

void LocalOppositeSpinTileAccumulator::accumulate_cross_response(
    PrimarySpin target_spin,
    const DirectionalOppositeSpinPairTile& partner_tile) {
  const bool target_alpha = target_spin == PrimarySpin::Alpha;
  const int n_primary = target_alpha
      ? selected_states_.n_unique_alpha
      : selected_states_.n_unique_beta;
  const auto& primary_pairs = target_alpha
      ? accepted_pair_cache_.alpha_pair_cache_ref()
      : accepted_pair_cache_.beta_pair_cache_ref();
  const auto& primary_determinants = target_alpha
      ? accepted_pair_cache_.alpha_reuse_table.unique_determinants
      : accepted_pair_cache_.beta_reuse_table.unique_determinants;
  const int n_electrons = primary_determinants.empty()
      ? 0
      : static_cast<int>(primary_determinants.front().size());
  const int extent = std::min(n_primary, kSameSpinTileExtent);
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(
      n_electrons, n_electrons);

  for (int left_begin = 0; left_begin < n_primary; left_begin += extent) {
    const int left_size = std::min(extent, n_primary - left_begin);
    for (int right_begin = 0; right_begin < n_primary; right_begin += extent) {
      const int right_size = std::min(extent, n_primary - right_begin);
      const std::size_t pair_count =
          static_cast<std::size_t>(left_size) * right_size;
      const std::size_t pair_weight_size =
          static_cast<std::size_t>(n_electrons) * n_electrons;
      std::vector<double> pair_weights(
          pair_count * pair_weight_size, 0.0);

      Eigen::MatrixXd weight_block(left_size, right_size);
      Eigen::MatrixXd raw_weight_block(left_size, right_size);
      for (int channel = 0; channel < n_packed_pairs_; ++channel) {
        const Eigen::MatrixXd& projected =
            partner_tile.projected_channel(channel);
        const Eigen::MatrixXd& raw = partner_tile.raw_channel(channel);
        const bool need_overlap = !projected.isZero(0.0);
        const bool need_packed = target_alpha && !raw.isZero(0.0);
        if (!need_overlap && !need_packed) {
          continue;
        }
        weight_block.setZero();
        raw_weight_block.setZero();
        for (const auto& state : selected_states_.states) {
          const auto& coefficients = state.coefficient_matrix;
          if (target_alpha) {
            const auto left_coefficients = coefficients.block(
                left_begin,
                partner_tile.left_begin,
                left_size,
                partner_tile.left_size);
            const auto right_coefficients = coefficients.block(
                right_begin,
                partner_tile.right_begin,
                right_size,
                partner_tile.right_size);
            if (need_overlap) {
              weight_block.noalias() += state.normalized_state_weight *
                  left_coefficients * projected *
                  right_coefficients.transpose();
            }
            if (need_packed) {
              raw_weight_block.noalias() += state.normalized_state_weight *
                  left_coefficients * raw * right_coefficients.transpose();
            }
          } else {
            if (need_overlap) {
              weight_block.noalias() += state.normalized_state_weight *
                  coefficients.block(
                      partner_tile.left_begin,
                      left_begin,
                      partner_tile.left_size,
                      left_size).transpose() *
                  projected *
                  coefficients.block(
                      partner_tile.right_begin,
                      right_begin,
                      partner_tile.right_size,
                      right_size);
            }
          }
        }
        for (int left_local = 0; left_local < left_size; ++left_local) {
          const auto& occupied_left =
              primary_determinants[left_begin + left_local];
          for (int right_local = 0; right_local < right_size; ++right_local) {
            const auto& occupied_right =
                primary_determinants[right_begin + right_local];
            const auto& accepted_projection = primary_pairs[
                ordered_spin_pair_storage_index(
                    left_begin + left_local,
                    right_begin + right_local,
                    n_primary)]
                                                  .opposite_spin_pair_cache
                                                  .first_order_cofactor_projection;
            if (need_packed) {
              add_accepted_projection_channel(
                  accepted_projection,
                  channel,
                  raw_weight_block(left_local, right_local),
                  &result_.packed_active_two_electron_gradient);
            }
            if (!need_overlap) {
              continue;
            }
            const std::size_t pair_offset =
                (static_cast<std::size_t>(left_local) * right_size +
                 right_local) * pair_weight_size;
            for (int left_electron = 0;
                 left_electron < static_cast<int>(occupied_left.size());
                 ++left_electron) {
              for (int right_electron = 0;
                   right_electron < static_cast<int>(occupied_right.size());
                   ++right_electron) {
                if (TwoElectronIndexer::packed_pair_index(
                        occupied_right[right_electron],
                        occupied_left[left_electron]) == channel) {
                  pair_weights[pair_offset + left_electron +
                      static_cast<std::size_t>(n_electrons) *
                          right_electron] +=
                      weight_block(left_local, right_local);
                }
              }
            }
          }
        }
      }

      for (int left_local = 0; left_local < left_size; ++left_local) {
        const int left = left_begin + left_local;
        const auto& occupied_left = primary_determinants[left];
        for (int right_local = 0; right_local < right_size; ++right_local) {
          const int right = right_begin + right_local;
          const auto& occupied_right = primary_determinants[right];
          const auto& accepted_pair = primary_pairs[
              ordered_spin_pair_storage_index(left, right, n_primary)];
          const std::size_t pair_offset =
              (static_cast<std::size_t>(left_local) * right_size +
               right_local) * pair_weight_size;
          const Eigen::Map<const Eigen::MatrixXd> pair_weight(
              pair_weights.data() + pair_offset,
              n_electrons,
              n_electrons);
          accumulate_pair_overlap_gradient_direction(
              occupied_left,
              occupied_right,
              cached_cofactor_differential(accepted_pair),
              zero,
              zero,
              pair_weight,
              n_active_orbitals_,
              &result_.active_orbital_overlap_gradient);
        }
      }
    }
  }
}

OppositeSpinBackwardContribution LocalOppositeSpinTileAccumulator::finish() {
  return std::move(result_);
}

}  // namespace xmvb::vb::detail
