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

void add_packed_outer_product(
    const OppositeSpinPackedPairProjection& primary,
    const Eigen::Ref<const Eigen::VectorXd>& partner_projection,
    std::vector<double>* packed_gradient) {
  for (std::size_t entry = 0;
       entry < primary.packed_pair_indices.size();
       ++entry) {
    const int primary_channel = primary.packed_pair_indices[entry];
    const double primary_value = primary.packed_pair_values[entry];
    for (int partner_channel = 0;
         partner_channel < partner_projection.size();
         ++partner_channel) {
      if (partner_projection[partner_channel] == 0.0) {
        continue;
      }
      (*packed_gradient)[TwoElectronIndexer::packed_pair_of_pairs_index(
          partner_channel, primary_channel)] +=
          primary_value * partner_projection[partner_channel];
    }
  }
}

void add_partner_projection_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    PrimarySpin primary_spin,
    int primary_left_begin,
    int primary_left_size,
    int primary_right_begin,
    int primary_right_size,
    const AcceptedSpinPairTile& partner_tile,
    Eigen::MatrixXd* raw_projection) {
  Eigen::MatrixXd coefficient_weight(
      primary_left_size, primary_right_size);
  for (int partner_left_local = 0;
       partner_left_local < partner_tile.left_size;
       ++partner_left_local) {
    const int partner_left =
        partner_tile.left_begin + partner_left_local;
    for (int partner_right_local = 0;
         partner_right_local < partner_tile.right_size;
         ++partner_right_local) {
      const int partner_right =
          partner_tile.right_begin + partner_right_local;
      coefficient_weight.setZero();
      for (const auto& state : selected_states.states) {
        const auto& coefficients = state.coefficient_matrix;
        if (primary_spin == PrimarySpin::Alpha) {
          coefficient_weight.noalias() += state.normalized_state_weight *
              coefficients.block(
                  primary_left_begin,
                  partner_left,
                  primary_left_size,
                  1) *
              coefficients.block(
                  primary_right_begin,
                  partner_right,
                  primary_right_size,
                  1).transpose();
        } else {
          coefficient_weight.noalias() += state.normalized_state_weight *
              coefficients.block(
                  partner_left,
                  primary_left_begin,
                  1,
                  primary_left_size).transpose() *
              coefficients.block(
                  partner_right,
                  primary_right_begin,
                  1,
                  primary_right_size);
        }
      }
      const auto& projection = partner_tile
          .pair(partner_left_local, partner_right_local)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        const int channel = projection.packed_pair_indices[entry];
        const double value = projection.packed_pair_values[entry];
        for (int primary_left_local = 0;
             primary_left_local < primary_left_size;
             ++primary_left_local) {
          for (int primary_right_local = 0;
               primary_right_local < primary_right_size;
               ++primary_right_local) {
            const int pair =
                primary_left_local * primary_right_size +
                primary_right_local;
            (*raw_projection)(channel, pair) += value *
                coefficient_weight(
                    primary_left_local, primary_right_local);
          }
        }
      }
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
    int n_active_orbitals,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron)
    : accepted_pair_cache_(accepted_pair_cache),
      selected_states_(selected_states),
      n_active_orbitals_(n_active_orbitals),
      n_packed_pairs_(packed_active_pair_count(n_active_orbitals)),
      active_overlap_(active_overlap),
      active_one_electron_(active_one_electron),
      active_two_electron_(active_two_electron),
      tile_extents_(plan_pair_tile_extents(
          accepted_pair_cache,
          n_active_orbitals,
          active_two_electron,
          true)) {
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
    const AcceptedSpinPairTile& accepted,
    const DirectionalOppositeSpinPairTileView& tile) {
  if (alpha_channel) {
    accumulate_primary(PrimarySpin::Alpha, accepted, tile, true);
    accumulate_cross_response(PrimarySpin::Beta, tile);
  }
  if (beta_channel) {
    accumulate_primary(PrimarySpin::Beta, accepted, tile, false);
    accumulate_cross_response(PrimarySpin::Alpha, tile);
  }
}

void LocalOppositeSpinTileAccumulator::accumulate_primary(
    PrimarySpin spin,
    const AcceptedSpinPairTile& accepted,
    const DirectionalOppositeSpinPairTileView& tile,
    bool accumulate_packed_gradient) {
  const bool alpha = spin == PrimarySpin::Alpha;
  const int n_partner = alpha
      ? selected_states_.n_unique_beta
      : selected_states_.n_unique_alpha;
  const auto& primary_determinants = alpha
      ? accepted_pair_cache_.alpha_reuse_table.unique_determinants
      : accepted_pair_cache_.beta_reuse_table.unique_determinants;
  const AcceptedPairTileProvider& partner_provider = alpha
      ? accepted_pair_cache_.beta_provider()
      : accepted_pair_cache_.alpha_provider();
  const int n_electrons = primary_determinants.empty()
      ? 0
      : static_cast<int>(primary_determinants.front().size());
  const int primary_pair_count = tile.left_size() * tile.right_size();
  Eigen::MatrixXd raw_projection = Eigen::MatrixXd::Zero(
      n_packed_pairs_, primary_pair_count);
  const int partner_extent = std::min(
      n_partner, alpha ? tile_extents_.beta : tile_extents_.alpha);
  for (int partner_left = 0;
       partner_left < n_partner;
       partner_left += partner_extent) {
    const int partner_left_end = std::min(
        n_partner, partner_left + partner_extent);
    for (int partner_right = 0;
         partner_right < n_partner;
         partner_right += partner_extent) {
      const int partner_right_end = std::min(
          n_partner, partner_right + partner_extent);
      const AcceptedSpinPairTile partner_tile = partner_provider.build(
          partner_left,
          partner_left_end,
          partner_right,
          partner_right_end,
          active_overlap_,
          active_one_electron_,
          active_two_electron_,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = false});
      add_partner_projection_tile(
          selected_states_,
          spin,
          tile.left_begin(),
          tile.left_size(),
          tile.right_begin(),
          tile.right_size(),
          partner_tile,
          &raw_projection);
    }
  }
  const Eigen::MatrixXd projected =
      apply_active_space_two_electron_kernel_block(
          active_two_electron_, n_active_orbitals_, raw_projection);
  Eigen::MatrixXd accepted_weight(n_electrons, n_electrons);
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(
      n_electrons, n_electrons);
  for (int left_local = 0; left_local < tile.left_size(); ++left_local) {
    const int left = tile.left_begin() + left_local;
    for (int right_local = 0; right_local < tile.right_size(); ++right_local) {
      const int right = tile.right_begin() + right_local;
      tile.with_pair(left_local, right_local, [&](const auto& directional_pair) {
      const int pair_index = left_local * tile.right_size() + right_local;
      if (accumulate_packed_gradient) {
        add_packed_outer_product(
            directional_pair.delta_first_order_cofactor_projection,
            raw_projection.col(pair_index),
            &result_.packed_active_two_electron_gradient);
      }

      const auto& occupied_left = primary_determinants[left];
      const auto& occupied_right = primary_determinants[right];
      for (int left_electron = 0;
           left_electron < n_electrons;
           ++left_electron) {
        for (int right_electron = 0;
             right_electron < n_electrons;
             ++right_electron) {
          const int channel = TwoElectronIndexer::packed_pair_index(
              occupied_right[right_electron],
              occupied_left[left_electron]);
          accepted_weight(left_electron, right_electron) =
              projected(channel, pair_index);
        }
      }
      const auto& accepted_pair = accepted.pair(left_local, right_local);
      accumulate_pair_overlap_gradient_direction(
          occupied_left,
          occupied_right,
          cached_cofactor_differential(accepted_pair),
          directional_pair.delta_overlap_submatrix,
          accepted_weight,
          zero,
          n_active_orbitals_,
          &result_.active_orbital_overlap_gradient);
      });
    }
  }
}

void LocalOppositeSpinTileAccumulator::accumulate_cross_response(
    PrimarySpin target_spin,
    const DirectionalOppositeSpinPairTileView& partner_tile) {
  const bool target_alpha = target_spin == PrimarySpin::Alpha;
  const int n_primary = target_alpha
      ? selected_states_.n_unique_alpha
      : selected_states_.n_unique_beta;
  const AcceptedPairTileProvider& primary_provider = target_alpha
      ? accepted_pair_cache_.alpha_provider()
      : accepted_pair_cache_.beta_provider();
  const auto& primary_determinants = target_alpha
      ? accepted_pair_cache_.alpha_reuse_table.unique_determinants
      : accepted_pair_cache_.beta_reuse_table.unique_determinants;
  const int n_electrons = primary_determinants.empty()
      ? 0
      : static_cast<int>(primary_determinants.front().size());
  const int extent = std::min(
      n_primary, target_alpha ? tile_extents_.alpha : tile_extents_.beta);
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(
      n_electrons, n_electrons);

  for (int left_begin = 0; left_begin < n_primary; left_begin += extent) {
    const int left_size = std::min(extent, n_primary - left_begin);
    for (int right_begin = 0; right_begin < n_primary; right_begin += extent) {
      const int right_size = std::min(extent, n_primary - right_begin);
      const AcceptedSpinPairTile accepted = primary_provider.build(
          left_begin,
          left_begin + left_size,
          right_begin,
          right_begin + right_size,
          active_overlap_,
          active_one_electron_,
          active_two_electron_,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true});
      const std::size_t pair_count =
          static_cast<std::size_t>(left_size) * right_size;
      const std::size_t pair_weight_size =
          static_cast<std::size_t>(n_electrons) * n_electrons;
      std::vector<double> pair_weights(
          pair_count * pair_weight_size, 0.0);

      Eigen::MatrixXd weight_block(left_size, right_size);
      Eigen::MatrixXd raw_weight_block(left_size, right_size);
      for (int channel = 0; channel < n_packed_pairs_; ++channel) {
        const auto projected = partner_tile.projected_channel(channel);
        const auto raw = partner_tile.raw_channel(channel);
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
                partner_tile.left_begin(),
                left_size,
                partner_tile.left_size());
            const auto right_coefficients = coefficients.block(
                right_begin,
                partner_tile.right_begin(),
                right_size,
                partner_tile.right_size());
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
                      partner_tile.left_begin(),
                      left_begin,
                      partner_tile.left_size(),
                      left_size).transpose() *
                  projected *
                  coefficients.block(
                      partner_tile.right_begin(),
                      right_begin,
                      partner_tile.right_size(),
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
            const auto& accepted_projection = accepted
                .pair(left_local, right_local)
                .opposite_spin_pair_cache.first_order_cofactor_projection;
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
          const auto& accepted_pair = accepted.pair(
              left_local, right_local);
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
