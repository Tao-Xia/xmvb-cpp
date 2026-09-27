#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {
namespace {

constexpr double kContributionTolerance = 1.0e-15;

enum class PrimarySpin { Alpha, Beta };

void validate_backward_inputs(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e) {
  if (!cache.has_pair_providers()) {
    throw std::invalid_argument(
        "tiled opposite-spin backward requires accepted-pair providers");
  }
  if (states.n_unique_alpha !=
          static_cast<int>(cache.alpha_reuse_table.unique_determinants.size()) ||
      states.n_unique_beta !=
          static_cast<int>(cache.beta_reuse_table.unique_determinants.size())) {
    throw std::invalid_argument(
        "selected-state dimensions do not match unique-spin tables");
  }
  if (h1e.rows() != n_active || h1e.cols() != n_active ||
      active_overlap.size() !=
          static_cast<std::size_t>(n_active) * n_active) {
    throw std::invalid_argument(
        "opposite-spin backward active-space dimensions are inconsistent");
  }
}

void validate_directional_states(
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices& directions) {
  if (states.n_unique_alpha != directions.n_unique_alpha ||
      states.n_unique_beta != directions.n_unique_beta ||
      states.n_determinants != directions.n_determinants ||
      states.selected_state_indices != directions.selected_state_indices ||
      states.states.size() != directions.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices do not match accepted dimensions");
  }
}

OppositeSpinBackwardContribution make_zero_contribution(int n_active) {
  OppositeSpinBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      static_cast<std::size_t>(n_active) * n_active, 0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active), 0.0);
  return result;
}

const AcceptedPairTileProvider& provider(
    const SameSpinPairCacheContext& cache,
    PrimarySpin spin) {
  return spin == PrimarySpin::Alpha
      ? cache.alpha_provider()
      : cache.beta_provider();
}

const std::vector<std::vector<int>>& strings(
    const SameSpinPairCacheContext& cache,
    PrimarySpin spin) {
  return spin == PrimarySpin::Alpha
      ? cache.alpha_reuse_table.unique_determinants
      : cache.beta_reuse_table.unique_determinants;
}

int spin_size(
    const SelectedStateDeterminantMatrices& states,
    PrimarySpin spin) {
  return spin == PrimarySpin::Alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
}

Eigen::MatrixXd build_raw_channel_block(
    const AcceptedSpinPairTile& tile,
    int n_packed_pairs) {
  Eigen::MatrixXd channels = Eigen::MatrixXd::Zero(
      tile.left_size * tile.right_size, n_packed_pairs);
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const int work = left + tile.left_size * right;
      const auto& projection = tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        channels(work, projection.packed_pair_indices[entry]) +=
            projection.packed_pair_values[entry];
      }
    }
  }
  return channels;
}

Eigen::MatrixXd coefficient_block(
    const Eigen::MatrixXd& coefficients,
    PrimarySpin primary_spin,
    int primary_begin,
    int primary_size,
    int partner_begin,
    int partner_size) {
  if (primary_spin == PrimarySpin::Alpha) {
    return coefficients.block(
        primary_begin, partner_begin, primary_size, partner_size);
  }
  return coefficients.block(
      partner_begin, primary_begin, partner_size, primary_size).transpose();
}

void accumulate_partner_tile_images(
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices* directions,
    PrimarySpin primary_spin,
    int primary_left_begin,
    int primary_left_size,
    int primary_right_begin,
    int primary_right_size,
    const AcceptedSpinPairTile& partner_tile,
    const Eigen::MatrixXd& partner_channels,
    Eigen::MatrixXd* raw_images) {
  using ChannelMap = Eigen::Map<
      const Eigen::MatrixXd,
      Eigen::Unaligned,
      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  const Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(
      partner_tile.left_size, 1);
  Eigen::MatrixXd weight(primary_left_size, primary_right_size);
  Eigen::MatrixXd push(primary_left_size, partner_tile.right_size);
  for (int channel = 0; channel < partner_channels.cols(); ++channel) {
    if (partner_channels.col(channel).isZero(0.0)) {
      continue;
    }
    const ChannelMap partner_channel(
        partner_channels.col(channel).data(),
        partner_tile.left_size,
        partner_tile.right_size,
        stride);
    weight.setZero();
    for (std::size_t state = 0; state < states.states.size(); ++state) {
      const auto& accepted = states.states[state];
      const Eigen::MatrixXd accepted_left = coefficient_block(
          accepted.coefficient_matrix,
          primary_spin,
          primary_left_begin,
          primary_left_size,
          partner_tile.left_begin,
          partner_tile.left_size);
      const Eigen::MatrixXd accepted_right = coefficient_block(
          accepted.coefficient_matrix,
          primary_spin,
          primary_right_begin,
          primary_right_size,
          partner_tile.right_begin,
          partner_tile.right_size);
      if (directions == nullptr) {
        push.noalias() = accepted_left * partner_channel;
        weight.noalias() += accepted.normalized_state_weight *
            push * accepted_right.transpose();
        continue;
      }
      const auto& direction = directions->states[state].coefficient_matrix;
      const Eigen::MatrixXd directional_left = coefficient_block(
          direction,
          primary_spin,
          primary_left_begin,
          primary_left_size,
          partner_tile.left_begin,
          partner_tile.left_size);
      const Eigen::MatrixXd directional_right = coefficient_block(
          direction,
          primary_spin,
          primary_right_begin,
          primary_right_size,
          partner_tile.right_begin,
          partner_tile.right_size);
      push.noalias() = directional_left * partner_channel;
      weight.noalias() += accepted.normalized_state_weight *
          push * accepted_right.transpose();
      push.noalias() = accepted_left * partner_channel;
      weight.noalias() += accepted.normalized_state_weight *
          push * directional_right.transpose();
    }
    raw_images->col(channel).noalias() +=
        Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  }
}

void accumulate_packed_gradient(
    const AcceptedSpinPairTile& primary_tile,
    const Eigen::MatrixXd& raw_partner_images,
    std::vector<double>* packed_gradient) {
  for (int left = 0; left < primary_tile.left_size; ++left) {
    for (int right = 0; right < primary_tile.right_size; ++right) {
      const int work = left + primary_tile.left_size * right;
      const auto& projection = primary_tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        const int alpha_channel = projection.packed_pair_indices[entry];
        const double alpha_value = projection.packed_pair_values[entry];
        for (int beta_channel = 0;
             beta_channel < raw_partner_images.cols();
             ++beta_channel) {
          const double contribution =
              alpha_value * raw_partner_images(work, beta_channel);
          if (std::abs(contribution) > kContributionTolerance) {
            (*packed_gradient)[
                TwoElectronIndexer::packed_pair_of_pairs_index(
                    beta_channel, alpha_channel)] += contribution;
          }
        }
      }
    }
  }
}

void accumulate_overlap_gradient(
    const AcceptedSpinPairTile& primary_tile,
    const std::vector<std::vector<int>>& primary_strings,
    const Eigen::MatrixXd& projected_partner_images,
    int n_active,
    std::vector<double>* overlap_gradient) {
  for (int left_local = 0;
       left_local < primary_tile.left_size;
       ++left_local) {
    const int left = primary_tile.left_begin + left_local;
    const auto& occupied_left = primary_strings[left];
    for (int right_local = 0;
         right_local < primary_tile.right_size;
         ++right_local) {
      const int right = primary_tile.right_begin + right_local;
      const auto& occupied_right = primary_strings[right];
      const int n_electrons = static_cast<int>(occupied_left.size());
      const int work = left_local + primary_tile.left_size * right_local;
      Eigen::MatrixXd cofactor_weight(n_electrons, n_electrons);
      for (int left_electron = 0;
           left_electron < n_electrons;
           ++left_electron) {
        for (int right_electron = 0;
             right_electron < n_electrons;
             ++right_electron) {
          const int channel = TwoElectronIndexer::packed_pair_index(
              occupied_right[right_electron],
              occupied_left[left_electron]);
          cofactor_weight(left_electron, right_electron) =
              projected_partner_images(work, channel);
        }
      }
      const Eigen::MatrixXd zero =
          Eigen::MatrixXd::Zero(n_electrons, n_electrons);
      detail::accumulate_pair_overlap_gradient_direction(
          occupied_left,
          occupied_right,
          cached_cofactor_differential(
              primary_tile.pair(left_local, right_local)),
          zero,
          zero,
          cofactor_weight,
          n_active,
          overlap_gradient);
    }
  }
}

void consume_primary_tiles(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices* directions,
    PrimarySpin primary_spin,
    bool include_packed_gradient,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    OppositeSpinBackwardContribution* result) {
  const PrimarySpin partner_spin = primary_spin == PrimarySpin::Alpha
      ? PrimarySpin::Beta
      : PrimarySpin::Alpha;
  const detail::OppositeSpinTilePlan tile_plan =
      detail::plan_opposite_spin_backward_tiles(
          cache, n_active, two_electron);
  const int primary_extent = primary_spin == PrimarySpin::Alpha
      ? tile_plan.primary.alpha
      : tile_plan.primary.beta;
  const int partner_extent = partner_spin == PrimarySpin::Alpha
      ? tile_plan.partner.alpha
      : tile_plan.partner.beta;
  const int n_primary = spin_size(states, primary_spin);
  const int n_partner = spin_size(states, partner_spin);
  const int n_pairs = packed_active_pair_count(n_active);

  for (int primary_left = 0;
       primary_left < n_primary;
       primary_left += primary_extent) {
    const int primary_left_end =
        std::min(n_primary, primary_left + primary_extent);
    for (int primary_right = 0;
         primary_right < n_primary;
         primary_right += primary_extent) {
      const int primary_right_end =
          std::min(n_primary, primary_right + primary_extent);
      const AcceptedSpinPairTile primary_tile = provider(cache, primary_spin).build(
          primary_left,
          primary_left_end,
          primary_right,
          primary_right_end,
          active_overlap,
          h1e,
          two_electron,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true,
              .populate_opposite_spin_projection = true});
      Eigen::MatrixXd raw_images = Eigen::MatrixXd::Zero(
          primary_tile.left_size * primary_tile.right_size, n_pairs);

      for (int partner_left = 0;
           partner_left < n_partner;
           partner_left += partner_extent) {
        const int partner_left_end =
            std::min(n_partner, partner_left + partner_extent);
        for (int partner_right = 0;
             partner_right < n_partner;
             partner_right += partner_extent) {
          const int partner_right_end =
              std::min(n_partner, partner_right + partner_extent);
          const AcceptedSpinPairTile partner_tile =
              provider(cache, partner_spin).build(
                  partner_left,
                  partner_left_end,
                  partner_right,
                  partner_right_end,
                  active_overlap,
                  h1e,
                  two_electron,
                  AcceptedPairTileBuildOptions{
                      .materialize_projected_pair_values = false,
                      .populate_response_payload = false,
                      .populate_opposite_spin_projection = true});
          const Eigen::MatrixXd partner_channels =
              build_raw_channel_block(partner_tile, n_pairs);
          accumulate_partner_tile_images(
              states,
              directions,
              primary_spin,
              primary_left,
              primary_tile.left_size,
              primary_right,
              primary_tile.right_size,
              partner_tile,
              partner_channels,
              &raw_images);
        }
      }
      if (include_packed_gradient) {
        accumulate_packed_gradient(
            primary_tile,
            raw_images,
            &result->packed_active_two_electron_gradient);
      }
      const Eigen::MatrixXd projected = apply_active_space_two_electron_kernel_block(
          two_electron, n_active, raw_images.transpose()).transpose();
      accumulate_overlap_gradient(
          primary_tile,
          strings(cache, primary_spin),
          projected,
          n_active,
          &result->active_orbital_overlap_gradient);
    }
  }
}

}  // namespace

OppositeSpinBackwardContribution build_opposite_spin_backward_contribution(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  validate_backward_inputs(cache, states, n_active, active_overlap, h1e);
  OppositeSpinBackwardContribution result = make_zero_contribution(n_active);
  consume_primary_tiles(
      cache,
      states,
      nullptr,
      PrimarySpin::Alpha,
      true,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &result);
  consume_primary_tiles(
      cache,
      states,
      nullptr,
      PrimarySpin::Beta,
      false,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &result);
  return result;
}

OppositeSpinBackwardContribution
build_directional_opposite_spin_backward_contribution(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices& directions,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  validate_backward_inputs(cache, states, n_active, active_overlap, h1e);
  validate_directional_states(states, directions);
  OppositeSpinBackwardContribution result = make_zero_contribution(n_active);
  auto start = std::chrono::steady_clock::now();
  consume_primary_tiles(
      cache,
      states,
      &directions,
      PrimarySpin::Alpha,
      true,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &result);
  result.timing.packed_gradient_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  start = std::chrono::steady_clock::now();
  consume_primary_tiles(
      cache,
      states,
      &directions,
      PrimarySpin::Beta,
      false,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &result);
  result.timing.beta_overlap_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace xmvb::vb
