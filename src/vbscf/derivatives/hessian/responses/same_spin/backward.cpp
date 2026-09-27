#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/matrix_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/weight_kernels_internal.hpp"

namespace xmvb::vb {

using detail::account_for_close_shell_spin_reuse;
using detail::accumulate_accepted_spin_backward_tile;
using detail::finalize_backward_contribution;
using detail::make_zero_backward_contribution;
using detail::PairTileExtents;
using detail::SameSpinAcceptedTileWeights;
using detail::validate_directional_selected_state_inputs;
using detail::validate_state_coefficient_matrix;

namespace {

enum class TargetSpin { Alpha, Beta };

void validate_backward_inputs(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e) {
  if (!cache.has_pair_providers()) {
    throw std::invalid_argument(
        "same-spin matrix backward requires accepted-pair providers");
  }
  if (states.states.size() != energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }
  if (static_cast<int>(
          cache.alpha_reuse_table.unique_determinants.size()) !=
          states.n_unique_alpha ||
      static_cast<int>(
          cache.beta_reuse_table.unique_determinants.size()) !=
          states.n_unique_beta) {
    throw std::invalid_argument(
        "same-spin context dimensions do not match selected-state matrices");
  }
  if (h1e.rows() != n_active || h1e.cols() != n_active ||
      active_overlap.size() !=
          static_cast<std::size_t>(n_active) * n_active) {
    throw std::invalid_argument(
        "accepted same-spin backward active-space dimensions are inconsistent");
  }
  for (const auto& state : states.states) {
    validate_state_coefficient_matrix(
        state, states.n_unique_alpha, states.n_unique_beta);
  }
}

Eigen::MatrixXd pack_partner_vectors(
    const SelectedStateDeterminantMatrices& states,
    TargetSpin target_spin) {
  const int columns_per_state = target_spin == TargetSpin::Alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
  const int partner_rows = target_spin == TargetSpin::Alpha
      ? states.n_unique_beta
      : states.n_unique_alpha;
  Eigen::MatrixXd packed(
      partner_rows,
      columns_per_state * static_cast<int>(states.states.size()));
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state].coefficient_matrix;
    if (target_spin == TargetSpin::Alpha) {
      packed.middleCols(
          static_cast<int>(state) * columns_per_state,
          columns_per_state) = coefficients.transpose();
    } else {
      packed.middleCols(
          static_cast<int>(state) * columns_per_state,
          columns_per_state) = coefficients;
    }
  }
  return packed;
}

const AcceptedPairTileProvider& partner_provider(
    const SameSpinPairCacheContext& cache,
    TargetSpin target_spin) {
  return target_spin == TargetSpin::Alpha
      ? cache.beta_provider()
      : cache.alpha_provider();
}

const AcceptedPairTileProvider& target_provider(
    const SameSpinPairCacheContext& cache,
    TargetSpin target_spin) {
  return target_spin == TargetSpin::Alpha
      ? cache.alpha_provider()
      : cache.beta_provider();
}

const std::vector<std::vector<int>>& target_strings(
    const SameSpinPairCacheContext& cache,
    TargetSpin target_spin) {
  return target_spin == TargetSpin::Alpha
      ? cache.alpha_reuse_table.unique_determinants
      : cache.beta_reuse_table.unique_determinants;
}

int target_size(
    const SelectedStateDeterminantMatrices& states,
    TargetSpin target_spin) {
  return target_spin == TargetSpin::Alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
}

int columns_per_state(
    const SelectedStateDeterminantMatrices& states,
    TargetSpin target_spin) {
  return target_size(states, target_spin);
}

Eigen::MatrixXd accepted_image_tile(
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& action,
    TargetSpin target_spin,
    int action_begin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size) {
  if (target_spin == TargetSpin::Alpha) {
    return action.middleCols(action_begin + left_begin, left_size).transpose() *
        coefficients.middleRows(right_begin, right_size).transpose();
  }
  return coefficients.middleCols(left_begin, left_size).transpose() *
      action.middleCols(action_begin + right_begin, right_size);
}

Eigen::MatrixXd directional_image_tile(
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& direction,
    const Eigen::MatrixXd& accepted_action,
    const Eigen::MatrixXd& directional_action,
    TargetSpin target_spin,
    int accepted_begin,
    int directional_begin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size) {
  if (target_spin == TargetSpin::Alpha) {
    Eigen::MatrixXd image = directional_action
        .middleCols(directional_begin + left_begin, left_size).transpose() *
        coefficients.middleRows(right_begin, right_size).transpose();
    image.noalias() += accepted_action
        .middleCols(accepted_begin + left_begin, left_size).transpose() *
        direction.middleRows(right_begin, right_size).transpose();
    return image;
  }
  Eigen::MatrixXd image =
      direction.middleCols(left_begin, left_size).transpose() *
      accepted_action.middleCols(accepted_begin + right_begin, right_size);
  image.noalias() +=
      coefficients.middleCols(left_begin, left_size).transpose() *
      directional_action.middleCols(
          directional_begin + right_begin, right_size);
  return image;
}

AcceptedSpinPairActionResult apply_partner_actions(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& accepted_states,
    const SelectedStateDeterminantMatrices* directional_states,
    TargetSpin target_spin,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  Eigen::MatrixXd vectors = pack_partner_vectors(accepted_states, target_spin);
  if (directional_states != nullptr) {
    const Eigen::MatrixXd directional =
        pack_partner_vectors(*directional_states, target_spin);
    const int accepted_columns = static_cast<int>(vectors.cols());
    vectors.conservativeResize(
        Eigen::NoChange, accepted_columns + directional.cols());
    vectors.rightCols(directional.cols()) = directional;
  }
  return apply_accepted_spin_pair_action(
      partner_provider(cache, target_spin),
      active_overlap,
      h1e,
      two_electron,
      vectors);
}

void build_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    const AcceptedSpinPairActionResult& action,
    TargetSpin target_spin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_size, right_size);
  const int state_columns = columns_per_state(states, target_spin);
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state];
    const int action_begin = static_cast<int>(state) * state_columns;
    const Eigen::MatrixXd overlap_image = accepted_image_tile(
        coefficients.coefficient_matrix,
        action.overlap,
        target_spin,
        action_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
    const double state_weight = coefficients.normalized_state_weight;
    weights->hamiltonian.noalias() += state_weight * overlap_image;
    weights->overlap.noalias() -=
        state_weight * energies[state] * overlap_image;
    weights->partner_total.noalias() += state_weight * accepted_image_tile(
        coefficients.coefficient_matrix,
        action.hamiltonian,
        target_spin,
        action_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
  }
}

void build_directional_tile_weights(
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices& directions,
    const std::vector<double>& energies,
    const std::vector<double>& directional_energies,
    const AcceptedSpinPairActionResult& action,
    TargetSpin target_spin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_size, right_size);
  const int state_columns = columns_per_state(states, target_spin);
  const int directional_offset =
      state_columns * static_cast<int>(states.states.size());
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state];
    const auto& direction = directions.states[state];
    const int accepted_begin = static_cast<int>(state) * state_columns;
    const int directional_begin = directional_offset + accepted_begin;
    const Eigen::MatrixXd base_overlap = accepted_image_tile(
        coefficients.coefficient_matrix,
        action.overlap,
        target_spin,
        accepted_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
    const Eigen::MatrixXd delta_overlap = directional_image_tile(
        coefficients.coefficient_matrix,
        direction.coefficient_matrix,
        action.overlap,
        action.overlap,
        target_spin,
        accepted_begin,
        directional_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
    const double state_weight = coefficients.normalized_state_weight;
    weights->hamiltonian.noalias() += state_weight * delta_overlap;
    weights->overlap.noalias() -= state_weight *
        (directional_energies[state] * base_overlap +
         energies[state] * delta_overlap);
    weights->partner_total.noalias() += state_weight * directional_image_tile(
        coefficients.coefficient_matrix,
        direction.coefficient_matrix,
        action.hamiltonian,
        action.hamiltonian,
        target_spin,
        accepted_begin,
        directional_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
  }
}

template <typename WeightBuilder>
void consume_backward_tiles(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    TargetSpin target_spin,
    int tile_extent,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    WeightBuilder&& build_weights,
    Eigen::MatrixXd* h1e_gradient,
    SameSpinMatrixBackwardContribution* result) {
  const int n_unique = target_size(states, target_spin);
  SameSpinAcceptedTileWeights weights;
  for (int left_begin = 0; left_begin < n_unique; left_begin += tile_extent) {
    const int left_end = std::min(n_unique, left_begin + tile_extent);
    for (int right_begin = 0;
         right_begin < n_unique;
         right_begin += tile_extent) {
      const int right_end = std::min(n_unique, right_begin + tile_extent);
      build_weights(
          left_begin,
          left_end - left_begin,
          right_begin,
          right_end - right_begin,
          &weights);
      const AcceptedSpinPairTile tile = target_provider(cache, target_spin).build(
          left_begin,
          left_end,
          right_begin,
          right_end,
          active_overlap,
          h1e,
          two_electron,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true,
              .populate_opposite_spin_projection = false});
      accumulate_accepted_spin_backward_tile(
          target_strings(cache, target_spin),
          tile,
          weights.hamiltonian,
          weights.overlap,
          weights.partner_total,
          n_active,
          h1e_gradient,
          &result->active_orbital_overlap_gradient,
          &result->packed_active_two_electron_gradient);
    }
  }
}

}  // namespace

SameSpinMatrixBackwardContribution build_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  validate_backward_inputs(
      cache, states, energies, n_active, active_overlap, h1e);

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active);
  Eigen::MatrixXd h1e_gradient =
      Eigen::MatrixXd::Zero(n_active, n_active);
  const PairTileExtents extents = detail::plan_pair_tile_extents(
      cache, n_active, two_electron, false);
  const bool close_shell = cache.close_shell_reuses_same_spin_pair_cache();

  const AcceptedSpinPairActionResult beta_action = apply_partner_actions(
      cache,
      states,
      nullptr,
      TargetSpin::Alpha,
      active_overlap,
      h1e,
      two_electron);
  consume_backward_tiles(
      cache,
      states,
      TargetSpin::Alpha,
      extents.alpha,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      [&](int left, int left_size, int right, int right_size,
          SameSpinAcceptedTileWeights* weights) {
        build_accepted_tile_weights(
            states,
            energies,
            beta_action,
            TargetSpin::Alpha,
            left,
            left_size,
            right,
            right_size,
            weights);
      },
      &h1e_gradient,
      &result);
  if (close_shell) {
    account_for_close_shell_spin_reuse(&h1e_gradient, &result);
  } else {
    const AcceptedSpinPairActionResult alpha_action = apply_partner_actions(
        cache,
        states,
        nullptr,
        TargetSpin::Beta,
        active_overlap,
        h1e,
        two_electron);
    consume_backward_tiles(
        cache,
        states,
        TargetSpin::Beta,
        extents.beta,
        n_active,
        active_overlap,
        h1e,
        two_electron,
        [&](int left, int left_size, int right, int right_size,
            SameSpinAcceptedTileWeights* weights) {
          build_accepted_tile_weights(
              states,
              energies,
              alpha_action,
              TargetSpin::Beta,
              left,
              left_size,
              right,
              right_size,
              weights);
        },
        &h1e_gradient,
        &result);
  }
  return finalize_backward_contribution(std::move(result), h1e_gradient);
}

SameSpinMatrixBackwardContribution
build_directional_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices& directions,
    const std::vector<double>& energies,
    const std::vector<double>& directional_energies,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  validate_backward_inputs(
      cache, states, energies, n_active, active_overlap, h1e);
  validate_directional_selected_state_inputs(
      states, directions, energies, directional_energies);
  for (const auto& state : directions.states) {
    validate_state_coefficient_matrix(
        state, directions.n_unique_alpha, directions.n_unique_beta);
  }

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active);
  Eigen::MatrixXd h1e_gradient =
      Eigen::MatrixXd::Zero(n_active, n_active);
  const PairTileExtents extents = detail::plan_pair_tile_extents(
      cache, n_active, two_electron, false);
  const bool close_shell = cache.close_shell_reuses_same_spin_pair_cache();

  const auto consume_spin = [&](TargetSpin spin, int extent) {
    const AcceptedSpinPairActionResult action = apply_partner_actions(
        cache,
        states,
        &directions,
        spin,
        active_overlap,
        h1e,
        two_electron);
    consume_backward_tiles(
        cache,
        states,
        spin,
        extent,
        n_active,
        active_overlap,
        h1e,
        two_electron,
        [&](int left, int left_size, int right, int right_size,
            SameSpinAcceptedTileWeights* weights) {
          build_directional_tile_weights(
              states,
              directions,
              energies,
              directional_energies,
              action,
              spin,
              left,
              left_size,
              right,
              right_size,
              weights);
        },
        &h1e_gradient,
        &result);
  };
  consume_spin(TargetSpin::Alpha, extents.alpha);
  if (close_shell) {
    account_for_close_shell_spin_reuse(&h1e_gradient, &result);
  } else {
    consume_spin(TargetSpin::Beta, extents.beta);
  }
  return finalize_backward_contribution(std::move(result), h1e_gradient);
}

}  // namespace xmvb::vb
