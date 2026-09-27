#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/weight_kernels_internal.hpp"

namespace xmvb::vb {

using detail::account_for_close_shell_spin_reuse;
using detail::accumulate_accepted_spin_backward_tile;
using detail::finalize_backward_contribution;
using detail::make_zero_backward_contribution;
using detail::SameSpinAcceptedTileWeights;
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

void validate_directional_inputs(
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices& directions,
    const std::vector<double>& energies,
    const std::vector<double>& directional_energies) {
  if (states.n_unique_alpha != directions.n_unique_alpha ||
      states.n_unique_beta != directions.n_unique_beta ||
      states.n_determinants != directions.n_determinants ||
      states.selected_state_indices != directions.selected_state_indices ||
      states.states.size() != directions.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices have inconsistent dimensions");
  }
  if (energies.size() != states.states.size() ||
      directional_energies.size() != states.states.size()) {
    throw std::invalid_argument(
        "selected-state energy directions have inconsistent dimensions");
  }
}

Eigen::MatrixXd pack_partner_vectors(
    const SelectedStateDeterminantMatrices& accepted_states,
    const SelectedStateDeterminantMatrices* directional_states,
    TargetSpin target_spin,
    int panel_begin,
    int panel_size) {
  const int target_size = target_spin == TargetSpin::Alpha
      ? accepted_states.n_unique_alpha
      : accepted_states.n_unique_beta;
  const int partner_rows = target_spin == TargetSpin::Alpha
      ? accepted_states.n_unique_beta
      : accepted_states.n_unique_alpha;
  if (panel_begin < 0 || panel_size <= 0 ||
      panel_begin + panel_size > target_size) {
    throw std::invalid_argument("partner-action panel bounds are invalid");
  }
  const int columns_per_set =
      panel_size * static_cast<int>(accepted_states.states.size());
  Eigen::MatrixXd packed(
      partner_rows,
      columns_per_set * (directional_states == nullptr ? 1 : 2));
  const auto pack_set = [&](
                            const SelectedStateDeterminantMatrices& states,
                            int column_offset) {
    for (std::size_t state = 0; state < states.states.size(); ++state) {
      const auto& coefficients = states.states[state].coefficient_matrix;
      auto output = packed.middleCols(
          column_offset + static_cast<int>(state) * panel_size,
          panel_size);
      if (target_spin == TargetSpin::Alpha) {
        output = coefficients.middleRows(
            panel_begin, panel_size).transpose();
      } else {
        output = coefficients.middleCols(panel_begin, panel_size);
      }
    }
  };
  pack_set(accepted_states, 0);
  if (directional_states != nullptr) {
    pack_set(*directional_states, columns_per_set);
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

Eigen::MatrixXd accepted_panel_image(
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& action,
    TargetSpin target_spin,
    int action_begin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size) {
  if (target_spin == TargetSpin::Alpha) {
    return action.middleCols(action_begin, left_size).transpose() *
        coefficients.middleRows(right_begin, right_size).transpose();
  }
  return coefficients.middleCols(left_begin, left_size).transpose() *
      action.middleCols(action_begin, right_size);
}

Eigen::MatrixXd directional_panel_image(
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
        .middleCols(directional_begin, left_size).transpose() *
        coefficients.middleRows(right_begin, right_size).transpose();
    image.noalias() += accepted_action
        .middleCols(accepted_begin, left_size).transpose() *
        direction.middleRows(right_begin, right_size).transpose();
    return image;
  }
  Eigen::MatrixXd image =
      direction.middleCols(left_begin, left_size).transpose() *
      accepted_action.middleCols(accepted_begin, right_size);
  image.noalias() +=
      coefficients.middleCols(left_begin, left_size).transpose() *
      directional_action.middleCols(directional_begin, right_size);
  return image;
}

AcceptedSpinPairActionResult apply_partner_action_panel(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& accepted_states,
    const SelectedStateDeterminantMatrices* directional_states,
    TargetSpin target_spin,
    int panel_begin,
    int panel_size,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    std::size_t workspace_bytes) {
  const Eigen::MatrixXd vectors = pack_partner_vectors(
      accepted_states,
      directional_states,
      target_spin,
      panel_begin,
      panel_size);
  return apply_accepted_spin_pair_action(
      partner_provider(cache, target_spin),
      active_overlap,
      h1e,
      two_electron,
      vectors,
      workspace_bytes);
}

void build_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    const AcceptedSpinPairActionResult& action,
    TargetSpin target_spin,
    int panel_size,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
  SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_size, right_size);
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state];
    const int action_begin = static_cast<int>(state) * panel_size;
    const Eigen::MatrixXd overlap_image = accepted_panel_image(
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
    weights->partner_total.noalias() += state_weight * accepted_panel_image(
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
    int panel_size,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
  SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_size, right_size);
  const int directional_offset =
      panel_size * static_cast<int>(states.states.size());
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state];
    const auto& direction = directions.states[state];
    const int accepted_begin = static_cast<int>(state) * panel_size;
    const int directional_begin = directional_offset + accepted_begin;
    const Eigen::MatrixXd base_overlap = accepted_panel_image(
        coefficients.coefficient_matrix,
        action.overlap,
        target_spin,
        accepted_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
    const Eigen::MatrixXd delta_overlap = directional_panel_image(
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
    weights->partner_total.noalias() += state_weight * directional_panel_image(
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

void consume_spin_panels(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const SelectedStateDeterminantMatrices* directions,
    const std::vector<double>& energies,
    const std::vector<double>* directional_energies,
    TargetSpin target_spin,
    int tile_extent,
    std::size_t action_workspace_bytes,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    Eigen::MatrixXd* h1e_gradient,
    SameSpinMatrixBackwardContribution* result) {
  const int n_unique = target_size(states, target_spin);
  SameSpinAcceptedTileWeights weights;
  for (int panel_begin = 0; panel_begin < n_unique;
       panel_begin += tile_extent) {
    const int panel_size = std::min(tile_extent, n_unique - panel_begin);
    const AcceptedSpinPairActionResult action = apply_partner_action_panel(
        cache,
        states,
        directions,
        target_spin,
        panel_begin,
        panel_size,
        active_overlap,
        h1e,
        two_electron,
        action_workspace_bytes);
    const auto consume_tile = [&](
                                  int left_begin,
                                  int left_size,
                                  int right_begin,
                                  int right_size) {
      if (directions == nullptr) {
        build_accepted_tile_weights(
            states,
            energies,
            action,
            target_spin,
            panel_size,
            left_begin,
            left_size,
            right_begin,
            right_size,
            &weights);
      } else {
        if (directional_energies == nullptr) {
          throw std::logic_error(
              "directional same-spin panels require energy directions");
        }
        build_directional_tile_weights(
            states,
            *directions,
            energies,
            *directional_energies,
            action,
            target_spin,
            panel_size,
            left_begin,
            left_size,
            right_begin,
            right_size,
            &weights);
      }
      const AcceptedSpinPairTile tile = target_provider(cache, target_spin).build(
          left_begin,
          left_begin + left_size,
          right_begin,
          right_begin + right_size,
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
    };
    if (target_spin == TargetSpin::Alpha) {
      for (int right_begin = 0; right_begin < n_unique;
           right_begin += tile_extent) {
        consume_tile(
            panel_begin,
            panel_size,
            right_begin,
            std::min(tile_extent, n_unique - right_begin));
      }
    } else {
      for (int left_begin = 0; left_begin < n_unique;
           left_begin += tile_extent) {
        consume_tile(
            left_begin,
            std::min(tile_extent, n_unique - left_begin),
            panel_begin,
            panel_size);
      }
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
  const detail::LocalResponseTilePlan tile_plan =
      detail::plan_local_response_tiles(
          cache,
          n_active,
          two_electron,
          static_cast<int>(states.states.size()));
  const bool close_shell = cache.close_shell_reuses_same_spin_pair_cache();
  consume_spin_panels(
      cache,
      states,
      nullptr,
      energies,
      nullptr,
      TargetSpin::Alpha,
      tile_plan.extents.alpha,
      tile_plan.alpha_partner_action_bytes,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &h1e_gradient,
      &result);
  if (close_shell) {
    account_for_close_shell_spin_reuse(&h1e_gradient, &result);
  } else {
    consume_spin_panels(
        cache,
        states,
        nullptr,
        energies,
        nullptr,
        TargetSpin::Beta,
        tile_plan.extents.beta,
        tile_plan.beta_partner_action_bytes,
        n_active,
        active_overlap,
        h1e,
        two_electron,
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
  validate_directional_inputs(
      states, directions, energies, directional_energies);
  for (const auto& state : directions.states) {
    validate_state_coefficient_matrix(
        state, directions.n_unique_alpha, directions.n_unique_beta);
  }

  SameSpinMatrixBackwardContribution result =
      make_zero_backward_contribution(n_active);
  Eigen::MatrixXd h1e_gradient =
      Eigen::MatrixXd::Zero(n_active, n_active);
  const detail::LocalResponseTilePlan tile_plan =
      detail::plan_local_response_tiles(
          cache,
          n_active,
          two_electron,
          static_cast<int>(states.states.size()),
          2);
  const bool close_shell = cache.close_shell_reuses_same_spin_pair_cache();
  consume_spin_panels(
      cache,
      states,
      &directions,
      energies,
      &directional_energies,
      TargetSpin::Alpha,
      tile_plan.extents.alpha,
      tile_plan.alpha_partner_action_bytes,
      n_active,
      active_overlap,
      h1e,
      two_electron,
      &h1e_gradient,
      &result);
  if (close_shell) {
    account_for_close_shell_spin_reuse(&h1e_gradient, &result);
  } else {
    consume_spin_panels(
        cache,
        states,
        &directions,
        energies,
        &directional_energies,
        TargetSpin::Beta,
        tile_plan.extents.beta,
        tile_plan.beta_partner_action_bytes,
        n_active,
        active_overlap,
        h1e,
        two_electron,
        &h1e_gradient,
        &result);
  }
  return finalize_backward_contribution(std::move(result), h1e_gradient);
}

}  // namespace xmvb::vb
