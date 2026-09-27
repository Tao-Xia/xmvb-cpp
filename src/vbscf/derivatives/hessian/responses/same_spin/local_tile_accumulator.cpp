#include "vbscf/derivatives/hessian/responses/same_spin/local_tile_accumulator_internal.hpp"

#include <algorithm>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/same_spin/backward_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"

namespace xmvb::vb::detail {
namespace {

Eigen::MatrixXd pack_partner_vectors(
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha) {
  const int target_size = target_alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
  const int partner_size = target_alpha
      ? states.n_unique_beta
      : states.n_unique_alpha;
  Eigen::MatrixXd packed(
      partner_size,
      target_size * static_cast<int>(states.states.size()));
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state].coefficient_matrix;
    packed.middleCols(
        static_cast<int>(state) * target_size,
        target_size) = target_alpha
        ? coefficients.transpose()
        : coefficients;
  }
  return packed;
}

AcceptedSpinPairActionResult build_partner_action(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  return apply_accepted_spin_pair_action(
      target_alpha ? cache.beta_provider() : cache.alpha_provider(),
      active_overlap,
      h1e,
      two_electron,
      pack_partner_vectors(states, target_alpha));
}

Eigen::MatrixXd partner_image_tile(
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& action,
    bool target_alpha,
    int action_begin,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size) {
  if (target_alpha) {
    return action.middleCols(action_begin + left_begin, left_size).transpose() *
        coefficients.middleRows(right_begin, right_size).transpose();
  }
  return coefficients.middleCols(left_begin, left_size).transpose() *
      action.middleCols(action_begin + right_begin, right_size);
}

void build_accepted_weights(
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    const AcceptedSpinPairActionResult& action,
    bool target_alpha,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_size, right_size);
  const int target_size = target_alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& selected = states.states[state];
    const int action_begin = static_cast<int>(state) * target_size;
    const Eigen::MatrixXd overlap = partner_image_tile(
        selected.coefficient_matrix,
        action.overlap,
        target_alpha,
        action_begin,
        left_begin,
        left_size,
        right_begin,
        right_size);
    weights->hamiltonian.noalias() +=
        selected.normalized_state_weight * overlap;
    weights->overlap.noalias() -=
        selected.normalized_state_weight * energies[state] * overlap;
    weights->partner_total.noalias() += selected.normalized_state_weight *
        partner_image_tile(
            selected.coefficient_matrix,
            action.hamiltonian,
            target_alpha,
            action_begin,
            left_begin,
            left_size,
            right_begin,
            right_size);
  }
}

void add_weight_product(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_product,
    const Eigen::Ref<const Eigen::MatrixXd>& hamiltonian_product,
    double state_weight,
    double state_energy,
    SameSpinAcceptedTileWeights* weights) {
  weights->hamiltonian.noalias() += state_weight * overlap_product;
  weights->overlap.noalias() -=
      state_weight * state_energy * overlap_product;
  weights->partner_total.noalias() += state_weight * hamiltonian_product;
}

}  // namespace

LocalSameSpinTileAccumulator::LocalSameSpinTileAccumulator(
    const SameSpinPairCacheContext& accepted_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const std::vector<double>& active_overlap,
    const Eigen::MatrixXd& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    const ActiveSpaceIntegralDirectionView& direction)
    : accepted_pair_cache_(accepted_pair_cache),
      selected_states_(selected_states),
      selected_state_energies_(selected_state_energies),
      n_active_orbitals_(n_active_orbitals),
      active_overlap_(active_overlap),
      active_one_electron_(active_one_electron),
      active_two_electron_(active_two_electron),
      direction_(direction),
      close_shell_same_spin_(
          accepted_pair_cache.close_shell_reuses_same_spin_pair_cache()),
      tile_extents_(plan_pair_tile_extents(
          accepted_pair_cache,
          n_active_orbitals,
          active_two_electron,
          true)),
      alpha_partner_action_(build_partner_action(
          accepted_pair_cache,
          selected_states,
          true,
          active_overlap,
          active_one_electron,
          active_two_electron)),
      beta_partner_action_(close_shell_same_spin_
          ? AcceptedSpinPairActionResult{}
          : build_partner_action(
                accepted_pair_cache,
                selected_states,
                false,
                active_overlap,
                active_one_electron,
                active_two_electron)),
      result_(make_zero_backward_contribution(n_active_orbitals)),
      one_electron_gradient_(
          Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals)) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "tiled local same-spin state energies have inconsistent dimensions");
  }
}

void LocalSameSpinTileAccumulator::consume(
    bool alpha_channel,
    bool beta_channel,
    const AcceptedSpinPairTile& accepted,
    const SameSpinDirectionalPairTileView& tile) {
  if (close_shell_same_spin_) {
    if (!alpha_channel || !beta_channel) {
      throw std::invalid_argument(
          "close-shell same-spin tile must represent both spin channels");
    }
    consume_alpha_primary(accepted, tile);
    accumulate_alpha_weight_response(tile);
    return;
  }
  if (alpha_channel) {
    consume_alpha_primary(accepted, tile);
    accumulate_beta_weight_response(tile);
  }
  if (beta_channel) {
    consume_beta_primary(accepted, tile);
    accumulate_alpha_weight_response(tile);
  }
}

void LocalSameSpinTileAccumulator::consume_alpha_primary(
    const AcceptedSpinPairTile& accepted,
    const SameSpinDirectionalPairTileView& tile) {
  SameSpinAcceptedTileWeights weights;
  build_accepted_weights(
      selected_states_,
      selected_state_energies_,
      alpha_partner_action_,
      true,
      tile.left_begin(),
      tile.left_size(),
      tile.right_begin(),
      tile.right_size(),
      &weights);
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.alpha_reuse_table.unique_determinants,
      accepted,
      tile,
      weights,
      n_active_orbitals_,
      active_one_electron_,
      active_two_electron_,
      direction_.overlap,
      &one_electron_gradient_,
      &result_.active_orbital_overlap_gradient,
      &result_.packed_active_two_electron_gradient);
}

void LocalSameSpinTileAccumulator::consume_beta_primary(
    const AcceptedSpinPairTile& accepted,
    const SameSpinDirectionalPairTileView& tile) {
  SameSpinAcceptedTileWeights weights;
  build_accepted_weights(
      selected_states_,
      selected_state_energies_,
      beta_partner_action_,
      false,
      tile.left_begin(),
      tile.left_size(),
      tile.right_begin(),
      tile.right_size(),
      &weights);
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.beta_reuse_table.unique_determinants,
      accepted,
      tile,
      weights,
      n_active_orbitals_,
      active_one_electron_,
      active_two_electron_,
      direction_.overlap,
      &one_electron_gradient_,
      &result_.active_orbital_overlap_gradient,
      &result_.packed_active_two_electron_gradient);
}

void LocalSameSpinTileAccumulator::accumulate_beta_weight_response(
    const SameSpinDirectionalPairTileView& alpha_tile) {
  const Eigen::MatrixXd delta_hamiltonian =
      alpha_tile.delta_regular_hamiltonian() +
      alpha_tile.delta_singular_hamiltonian();
  const int extent = std::min(
      selected_states_.n_unique_beta, tile_extents_.beta);
  const int n_left_tiles =
      (selected_states_.n_unique_beta + extent - 1) / extent;
  std::vector<SameSpinAcceptedTileWeights> weights(n_left_tiles);
  for (int right = 0; right < selected_states_.n_unique_beta;
       right += extent) {
    const int right_size = std::min(
        extent, selected_states_.n_unique_beta - right);
    for (int left_tile = 0; left_tile < n_left_tiles; ++left_tile) {
      const int left = left_tile * extent;
      weights[left_tile].reset(
          std::min(extent, selected_states_.n_unique_beta - left),
          right_size);
    }
    for (std::size_t state = 0; state < selected_states_.states.size();
         ++state) {
      const auto& selected_state = selected_states_.states[state];
      const auto& coefficients = selected_state.coefficient_matrix;
      const auto right_coefficients = coefficients.block(
          alpha_tile.right_begin(),
          right,
          alpha_tile.right_size(),
          right_size);
      const Eigen::MatrixXd overlap_right =
          alpha_tile.delta_overlap() * right_coefficients;
      const Eigen::MatrixXd hamiltonian_right =
          delta_hamiltonian * right_coefficients;
      for (int left_tile = 0; left_tile < n_left_tiles; ++left_tile) {
        const int left = left_tile * extent;
        const int left_size = std::min(
            extent, selected_states_.n_unique_beta - left);
        const auto left_coefficients = coefficients.block(
            alpha_tile.left_begin(),
            left,
            alpha_tile.left_size(),
            left_size);
        add_weight_product(
            left_coefficients.transpose() * overlap_right,
            left_coefficients.transpose() * hamiltonian_right,
            selected_state.normalized_state_weight,
            selected_state_energies_[state],
            &weights[left_tile]);
      }
    }
    for (int left_tile = 0; left_tile < n_left_tiles; ++left_tile) {
      const int left = left_tile * extent;
      const int left_size = std::min(
          extent, selected_states_.n_unique_beta - left);
      const AcceptedSpinPairTile accepted =
          accepted_pair_cache_.beta_provider().build(
              left,
              left + left_size,
              right,
              right + right_size,
              active_overlap_,
              active_one_electron_,
              active_two_electron_,
              AcceptedPairTileBuildOptions{
                  .materialize_projected_pair_values = false,
                  .populate_response_payload = true,
                  .populate_opposite_spin_projection = false});
      accumulate_accepted_spin_backward_tile(
          accepted_pair_cache_.beta_reuse_table.unique_determinants,
          accepted,
          weights[left_tile].hamiltonian,
          weights[left_tile].overlap,
          weights[left_tile].partner_total,
          n_active_orbitals_,
          &one_electron_gradient_,
          &result_.active_orbital_overlap_gradient,
          &result_.packed_active_two_electron_gradient);
    }
  }
}

void LocalSameSpinTileAccumulator::accumulate_alpha_weight_response(
    const SameSpinDirectionalPairTileView& beta_tile) {
  const Eigen::MatrixXd delta_hamiltonian =
      beta_tile.delta_regular_hamiltonian() +
      beta_tile.delta_singular_hamiltonian();
  const int extent = std::min(
      selected_states_.n_unique_alpha, tile_extents_.alpha);
  const int n_right_tiles =
      (selected_states_.n_unique_alpha + extent - 1) / extent;
  std::vector<SameSpinAcceptedTileWeights> weights(n_right_tiles);
  for (int left = 0; left < selected_states_.n_unique_alpha; left += extent) {
    const int left_size = std::min(
        extent, selected_states_.n_unique_alpha - left);
    for (int right_tile = 0; right_tile < n_right_tiles; ++right_tile) {
      const int right = right_tile * extent;
      weights[right_tile].reset(
          left_size,
          std::min(extent, selected_states_.n_unique_alpha - right));
    }
    for (std::size_t state = 0; state < selected_states_.states.size();
         ++state) {
      const auto& selected_state = selected_states_.states[state];
      const auto& coefficients = selected_state.coefficient_matrix;
      const auto left_coefficients = coefficients.block(
          left,
          beta_tile.left_begin(),
          left_size,
          beta_tile.left_size());
      const Eigen::MatrixXd overlap_left =
          left_coefficients * beta_tile.delta_overlap();
      const Eigen::MatrixXd hamiltonian_left =
          left_coefficients * delta_hamiltonian;
      for (int right_tile = 0; right_tile < n_right_tiles; ++right_tile) {
        const int right = right_tile * extent;
        const int right_size = std::min(
            extent, selected_states_.n_unique_alpha - right);
        const auto right_coefficients = coefficients.block(
            right,
            beta_tile.right_begin(),
            right_size,
            beta_tile.right_size());
        add_weight_product(
            overlap_left * right_coefficients.transpose(),
            hamiltonian_left * right_coefficients.transpose(),
            selected_state.normalized_state_weight,
            selected_state_energies_[state],
            &weights[right_tile]);
      }
    }
    for (int right_tile = 0; right_tile < n_right_tiles; ++right_tile) {
      const int right = right_tile * extent;
      const int right_size = std::min(
          extent, selected_states_.n_unique_alpha - right);
      const AcceptedSpinPairTile accepted =
          accepted_pair_cache_.alpha_provider().build(
              left,
              left + left_size,
              right,
              right + right_size,
              active_overlap_,
              active_one_electron_,
              active_two_electron_,
              AcceptedPairTileBuildOptions{
                  .materialize_projected_pair_values = false,
                  .populate_response_payload = true,
                  .populate_opposite_spin_projection = false});
      accumulate_accepted_spin_backward_tile(
          accepted_pair_cache_.alpha_reuse_table.unique_determinants,
          accepted,
          weights[right_tile].hamiltonian,
          weights[right_tile].overlap,
          weights[right_tile].partner_total,
          n_active_orbitals_,
          &one_electron_gradient_,
          &result_.active_orbital_overlap_gradient,
          &result_.packed_active_two_electron_gradient);
    }
  }
}

SameSpinMatrixBackwardContribution LocalSameSpinTileAccumulator::finish() {
  if (close_shell_same_spin_) {
    account_for_close_shell_spin_reuse(&one_electron_gradient_, &result_);
  }
  return finalize_backward_contribution(
      std::move(result_), one_electron_gradient_);
}

}  // namespace xmvb::vb::detail
