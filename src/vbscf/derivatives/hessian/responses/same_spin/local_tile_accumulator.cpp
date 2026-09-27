#include "vbscf/derivatives/hessian/responses/same_spin/local_tile_accumulator_internal.hpp"

#include <algorithm>
#include <stdexcept>

#include "vbscf/derivatives/hessian/responses/same_spin/backward_kernels_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"

namespace xmvb::vb::detail {
namespace {

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
    const Eigen::MatrixXd& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    const ActiveSpaceIntegralDirectionView& direction)
    : accepted_pair_cache_(accepted_pair_cache),
      selected_states_(selected_states),
      selected_state_energies_(selected_state_energies),
      n_active_orbitals_(n_active_orbitals),
      active_one_electron_(active_one_electron),
      active_two_electron_(active_two_electron),
      direction_(direction),
      close_shell_same_spin_(
          accepted_pair_cache.close_shell_reuses_same_spin_pair_cache()),
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
    const SameSpinDirectionalPairTile& tile) {
  if (close_shell_same_spin_) {
    if (!alpha_channel || !beta_channel) {
      throw std::invalid_argument(
          "close-shell same-spin tile must represent both spin channels");
    }
    consume_alpha_primary(tile);
    accumulate_alpha_weight_response(tile);
    return;
  }
  if (alpha_channel) {
    consume_alpha_primary(tile);
    accumulate_beta_weight_response(tile);
  }
  if (beta_channel) {
    consume_beta_primary(tile);
    accumulate_alpha_weight_response(tile);
  }
}

void LocalSameSpinTileAccumulator::consume_alpha_primary(
    const SameSpinDirectionalPairTile& tile) {
  SameSpinAcceptedTileWeights weights;
  accumulate_alpha_accepted_tile_weights(
      selected_states_,
      selected_state_energies_,
      accepted_pair_cache_.beta_pair_cache_ref(),
      selected_states_.n_unique_beta,
      tile.left_begin,
      tile.left_begin + tile.left_size(),
      tile.right_begin,
      tile.right_begin + tile.right_size(),
      &weights);
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.alpha_reuse_table.unique_determinants,
      accepted_pair_cache_.alpha_pair_cache_ref(),
      tile,
      weights,
      selected_states_.n_unique_alpha,
      n_active_orbitals_,
      active_one_electron_,
      active_two_electron_,
      direction_.overlap,
      &one_electron_gradient_,
      &result_.active_orbital_overlap_gradient,
      &result_.packed_active_two_electron_gradient);
}

void LocalSameSpinTileAccumulator::consume_beta_primary(
    const SameSpinDirectionalPairTile& tile) {
  SameSpinAcceptedTileWeights weights;
  accumulate_beta_accepted_tile_weights(
      selected_states_,
      selected_state_energies_,
      accepted_pair_cache_.alpha_pair_cache_ref(),
      selected_states_.n_unique_alpha,
      tile.left_begin,
      tile.left_begin + tile.left_size(),
      tile.right_begin,
      tile.right_begin + tile.right_size(),
      &weights);
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.beta_reuse_table.unique_determinants,
      accepted_pair_cache_.beta_pair_cache_ref(),
      tile,
      weights,
      selected_states_.n_unique_beta,
      n_active_orbitals_,
      active_one_electron_,
      active_two_electron_,
      direction_.overlap,
      &one_electron_gradient_,
      &result_.active_orbital_overlap_gradient,
      &result_.packed_active_two_electron_gradient);
}

void LocalSameSpinTileAccumulator::accumulate_beta_weight_response(
    const SameSpinDirectionalPairTile& alpha_tile) {
  const Eigen::MatrixXd delta_hamiltonian =
      alpha_tile.delta_regular_hamiltonian +
      alpha_tile.delta_singular_hamiltonian;
  const int extent = std::min(
      selected_states_.n_unique_beta, kSameSpinTileExtent);
  SameSpinAcceptedTileWeights weights;
  for (int left = 0; left < selected_states_.n_unique_beta; left += extent) {
    const int left_size = std::min(
        extent, selected_states_.n_unique_beta - left);
    for (int right = 0; right < selected_states_.n_unique_beta;
         right += extent) {
      const int right_size = std::min(
          extent, selected_states_.n_unique_beta - right);
      weights.reset(left_size, right_size);
      for (std::size_t state = 0; state < selected_states_.states.size();
           ++state) {
        const auto& coefficients =
            selected_states_.states[state].coefficient_matrix;
        const auto left_coefficients = coefficients.block(
            alpha_tile.left_begin,
            left,
            alpha_tile.left_size(),
            left_size);
        const auto right_coefficients = coefficients.block(
            alpha_tile.right_begin,
            right,
            alpha_tile.right_size(),
            right_size);
        const Eigen::MatrixXd overlap_product =
            left_coefficients.transpose() * alpha_tile.delta_overlap *
            right_coefficients;
        const Eigen::MatrixXd hamiltonian_product =
            left_coefficients.transpose() * delta_hamiltonian *
            right_coefficients;
        add_weight_product(
            overlap_product,
            hamiltonian_product,
            selected_states_.states[state].normalized_state_weight,
            selected_state_energies_[state],
            &weights);
      }
      accumulate_accepted_pair_weight_response_tile(
          accepted_pair_cache_.beta_reuse_table.unique_determinants,
          accepted_pair_cache_.beta_pair_cache_ref(),
          weights,
          left,
          right,
          selected_states_.n_unique_beta,
          n_active_orbitals_,
          &one_electron_gradient_,
          &result_.active_orbital_overlap_gradient,
          &result_.packed_active_two_electron_gradient);
    }
  }
}

void LocalSameSpinTileAccumulator::accumulate_alpha_weight_response(
    const SameSpinDirectionalPairTile& beta_tile) {
  const Eigen::MatrixXd delta_hamiltonian =
      beta_tile.delta_regular_hamiltonian +
      beta_tile.delta_singular_hamiltonian;
  const int extent = std::min(
      selected_states_.n_unique_alpha, kSameSpinTileExtent);
  SameSpinAcceptedTileWeights weights;
  for (int left = 0; left < selected_states_.n_unique_alpha; left += extent) {
    const int left_size = std::min(
        extent, selected_states_.n_unique_alpha - left);
    for (int right = 0; right < selected_states_.n_unique_alpha;
         right += extent) {
      const int right_size = std::min(
          extent, selected_states_.n_unique_alpha - right);
      weights.reset(left_size, right_size);
      for (std::size_t state = 0; state < selected_states_.states.size();
           ++state) {
        const auto& coefficients =
            selected_states_.states[state].coefficient_matrix;
        const auto left_coefficients = coefficients.block(
            left,
            beta_tile.left_begin,
            left_size,
            beta_tile.left_size());
        const auto right_coefficients = coefficients.block(
            right,
            beta_tile.right_begin,
            right_size,
            beta_tile.right_size());
        const Eigen::MatrixXd overlap_product =
            left_coefficients * beta_tile.delta_overlap *
            right_coefficients.transpose();
        const Eigen::MatrixXd hamiltonian_product =
            left_coefficients * delta_hamiltonian *
            right_coefficients.transpose();
        add_weight_product(
            overlap_product,
            hamiltonian_product,
            selected_states_.states[state].normalized_state_weight,
            selected_state_energies_[state],
            &weights);
      }
      accumulate_accepted_pair_weight_response_tile(
          accepted_pair_cache_.alpha_reuse_table.unique_determinants,
          accepted_pair_cache_.alpha_pair_cache_ref(),
          weights,
          left,
          right,
          selected_states_.n_unique_alpha,
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
