#include "vbscf/derivatives/hessian/responses/same_spin/local_tile_accumulator_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
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

Eigen::MatrixXd pack_partner_panel_vectors(
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha,
    int panel_begin,
    int panel_size) {
  const int target_size = target_alpha
      ? states.n_unique_alpha
      : states.n_unique_beta;
  const int partner_size = target_alpha
      ? states.n_unique_beta
      : states.n_unique_alpha;
  if (panel_begin < 0 || panel_size <= 0 ||
      panel_begin + panel_size > target_size) {
    throw std::invalid_argument("partner-action panel bounds are invalid");
  }
  Eigen::MatrixXd packed(
      partner_size,
      panel_size * static_cast<int>(states.states.size()));
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& coefficients = states.states[state].coefficient_matrix;
    auto output = packed.middleCols(
        static_cast<int>(state) * panel_size,
        panel_size);
    if (target_alpha) {
      output = coefficients.middleRows(
          panel_begin, panel_size).transpose();
    } else {
      output = coefficients.middleCols(panel_begin, panel_size);
    }
  }
  return packed;
}

AcceptedSpinPairActionResult build_partner_action(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    std::size_t workspace_bytes) {
  return apply_accepted_spin_pair_action(
      target_alpha ? cache.beta_provider() : cache.alpha_provider(),
      active_overlap,
      h1e,
      two_electron,
      pack_partner_vectors(states, target_alpha),
      workspace_bytes);
}

AcceptedSpinPairActionResult build_partner_action_panel(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha,
    int panel_begin,
    int panel_size,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron) {
  return apply_accepted_spin_pair_action(
      target_alpha ? cache.beta_provider() : cache.alpha_provider(),
      active_overlap,
      h1e,
      two_electron,
      pack_partner_panel_vectors(
          states,
          target_alpha,
          panel_begin,
          panel_size),
      kPairTileWorkspaceBytes);
}

bool is_symmetric(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols() || !matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, matrix.cwiseAbs().maxCoeff());
  return (matrix - matrix.transpose()).cwiseAbs().maxCoeff() <=
      64.0 * std::numeric_limits<double>::epsilon() *
          static_cast<double>(std::max<Eigen::Index>(1, matrix.rows())) *
          scale;
}

bool accepted_kernels_are_symmetric(
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e) {
  const int n_active = static_cast<int>(h1e.rows());
  if (active_overlap.size() !=
      static_cast<std::size_t>(n_active) * n_active) {
    return false;
  }
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(), n_active, n_active);
  return is_symmetric(overlap) && is_symmetric(h1e);
}

bool product_fits(
    std::size_t left,
    std::size_t right,
    std::size_t limit) {
  return left == 0 || right <= limit / left;
}

struct PartnerActionPlan {
  bool retain = false;
  std::size_t pair_workspace_bytes = kPairTileWorkspaceBytes;
};

/** Determines whether dense partner images fit beside the pair-tile stream. */
PartnerActionPlan plan_partner_actions(
    const SelectedStateDeterminantMatrices& states,
    bool close_shell,
    std::size_t workspace_bytes) {
  if (workspace_bytes == 0) {
    return {};
  }
  const std::size_t n_alpha = static_cast<std::size_t>(
      std::max(0, states.n_unique_alpha));
  const std::size_t n_beta = static_cast<std::size_t>(
      std::max(0, states.n_unique_beta));
  const std::size_t n_states = states.states.size();
  if (!product_fits(n_alpha, n_beta, workspace_bytes) ||
      !product_fits(n_alpha * n_beta, n_states, workspace_bytes) ||
      !product_fits(
          n_alpha * n_beta * n_states,
          sizeof(double),
          workspace_bytes)) {
    return {};
  }

  const std::size_t state_product_bytes =
      n_alpha * n_beta * n_states * sizeof(double);
  // During construction, the packed coefficient block coexists with both
  // output images.  For an open shell, the already accepted first action also
  // remains live while the second action is built.
  const std::size_t dense_peak_factor = close_shell ? 3 : 5;
  if (!product_fits(
          state_product_bytes,
          dense_peak_factor,
          workspace_bytes / 2)) {
    return {};
  }
  const std::size_t dense_peak_bytes =
      state_product_bytes * dense_peak_factor;
  return {
      true,
      std::max<std::size_t>(
          1, workspace_bytes - dense_peak_bytes)};
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

void build_cached_accepted_weights(
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

void build_panel_accepted_weights(
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    const SameSpinPartnerActionPanel& panel,
    bool target_alpha,
    int left_begin,
    int left_size,
    int right_begin,
    int right_size,
    SameSpinAcceptedTileWeights* weights) {
  if (left_begin < panel.begin ||
      left_begin + left_size > panel.begin + panel.size) {
    throw std::logic_error(
        "accepted weight tile is outside its partner-action panel");
  }
  weights->reset(left_size, right_size);
  const int panel_offset = left_begin - panel.begin;
  for (std::size_t state = 0; state < states.states.size(); ++state) {
    const auto& selected = states.states[state];
    const int action_begin = static_cast<int>(state) * panel.size;
    const auto overlap_image = panel.action.overlap.middleCols(
        action_begin + panel_offset, left_size);
    const auto hamiltonian_image = panel.action.hamiltonian.middleCols(
        action_begin + panel_offset, left_size);
    Eigen::MatrixXd product(left_size, right_size);
    if (target_alpha) {
      const auto right_coefficients = selected.coefficient_matrix.middleRows(
          right_begin, right_size);
      product.noalias() =
          overlap_image.transpose() * right_coefficients.transpose();
      weights->hamiltonian.noalias() +=
          selected.normalized_state_weight * product;
      weights->overlap.noalias() -=
          selected.normalized_state_weight * energies[state] * product;
      product.noalias() =
          hamiltonian_image.transpose() * right_coefficients.transpose();
    } else {
      const auto right_coefficients = selected.coefficient_matrix.middleCols(
          right_begin, right_size);
      product.noalias() =
          overlap_image.transpose() * right_coefficients;
      weights->hamiltonian.noalias() +=
          selected.normalized_state_weight * product;
      weights->overlap.noalias() -=
          selected.normalized_state_weight * energies[state] * product;
      product.noalias() =
          hamiltonian_image.transpose() * right_coefficients;
    }
    weights->partner_total.noalias() +=
        selected.normalized_state_weight * product;
  }
}

bool is_reverse_of(
    const SameSpinAcceptedWeightTile& cached,
    const SameSpinDirectionalPairTileView& tile) {
  return tile.transposed() &&
      cached.left_begin == tile.right_begin() &&
      cached.right_begin == tile.left_begin() &&
      cached.weights.hamiltonian.rows() == tile.right_size() &&
      cached.weights.hamiltonian.cols() == tile.left_size();
}

SameSpinAcceptedTileWeights transpose_weights(
    const SameSpinAcceptedTileWeights& source) {
  SameSpinAcceptedTileWeights result;
  result.hamiltonian = source.hamiltonian.transpose();
  result.overlap = source.overlap.transpose();
  result.partner_total = source.partner_total.transpose();
  return result;
}

void refresh_partner_panel(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    bool target_alpha,
    int panel_begin,
    int panel_size,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    SameSpinPartnerActionPanel* panel,
    std::size_t* build_count) {
  if (panel->begin == panel_begin && panel->size == panel_size) {
    return;
  }
  panel->begin = panel_begin;
  panel->size = panel_size;
  panel->action = build_partner_action_panel(
      cache,
      states,
      target_alpha,
      panel_begin,
      panel_size,
      active_overlap,
      h1e,
      two_electron);
  ++(*build_count);
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
    const ActiveSpaceIntegralDirectionView& direction,
    std::size_t retained_action_budget_bytes)
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
      accepted_kernels_are_symmetric_(accepted_kernels_are_symmetric(
          active_overlap,
          active_one_electron)),
      tile_extents_(plan_pair_tile_extents(
          accepted_pair_cache,
          n_active_orbitals,
          active_two_electron,
          true)),
      result_(make_zero_backward_contribution(n_active_orbitals)),
      one_electron_gradient_(
          Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals)) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "tiled local same-spin state energies have inconsistent dimensions");
  }
  const PartnerActionPlan action_plan = plan_partner_actions(
      selected_states_,
      close_shell_same_spin_,
      retained_action_budget_bytes);
  if (action_plan.retain) {
    alpha_partner_action_.emplace(build_partner_action(
        accepted_pair_cache_,
        selected_states_,
        true,
        active_overlap_,
        active_one_electron_,
        active_two_electron_,
        action_plan.pair_workspace_bytes));
    if (!close_shell_same_spin_) {
      beta_partner_action_.emplace(build_partner_action(
          accepted_pair_cache_,
          selected_states_,
          false,
          active_overlap_,
          active_one_electron_,
          active_two_electron_,
          action_plan.pair_workspace_bytes));
    }
  } else if (!accepted_kernels_are_symmetric_) {
    throw std::invalid_argument(
        "panelized same-spin response requires symmetric accepted S/H kernels");
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
  SameSpinAcceptedTileWeights transient_weights;
  const SameSpinAcceptedTileWeights* weights = nullptr;
  if (alpha_partner_action_.has_value()) {
    build_cached_accepted_weights(
        selected_states_,
        selected_state_energies_,
        *alpha_partner_action_,
        true,
        tile.left_begin(),
        tile.left_size(),
        tile.right_begin(),
        tile.right_size(),
        &transient_weights);
    weights = &transient_weights;
  } else {
    if (tile.transposed()) {
      if (!last_alpha_weights_.has_value() ||
          !is_reverse_of(*last_alpha_weights_, tile)) {
        throw std::logic_error(
            "reverse alpha tile is not adjacent to its forward tile");
      }
      transient_weights = transpose_weights(last_alpha_weights_->weights);
      weights = &transient_weights;
    } else {
      refresh_partner_panel(
          accepted_pair_cache_,
          selected_states_,
          true,
          tile.left_begin(),
          tile.left_size(),
          active_overlap_,
          active_one_electron_,
          active_two_electron_,
          &alpha_partner_panel_,
          &alpha_partner_panel_build_count_);
      last_alpha_weights_.emplace();
      last_alpha_weights_->left_begin = tile.left_begin();
      last_alpha_weights_->right_begin = tile.right_begin();
      build_panel_accepted_weights(
          selected_states_,
          selected_state_energies_,
          alpha_partner_panel_,
          true,
          tile.left_begin(),
          tile.left_size(),
          tile.right_begin(),
          tile.right_size(),
          &last_alpha_weights_->weights);
      weights = &last_alpha_weights_->weights;
    }
  }
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.alpha_reuse_table.unique_determinants,
      accepted,
      tile,
      *weights,
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
  SameSpinAcceptedTileWeights transient_weights;
  const SameSpinAcceptedTileWeights* weights = nullptr;
  if (beta_partner_action_.has_value()) {
    build_cached_accepted_weights(
        selected_states_,
        selected_state_energies_,
        *beta_partner_action_,
        false,
        tile.left_begin(),
        tile.left_size(),
        tile.right_begin(),
        tile.right_size(),
        &transient_weights);
    weights = &transient_weights;
  } else {
    if (tile.transposed()) {
      if (!last_beta_weights_.has_value() ||
          !is_reverse_of(*last_beta_weights_, tile)) {
        throw std::logic_error(
            "reverse beta tile is not adjacent to its forward tile");
      }
      transient_weights = transpose_weights(last_beta_weights_->weights);
      weights = &transient_weights;
    } else {
      refresh_partner_panel(
          accepted_pair_cache_,
          selected_states_,
          false,
          tile.left_begin(),
          tile.left_size(),
          active_overlap_,
          active_one_electron_,
          active_two_electron_,
          &beta_partner_panel_,
          &beta_partner_panel_build_count_);
      last_beta_weights_.emplace();
      last_beta_weights_->left_begin = tile.left_begin();
      last_beta_weights_->right_begin = tile.right_begin();
      build_panel_accepted_weights(
          selected_states_,
          selected_state_energies_,
          beta_partner_panel_,
          false,
          tile.left_begin(),
          tile.left_size(),
          tile.right_begin(),
          tile.right_size(),
          &last_beta_weights_->weights);
      weights = &last_beta_weights_->weights;
    }
  }
  accumulate_local_primary_pair_tile(
      accepted_pair_cache_.beta_reuse_table.unique_determinants,
      accepted,
      tile,
      *weights,
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
