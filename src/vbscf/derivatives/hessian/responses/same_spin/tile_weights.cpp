#include "vbscf/derivatives/hessian/responses/same_spin/tile_weights_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/weight_kernels_internal.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/structures/assembly/local_contractions.hpp"

namespace xmvb::vb::detail {

namespace {

constexpr double kContributionTolerance = 1.0e-15;

double same_spin_pair_overlap_scalar(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int row,
    int column) {
  const int left_id = std::min(row, column);
  const int right_id = std::max(row, column);
  const auto& pair_evaluation =
      ordered_pair_cache[ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants)];
  return pair_evaluation.overlap_result.overlap_determinant;
}

double same_spin_pair_total_scalar(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int row,
    int column) {
  const int left_id = std::min(row, column);
  const int right_id = std::max(row, column);
  const auto& pair_evaluation =
      ordered_pair_cache[ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants)];
  return pair_evaluation.total_hamiltonian;
}

template <typename PartnerTileBuilder>
void accumulate_alpha_single_kernel_image_tile(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    int beta_tile_size,
    PartnerTileBuilder&& build_partner_tile,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const SupportWindow alpha_left_window =
      find_support_window(alpha_support, alpha_left_begin, alpha_left_end);
  const SupportWindow alpha_right_window =
      find_support_window(alpha_support, alpha_right_begin, alpha_right_end);
  if (alpha_left_window.empty() ||
      alpha_right_window.empty() ||
      beta_support.empty()) {
    return;
  }

  // This forms one alpha-side tile of C K_beta C^T. Only the requested
  // alpha rows/columns are materialized; the partner beta support is streamed
  // in bounded tiles so directional overlap-energy terms do not allocate a
  // full unique-spin weight matrix.
  Eigen::MatrixXd partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd image;
  const int n_beta_support = static_cast<int>(beta_support.size());
  for (int beta_left_begin_local = 0;
       beta_left_begin_local < n_beta_support;
       beta_left_begin_local += beta_tile_size) {
    const SupportWindow beta_left_window{
        beta_left_begin_local,
        std::min(n_beta_support, beta_left_begin_local + beta_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int beta_right_begin_local = 0;
         beta_right_begin_local < n_beta_support;
         beta_right_begin_local += beta_tile_size) {
      const SupportWindow beta_right_window{
          beta_right_begin_local,
          std::min(n_beta_support, beta_right_begin_local + beta_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tile(
          beta_support,
          beta_left_window,
          beta_support,
          beta_right_window,
          &partner_tile);
      partner_push.noalias() = left_block * partner_tile;
      image.noalias() = partner_push * right_block.transpose();
      scatter_add_dense_submatrix_to_tile(
          image,
          alpha_support,
          alpha_left_window,
          alpha_left_begin,
          alpha_support,
          alpha_right_window,
          alpha_right_begin,
          scale,
          tile_matrix);
    }
  }
}

template <typename PartnerTileBuilder>
void accumulate_beta_single_kernel_image_tile(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    int alpha_tile_size,
    PartnerTileBuilder&& build_partner_tile,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const SupportWindow beta_left_window =
      find_support_window(beta_support, beta_left_begin, beta_left_end);
  const SupportWindow beta_right_window =
      find_support_window(beta_support, beta_right_begin, beta_right_end);
  if (beta_left_window.empty() ||
      beta_right_window.empty() ||
      alpha_support.empty()) {
    return;
  }

  // Beta-side analogue of the single-kernel alpha tile above. It forms
  // C^T K_alpha C only on the requested beta tile and streams the alpha
  // partner support in bounded chunks.
  Eigen::MatrixXd partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd image;
  const int n_alpha_support = static_cast<int>(alpha_support.size());
  for (int alpha_left_begin_local = 0;
       alpha_left_begin_local < n_alpha_support;
       alpha_left_begin_local += alpha_tile_size) {
    const SupportWindow alpha_left_window{
        alpha_left_begin_local,
        std::min(n_alpha_support, alpha_left_begin_local + alpha_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int alpha_right_begin_local = 0;
         alpha_right_begin_local < n_alpha_support;
         alpha_right_begin_local += alpha_tile_size) {
      const SupportWindow alpha_right_window{
          alpha_right_begin_local,
          std::min(n_alpha_support, alpha_right_begin_local + alpha_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tile(
          alpha_support,
          alpha_left_window,
          alpha_support,
          alpha_right_window,
          &partner_tile);
      partner_push.noalias() = left_block.transpose() * partner_tile;
      image.noalias() = partner_push * right_block;
      scatter_add_dense_submatrix_to_tile(
          image,
          beta_support,
          beta_left_window,
          beta_left_begin,
          beta_support,
          beta_right_window,
          beta_right_begin,
          scale,
          tile_matrix);
    }
  }
}

}  // namespace

void accumulate_alpha_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;
    auto build_partner_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };
    auto consume_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };

    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_beta_accepted_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;
    auto build_partner_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };
    auto consume_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };

    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_alpha_local_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    const SameSpinDirectionalScalarMatrices& partner_directional_scalars,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinLocalTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    // First contraction: accepted and directional overlap-determinant partner
    // kernels. They produce W_H/W_S and dW_H/dW_S on the current alpha tile.
    auto build_overlap_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* delta_overlap_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                .delta_overlap_determinant_matrix(row, column);
          },
          delta_overlap_tile);
    };
    auto consume_overlap_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& delta_overlap_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->delta_overlap);
    };
    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tiles,
        consume_overlap_images);

    // Second contraction: accepted and directional same-spin total partner
    // kernels. The directional total is read as regular + singular without
    // materializing an additional dense sum matrix.
    auto build_total_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* total_tile,
        Eigen::MatrixXd* delta_total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                       .delta_regular_total_hamiltonian_matrix(row, column) +
                partner_directional_scalars
                    .delta_singular_total_hamiltonian_matrix(row, column);
          },
          delta_total_tile);
    };
    auto consume_total_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& total_image,
        const Eigen::MatrixXd& delta_total_image) {
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
      scatter_add_dense_submatrix_to_tile(
          delta_total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_partner_total);
    };
    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_total_tiles,
        consume_total_images);
  }
}

void accumulate_beta_local_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    const SameSpinDirectionalScalarMatrices& partner_directional_scalars,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinLocalTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    auto build_overlap_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* delta_overlap_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                .delta_overlap_determinant_matrix(row, column);
          },
          delta_overlap_tile);
    };
    auto consume_overlap_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& delta_overlap_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          delta_overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          overlap_weight,
          &weights->delta_overlap);
    };
    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tiles,
        consume_overlap_images);

    auto build_total_tiles = [&](
        const std::vector<int>& support,
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        Eigen::MatrixXd* total_tile,
        Eigen::MatrixXd* delta_total_tile) {
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
      gather_scalar_block_from_support(
          support,
          left_window,
          support,
          right_window,
          [&](int row, int column) {
            return partner_directional_scalars
                       .delta_regular_total_hamiltonian_matrix(row, column) +
                partner_directional_scalars
                    .delta_singular_total_hamiltonian_matrix(row, column);
          },
          delta_total_tile);
    };
    auto consume_total_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& total_image,
        const Eigen::MatrixXd& delta_total_image) {
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
      scatter_add_dense_submatrix_to_tile(
          delta_total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->delta_partner_total);
    };
    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_total_tiles,
        consume_total_images);
  }
}

void accumulate_alpha_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    auto build_overlap_tile = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
    };
    accumulate_alpha_single_kernel_image_tile(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tile,
        -state_weight * directional_state_energy,
        &weights->overlap);

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };

    // Directional selected-state response at fixed partner kernels:
    // d(C K C^T) = dC K C^T + C K dC^T.
    auto consume_left_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          directional_state_coefficients.alpha_support,
          left_window,
          left_begin,
          state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_alpha_oriented_support_local_mixed_image_pair(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_left_directional_images);

    auto consume_right_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.alpha_support,
          left_window,
          left_begin,
          directional_state_coefficients.alpha_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_alpha_oriented_support_local_mixed_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_right_directional_images);
  }
}

void accumulate_beta_directional_tile_weights(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    const std::vector<SpinDeterminantPairEvaluation>& partner_pair_cache,
    int n_unique_partner,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    SameSpinAcceptedTileWeights* weights) {
  weights->reset(left_end - left_begin, right_end - right_begin);

  const int partner_tile_size = kSameSpinTileExtent;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    const double state_weight =
        state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    auto build_overlap_tile = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
    };
    accumulate_beta_single_kernel_image_tile(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_overlap_tile,
        -state_weight * directional_state_energy,
        &weights->overlap);

    if (!selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& left_support,
        const SupportWindow& left_window,
        const std::vector<int>& right_support,
        const SupportWindow& right_window,
        Eigen::MatrixXd* overlap_tile,
        Eigen::MatrixXd* total_tile) {
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_overlap_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          overlap_tile);
      gather_scalar_block_from_support(
          left_support,
          left_window,
          right_support,
          right_window,
          [&](int row, int column) {
            return same_spin_pair_total_scalar(
                partner_pair_cache,
                n_unique_partner,
                row,
                column);
          },
          total_tile);
    };

    auto consume_left_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          directional_state_coefficients.beta_support,
          left_window,
          left_begin,
          state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_beta_oriented_support_local_mixed_image_pair(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_left_directional_images);

    auto consume_right_directional_images = [&](
        const SupportWindow& left_window,
        const SupportWindow& right_window,
        const Eigen::MatrixXd& overlap_image,
        const Eigen::MatrixXd& total_image) {
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->hamiltonian);
      scatter_add_dense_submatrix_to_tile(
          overlap_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          -state_weight * state_energy,
          &weights->overlap);
      scatter_add_dense_submatrix_to_tile(
          total_image,
          state_coefficients.beta_support,
          left_window,
          left_begin,
          directional_state_coefficients.beta_support,
          right_window,
          right_begin,
          state_weight,
          &weights->partner_total);
    };
    for_each_beta_oriented_support_local_mixed_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        left_begin,
        left_end,
        right_begin,
        right_end,
        partner_tile_size,
        build_partner_tiles,
        consume_right_directional_images);
  }
}

}  // namespace xmvb::vb::detail
