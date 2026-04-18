#include "vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrix_utils.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

constexpr double kContributionTolerance = 1.0e-15;
constexpr int kSameSpinBackwardPairTileSize = 64;

struct SupportWindow {
  int begin = 0;
  int end = 0;

  int size() const {
    return end - begin;
  }

  bool empty() const {
    return begin >= end;
  }
};

std::size_t square_storage_size(int dimension) {
  return xmvb::product_size(dimension, dimension);
}

int same_spin_backward_pair_tile_size() {
  const char* env_value = std::getenv("XMVB_CPP_SAME_SPIN_BACKWARD_PAIR_TILE_SIZE");
  if (env_value == nullptr || env_value[0] == '\0') {
    return kSameSpinBackwardPairTileSize;
  }
  const int parsed_value = std::stoi(env_value);
  if (parsed_value <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_SAME_SPIN_BACKWARD_PAIR_TILE_SIZE must be positive");
  }
  return parsed_value;
}

double max_abs_dense_matrix(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.size() == 0) {
    return 0.0;
  }
  return matrix.cwiseAbs().maxCoeff();
}

void validate_full_matrix_same_spin_inputs(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "biorthogonal same-spin matrix backward requires an enabled same-spin cache");
  }
  if (selected_state_matrices.states.empty()) {
    throw std::invalid_argument("selected_state_matrices.states must not be empty");
  }
  if (static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) !=
          selected_state_matrices.n_unique_alpha ||
      static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size()) !=
          selected_state_matrices.n_unique_beta) {
    throw std::invalid_argument(
        "same-spin cache dimensions do not match biorthogonal selected-state matrices");
  }
  if (same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id !=
          selected_state_matrices.determinant_to_unique_alpha_id ||
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id !=
          selected_state_matrices.determinant_to_unique_beta_id) {
    throw std::invalid_argument(
        "determinant-to-unique reuse maps do not match selected-state matrices");
  }
}

SupportWindow find_support_window(
    const std::vector<int>& support,
    int global_begin,
    int global_end) {
  const auto begin_iterator =
      std::lower_bound(support.begin(), support.end(), global_begin);
  const auto end_iterator =
      std::lower_bound(begin_iterator, support.end(), global_end);
  return SupportWindow{
      static_cast<int>(std::distance(support.begin(), begin_iterator)),
      static_cast<int>(std::distance(support.begin(), end_iterator)),
  };
}

std::vector<int> build_support_global_indices(
    const std::vector<int>& support,
    const SupportWindow& window) {
  return std::vector<int>(
      support.begin() + window.begin,
      support.begin() + window.end);
}

std::vector<int> build_tile_offsets_from_support_window(
    const std::vector<int>& support,
    const SupportWindow& window,
    int tile_begin) {
  std::vector<int> offsets;
  offsets.reserve(xmvb::to_size(window.size()));
  for (int support_index = window.begin; support_index < window.end; ++support_index) {
    offsets.push_back(support[xmvb::to_size(support_index)] - tile_begin);
  }
  return offsets;
}

void scatter_add_dense_submatrix_to_tile(
    const Eigen::Ref<const Eigen::MatrixXd>& local_matrix,
    const std::vector<int>& row_tile_offsets,
    const std::vector<int>& column_tile_offsets,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (tile_matrix == nullptr) {
    throw std::invalid_argument("tile_matrix must not be null");
  }
  if (local_matrix.rows() != static_cast<int>(row_tile_offsets.size()) ||
      local_matrix.cols() != static_cast<int>(column_tile_offsets.size())) {
    throw std::invalid_argument(
        "local_matrix shape does not match tile scatter offsets");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  for (int column_local = 0;
       column_local < static_cast<int>(column_tile_offsets.size());
       ++column_local) {
    const int tile_column = column_tile_offsets[xmvb::to_size(column_local)];
    for (int row_local = 0;
         row_local < static_cast<int>(row_tile_offsets.size());
         ++row_local) {
      const int tile_row = row_tile_offsets[xmvb::to_size(row_local)];
      (*tile_matrix)(tile_row, tile_column) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

void gather_same_spin_scalar_blocks_from_ordered_cache(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* regular_total_block,
    Eigen::MatrixXd* singular_total_block) {
  if (overlap_block == nullptr) {
    throw std::invalid_argument("overlap_block must not be null");
  }
  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  if (regular_total_block != nullptr) {
    regular_total_block->resize(
        static_cast<int>(row_indices.size()),
        static_cast<int>(column_indices.size()));
  }
  if (singular_total_block != nullptr) {
    singular_total_block->resize(
        static_cast<int>(row_indices.size()),
        static_cast<int>(column_indices.size()));
  }

  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    if (column_global < 0 || column_global >= n_unique_determinants) {
      throw std::out_of_range("column index is out of range");
    }
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      if (row_global < 0 || row_global >= n_unique_determinants) {
        throw std::out_of_range("row index is out of range");
      }
      const auto& pair_evaluation =
          ordered_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
              row_global,
              column_global,
              n_unique_determinants)];
      (*overlap_block)(row_local, column_local) =
          pair_evaluation.overlap_result.overlap_determinant;
      if (regular_total_block != nullptr) {
        (*regular_total_block)(row_local, column_local) =
            pair_evaluation.overlap_result.nullity == 0
                ? pair_evaluation.total_hamiltonian
                : 0.0;
      }
      if (singular_total_block != nullptr) {
        (*singular_total_block)(row_local, column_local) =
            pair_evaluation.overlap_result.nullity == 0
                ? 0.0
                : pair_evaluation.total_hamiltonian;
      }
    }
  }
}

void validate_local_state_matrices(
    const BiorthogonalSelectedStateDeterminantCoefficients& state_coefficients) {
  validate_biorthogonal_state_matrix_shape(
      state_coefficients.local_left_coefficient_matrix,
      static_cast<int>(state_coefficients.alpha_support.size()),
      static_cast<int>(state_coefficients.beta_support.size()),
      "local_left_coefficient_matrix");
  validate_biorthogonal_state_matrix_shape(
      state_coefficients.local_right_coefficient_matrix,
      static_cast<int>(state_coefficients.alpha_support.size()),
      static_cast<int>(state_coefficients.beta_support.size()),
      "local_right_coefficient_matrix");
  validate_biorthogonal_state_matrix_shape(
      state_coefficients.local_residual_coefficient_matrix,
      static_cast<int>(state_coefficients.alpha_support.size()),
      static_cast<int>(state_coefficients.beta_support.size()),
      "local_residual_coefficient_matrix");
}

void accumulate_alpha_selected_state_weight_tile_from_partner_cache(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& beta_pair_cache,
    int n_unique_beta,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    bool include_partner_total,
    Eigen::MatrixXd* overlap_weight_tile,
    Eigen::MatrixXd* partner_total_tile,
    Eigen::MatrixXd* singular_partner_tile) {
  if (overlap_weight_tile == nullptr ||
      partner_total_tile == nullptr ||
      singular_partner_tile == nullptr) {
    throw std::invalid_argument("alpha weight-tile outputs must not be null");
  }

  const int tile_rows = alpha_left_end - alpha_left_begin;
  const int tile_cols = alpha_right_end - alpha_right_begin;
  *overlap_weight_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);
  *partner_total_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);
  *singular_partner_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);

  const int partner_tile_size =
      std::min(n_unique_beta, same_spin_backward_pair_tile_size());
  Eigen::MatrixXd beta_overlap_block;
  Eigen::MatrixXd beta_regular_total_block;
  Eigen::MatrixXd beta_singular_total_block;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_image;

  for (const auto& state_coefficients : selected_state_matrices.states) {
    const double state_weight = state_coefficients.normalized_state_weight;
    if (std::abs(state_weight) <= kContributionTolerance) {
      continue;
    }
    validate_local_state_matrices(state_coefficients);

    const SupportWindow alpha_left_window =
        find_support_window(
            state_coefficients.alpha_support,
            alpha_left_begin,
            alpha_left_end);
    const SupportWindow alpha_right_window =
        find_support_window(
            state_coefficients.alpha_support,
            alpha_right_begin,
            alpha_right_end);
    if (alpha_left_window.empty() || alpha_right_window.empty()) {
      continue;
    }

    const std::vector<int> alpha_left_offsets =
        build_tile_offsets_from_support_window(
            state_coefficients.alpha_support,
            alpha_left_window,
            alpha_left_begin);
    const std::vector<int> alpha_right_offsets =
        build_tile_offsets_from_support_window(
            state_coefficients.alpha_support,
            alpha_right_window,
            alpha_right_begin);

    const int n_beta_support =
        static_cast<int>(state_coefficients.beta_support.size());
    for (int beta_left_begin_local = 0;
         beta_left_begin_local < n_beta_support;
         beta_left_begin_local += partner_tile_size) {
      const SupportWindow beta_left_window{
          beta_left_begin_local,
          std::min(n_beta_support, beta_left_begin_local + partner_tile_size),
      };
      const std::vector<int> beta_left_indices =
          build_support_global_indices(
              state_coefficients.beta_support,
              beta_left_window);

      const auto left_block =
          state_coefficients.local_left_coefficient_matrix.block(
              alpha_left_window.begin,
              beta_left_window.begin,
              alpha_left_window.size(),
              beta_left_window.size());
      const auto residual_block =
          state_coefficients.local_residual_coefficient_matrix.block(
              alpha_left_window.begin,
              beta_left_window.begin,
              alpha_left_window.size(),
              beta_left_window.size());

      for (int beta_right_begin_local = 0;
           beta_right_begin_local < n_beta_support;
           beta_right_begin_local += partner_tile_size) {
        const SupportWindow beta_right_window{
            beta_right_begin_local,
            std::min(n_beta_support, beta_right_begin_local + partner_tile_size),
        };
        const std::vector<int> beta_right_indices =
            build_support_global_indices(
                state_coefficients.beta_support,
                beta_right_window);

        gather_same_spin_scalar_blocks_from_ordered_cache(
            beta_pair_cache,
            n_unique_beta,
            beta_left_indices,
            beta_right_indices,
            &beta_overlap_block,
            include_partner_total ? &beta_regular_total_block : nullptr,
            include_partner_total ? &beta_singular_total_block : nullptr);

        const auto right_block =
            state_coefficients.local_right_coefficient_matrix.block(
                alpha_right_window.begin,
                beta_right_window.begin,
                alpha_right_window.size(),
                beta_right_window.size());

        // This is the local streamed version of
        //   W_overlap(alpha_L, alpha_R) += Q * S_beta * R^T
        // and, when requested,
        //   W_partner(alpha_L, alpha_R) += L * H_beta * R^T.
        beta_push.noalias() = residual_block * beta_overlap_block;
        alpha_image.noalias() = beta_push * right_block.transpose();
        scatter_add_dense_submatrix_to_tile(
            alpha_image,
            alpha_left_offsets,
            alpha_right_offsets,
            state_weight,
            overlap_weight_tile);

        if (!include_partner_total) {
          continue;
        }

        beta_push.noalias() = left_block * beta_regular_total_block;
        alpha_image.noalias() = beta_push * right_block.transpose();
        scatter_add_dense_submatrix_to_tile(
            alpha_image,
            alpha_left_offsets,
            alpha_right_offsets,
            state_weight,
            partner_total_tile);

        beta_push.noalias() = left_block * beta_singular_total_block;
        alpha_image.noalias() = beta_push * right_block.transpose();
        scatter_add_dense_submatrix_to_tile(
            alpha_image,
            alpha_left_offsets,
            alpha_right_offsets,
            state_weight,
            singular_partner_tile);
      }
    }
  }
}

void accumulate_beta_selected_state_weight_tile_from_partner_cache(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& alpha_pair_cache,
    int n_unique_alpha,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    bool include_partner_total,
    Eigen::MatrixXd* overlap_weight_tile,
    Eigen::MatrixXd* partner_total_tile,
    Eigen::MatrixXd* singular_partner_tile) {
  if (overlap_weight_tile == nullptr ||
      partner_total_tile == nullptr ||
      singular_partner_tile == nullptr) {
    throw std::invalid_argument("beta weight-tile outputs must not be null");
  }

  const int tile_rows = beta_left_end - beta_left_begin;
  const int tile_cols = beta_right_end - beta_right_begin;
  *overlap_weight_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);
  *partner_total_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);
  *singular_partner_tile = Eigen::MatrixXd::Zero(tile_rows, tile_cols);

  const int partner_tile_size =
      std::min(n_unique_alpha, same_spin_backward_pair_tile_size());
  Eigen::MatrixXd alpha_overlap_block;
  Eigen::MatrixXd alpha_regular_total_block;
  Eigen::MatrixXd alpha_singular_total_block;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd beta_image;

  for (const auto& state_coefficients : selected_state_matrices.states) {
    const double state_weight = state_coefficients.normalized_state_weight;
    if (std::abs(state_weight) <= kContributionTolerance) {
      continue;
    }
    validate_local_state_matrices(state_coefficients);

    const SupportWindow beta_left_window =
        find_support_window(
            state_coefficients.beta_support,
            beta_left_begin,
            beta_left_end);
    const SupportWindow beta_right_window =
        find_support_window(
            state_coefficients.beta_support,
            beta_right_begin,
            beta_right_end);
    if (beta_left_window.empty() || beta_right_window.empty()) {
      continue;
    }

    const std::vector<int> beta_left_offsets =
        build_tile_offsets_from_support_window(
            state_coefficients.beta_support,
            beta_left_window,
            beta_left_begin);
    const std::vector<int> beta_right_offsets =
        build_tile_offsets_from_support_window(
            state_coefficients.beta_support,
            beta_right_window,
            beta_right_begin);

    const int n_alpha_support =
        static_cast<int>(state_coefficients.alpha_support.size());
    for (int alpha_left_begin_local = 0;
         alpha_left_begin_local < n_alpha_support;
         alpha_left_begin_local += partner_tile_size) {
      const SupportWindow alpha_left_window{
          alpha_left_begin_local,
          std::min(n_alpha_support, alpha_left_begin_local + partner_tile_size),
      };
      const std::vector<int> alpha_left_indices =
          build_support_global_indices(
              state_coefficients.alpha_support,
              alpha_left_window);

      const auto left_block =
          state_coefficients.local_left_coefficient_matrix.block(
              alpha_left_window.begin,
              beta_left_window.begin,
              alpha_left_window.size(),
              beta_left_window.size());
      const auto residual_block =
          state_coefficients.local_residual_coefficient_matrix.block(
              alpha_left_window.begin,
              beta_left_window.begin,
              alpha_left_window.size(),
              beta_left_window.size());

      for (int alpha_right_begin_local = 0;
           alpha_right_begin_local < n_alpha_support;
           alpha_right_begin_local += partner_tile_size) {
        const SupportWindow alpha_right_window{
            alpha_right_begin_local,
            std::min(n_alpha_support, alpha_right_begin_local + partner_tile_size),
        };
        const std::vector<int> alpha_right_indices =
            build_support_global_indices(
                state_coefficients.alpha_support,
                alpha_right_window);

        gather_same_spin_scalar_blocks_from_ordered_cache(
            alpha_pair_cache,
            n_unique_alpha,
            alpha_left_indices,
            alpha_right_indices,
            &alpha_overlap_block,
            include_partner_total ? &alpha_regular_total_block : nullptr,
            include_partner_total ? &alpha_singular_total_block : nullptr);

        const auto right_block =
            state_coefficients.local_right_coefficient_matrix.block(
                alpha_right_window.begin,
                beta_right_window.begin,
                alpha_right_window.size(),
                beta_right_window.size());

        // This is the local streamed version of
        //   W_overlap(beta_L, beta_R) += Q^T * S_alpha * R
        // and, when requested,
        //   W_partner(beta_L, beta_R) += L^T * H_alpha * R.
        alpha_push.noalias() = residual_block.transpose() * alpha_overlap_block;
        beta_image.noalias() = alpha_push * right_block;
        scatter_add_dense_submatrix_to_tile(
            beta_image,
            beta_left_offsets,
            beta_right_offsets,
            state_weight,
            overlap_weight_tile);

        if (!include_partner_total) {
          continue;
        }

        alpha_push.noalias() = left_block.transpose() * alpha_regular_total_block;
        beta_image.noalias() = alpha_push * right_block;
        scatter_add_dense_submatrix_to_tile(
            beta_image,
            beta_left_offsets,
            beta_right_offsets,
            state_weight,
            partner_total_tile);

        alpha_push.noalias() = left_block.transpose() * alpha_singular_total_block;
        beta_image.noalias() = alpha_push * right_block;
        scatter_add_dense_submatrix_to_tile(
            beta_image,
            beta_left_offsets,
            beta_right_offsets,
            state_weight,
            singular_partner_tile);
      }
    }
  }
}

void throw_if_singular_partner_tile_is_used(
    const Eigen::Ref<const Eigen::MatrixXd>& singular_partner_tile,
    const char* spin_label) {
  if (max_abs_dense_matrix(singular_partner_tile) > kContributionTolerance) {
    throw std::runtime_error(
        std::string("biorthogonal same-spin matrix backward encountered non-regular ") +
        spin_label + " partner pairs with nonzero selected-state weight");
  }
}

void accumulate_spin_overlap_backward_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int tile_left_begin,
    int tile_left_end,
    int tile_right_begin,
    int tile_right_end,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_weight_tile,
    const Eigen::Ref<const Eigen::MatrixXd>& partner_total_tile,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (overlap_weight_tile.rows() != tile_left_end - tile_left_begin ||
      overlap_weight_tile.cols() != tile_right_end - tile_right_begin ||
      partner_total_tile.rows() != tile_left_end - tile_left_begin ||
      partner_total_tile.cols() != tile_right_end - tile_right_begin) {
    throw std::invalid_argument("same-spin weight tile shape is inconsistent");
  }

  Eigen::MatrixXd scaled_inverse_overlap_gradient;
  for (int left_id = tile_left_begin; left_id < tile_left_end; ++left_id) {
    const int row_local = left_id - tile_left_begin;
    for (int right_id = tile_right_begin; right_id < tile_right_end; ++right_id) {
      const int column_local = right_id - tile_right_begin;
      const double determinant_overlap_weight =
          overlap_weight_tile(row_local, column_local) +
          partner_total_tile(row_local, column_local);
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      const auto& pair_evaluation =
          ordered_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      if (pair_evaluation.overlap_result.nullity != 0 ||
          pair_evaluation.overlap_result.overlap_determinant == 0.0) {
        throw std::runtime_error(
            "biorthogonal same-spin matrix backward requires nullity == 0");
      }
      if (!pair_evaluation.has_same_spin_phi_cache) {
        throw std::runtime_error(
            "biorthogonal same-spin matrix backward requires cached same-spin phi payloads");
      }

      const auto& occ_L = xmvb::index_at(unique_determinants, left_id);
      const auto& occ_R = xmvb::index_at(unique_determinants, right_id);
      const int n_electrons = static_cast<int>(occ_L.size());
      scaled_inverse_overlap_gradient.setZero(n_electrons, n_electrons);
      xmvb::vb::accumulate_spin_overlap_gradient(
          occ_L,
          occ_R,
          pair_evaluation.overlap_result,
          determinant_overlap_weight,
          scaled_inverse_overlap_gradient,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

template <typename TileBuilder>
void accumulate_streamed_same_spin_overlap_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const char* singular_partner_spin_label,
    TileBuilder&& build_weight_tiles,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  Eigen::MatrixXd overlap_weight_tile;
  Eigen::MatrixXd partner_total_tile;
  Eigen::MatrixXd singular_partner_tile;

  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);

      build_weight_tiles(
          left_begin,
          left_end,
          right_begin,
          right_end,
          &overlap_weight_tile,
          &partner_total_tile,
          &singular_partner_tile);
      throw_if_singular_partner_tile_is_used(
          singular_partner_tile,
          singular_partner_spin_label);
      if (max_abs_dense_matrix(overlap_weight_tile) <= kContributionTolerance &&
          max_abs_dense_matrix(partner_total_tile) <= kContributionTolerance) {
        continue;
      }

      accumulate_spin_overlap_backward_tile(
          unique_determinants,
          ordered_pair_cache,
          n_unique_determinants,
          left_begin,
          left_end,
          right_begin,
          right_end,
          overlap_weight_tile,
          partner_total_tile,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

}  // namespace

xmvb::vb::SameSpinMatrixBackwardContribution
build_biorthogonal_same_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_state_matrices);

  xmvb::vb::SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      xmvb::vb::packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  // The current exact biorthogonal same-spin implementation only consumes the
  // explicit overlap-side adjoint on this path, so we stream the overlap and
  // partner-overlap-response tiles directly into the final AO-overlap gradient
  // instead of materializing global `N_unique x N_unique` weight matrices.
  accumulate_streamed_same_spin_overlap_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      selected_state_matrices.n_unique_alpha,
      n_active_orbitals,
      "beta",
      [&](int left_begin,
          int left_end,
          int right_begin,
          int right_end,
          Eigen::MatrixXd* overlap_weight_tile,
          Eigen::MatrixXd* partner_total_tile,
          Eigen::MatrixXd* singular_partner_tile) {
        accumulate_alpha_selected_state_weight_tile_from_partner_cache(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_state_matrices.n_unique_beta,
            selected_state_matrices,
            left_begin,
            left_end,
            right_begin,
            right_end,
            true,
            overlap_weight_tile,
            partner_total_tile,
            singular_partner_tile);
      },
      &result.active_orbital_overlap_gradient);
  accumulate_streamed_same_spin_overlap_backward(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_state_matrices.n_unique_beta,
      n_active_orbitals,
      "alpha",
      [&](int left_begin,
          int left_end,
          int right_begin,
          int right_end,
          Eigen::MatrixXd* overlap_weight_tile,
          Eigen::MatrixXd* partner_total_tile,
          Eigen::MatrixXd* singular_partner_tile) {
        accumulate_beta_selected_state_weight_tile_from_partner_cache(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_state_matrices.n_unique_alpha,
            selected_state_matrices,
            left_begin,
            left_end,
            right_begin,
            right_end,
            true,
            overlap_weight_tile,
            partner_total_tile,
            singular_partner_tile);
      },
      &result.active_orbital_overlap_gradient);

  return result;
}

xmvb::vb::SameSpinMatrixBackwardContribution
build_biorthogonal_same_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalForwardSpinPairTileProvider& alpha_biorthogonal_provider,
    const BiorthogonalForwardSpinPairTileProvider& beta_biorthogonal_provider,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_state_matrices);
  if (alpha_biorthogonal_provider.n_unique_determinants() !=
          selected_state_matrices.n_unique_alpha ||
      beta_biorthogonal_provider.n_unique_determinants() !=
          selected_state_matrices.n_unique_beta) {
    throw std::invalid_argument(
        "biorthogonal same-spin tile providers do not match selected-state dimensions");
  }

  xmvb::vb::SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      xmvb::vb::packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  // The current exact selected-space gradient path uses this overload only for
  // the explicit same-spin overlap adjoint. The partner Hamiltonian channel is
  // still supplied by the dedicated transform backpropagators, so this streamed
  // overlap contraction validates provider dimensions but does not materialize
  // any biorthogonal same-spin scalar matrices from the forward tiles.
  accumulate_streamed_same_spin_overlap_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      selected_state_matrices.n_unique_alpha,
      n_active_orbitals,
      "beta",
      [&](int left_begin,
          int left_end,
          int right_begin,
          int right_end,
          Eigen::MatrixXd* overlap_weight_tile,
          Eigen::MatrixXd* partner_total_tile,
          Eigen::MatrixXd* singular_partner_tile) {
        accumulate_alpha_selected_state_weight_tile_from_partner_cache(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_state_matrices.n_unique_beta,
            selected_state_matrices,
            left_begin,
            left_end,
            right_begin,
            right_end,
            false,
            overlap_weight_tile,
            partner_total_tile,
            singular_partner_tile);
      },
      &result.active_orbital_overlap_gradient);
  accumulate_streamed_same_spin_overlap_backward(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      selected_state_matrices.n_unique_beta,
      n_active_orbitals,
      "alpha",
      [&](int left_begin,
          int left_end,
          int right_begin,
          int right_end,
          Eigen::MatrixXd* overlap_weight_tile,
          Eigen::MatrixXd* partner_total_tile,
          Eigen::MatrixXd* singular_partner_tile) {
        accumulate_beta_selected_state_weight_tile_from_partner_cache(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_state_matrices.n_unique_alpha,
            selected_state_matrices,
            left_begin,
            left_end,
            right_begin,
            right_end,
            false,
            overlap_weight_tile,
            partner_total_tile,
            singular_partner_tile);
      },
      &result.active_orbital_overlap_gradient);

  return result;
}

}  // namespace xmvb::vb::biorthogonal_vbscf
