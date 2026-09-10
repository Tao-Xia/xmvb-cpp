#include "vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions_internal.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/tile_kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {

using detail::DirectionalOppositeSpinPairData;
using detail::kOppositeSpinUniqueTileSize;

namespace {

constexpr double kContributionTolerance = 1.0e-15;

std::vector<int> build_retained_minor_indices_local(
    int dimension,
    const std::vector<int>& deleted_indices) {
  std::vector<int> retained_indices;
  retained_indices.reserve(
      dimension - static_cast<int>(deleted_indices.size()));
  for (int index = 0; index < dimension; ++index) {
    if (std::find(deleted_indices.begin(), deleted_indices.end(), index) ==
        deleted_indices.end()) {
      retained_indices.push_back(index);
    }
  }
  return retained_indices;
}

void scatter_minor_cofactor_to_overlap_block_gradient_local(
    const Eigen::MatrixXd& minor_cofactor,
    const std::vector<int>& retained_rows,
    const std::vector<int>& retained_cols,
    double scale,
    Eigen::MatrixXd* overlap_block_gradient) {
  if (overlap_block_gradient == nullptr) {
    throw std::invalid_argument("overlap_block_gradient must not be null");
  }
  if (minor_cofactor.rows() != static_cast<int>(retained_rows.size()) ||
      minor_cofactor.cols() != static_cast<int>(retained_cols.size())) {
    throw std::invalid_argument(
        "minor cofactor dimensions do not match retained deleted-minor indices");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  for (int retained_col = 0;
       retained_col < static_cast<int>(retained_cols.size());
       ++retained_col) {
    const int overlap_col = retained_cols[retained_col];
    for (int retained_row = 0;
         retained_row < static_cast<int>(retained_rows.size());
         ++retained_row) {
      const int overlap_row = retained_rows[retained_row];
      (*overlap_block_gradient)(overlap_row, overlap_col) +=
          scale * minor_cofactor(retained_row, retained_col);
    }
  }
}

void accumulate_deleted_minor_pullback_to_overlap_block_gradient_local(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    double scale,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* overlap_block_gradient) {
  if (overlap_block_gradient == nullptr) {
    throw std::invalid_argument("overlap_block_gradient must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }

  const std::vector<int> retained_rows =
      build_retained_minor_indices_local(overlap_block.rows(), deleted_rows);
  const std::vector<int> retained_cols =
      build_retained_minor_indices_local(overlap_block.cols(), deleted_cols);
  if (retained_rows.empty() || retained_cols.empty()) {
    return;
  }

  const Eigen::MatrixXd minor =
      build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
  const DeterminantOverlapResult minor_result =
      overlap_resolver.resolve_matrix(minor);
  const Eigen::MatrixXd minor_cofactor =
      calc_cofactor_1st(minor_result);
  if (minor_cofactor.size() == 0) {
    return;
  }

  scatter_minor_cofactor_to_overlap_block_gradient_local(
      minor_cofactor,
      retained_rows,
      retained_cols,
      scale * calc_deleted_minor_sign(
                  overlap_block.rows(),
                  overlap_block.cols(),
                  deleted_rows,
                  deleted_cols),
      overlap_block_gradient);
}

void accumulate_singular_spin_overlap_gradient_from_dense_image_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& overlap_result,
    const Eigen::MatrixXd& pair_dense_image,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (pair_dense_image.rows() != static_cast<int>(occ_R.size()) ||
      pair_dense_image.cols() != static_cast<int>(occ_L.size())) {
    throw std::invalid_argument(
        "pair_dense_image shape does not match the singular same-spin overlap block");
  }
  if (overlap_result.nullity != 1) {
    return;
  }

  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  Eigen::MatrixXd overlap_block_gradient =
      Eigen::MatrixXd::Zero(overlap_block.rows(), overlap_block.cols());
  const DeterminantOverlapResolver overlap_resolver;
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const double coefficient = pair_dense_image(right_row, left_column);
      if (std::abs(coefficient) <= kContributionTolerance) {
        continue;
      }
      accumulate_deleted_minor_pullback_to_overlap_block_gradient_local(
          overlap_block,
          {right_row},
          {left_column},
          coefficient,
          overlap_resolver,
          &overlap_block_gradient);
    }
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_block_gradient(right_row, left_column);
    }
  }
}

}  // namespace

namespace detail {

void accumulate_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_alpha_determinants =
      same_spin_pair_cache.alpha_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd pair_dense_image;
    Eigen::MatrixXd inverse_overlap_gradient;
    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        // W_P[I,J] = sum_s w_s C_s[I,:] V_beta(P) C_s[J,:]^T.
        // The tile is consumed immediately by alpha pair-cache overlap adjoints,
        // so no packed_pair -> N_unique^2 dense image is materialized.
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const auto alpha_pair_weight_tiles =
            build_alpha_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                n_packed_active_pairs,
                alpha_left_begin,
                alpha_left_end,
                alpha_right_begin,
                alpha_right_end);

        for (int alpha_left_id = alpha_left_begin;
             alpha_left_id < alpha_left_end;
             ++alpha_left_id) {
          const int alpha_left_local = alpha_left_id - alpha_left_begin;
          for (int alpha_right_id = alpha_right_begin;
               alpha_right_id < alpha_right_end;
               ++alpha_right_id) {
            const int alpha_right_local = alpha_right_id - alpha_right_begin;
            const int alpha_tile_index =
                alpha_left_local + alpha_tile_left_size * alpha_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    alpha_left_id,
                    alpha_right_id,
                    selected_states.n_unique_alpha);
            const auto& alpha_pair_evaluation =
                same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
            const auto& occ_L = unique_alpha_determinants[alpha_left_id];
            const auto& occ_R = unique_alpha_determinants[alpha_right_id];

            if (alpha_pair_evaluation.overlap_result.nullity == 1) {
              build_local_packed_pair_dense_image_matrix_from_tile_matrix(
                  occ_L,
                  occ_R,
                  alpha_pair_weight_tiles,
                  alpha_tile_index,
                  &pair_dense_image);
              accumulate_singular_spin_overlap_gradient_from_dense_image_local(
                  occ_L,
                  occ_R,
                  alpha_pair_evaluation.overlap_result,
                  pair_dense_image,
                  n_active_orbitals,
                  active_orbital_overlap_gradient);
              continue;
            }

            const auto& inverse_projection =
                alpha_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() ||
                inverse_projection.projected_pair_values.empty()) {
              continue;
            }
            if (alpha_pair_evaluation.overlap_result.nullity != 0) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  alpha_pair_weight_tiles(alpha_tile_index, packed_pair_index);
            }
            if (std::abs(determinant_overlap_weight) <=
                kContributionTolerance) {
              continue;
            }

            build_inverse_overlap_gradient_from_tile_matrix(
                occ_L,
                occ_R,
                alpha_pair_weight_tiles,
                alpha_tile_index,
                &inverse_overlap_gradient);
            accumulate_spin_overlap_gradient(
                occ_L,
                occ_R,
                alpha_pair_evaluation.overlap_result,
                determinant_overlap_weight,
                inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

void accumulate_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.beta_pair_cache_ref(),
          "beta");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_beta_determinants =
      same_spin_pair_cache.beta_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd pair_dense_image;
    Eigen::MatrixXd inverse_overlap_gradient;
    for (int beta_left_begin = 0;
         beta_left_begin < selected_states.n_unique_beta;
         beta_left_begin += unique_tile_size) {
      const int beta_left_end =
          std::min(
              selected_states.n_unique_beta,
              beta_left_begin + unique_tile_size);
      for (int beta_right_begin = 0;
           beta_right_begin < selected_states.n_unique_beta;
           beta_right_begin += unique_tile_size) {
        const int beta_right_end =
            std::min(
                selected_states.n_unique_beta,
                beta_right_begin + unique_tile_size);

        // W_P[K,L] = sum_s w_s C_s[:,K]^T V_alpha(P) C_s[:,L].
        // The current beta tile replaces the old packed_pair -> dense
        // N_unique_beta^2 image cache.
        const int beta_tile_left_size = beta_left_end - beta_left_begin;
        const auto beta_pair_weight_tiles =
            build_beta_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                n_packed_active_pairs,
                beta_left_begin,
                beta_left_end,
                beta_right_begin,
                beta_right_end);

        for (int beta_left_id = beta_left_begin;
             beta_left_id < beta_left_end;
             ++beta_left_id) {
          const int beta_left_local = beta_left_id - beta_left_begin;
          for (int beta_right_id = beta_right_begin;
               beta_right_id < beta_right_end;
               ++beta_right_id) {
            const int beta_right_local = beta_right_id - beta_right_begin;
            const int beta_tile_index =
                beta_left_local + beta_tile_left_size * beta_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    beta_left_id,
                    beta_right_id,
                    selected_states.n_unique_beta);
            const auto& beta_pair_evaluation =
                same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
            const auto& occ_L = unique_beta_determinants[beta_left_id];
            const auto& occ_R = unique_beta_determinants[beta_right_id];

            if (beta_pair_evaluation.overlap_result.nullity == 1) {
              build_local_packed_pair_dense_image_matrix_from_tile_matrix(
                  occ_L,
                  occ_R,
                  beta_pair_weight_tiles,
                  beta_tile_index,
                  &pair_dense_image);
              accumulate_singular_spin_overlap_gradient_from_dense_image_local(
                  occ_L,
                  occ_R,
                  beta_pair_evaluation.overlap_result,
                  pair_dense_image,
                  n_active_orbitals,
                  active_orbital_overlap_gradient);
              continue;
            }

            const auto& inverse_projection =
                beta_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() ||
                inverse_projection.projected_pair_values.empty()) {
              continue;
            }
            if (beta_pair_evaluation.overlap_result.nullity != 0) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  beta_pair_weight_tiles(beta_tile_index, packed_pair_index);
            }
            if (std::abs(determinant_overlap_weight) <=
                kContributionTolerance) {
              continue;
            }

            build_inverse_overlap_gradient_from_tile_matrix(
                occ_L,
                occ_R,
                beta_pair_weight_tiles,
                beta_tile_index,
                &inverse_overlap_gradient);
            accumulate_spin_overlap_gradient(
                occ_L,
                occ_R,
                beta_pair_evaluation.overlap_result,
                determinant_overlap_weight,
                inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

void accumulate_directional_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_alpha_determinants =
      same_spin_pair_cache.alpha_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd inverse_overlap_gradient;
    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const auto alpha_pair_weight_tiles =
            build_directional_alpha_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                n_packed_active_pairs,
                alpha_left_begin,
                alpha_left_end,
                alpha_right_begin,
                alpha_right_end);

        for (int alpha_left_id = alpha_left_begin;
             alpha_left_id < alpha_left_end;
             ++alpha_left_id) {
          const int alpha_left_local = alpha_left_id - alpha_left_begin;
          for (int alpha_right_id = alpha_right_begin;
               alpha_right_id < alpha_right_end;
               ++alpha_right_id) {
            const int alpha_right_local = alpha_right_id - alpha_right_begin;
            const int alpha_tile_index =
                alpha_left_local + alpha_tile_left_size * alpha_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    alpha_left_id,
                    alpha_right_id,
                    selected_states.n_unique_alpha);
            const auto& alpha_pair_evaluation =
                same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
            const auto& inverse_projection =
                alpha_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() ||
                inverse_projection.projected_pair_values.empty()) {
              continue;
            }
            if (alpha_pair_evaluation.overlap_result.nullity != 0) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  alpha_pair_weight_tiles(alpha_tile_index, packed_pair_index);
            }
            if (std::abs(determinant_overlap_weight) <=
                kContributionTolerance) {
              continue;
            }

            build_inverse_overlap_gradient_from_tile_matrix(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                alpha_pair_weight_tiles,
                alpha_tile_index,
                &inverse_overlap_gradient);
            accumulate_spin_overlap_gradient(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                alpha_pair_evaluation.overlap_result,
                determinant_overlap_weight,
                inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

void accumulate_directional_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.beta_pair_cache_ref(),
          "beta");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_beta_determinants =
      same_spin_pair_cache.beta_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd inverse_overlap_gradient;
    for (int beta_left_begin = 0;
         beta_left_begin < selected_states.n_unique_beta;
         beta_left_begin += unique_tile_size) {
      const int beta_left_end =
          std::min(
              selected_states.n_unique_beta,
              beta_left_begin + unique_tile_size);
      for (int beta_right_begin = 0;
           beta_right_begin < selected_states.n_unique_beta;
           beta_right_begin += unique_tile_size) {
        const int beta_right_end =
            std::min(
                selected_states.n_unique_beta,
                beta_right_begin + unique_tile_size);

        const int beta_tile_left_size = beta_left_end - beta_left_begin;
        const auto beta_pair_weight_tiles =
            build_directional_beta_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                n_packed_active_pairs,
                beta_left_begin,
                beta_left_end,
                beta_right_begin,
                beta_right_end);

        for (int beta_left_id = beta_left_begin;
             beta_left_id < beta_left_end;
             ++beta_left_id) {
          const int beta_left_local = beta_left_id - beta_left_begin;
          for (int beta_right_id = beta_right_begin;
               beta_right_id < beta_right_end;
               ++beta_right_id) {
            const int beta_right_local = beta_right_id - beta_right_begin;
            const int beta_tile_index =
                beta_left_local + beta_tile_left_size * beta_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    beta_left_id,
                    beta_right_id,
                    selected_states.n_unique_beta);
            const auto& beta_pair_evaluation =
                same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
            const auto& inverse_projection =
                beta_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() ||
                inverse_projection.projected_pair_values.empty()) {
              continue;
            }
            if (beta_pair_evaluation.overlap_result.nullity != 0) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  beta_pair_weight_tiles(beta_tile_index, packed_pair_index);
            }
            if (std::abs(determinant_overlap_weight) <=
                kContributionTolerance) {
              continue;
            }

            build_inverse_overlap_gradient_from_tile_matrix(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                beta_pair_weight_tiles,
                beta_tile_index,
                &inverse_overlap_gradient);
            accumulate_spin_overlap_gradient(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                beta_pair_evaluation.overlap_result,
                determinant_overlap_weight,
                inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

void accumulate_local_alpha_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_alpha_determinants =
      same_spin_pair_cache.alpha_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd inverse_overlap_gradient;
    Eigen::MatrixXd delta_inverse_overlap_gradient;
    for (int alpha_left_begin = 0;
         alpha_left_begin < selected_states.n_unique_alpha;
         alpha_left_begin += unique_tile_size) {
      const int alpha_left_end =
          std::min(
              selected_states.n_unique_alpha,
              alpha_left_begin + unique_tile_size);
      for (int alpha_right_begin = 0;
           alpha_right_begin < selected_states.n_unique_alpha;
           alpha_right_begin += unique_tile_size) {
        const int alpha_right_end =
            std::min(
                selected_states.n_unique_alpha,
                alpha_right_begin + unique_tile_size);

        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const auto alpha_pair_weight_tiles =
            build_alpha_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                n_packed_active_pairs,
                alpha_left_begin,
                alpha_left_end,
                alpha_right_begin,
                alpha_right_end);
        const auto directional_alpha_pair_weight_tiles =
            build_local_directional_alpha_overlap_weight_tile_matrix(
                beta_directional_pair_data,
                selected_states,
                n_packed_active_pairs,
                alpha_left_begin,
                alpha_left_end,
                alpha_right_begin,
                alpha_right_end);

        for (int alpha_left_id = alpha_left_begin;
             alpha_left_id < alpha_left_end;
             ++alpha_left_id) {
          const int alpha_left_local = alpha_left_id - alpha_left_begin;
          for (int alpha_right_id = alpha_right_begin;
               alpha_right_id < alpha_right_end;
               ++alpha_right_id) {
            const int alpha_right_local = alpha_right_id - alpha_right_begin;
            const int alpha_tile_index =
                alpha_left_local + alpha_tile_left_size * alpha_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    alpha_left_id,
                    alpha_right_id,
                    selected_states.n_unique_alpha);
            const auto& alpha_pair_evaluation =
                same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
            build_inverse_overlap_gradient_from_tile_matrix(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                alpha_pair_weight_tiles,
                alpha_tile_index,
                &inverse_overlap_gradient);
            build_inverse_overlap_gradient_from_tile_matrix(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                directional_alpha_pair_weight_tiles,
                alpha_tile_index,
                &delta_inverse_overlap_gradient);
            accumulate_spin_overlap_gradient_direction_local(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                cached_cofactor_differential(alpha_pair_evaluation),
                alpha_directional_pair_data[ordered_pair_index].delta_overlap_submatrix,
                inverse_overlap_gradient,
                delta_inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

void accumulate_local_beta_overlap_gradient(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.beta_pair_cache_ref(),
          "beta");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_beta_determinants =
      same_spin_pair_cache.beta_reuse_table.unique_determinants;
  {
    const int unique_tile_size = kOppositeSpinUniqueTileSize;
    Eigen::MatrixXd inverse_overlap_gradient;
    Eigen::MatrixXd delta_inverse_overlap_gradient;
    for (int beta_left_begin = 0;
         beta_left_begin < selected_states.n_unique_beta;
         beta_left_begin += unique_tile_size) {
      const int beta_left_end =
          std::min(
              selected_states.n_unique_beta,
              beta_left_begin + unique_tile_size);
      for (int beta_right_begin = 0;
           beta_right_begin < selected_states.n_unique_beta;
           beta_right_begin += unique_tile_size) {
        const int beta_right_end =
            std::min(
                selected_states.n_unique_beta,
                beta_right_begin + unique_tile_size);

        const int beta_tile_left_size = beta_left_end - beta_left_begin;
        const auto beta_pair_weight_tiles =
            build_beta_overlap_weight_tile_matrix(
                same_spin_pair_cache,
                selected_states,
                n_packed_active_pairs,
                beta_left_begin,
                beta_left_end,
                beta_right_begin,
                beta_right_end);
        const auto directional_beta_pair_weight_tiles =
            build_local_directional_beta_overlap_weight_tile_matrix(
                alpha_directional_pair_data,
                selected_states,
                n_packed_active_pairs,
                beta_left_begin,
                beta_left_end,
                beta_right_begin,
                beta_right_end);

        for (int beta_left_id = beta_left_begin;
             beta_left_id < beta_left_end;
             ++beta_left_id) {
          const int beta_left_local = beta_left_id - beta_left_begin;
          for (int beta_right_id = beta_right_begin;
               beta_right_id < beta_right_end;
               ++beta_right_id) {
            const int beta_right_local = beta_right_id - beta_right_begin;
            const int beta_tile_index =
                beta_left_local + beta_tile_left_size * beta_right_local;
            const std::size_t ordered_pair_index =
                ordered_spin_pair_storage_index(
                    beta_left_id,
                    beta_right_id,
                    selected_states.n_unique_beta);
            const auto& beta_pair_evaluation =
                same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
            build_inverse_overlap_gradient_from_tile_matrix(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                beta_pair_weight_tiles,
                beta_tile_index,
                &inverse_overlap_gradient);
            build_inverse_overlap_gradient_from_tile_matrix(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                directional_beta_pair_weight_tiles,
                beta_tile_index,
                &delta_inverse_overlap_gradient);
            accumulate_spin_overlap_gradient_direction_local(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                cached_cofactor_differential(beta_pair_evaluation),
                beta_directional_pair_data[ordered_pair_index].delta_overlap_submatrix,
                inverse_overlap_gradient,
                delta_inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}
}  // namespace detail

}  // namespace xmvb::vb
