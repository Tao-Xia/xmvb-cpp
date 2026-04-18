#include "vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrix_utils.hpp"
#include "vb/matrices/structure_block_kernels.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {
constexpr double kContributionTolerance = 1.0e-15;
constexpr int kOppositeSpinBackwardSparseBlockSize = 32;
constexpr int kOppositeSpinBackwardDenseBatchSize = 8;

bool dense_matrix_is_effectively_zero(const Eigen::MatrixXd& matrix) {
  return matrix.size() == 0 || matrix.cwiseAbs().maxCoeff() <= kContributionTolerance;
}

void validate_local_state_matrices(
    const BiorthogonalSelectedStateDeterminantCoefficients& state_coefficients);

void validate_matrix_backward_inputs(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_states) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "biorthogonal matrix-form opposite-spin backward requires an enabled same-spin cache");
  }
  if (selected_states.n_unique_alpha !=
          static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) ||
      selected_states.n_unique_beta !=
          static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size())) {
    throw std::invalid_argument(
        "biorthogonal selected-state dimensions do not match same-spin cache reuse tables");
  }

  for (const auto& state_coefficients : selected_states.states) {
    validate_local_state_matrices(state_coefficients);
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

std::vector<Eigen::MatrixXd> build_local_first_order_dense_matrix_block(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    const std::vector<int>& support,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 || packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::MatrixXd> local_matrices;
  local_matrices.reserve(xmvb::to_size(block_size));
  for (int local_index = 0; local_index < block_size; ++local_index) {
    local_matrices.emplace_back(
        Eigen::MatrixXd::Zero(
            static_cast<int>(support.size()),
            static_cast<int>(support.size())));
  }
  if (block_size == 0) {
    return local_matrices;
  }

  for (int column_local = 0;
       column_local < static_cast<int>(support.size());
       ++column_local) {
    const int right_unique_index = support[xmvb::to_size(column_local)];
    if (right_unique_index < 0 || right_unique_index >= n_unique_determinants) {
      throw std::out_of_range("support right index is out of range");
    }
    for (int row_local = 0;
         row_local < static_cast<int>(support.size());
         ++row_local) {
      const int left_unique_index = support[xmvb::to_size(row_local)];
      if (left_unique_index < 0 || left_unique_index >= n_unique_determinants) {
        throw std::out_of_range("support left index is out of range");
      }
      const auto& pair_evaluation =
          ordered_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < packed_pair_begin || packed_pair_index >= packed_pair_end) {
          continue;
        }

        const double packed_pair_value = projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <= kContributionTolerance) {
          continue;
        }
        local_matrices[xmvb::to_size(
            packed_pair_index - packed_pair_begin)](row_local, column_local) +=
            packed_pair_value;
      }
    }
  }

  return local_matrices;
}

std::vector<int> build_nonzero_dense_matrix_local_indices(
    const std::vector<Eigen::MatrixXd>& dense_matrices) {
  std::vector<int> local_indices;
  for (int local_index = 0;
       local_index < static_cast<int>(dense_matrices.size());
       ++local_index) {
    if (!dense_matrix_is_effectively_zero(
            dense_matrices[xmvb::to_size(local_index)])) {
      local_indices.push_back(local_index);
    }
  }
  return local_indices;
}

int find_support_local_index(
    const std::vector<int>& support,
    int global_index) {
  const auto iterator =
      std::lower_bound(support.begin(), support.end(), global_index);
  if (iterator == support.end() || *iterator != global_index) {
    return -1;
  }
  return static_cast<int>(iterator - support.begin());
}

std::vector<int> build_needed_packed_pair_indices(
    const OppositeSpinPackedPairProjection& inverse_projection,
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R) {
  std::vector<int> packed_pair_indices = inverse_projection.packed_pair_indices;
  packed_pair_indices.reserve(
      packed_pair_indices.size() + xmvb::product_size(occ_L.size(), occ_R.size()));
  for (const int orbital_index_left : occ_L) {
    for (const int orbital_index_right : occ_R) {
      packed_pair_indices.push_back(
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              orbital_index_right,
              orbital_index_left));
    }
  }
  std::sort(packed_pair_indices.begin(), packed_pair_indices.end());
  packed_pair_indices.erase(
      std::unique(packed_pair_indices.begin(), packed_pair_indices.end()),
      packed_pair_indices.end());
  return packed_pair_indices;
}

double lookup_projected_value_in_subset(
    const std::vector<int>& packed_pair_indices,
    const Eigen::Ref<const Eigen::VectorXd>& projected_values,
    int packed_pair_index) {
  if (static_cast<int>(packed_pair_indices.size()) != projected_values.size()) {
    throw std::invalid_argument(
        "packed_pair_indices and projected_values sizes do not match");
  }
  const auto iterator =
      std::lower_bound(
          packed_pair_indices.begin(),
          packed_pair_indices.end(),
          packed_pair_index);
  if (iterator == packed_pair_indices.end() || *iterator != packed_pair_index) {
    throw std::out_of_range("packed_pair_index is not present in projected subset");
  }
  return projected_values(static_cast<int>(iterator - packed_pair_indices.begin()));
}

double contract_sparse_projection_with_projected_subset(
    const OppositeSpinPackedPairProjection& sparse_projection,
    const std::vector<int>& packed_pair_indices,
    const Eigen::Ref<const Eigen::VectorXd>& projected_values) {
  double contraction = 0.0;
  for (std::size_t entry_index = 0;
       entry_index < sparse_projection.packed_pair_indices.size();
       ++entry_index) {
    contraction +=
        sparse_projection.packed_pair_values[entry_index] *
        lookup_projected_value_in_subset(
            packed_pair_indices,
            projected_values,
            sparse_projection.packed_pair_indices[entry_index]);
  }
  return contraction;
}

void build_inverse_overlap_gradient_from_projected_subset(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<int>& packed_pair_indices,
    const Eigen::Ref<const Eigen::VectorXd>& projected_values,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (inverse_overlap_gradient == nullptr) {
    throw std::invalid_argument("inverse_overlap_gradient must not be null");
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient->setZero(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[xmvb::to_size(right_row)];
      const int packed_pair_index = xmvb::vb::TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      (*inverse_overlap_gradient)(left_column, right_row) =
          lookup_projected_value_in_subset(
              packed_pair_indices,
              projected_values,
              packed_pair_index);
    }
  }
}

void accumulate_needed_projected_values_from_beta_states_for_alpha_pair(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& beta_pair_cache,
    int n_unique_beta,
    const BiorthogonalSelectedStateMatrices& selected_states,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals,
    int alpha_left_id,
    int alpha_right_id,
    const std::vector<int>& packed_pair_indices,
    Eigen::VectorXd* projected_values) {
  if (projected_values == nullptr) {
    throw std::invalid_argument("projected_values must not be null");
  }
  *projected_values = Eigen::VectorXd::Zero(static_cast<int>(packed_pair_indices.size()));
  Eigen::VectorXd local_projected_values;

  for (const auto& state_coefficients : selected_states.states) {
    if (std::abs(state_coefficients.normalized_state_weight) <= kContributionTolerance ||
        state_coefficients.local_left_coefficient_matrix.size() == 0) {
      continue;
    }

    const int alpha_left_local =
        find_support_local_index(state_coefficients.alpha_support, alpha_left_id);
    const int alpha_right_local =
        find_support_local_index(state_coefficients.alpha_support, alpha_right_id);
    if (alpha_left_local < 0 || alpha_right_local < 0) {
      continue;
    }

    const auto left_row =
        state_coefficients.local_left_coefficient_matrix.row(alpha_left_local);
    const auto right_row =
        state_coefficients.local_right_coefficient_matrix.row(alpha_right_local);
    for (int beta_right_local = 0;
         beta_right_local < static_cast<int>(state_coefficients.beta_support.size());
         ++beta_right_local) {
      const double right_value = right_row(beta_right_local);
      if (std::abs(right_value) <= kContributionTolerance) {
        continue;
      }
      const int beta_right_id =
          state_coefficients.beta_support[xmvb::to_size(beta_right_local)];
      for (int beta_left_local = 0;
           beta_left_local < static_cast<int>(state_coefficients.beta_support.size());
           ++beta_left_local) {
        const double coefficient =
            state_coefficients.normalized_state_weight *
            left_row(beta_left_local) *
            right_value;
        if (std::abs(coefficient) <= kContributionTolerance) {
          continue;
        }

        const int beta_left_id =
            state_coefficients.beta_support[xmvb::to_size(beta_left_local)];
        const auto& projection =
            beta_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
                beta_left_id,
                beta_right_id,
                n_unique_beta)]
                .opposite_spin_pair_cache
                .first_order_cofactor_projection;
        if (projection.packed_pair_indices.empty()) {
          continue;
        }
        local_projected_values =
            xmvb::vb::gather_projected_values_for_packed_pair_indices(
                projection,
                packed_pair_indices,
                active_space_two_electron_view,
                n_active_orbitals);
        projected_values->noalias() += coefficient * local_projected_values;
      }
    }
  }
}

void accumulate_needed_projected_values_from_alpha_states_for_beta_pair(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& alpha_pair_cache,
    int n_unique_alpha,
    const BiorthogonalSelectedStateMatrices& selected_states,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals,
    int beta_left_id,
    int beta_right_id,
    const std::vector<int>& packed_pair_indices,
    Eigen::VectorXd* projected_values) {
  if (projected_values == nullptr) {
    throw std::invalid_argument("projected_values must not be null");
  }
  *projected_values = Eigen::VectorXd::Zero(static_cast<int>(packed_pair_indices.size()));
  Eigen::VectorXd local_projected_values;

  for (const auto& state_coefficients : selected_states.states) {
    if (std::abs(state_coefficients.normalized_state_weight) <= kContributionTolerance ||
        state_coefficients.local_left_coefficient_matrix.size() == 0) {
      continue;
    }

    const int beta_left_local =
        find_support_local_index(state_coefficients.beta_support, beta_left_id);
    const int beta_right_local =
        find_support_local_index(state_coefficients.beta_support, beta_right_id);
    if (beta_left_local < 0 || beta_right_local < 0) {
      continue;
    }

    const auto left_column =
        state_coefficients.local_left_coefficient_matrix.col(beta_left_local);
    const auto right_column =
        state_coefficients.local_right_coefficient_matrix.col(beta_right_local);
    for (int alpha_right_local = 0;
         alpha_right_local < static_cast<int>(state_coefficients.alpha_support.size());
         ++alpha_right_local) {
      const double right_value = right_column(alpha_right_local);
      if (std::abs(right_value) <= kContributionTolerance) {
        continue;
      }
      const int alpha_right_id =
          state_coefficients.alpha_support[xmvb::to_size(alpha_right_local)];
      for (int alpha_left_local = 0;
           alpha_left_local < static_cast<int>(state_coefficients.alpha_support.size());
           ++alpha_left_local) {
        const double coefficient =
            state_coefficients.normalized_state_weight *
            left_column(alpha_left_local) *
            right_value;
        if (std::abs(coefficient) <= kContributionTolerance) {
          continue;
        }

        const int alpha_left_id =
            state_coefficients.alpha_support[xmvb::to_size(alpha_left_local)];
        const auto& projection =
            alpha_pair_cache[xmvb::vb::ordered_spin_pair_storage_index(
                alpha_left_id,
                alpha_right_id,
                n_unique_alpha)]
                .opposite_spin_pair_cache
                .first_order_cofactor_projection;
        if (projection.packed_pair_indices.empty()) {
          continue;
        }
        local_projected_values =
            xmvb::vb::gather_projected_values_for_packed_pair_indices(
                projection,
                packed_pair_indices,
                active_space_two_electron_view,
                n_active_orbitals);
        projected_values->noalias() += coefficient * local_projected_values;
      }
    }
  }
}

void accumulate_alpha_overlap_gradient(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_states,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      xmvb::vb::infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_alpha_determinants =
      same_spin_pair_cache.alpha_reuse_table.unique_determinants;
  std::vector<int> packed_pair_indices;
  Eigen::VectorXd projected_values;
  Eigen::MatrixXd inverse_overlap_gradient;
  for (int alpha_left_id = 0;
       alpha_left_id < selected_states.n_unique_alpha;
       ++alpha_left_id) {
    for (int alpha_right_id = 0;
         alpha_right_id < selected_states.n_unique_alpha;
         ++alpha_right_id) {
      const std::size_t ordered_pair_index = xmvb::vb::ordered_spin_pair_storage_index(
          alpha_left_id,
          alpha_right_id,
          selected_states.n_unique_alpha);
      const auto& alpha_pair_evaluation =
          same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
      const auto& inverse_projection =
          alpha_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
      if (inverse_projection.packed_pair_indices.empty()) {
        continue;
      }
      if (alpha_pair_evaluation.overlap_result.nullity != 0) {
        continue;
      }

      packed_pair_indices =
          build_needed_packed_pair_indices(
              inverse_projection,
              unique_alpha_determinants[xmvb::to_size(alpha_left_id)],
              unique_alpha_determinants[xmvb::to_size(alpha_right_id)]);
      accumulate_needed_projected_values_from_beta_states_for_alpha_pair(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta,
          selected_states,
          active_space_two_electron_view,
          n_active_orbitals,
          alpha_left_id,
          alpha_right_id,
          packed_pair_indices,
          &projected_values);
      const double determinant_overlap_weight =
          contract_sparse_projection_with_projected_subset(
              inverse_projection,
              packed_pair_indices,
              projected_values);
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      build_inverse_overlap_gradient_from_projected_subset(
          unique_alpha_determinants[xmvb::to_size(alpha_left_id)],
          unique_alpha_determinants[xmvb::to_size(alpha_right_id)],
          packed_pair_indices,
          projected_values,
          &inverse_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          unique_alpha_determinants[xmvb::to_size(alpha_left_id)],
          unique_alpha_determinants[xmvb::to_size(alpha_right_id)],
          alpha_pair_evaluation.overlap_result,
          determinant_overlap_weight,
          inverse_overlap_gradient,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

void accumulate_beta_overlap_gradient(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_states,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  const int n_packed_active_pairs =
      xmvb::vb::infer_n_packed_active_pairs(
          same_spin_pair_cache.beta_pair_cache_ref(),
          "beta");
  if (n_packed_active_pairs == 0) {
    return;
  }

  const auto& unique_beta_determinants =
      same_spin_pair_cache.beta_reuse_table.unique_determinants;
  std::vector<int> packed_pair_indices;
  Eigen::VectorXd projected_values;
  Eigen::MatrixXd inverse_overlap_gradient;
  for (int beta_left_id = 0;
       beta_left_id < selected_states.n_unique_beta;
       ++beta_left_id) {
    for (int beta_right_id = 0;
         beta_right_id < selected_states.n_unique_beta;
         ++beta_right_id) {
      const std::size_t ordered_pair_index = xmvb::vb::ordered_spin_pair_storage_index(
          beta_left_id,
          beta_right_id,
          selected_states.n_unique_beta);
      const auto& beta_pair_evaluation =
          same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
      const auto& inverse_projection =
          beta_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
      if (inverse_projection.packed_pair_indices.empty()) {
        continue;
      }
      if (beta_pair_evaluation.overlap_result.nullity != 0) {
        continue;
      }

      packed_pair_indices =
          build_needed_packed_pair_indices(
              inverse_projection,
              unique_beta_determinants[xmvb::to_size(beta_left_id)],
              unique_beta_determinants[xmvb::to_size(beta_right_id)]);
      accumulate_needed_projected_values_from_alpha_states_for_beta_pair(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha,
          selected_states,
          active_space_two_electron_view,
          n_active_orbitals,
          beta_left_id,
          beta_right_id,
          packed_pair_indices,
          &projected_values);
      const double determinant_overlap_weight =
          contract_sparse_projection_with_projected_subset(
              inverse_projection,
              packed_pair_indices,
              projected_values);
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      build_inverse_overlap_gradient_from_projected_subset(
          unique_beta_determinants[xmvb::to_size(beta_left_id)],
          unique_beta_determinants[xmvb::to_size(beta_right_id)],
          packed_pair_indices,
          projected_values,
          &inverse_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          unique_beta_determinants[xmvb::to_size(beta_left_id)],
          unique_beta_determinants[xmvb::to_size(beta_right_id)],
          beta_pair_evaluation.overlap_result,
          determinant_overlap_weight,
          inverse_overlap_gradient,
          n_active_orbitals,
          active_orbital_overlap_gradient);
    }
  }
}

}  // namespace

xmvb::vb::OppositeSpinMatrixBackwardContribution
build_biorthogonal_opposite_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_states,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals) {
  validate_matrix_backward_inputs(same_spin_pair_cache, selected_states);

  xmvb::vb::OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      xmvb::product_size(n_active_orbitals, n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      xmvb::vb::packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  const int n_alpha_packed_active_pairs =
      xmvb::vb::infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  const int n_beta_packed_active_pairs =
      xmvb::vb::infer_n_packed_active_pairs(
          same_spin_pair_cache.beta_pair_cache_ref(),
          "beta");
  if (n_alpha_packed_active_pairs != n_beta_packed_active_pairs) {
    throw std::invalid_argument(
        "alpha/beta same-spin caches disagree on n_packed_active_pairs");
  }

  const int n_packed_active_pairs = n_alpha_packed_active_pairs;
  if (n_packed_active_pairs == 0) {
    return result;
  }

  const int sparse_block_size = std::min(
      n_packed_active_pairs,
      kOppositeSpinBackwardSparseBlockSize);
  const int beta_batch_size = std::min(
      n_packed_active_pairs,
      kOppositeSpinBackwardDenseBatchSize);

  // Opposite-spin packed 2e adjoint now contracts directly on each selected
  // state's touched unique-spin support. This avoids both the old full
  // `N_unique^2` dense weight images and the repeated global sparse-matrix
  // gather back down to the local support.
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd image;
  for (const auto& state_coefficients : selected_states.states) {
    if (std::abs(state_coefficients.normalized_state_weight) <= kContributionTolerance ||
        state_coefficients.local_left_coefficient_matrix.size() == 0) {
      continue;
    }

    for (int beta_batch_begin = 0;
         beta_batch_begin < n_packed_active_pairs;
         beta_batch_begin += beta_batch_size) {
      const int beta_batch_end =
          std::min(n_packed_active_pairs, beta_batch_begin + beta_batch_size);
      const auto beta_local_block = build_local_first_order_dense_matrix_block(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta,
          state_coefficients.beta_support,
          beta_batch_begin,
          beta_batch_end);
      const std::vector<int> nonzero_beta_local_indices =
          build_nonzero_dense_matrix_local_indices(beta_local_block);
      if (nonzero_beta_local_indices.empty()) {
        continue;
      }

      for (int alpha_block_begin = 0;
           alpha_block_begin < n_packed_active_pairs;
           alpha_block_begin += sparse_block_size) {
        const int alpha_block_end =
            std::min(n_packed_active_pairs, alpha_block_begin + sparse_block_size);
        const auto alpha_local_block = build_local_first_order_dense_matrix_block(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            state_coefficients.alpha_support,
            alpha_block_begin,
            alpha_block_end);
        const std::vector<int> nonzero_alpha_local_indices =
            build_nonzero_dense_matrix_local_indices(alpha_local_block);
        if (nonzero_alpha_local_indices.empty()) {
          continue;
        }

        for (const int beta_local_index : nonzero_beta_local_indices) {
          const int beta_packed_pair_index = beta_batch_begin + beta_local_index;
          for (const int alpha_local_index : nonzero_alpha_local_indices) {
            const double packed_gradient_value =
                state_coefficients.normalized_state_weight *
                xmvb::vb::contract_dense_structure_pair_kernel(
                    state_coefficients.local_left_coefficient_matrix,
                    state_coefficients.local_right_coefficient_matrix,
                    alpha_local_block[xmvb::to_size(alpha_local_index)],
                    beta_local_block[xmvb::to_size(beta_local_index)],
                    &beta_push,
                    &image);
            if (std::abs(packed_gradient_value) <= kContributionTolerance) {
              continue;
            }

            const int alpha_packed_pair_index = alpha_block_begin + alpha_local_index;
            const int packed_pair_of_pairs_index =
                xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                    beta_packed_pair_index,
                    alpha_packed_pair_index);
            result.packed_active_two_electron_gradient[xmvb::to_size(
                packed_pair_of_pairs_index)] += packed_gradient_value;
          }
        }
      }
    }
  }

  accumulate_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      active_space_two_electron_view,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  accumulate_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      active_space_two_electron_view,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);

  return result;
}

}  // namespace xmvb::vb::biorthogonal_vbscf
