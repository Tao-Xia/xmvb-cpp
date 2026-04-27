#include "vb/scf/opposite_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/support_local_contraction_kernels.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {


constexpr double kContributionTolerance = 1.0e-15;
constexpr int kOppositeSpinBackwardSparseBlockSize = 32;
constexpr int kOppositeSpinBackwardDenseBatchSize = 8;
constexpr int kOppositeSpinBackwardOverlapBlockSize = 32;
constexpr int kOppositeSpinBackwardUniqueTileSize = 64;

struct RegularSpinDirectionalOverlapData {
  double overlap_determinant = 0.0;
  double delta_overlap_determinant = 0.0;
  Eigen::MatrixXd inverse_overlap_submatrix;
  Eigen::MatrixXd delta_inverse_overlap_submatrix;
  Eigen::MatrixXd cofactor_1st;
  Eigen::MatrixXd delta_cofactor_1st;
};

struct DirectionalOppositeSpinPairData {
  double delta_overlap_determinant = 0.0;
  Eigen::MatrixXd delta_inverse_overlap_submatrix;
  OppositeSpinPackedPairProjection delta_first_order_cofactor_projection;
  OppositeSpinPackedPairProjection delta_inverse_overlap_projection;
};

int positive_env_override(
    const char* env_name,
    int default_value) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return default_value;
  }
  const int parsed_value = std::stoi(env_value);
  if (parsed_value <= 0) {
    throw std::invalid_argument(
        std::string(env_name) + " must be positive");
  }
  return parsed_value;
}

int opposite_spin_backward_sparse_block_size() {
  return positive_env_override(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_SPARSE_BLOCK_SIZE",
      kOppositeSpinBackwardSparseBlockSize);
}

int opposite_spin_backward_dense_batch_size() {
  return positive_env_override(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_DENSE_BATCH_SIZE",
      kOppositeSpinBackwardDenseBatchSize);
}

int opposite_spin_backward_overlap_block_size() {
  return positive_env_override(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_OVERLAP_BLOCK_SIZE",
      kOppositeSpinBackwardOverlapBlockSize);
}

int opposite_spin_backward_unique_tile_size() {
  return positive_env_override(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_UNIQUE_TILE_SIZE",
      kOppositeSpinBackwardUniqueTileSize);
}

bool selected_state_has_local_support(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  return !state_coefficients.alpha_support.empty() &&
      !state_coefficients.beta_support.empty() &&
      state_coefficients.local_coefficient_matrix.size() != 0;
}

void validate_local_state_coefficient_matrix(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  if (state_coefficients.local_coefficient_matrix.rows() !=
          static_cast<int>(state_coefficients.alpha_support.size()) ||
      state_coefficients.local_coefficient_matrix.cols() !=
          static_cast<int>(state_coefficients.beta_support.size())) {
    throw std::invalid_argument(
        "selected-state local_coefficient_matrix shape does not match support dimensions");
  }
}

Eigen::MatrixXd build_local_overlap_direction_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    int n_active_orbitals) {
  return build_overlap_submatrix(
      occ_L,
      occ_R,
      delta_active_orbital_overlap_matrix,
      n_active_orbitals);
}

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

RegularSpinDirectionalOverlapData build_regular_spin_directional_overlap_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix) {
  const auto& overlap_result = pair_evaluation.overlap_result;
  if (overlap_result.nullity != 0 || overlap_result.overlap_determinant == 0.0) {
    throw std::runtime_error(
        "opposite-spin local-response requires a non-singular same-spin pair");
  }

  RegularSpinDirectionalOverlapData result;
  result.overlap_determinant = overlap_result.overlap_determinant;
  result.inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(overlap_result);
  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  result.delta_overlap_determinant =
      result.overlap_determinant *
      (result.inverse_overlap_submatrix * delta_overlap_submatrix).trace();
  result.delta_inverse_overlap_submatrix =
      -result.inverse_overlap_submatrix *
      delta_overlap_submatrix *
      result.inverse_overlap_submatrix;
  result.cofactor_1st = calc_cofactor_1st(overlap_result);
  result.delta_cofactor_1st =
      result.delta_overlap_determinant *
      result.inverse_overlap_submatrix.transpose();
  result.delta_cofactor_1st.noalias() +=
      result.overlap_determinant *
      result.delta_inverse_overlap_submatrix.transpose();
  return result;
}

OppositeSpinPackedPairProjection build_sparse_packed_pair_projection_coefficients(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals) {
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_values(
      n_packed_active_pairs,
      0.0);
  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0);
  std::vector<int> touched_indices;
  touched_indices.reserve(occ_L.size() * occ_R.size());

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      if (touched_mask[packed_pair_index] == 0) {
        touched_mask[packed_pair_index] = 1;
        touched_indices.push_back(packed_pair_index);
      }
      const double coefficient =
          coefficient_matrix_is_right_by_left
              ? coefficient_matrix(right_row, left_column)
              : coefficient_matrix(left_column, right_row);
      dense_pair_values[packed_pair_index] += coefficient;
    }
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double packed_pair_value =
        dense_pair_values[packed_pair_index];
    if (std::abs(packed_pair_value) <= kContributionTolerance) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }
  return projection;
}

std::vector<double> apply_directional_active_two_electron_kernel_to_sparse_projection(
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> projected_pair_values(
      n_packed_active_pairs,
      0.0);
  if (delta_packed_active_two_electron_integrals.empty() ||
      packed_pair_indices.empty()) {
    return projected_pair_values;
  }
  for (std::size_t entry_index = 0;
       entry_index < packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = packed_pair_indices[entry_index];
    const double packed_pair_value = packed_pair_values[entry_index];
    for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_pair,
              packed_pair_index);
      projected_pair_values[row_pair] +=
          delta_packed_active_two_electron_integrals[
              packed_pair_of_pairs_index] *
          packed_pair_value;
    }
  }
  return projected_pair_values;
}

bool dense_matrix_is_effectively_zero(const Eigen::MatrixXd& matrix) {
  return matrix.size() == 0 || matrix.cwiseAbs().maxCoeff() <= kContributionTolerance;
}

std::vector<DirectionalOppositeSpinPairData>
build_directional_opposite_spin_pair_data(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const std::size_t expected_size =
      n_unique_determinants *
      n_unique_determinants;
  if (ordered_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        "directional opposite-spin pair data requires a full ordered pair cache");
  }

  std::vector<DirectionalOppositeSpinPairData> directional_pair_data(expected_size);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  for (int left_unique_index = 0;
       left_unique_index < n_unique_determinants;
       ++left_unique_index) {
    for (int right_unique_index = 0;
         right_unique_index < n_unique_determinants;
         ++right_unique_index) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          left_unique_index,
          right_unique_index,
          n_unique_determinants);
      const auto& pair_evaluation = ordered_pair_cache[ordered_pair_index];
      auto& directional_entry = directional_pair_data[ordered_pair_index];

      if (unique_determinants[left_unique_index].empty()) {
        continue;
      }
      if (pair_evaluation.overlap_result.nullity != 0 ||
          pair_evaluation.overlap_result.overlap_determinant == 0.0) {
        continue;
      }

      const RegularSpinDirectionalOverlapData overlap_data =
          build_regular_spin_directional_overlap_data(
              unique_determinants[left_unique_index],
              unique_determinants[right_unique_index],
              pair_evaluation,
              n_active_orbitals,
              delta_active_orbital_overlap_matrix);
      directional_entry.delta_overlap_determinant =
          overlap_data.delta_overlap_determinant;
      directional_entry.delta_inverse_overlap_submatrix =
          overlap_data.delta_inverse_overlap_submatrix;

      directional_entry.delta_first_order_cofactor_projection =
          build_sparse_packed_pair_projection_coefficients(
              unique_determinants[left_unique_index],
              unique_determinants[right_unique_index],
              overlap_data.delta_cofactor_1st,
              true,
              n_active_orbitals);
      auto& directional_first_order_projection =
          directional_entry.delta_first_order_cofactor_projection;
      directional_first_order_projection.projected_pair_values =
          apply_active_space_two_electron_kernel_to_sparse_projection(
              two_electron_view,
              n_active_orbitals,
              directional_first_order_projection.packed_pair_indices,
              directional_first_order_projection.packed_pair_values);
      const auto& accepted_first_order_projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      const std::vector<double> delta_kernel_times_first_order =
          apply_directional_active_two_electron_kernel_to_sparse_projection(
              n_active_orbitals,
              accepted_first_order_projection.packed_pair_indices,
              accepted_first_order_projection.packed_pair_values,
              delta_packed_active_two_electron_integrals);
      if (directional_first_order_projection.projected_pair_values.size() <
          delta_kernel_times_first_order.size()) {
        directional_first_order_projection.projected_pair_values.resize(
            delta_kernel_times_first_order.size(),
            0.0);
      }
      for (std::size_t pair_index = 0;
           pair_index < delta_kernel_times_first_order.size();
           ++pair_index) {
        directional_first_order_projection.projected_pair_values[pair_index] +=
            delta_kernel_times_first_order[pair_index];
      }

      directional_entry.delta_inverse_overlap_projection =
          build_sparse_packed_pair_projection_coefficients(
              unique_determinants[left_unique_index],
              unique_determinants[right_unique_index],
              overlap_data.delta_inverse_overlap_submatrix,
              false,
              n_active_orbitals);
      auto& directional_inverse_projection =
          directional_entry.delta_inverse_overlap_projection;
      directional_inverse_projection.projected_pair_values =
          apply_active_space_two_electron_kernel_to_sparse_projection(
              two_electron_view,
              n_active_orbitals,
              directional_inverse_projection.packed_pair_indices,
              directional_inverse_projection.packed_pair_values);
      const auto& accepted_inverse_projection =
          pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
      const std::vector<double> delta_kernel_times_inverse =
          apply_directional_active_two_electron_kernel_to_sparse_projection(
              n_active_orbitals,
              accepted_inverse_projection.packed_pair_indices,
              accepted_inverse_projection.packed_pair_values,
              delta_packed_active_two_electron_integrals);
      if (directional_inverse_projection.projected_pair_values.size() <
          delta_kernel_times_inverse.size()) {
        directional_inverse_projection.projected_pair_values.resize(
            delta_kernel_times_inverse.size(),
            0.0);
      }
      for (std::size_t pair_index = 0;
           pair_index < delta_kernel_times_inverse.size();
           ++pair_index) {
        directional_inverse_projection.projected_pair_values[pair_index] +=
            delta_kernel_times_inverse[pair_index];
      }
    }
  }
  return directional_pair_data;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> build_first_order_sparse_matrix_block(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> sparse_matrices;
  sparse_matrices.resize(block_size);
  if (block_size == 0) {
    return sparse_matrices;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    sparse_matrices[local_index].resize(
        n_unique_determinants,
        n_unique_determinants);
  }

  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < packed_pair_begin ||
            packed_pair_index >= packed_pair_end) {
          continue;
        }

        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[
            packed_pair_index - packed_pair_begin].emplace_back(
                left_unique_index,
                right_unique_index,
                packed_pair_value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& sparse_matrix = sparse_matrices[local_index];
    const auto& triplets =
        triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return sparse_matrices;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> build_directional_first_order_sparse_matrix_block(
    const std::vector<DirectionalOppositeSpinPairData>& directional_pair_data,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> sparse_matrices;
  sparse_matrices.resize(block_size);
  if (block_size == 0) {
    return sparse_matrices;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    sparse_matrices[local_index].resize(
        n_unique_determinants,
        n_unique_determinants);
  }

  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& directional_entry =
          directional_pair_data[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          directional_entry.delta_first_order_cofactor_projection;
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < packed_pair_begin ||
            packed_pair_index >= packed_pair_end) {
          continue;
        }

        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[
            packed_pair_index - packed_pair_begin].emplace_back(
                left_unique_index,
                right_unique_index,
                packed_pair_value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& sparse_matrix = sparse_matrices[local_index];
    const auto& triplets =
        triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return sparse_matrices;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>
build_weighted_cofactor_projected_sparse_matrix_block(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair sparse-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> weighted_images(
      block_size);
  for (auto& weighted_image : weighted_images) {
    weighted_image.resize(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_unique_index,
              right_unique_index,
              n_unique_determinants)];
      const auto& projection =
          pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection;
      if (projection.projected_pair_values.empty()) {
        continue;
      }

      const int local_end = std::min(
          block_size,
          static_cast<int>(projection.projected_pair_values.size()) -
              packed_pair_begin);
      for (int local_index = 0; local_index < local_end; ++local_index) {
        const double value =
            projection.projected_pair_values[packed_pair_begin + local_index];
        if (std::abs(value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[local_index].emplace_back(
            left_unique_index,
            right_unique_index,
            value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& weighted_image = weighted_images[local_index];
    const auto& triplets = triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      weighted_image.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return weighted_images;
}

std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>
build_directional_weighted_cofactor_projected_sparse_matrix_block(
    const std::vector<DirectionalOppositeSpinPairData>& directional_pair_data,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "directional packed-pair sparse-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>> weighted_images(
      block_size);
  for (auto& weighted_image : weighted_images) {
    weighted_image.resize(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
  }

  std::vector<std::vector<Eigen::Triplet<double, int>>> triplets_by_local_index(
      block_size);
  for (int right_unique_index = 0;
       right_unique_index < n_unique_determinants;
       ++right_unique_index) {
    for (int left_unique_index = 0;
         left_unique_index < n_unique_determinants;
         ++left_unique_index) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          left_unique_index,
          right_unique_index,
          n_unique_determinants);
      const auto& directional_projection =
          directional_pair_data[ordered_pair_index]
              .delta_first_order_cofactor_projection;
      if (directional_projection.projected_pair_values.empty()) {
        continue;
      }

      const int local_end = std::min(
          block_size,
          static_cast<int>(directional_projection.projected_pair_values.size()) -
              packed_pair_begin);
      for (int local_index = 0; local_index < local_end; ++local_index) {
        const double value =
            directional_projection.projected_pair_values[
                packed_pair_begin + local_index];
        if (std::abs(value) <= kContributionTolerance) {
          continue;
        }
        triplets_by_local_index[local_index].emplace_back(
            left_unique_index,
            right_unique_index,
            value);
      }
    }
  }

  for (int local_index = 0; local_index < block_size; ++local_index) {
    auto& weighted_image = weighted_images[local_index];
    const auto& triplets = triplets_by_local_index[local_index];
    if (!triplets.empty()) {
      weighted_image.setFromTriplets(triplets.begin(), triplets.end());
    }
  }
  return weighted_images;
}

double contract_sparse_matrix_tile_with_dense_tile_matrix(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& sparse_matrix,
    const Eigen::MatrixXd& dense_tile_matrix,
    int dense_tile_column,
    int tile_left_size,
    int row_begin,
    int column_begin) {
  const int row_end = row_begin + tile_left_size;
  const int column_end =
      column_begin + dense_tile_matrix.rows() / tile_left_size;
  double contraction = 0.0;
  for (int column = column_begin; column < column_end; ++column) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator iterator(
             sparse_matrix,
             column);
         iterator;
         ++iterator) {
      const int row = iterator.row();
      if (row < row_begin || row >= row_end) {
        continue;
      }
      const int tile_index =
          (row - row_begin) + tile_left_size * (column - column_begin);
      contraction +=
          iterator.value() *
          dense_tile_matrix(tile_index, dense_tile_column);
    }
  }
  return contraction;
}

void accumulate_alpha_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& beta_pair_matrix,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    Eigen::MatrixXd* alpha_pair_tile) {
  alpha_pair_tile->setZero(
      alpha_left_end - alpha_left_begin,
      alpha_right_end - alpha_right_begin);

  const int beta_tile_size = opposite_spin_backward_unique_tile_size();
  for (const auto& state_coefficients : selected_states.states) {
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& beta_support,
        const SupportWindow& beta_left_window,
        const SupportWindow& beta_right_window,
        Eigen::MatrixXd* beta_tile,
        Eigen::MatrixXd* unused_tile) {
      gather_scalar_block_from_support(
          beta_support,
          beta_left_window,
          beta_support,
          beta_right_window,
          [&](int row, int column) {
            return beta_pair_matrix.coeff(row, column);
          },
          beta_tile);
      unused_tile->setZero(beta_tile->rows(), beta_tile->cols());
    };
    auto consume_images = [&](
        const SupportWindow& alpha_left_window,
        const SupportWindow& alpha_right_window,
        const Eigen::MatrixXd& alpha_image,
        const Eigen::MatrixXd& unused_image) {
      (void)unused_image;
      scatter_add_dense_submatrix_to_tile(
          alpha_image,
          state_coefficients.alpha_support,
          alpha_left_window,
          alpha_left_begin,
          state_coefficients.alpha_support,
          alpha_right_window,
          alpha_right_begin,
          state_coefficients.normalized_state_weight,
          alpha_pair_tile);
    };

    for_each_alpha_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        alpha_left_begin,
        alpha_left_end,
        alpha_right_begin,
        alpha_right_end,
        beta_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_directional_alpha_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& beta_pair_matrix,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    Eigen::MatrixXd* alpha_pair_tile) {
  alpha_pair_tile->setZero(
      alpha_left_end - alpha_left_begin,
      alpha_right_end - alpha_right_begin);

  const int beta_tile_size = opposite_spin_backward_unique_tile_size();
  Eigen::MatrixXd beta_tile;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients) ||
        !selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    // Directional selected-state packed-gradient weight:
    //   d(C U_beta C^T) = dC U_beta C^T + C U_beta dC^T.
    // The accepted and directional coefficient supports may differ, so both
    // product-rule terms are streamed with their own left/right local supports
    // instead of materializing a union-support dense coefficient block.
    auto accumulate_mixed_image = [&](
        const Eigen::MatrixXd& left_coefficients,
        const std::vector<int>& left_alpha_support,
        const std::vector<int>& left_beta_support,
        const Eigen::MatrixXd& right_coefficients,
        const std::vector<int>& right_alpha_support,
        const std::vector<int>& right_beta_support) {
      const SupportWindow alpha_left_window =
          find_support_window(
              left_alpha_support,
              alpha_left_begin,
              alpha_left_end);
      const SupportWindow alpha_right_window =
          find_support_window(
              right_alpha_support,
              alpha_right_begin,
              alpha_right_end);
      if (alpha_left_window.empty() ||
          alpha_right_window.empty() ||
          left_beta_support.empty() ||
          right_beta_support.empty()) {
        return;
      }

      const int n_left_beta_support =
          static_cast<int>(left_beta_support.size());
      const int n_right_beta_support =
          static_cast<int>(right_beta_support.size());
      for (int beta_left_begin_local = 0;
           beta_left_begin_local < n_left_beta_support;
           beta_left_begin_local += beta_tile_size) {
        const SupportWindow beta_left_window{
            beta_left_begin_local,
            std::min(n_left_beta_support, beta_left_begin_local + beta_tile_size),
        };
        const auto left_block =
            left_coefficients.block(
                alpha_left_window.begin,
                beta_left_window.begin,
                alpha_left_window.size(),
                beta_left_window.size());

        for (int beta_right_begin_local = 0;
             beta_right_begin_local < n_right_beta_support;
             beta_right_begin_local += beta_tile_size) {
          const SupportWindow beta_right_window{
              beta_right_begin_local,
              std::min(
                  n_right_beta_support,
                  beta_right_begin_local + beta_tile_size),
          };
          const auto right_block =
              right_coefficients.block(
                  alpha_right_window.begin,
                  beta_right_window.begin,
                  alpha_right_window.size(),
                  beta_right_window.size());
          gather_scalar_block_from_support(
              left_beta_support,
              beta_left_window,
              right_beta_support,
              beta_right_window,
              [&](int row, int column) {
                return beta_pair_matrix.coeff(row, column);
              },
              &beta_tile);
          beta_push.noalias() = left_block * beta_tile;
          alpha_image.noalias() = beta_push * right_block.transpose();
          scatter_add_dense_submatrix_to_tile(
              alpha_image,
              left_alpha_support,
              alpha_left_window,
              alpha_left_begin,
              right_alpha_support,
              alpha_right_window,
              alpha_right_begin,
              state_coefficients.normalized_state_weight,
              alpha_pair_tile);
        }
      }
    };

    accumulate_mixed_image(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support);
    accumulate_mixed_image(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support);
  }
}

void accumulate_beta_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& alpha_pair_matrix,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    Eigen::MatrixXd* beta_pair_tile) {
  beta_pair_tile->setZero(
      beta_left_end - beta_left_begin,
      beta_right_end - beta_right_begin);

  const int alpha_tile_size = opposite_spin_backward_unique_tile_size();
  for (const auto& state_coefficients : selected_states.states) {
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);

    auto build_partner_tiles = [&](
        const std::vector<int>& alpha_support,
        const SupportWindow& alpha_left_window,
        const SupportWindow& alpha_right_window,
        Eigen::MatrixXd* alpha_tile,
        Eigen::MatrixXd* unused_tile) {
      gather_scalar_block_from_support(
          alpha_support,
          alpha_left_window,
          alpha_support,
          alpha_right_window,
          [&](int row, int column) {
            return alpha_pair_matrix.coeff(row, column);
          },
          alpha_tile);
      unused_tile->setZero(alpha_tile->rows(), alpha_tile->cols());
    };
    auto consume_images = [&](
        const SupportWindow& beta_left_window,
        const SupportWindow& beta_right_window,
        const Eigen::MatrixXd& beta_image,
        const Eigen::MatrixXd& unused_image) {
      (void)unused_image;
      scatter_add_dense_submatrix_to_tile(
          beta_image,
          state_coefficients.beta_support,
          beta_left_window,
          beta_left_begin,
          state_coefficients.beta_support,
          beta_right_window,
          beta_right_begin,
          state_coefficients.normalized_state_weight,
          beta_pair_tile);
    };

    for_each_beta_oriented_support_local_image_pair(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        beta_left_begin,
        beta_left_end,
        beta_right_begin,
        beta_right_end,
        alpha_tile_size,
        build_partner_tiles,
        consume_images);
  }
}

void accumulate_directional_beta_pair_matrix_tile(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& alpha_pair_matrix,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    Eigen::MatrixXd* beta_pair_tile) {
  beta_pair_tile->setZero(
      beta_left_end - beta_left_begin,
      beta_right_end - beta_right_begin);

  const int alpha_tile_size = opposite_spin_backward_unique_tile_size();
  Eigen::MatrixXd alpha_tile;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients) ||
        !selected_state_has_local_support(directional_state_coefficients)) {
      continue;
    }
    validate_local_state_coefficient_matrix(state_coefficients);
    validate_local_state_coefficient_matrix(directional_state_coefficients);

    // Beta-side product rule:
    //   d(C^T U_alpha C) = dC^T U_alpha C + C^T U_alpha dC.
    auto accumulate_mixed_image = [&](
        const Eigen::MatrixXd& left_coefficients,
        const std::vector<int>& left_alpha_support,
        const std::vector<int>& left_beta_support,
        const Eigen::MatrixXd& right_coefficients,
        const std::vector<int>& right_alpha_support,
        const std::vector<int>& right_beta_support) {
      const SupportWindow beta_left_window =
          find_support_window(
              left_beta_support,
              beta_left_begin,
              beta_left_end);
      const SupportWindow beta_right_window =
          find_support_window(
              right_beta_support,
              beta_right_begin,
              beta_right_end);
      if (beta_left_window.empty() ||
          beta_right_window.empty() ||
          left_alpha_support.empty() ||
          right_alpha_support.empty()) {
        return;
      }

      const int n_left_alpha_support =
          static_cast<int>(left_alpha_support.size());
      const int n_right_alpha_support =
          static_cast<int>(right_alpha_support.size());
      for (int alpha_left_begin_local = 0;
           alpha_left_begin_local < n_left_alpha_support;
           alpha_left_begin_local += alpha_tile_size) {
        const SupportWindow alpha_left_window{
            alpha_left_begin_local,
            std::min(
                n_left_alpha_support,
                alpha_left_begin_local + alpha_tile_size),
        };
        const auto left_block =
            left_coefficients.block(
                alpha_left_window.begin,
                beta_left_window.begin,
                alpha_left_window.size(),
                beta_left_window.size());

        for (int alpha_right_begin_local = 0;
             alpha_right_begin_local < n_right_alpha_support;
             alpha_right_begin_local += alpha_tile_size) {
          const SupportWindow alpha_right_window{
              alpha_right_begin_local,
              std::min(
                  n_right_alpha_support,
                  alpha_right_begin_local + alpha_tile_size),
          };
          const auto right_block =
              right_coefficients.block(
                  alpha_right_window.begin,
                  beta_right_window.begin,
                  alpha_right_window.size(),
                  beta_right_window.size());
          gather_scalar_block_from_support(
              left_alpha_support,
              alpha_left_window,
              right_alpha_support,
              alpha_right_window,
              [&](int row, int column) {
                return alpha_pair_matrix.coeff(row, column);
              },
              &alpha_tile);
          alpha_push.noalias() = left_block.transpose() * alpha_tile;
          beta_image.noalias() = alpha_push * right_block;
          scatter_add_dense_submatrix_to_tile(
              beta_image,
              left_beta_support,
              beta_left_window,
              beta_left_begin,
              right_beta_support,
              beta_right_window,
              beta_right_begin,
              state_coefficients.normalized_state_weight,
              beta_pair_tile);
        }
      }
    };

    accumulate_mixed_image(
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support,
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support);
    accumulate_mixed_image(
        state_coefficients.local_coefficient_matrix,
        state_coefficients.alpha_support,
        state_coefficients.beta_support,
        directional_state_coefficients.local_coefficient_matrix,
        directional_state_coefficients.alpha_support,
        directional_state_coefficients.beta_support);
  }
}

void validate_directional_selected_state_inputs(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states) {
  if (selected_states.n_unique_alpha != directional_selected_states.n_unique_alpha ||
      selected_states.n_unique_beta != directional_selected_states.n_unique_beta ||
      selected_states.n_determinants != directional_selected_states.n_determinants ||
      selected_states.selected_state_indices !=
          directional_selected_states.selected_state_indices ||
      selected_states.states.size() != directional_selected_states.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices do not match accepted-point dimensions");
  }
}

void accumulate_regular_spin_overlap_gradient_direction_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    double overlap_determinant,
    double delta_overlap_determinant,
    const Eigen::MatrixXd& inverse_overlap_submatrix,
    const Eigen::MatrixXd& delta_inverse_overlap_submatrix,
    double determinant_overlap_weight,
    double delta_determinant_overlap_weight,
    const Eigen::MatrixXd& inverse_overlap_gradient,
    const Eigen::MatrixXd& delta_inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (overlap_determinant == 0.0) {
    throw std::invalid_argument(
        "directional overlap gradient requires non-zero overlap_determinant");
  }

  const Eigen::MatrixXd inverse_overlap_transpose = inverse_overlap_submatrix.transpose();
  const Eigen::MatrixXd delta_inverse_overlap_transpose =
      delta_inverse_overlap_submatrix.transpose();
  Eigen::MatrixXd overlap_submatrix_gradient_direction =
      (delta_determinant_overlap_weight * overlap_determinant +
       determinant_overlap_weight * delta_overlap_determinant) *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() +=
      determinant_overlap_weight * overlap_determinant *
      delta_inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      delta_overlap_determinant *
      inverse_overlap_transpose *
      inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      delta_inverse_overlap_transpose *
      inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      inverse_overlap_transpose *
      delta_inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      inverse_overlap_transpose *
      inverse_overlap_gradient *
      delta_inverse_overlap_transpose;

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_submatrix_gradient_direction(right_row, left_column);
    }
  }
}

void validate_matrix_backward_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "matrix-form opposite-spin backward requires an enabled same-spin cache");
  }
  if (selected_states.n_unique_alpha !=
          static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) ||
      selected_states.n_unique_beta !=
          static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size())) {
    throw std::invalid_argument(
        "selected-state dimensions do not match same-spin cache reuse tables");
  }
}

void accumulate_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = opposite_spin_backward_unique_tile_size();
  Eigen::MatrixXd alpha_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);

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

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          if (dense_matrix_is_effectively_zero(alpha_pair_weight_tile)) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              const double packed_gradient_value =
                  contract_sparse_matrix_tile_with_dense_tile_matrix(
                      alpha_sparse_block[alpha_local_index],
                      alpha_pair_weight_tiles,
                      static_cast<int>(beta_active_index),
                      alpha_tile_left_size,
                      alpha_left_begin,
                      alpha_right_begin);
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

void accumulate_directional_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = opposite_spin_backward_unique_tile_size();
  Eigen::MatrixXd alpha_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);

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

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_directional_alpha_pair_matrix_tile(
              selected_states,
              directional_selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          if (dense_matrix_is_effectively_zero(alpha_pair_weight_tile)) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              const double packed_gradient_value =
                  contract_sparse_matrix_tile_with_dense_tile_matrix(
                      alpha_sparse_block[alpha_local_index],
                      alpha_pair_weight_tiles,
                      static_cast<int>(beta_active_index),
                      alpha_tile_left_size,
                      alpha_left_begin,
                      alpha_right_begin);
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

void accumulate_local_opposite_spin_packed_gradient_by_tiles(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int sparse_block_size,
    int dense_batch_size,
    std::vector<double>* packed_active_two_electron_gradient) {
  const int unique_tile_size = opposite_spin_backward_unique_tile_size();
  Eigen::MatrixXd alpha_pair_weight_tile;
  Eigen::MatrixXd alpha_directional_pair_weight_tile;
  for (int beta_batch_begin = 0;
       beta_batch_begin < n_packed_active_pairs;
       beta_batch_begin += dense_batch_size) {
    const int beta_batch_end =
        std::min(n_packed_active_pairs, beta_batch_begin + dense_batch_size);
    const auto beta_sparse_batch = build_first_order_sparse_matrix_block(
        same_spin_pair_cache.beta_pair_cache_ref(),
        selected_states.n_unique_beta,
        beta_batch_begin,
        beta_batch_end);
    const auto beta_directional_sparse_batch =
        build_directional_first_order_sparse_matrix_block(
            beta_directional_pair_data,
            selected_states.n_unique_beta,
            beta_batch_begin,
            beta_batch_end);

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

        std::vector<int> active_beta_packed_pair_indices;
        const int alpha_tile_left_size = alpha_left_end - alpha_left_begin;
        const int alpha_tile_area =
            alpha_tile_left_size * (alpha_right_end - alpha_right_begin);
        Eigen::MatrixXd alpha_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        Eigen::MatrixXd alpha_directional_pair_weight_tiles(
            alpha_tile_area,
            beta_batch_end - beta_batch_begin);
        std::vector<unsigned char> accepted_tile_nonzero;
        std::vector<unsigned char> directional_tile_nonzero;
        active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
        accepted_tile_nonzero.reserve(beta_sparse_batch.size());
        directional_tile_nonzero.reserve(beta_sparse_batch.size());
        int active_beta_count = 0;
        for (int beta_local_index = 0;
             beta_local_index < beta_batch_end - beta_batch_begin;
             ++beta_local_index) {
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_pair_weight_tile);
          accumulate_alpha_pair_matrix_tile(
              selected_states,
              beta_directional_sparse_batch[beta_local_index],
              alpha_left_begin,
              alpha_left_end,
              alpha_right_begin,
              alpha_right_end,
              &alpha_directional_pair_weight_tile);
          const bool accepted_nonzero =
              !dense_matrix_is_effectively_zero(alpha_pair_weight_tile);
          const bool directional_nonzero =
              !dense_matrix_is_effectively_zero(
                  alpha_directional_pair_weight_tile);
          if (!accepted_nonzero && !directional_nonzero) {
            continue;
          }
          active_beta_packed_pair_indices.push_back(
              beta_batch_begin + beta_local_index);
          alpha_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_pair_weight_tile.data(),
                  alpha_pair_weight_tile.size());
          alpha_directional_pair_weight_tiles.col(active_beta_count) =
              Eigen::Map<const Eigen::VectorXd>(
                  alpha_directional_pair_weight_tile.data(),
                  alpha_directional_pair_weight_tile.size());
          accepted_tile_nonzero.push_back(accepted_nonzero ? 1u : 0u);
          directional_tile_nonzero.push_back(directional_nonzero ? 1u : 0u);
          ++active_beta_count;
        }
        if (active_beta_packed_pair_indices.empty()) {
          continue;
        }

        for (int alpha_block_begin = 0;
             alpha_block_begin < n_packed_active_pairs;
             alpha_block_begin += sparse_block_size) {
          const int alpha_block_end =
              std::min(
                  n_packed_active_pairs,
                  alpha_block_begin + sparse_block_size);
          const auto alpha_sparse_block = build_first_order_sparse_matrix_block(
              same_spin_pair_cache.alpha_pair_cache_ref(),
              selected_states.n_unique_alpha,
              alpha_block_begin,
              alpha_block_end);
          const auto alpha_directional_sparse_block =
              build_directional_first_order_sparse_matrix_block(
                  alpha_directional_pair_data,
                  selected_states.n_unique_alpha,
                  alpha_block_begin,
                  alpha_block_end);

          for (std::size_t beta_active_index = 0;
               beta_active_index < active_beta_packed_pair_indices.size();
               ++beta_active_index) {
            const int beta_packed_pair_index =
                active_beta_packed_pair_indices[beta_active_index];
            for (int alpha_local_index = 0;
                 alpha_local_index < alpha_block_end - alpha_block_begin;
                 ++alpha_local_index) {
              double packed_gradient_value = 0.0;
              if (accepted_tile_nonzero[beta_active_index] != 0u) {
                packed_gradient_value +=
                    contract_sparse_matrix_tile_with_dense_tile_matrix(
                        alpha_directional_sparse_block[alpha_local_index],
                        alpha_pair_weight_tiles,
                        static_cast<int>(beta_active_index),
                        alpha_tile_left_size,
                        alpha_left_begin,
                        alpha_right_begin);
              }
              if (directional_tile_nonzero[beta_active_index] != 0u) {
                packed_gradient_value +=
                    contract_sparse_matrix_tile_with_dense_tile_matrix(
                        alpha_sparse_block[alpha_local_index],
                        alpha_directional_pair_weight_tiles,
                        static_cast<int>(beta_active_index),
                        alpha_tile_left_size,
                        alpha_left_begin,
                        alpha_right_begin);
              }
              if (std::abs(packed_gradient_value) <= kContributionTolerance) {
                continue;
              }

              const int alpha_packed_pair_index =
                  alpha_block_begin + alpha_local_index;
              const int packed_pair_of_pairs_index =
                  TwoElectronIndexer::packed_pair_of_pairs_index(
                      beta_packed_pair_index,
                      alpha_packed_pair_index);
              (*packed_active_two_electron_gradient)[
                  packed_pair_of_pairs_index] += packed_gradient_value;
            }
          }
        }
      }
    }
  }
}

// Store the overlap-adjoint tile bundle as one column-major matrix. Column P is
// W_P(left_local, right_local) with left_local as the fastest index, which
// preserves the mathematical tile layout while avoiding one allocation per
// packed active pair.
Eigen::MatrixXd build_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_alpha_pair_matrix_tile(
          selected_states,
          beta_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_beta_pair_matrix_tile(
          selected_states,
          alpha_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_directional_alpha_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.beta_pair_cache_ref(),
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_directional_alpha_pair_matrix_tile(
          selected_states,
          directional_selected_states,
          beta_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_directional_beta_overlap_weight_tile_matrix(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_weighted_sparse_block =
        build_weighted_cofactor_projected_sparse_matrix_block(
            same_spin_pair_cache.alpha_pair_cache_ref(),
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_directional_beta_pair_matrix_tile(
          selected_states,
          directional_selected_states,
          alpha_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_local_directional_alpha_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>& beta_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end) {
  const int n_left = alpha_left_end - alpha_left_begin;
  const int n_right = alpha_right_end - alpha_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto beta_directional_weighted_sparse_block =
        build_directional_weighted_cofactor_projected_sparse_matrix_block(
            beta_directional_pair_data,
            selected_states.n_unique_beta,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_alpha_pair_matrix_tile(
          selected_states,
          beta_directional_weighted_sparse_block[local_index],
          alpha_left_begin,
          alpha_left_end,
          alpha_right_begin,
          alpha_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

Eigen::MatrixXd build_local_directional_beta_overlap_weight_tile_matrix(
    const std::vector<DirectionalOppositeSpinPairData>& alpha_directional_pair_data,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_packed_active_pairs,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end) {
  const int n_left = beta_left_end - beta_left_begin;
  const int n_right = beta_right_end - beta_right_begin;
  Eigen::MatrixXd tile_matrix =
      Eigen::MatrixXd::Zero(n_left * n_right, n_packed_active_pairs);

  Eigen::MatrixXd tile;
  const int block_size =
      std::min(n_packed_active_pairs, opposite_spin_backward_overlap_block_size());
  for (int packed_pair_begin = 0;
       packed_pair_begin < n_packed_active_pairs;
       packed_pair_begin += block_size) {
    const int packed_pair_end =
        std::min(n_packed_active_pairs, packed_pair_begin + block_size);
    const auto alpha_directional_weighted_sparse_block =
        build_directional_weighted_cofactor_projected_sparse_matrix_block(
            alpha_directional_pair_data,
            selected_states.n_unique_alpha,
            packed_pair_begin,
            packed_pair_end);
    for (int local_index = 0;
         local_index < packed_pair_end - packed_pair_begin;
         ++local_index) {
      accumulate_beta_pair_matrix_tile(
          selected_states,
          alpha_directional_weighted_sparse_block[local_index],
          beta_left_begin,
          beta_left_end,
          beta_right_begin,
          beta_right_end,
          &tile);
      if (!dense_matrix_is_effectively_zero(tile)) {
        tile_matrix.col(packed_pair_begin + local_index) =
            Eigen::Map<const Eigen::VectorXd>(tile.data(), tile.size());
      }
    }
  }
  return tile_matrix;
}

void build_local_packed_pair_dense_image_matrix_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* pair_dense_image) {
  pair_dense_image->resize(
      static_cast<int>(occ_R.size()),
      static_cast<int>(occ_L.size()));
  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(occ_R.size());
         ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index =
          TwoElectronIndexer::packed_pair_index(
              orbital_index_right,
              orbital_index_left);
      (*pair_dense_image)(right_row, left_column) =
          packed_pair_tile_matrix(tile_index, packed_pair_index);
    }
  }
}

void build_inverse_overlap_gradient_from_tile_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& packed_pair_tile_matrix,
    int tile_index,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient->setZero(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index =
          TwoElectronIndexer::packed_pair_index(
              orbital_index_right,
              orbital_index_left);
      (*inverse_overlap_gradient)(left_column, right_row) =
          packed_pair_tile_matrix(tile_index, packed_pair_index);
    }
  }
}

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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
            if (alpha_pair_evaluation.overlap_result.nullity != 0 ||
                alpha_pair_evaluation.overlap_result.overlap_determinant ==
                    0.0) {
              continue;
            }

            const auto& inverse_projection =
                alpha_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            const auto& directional_inverse_projection =
                alpha_directional_pair_data[ordered_pair_index]
                    .delta_inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() &&
                directional_inverse_projection.packed_pair_indices.empty()) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            double delta_determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              const double accepted_dense_value =
                  alpha_pair_weight_tiles(alpha_tile_index, packed_pair_index);
              const double directional_dense_value =
                  directional_alpha_pair_weight_tiles(
                      alpha_tile_index,
                      packed_pair_index);
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  accepted_dense_value;
              delta_determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  directional_dense_value;
            }
            for (std::size_t entry_index = 0;
                 entry_index <
                     directional_inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  directional_inverse_projection.packed_pair_indices[entry_index];
              const double accepted_dense_value =
                  alpha_pair_weight_tiles(alpha_tile_index, packed_pair_index);
              delta_determinant_overlap_weight +=
                  directional_inverse_projection.packed_pair_values[entry_index] *
                  accepted_dense_value;
            }

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
            const Eigen::MatrixXd inverse_overlap_submatrix =
                build_inverse_overlap_submatrix_from_result(
                    alpha_pair_evaluation.overlap_result);
            accumulate_regular_spin_overlap_gradient_direction_local(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                alpha_pair_evaluation.overlap_result.overlap_determinant,
                alpha_directional_pair_data[ordered_pair_index]
                    .delta_overlap_determinant,
                inverse_overlap_submatrix,
                alpha_directional_pair_data[ordered_pair_index]
                    .delta_inverse_overlap_submatrix,
                determinant_overlap_weight,
                delta_determinant_overlap_weight,
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
    const int unique_tile_size = opposite_spin_backward_unique_tile_size();
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
            if (beta_pair_evaluation.overlap_result.nullity != 0 ||
                beta_pair_evaluation.overlap_result.overlap_determinant ==
                    0.0) {
              continue;
            }

            const auto& inverse_projection =
                beta_pair_evaluation
                    .opposite_spin_pair_cache
                    .inverse_overlap_projection;
            const auto& directional_inverse_projection =
                beta_directional_pair_data[ordered_pair_index]
                    .delta_inverse_overlap_projection;
            if (inverse_projection.packed_pair_indices.empty() &&
                directional_inverse_projection.packed_pair_indices.empty()) {
              continue;
            }

            double determinant_overlap_weight = 0.0;
            double delta_determinant_overlap_weight = 0.0;
            for (std::size_t entry_index = 0;
                 entry_index < inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  inverse_projection.packed_pair_indices[entry_index];
              const double accepted_dense_value =
                  beta_pair_weight_tiles(beta_tile_index, packed_pair_index);
              const double directional_dense_value =
                  directional_beta_pair_weight_tiles(
                      beta_tile_index,
                      packed_pair_index);
              determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  accepted_dense_value;
              delta_determinant_overlap_weight +=
                  inverse_projection.packed_pair_values[entry_index] *
                  directional_dense_value;
            }
            for (std::size_t entry_index = 0;
                 entry_index <
                     directional_inverse_projection.packed_pair_indices.size();
                 ++entry_index) {
              const int packed_pair_index =
                  directional_inverse_projection.packed_pair_indices[entry_index];
              const double accepted_dense_value =
                  beta_pair_weight_tiles(beta_tile_index, packed_pair_index);
              delta_determinant_overlap_weight +=
                  directional_inverse_projection.packed_pair_values[entry_index] *
                  accepted_dense_value;
            }

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
            const Eigen::MatrixXd inverse_overlap_submatrix =
                build_inverse_overlap_submatrix_from_result(
                    beta_pair_evaluation.overlap_result);
            accumulate_regular_spin_overlap_gradient_direction_local(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                beta_pair_evaluation.overlap_result.overlap_determinant,
                beta_directional_pair_data[ordered_pair_index]
                    .delta_overlap_determinant,
                inverse_overlap_submatrix,
                beta_directional_pair_data[ordered_pair_index]
                    .delta_inverse_overlap_submatrix,
                determinant_overlap_weight,
                delta_determinant_overlap_weight,
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

}  // namespace

OppositeSpinMatrixBackwardContribution build_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals) {
  validate_matrix_backward_inputs(same_spin_pair_cache, selected_states);

  OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      n_active_orbitals *
          n_active_orbitals,
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  const int n_alpha_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  const int n_beta_packed_active_pairs =
      infer_n_packed_active_pairs(
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

  // Rebuild opposite-spin packed-pair channel matrices on demand from the
  // unique same-spin cache. Sparse packed-pair blocks and unique-spin tiles
  // bound the working set without the old dense provider layer.
  const int sparse_block_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_sparse_block_size());
  const int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());

  // Opposite-spin packed 2e adjoint:
  //   dE / dG(Q,P) = sum_n w_n < U_alpha(P), C^(n) U_beta(Q) [C^(n)]^T >.
  //
  // Build packed-pair sparse families in bounded blocks and contract each
  // selected-state image on a unique-spin tile. No production path builds the
  // old packed_pair -> N_unique^2 dense image.
  accumulate_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      selected_states,
      n_packed_active_pairs,
      sparse_block_size,
      dense_batch_size,
      &result.packed_active_two_electron_gradient);

  // Opposite-spin overlap adjoint:
  // alpha-side uses S_beta .* (G x_beta), beta-side uses S_alpha .* (G x_alpha).
  accumulate_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  accumulate_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);

  return result;
}

OppositeSpinMatrixBackwardContribution
build_directional_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals) {
  validate_matrix_backward_inputs(same_spin_pair_cache, selected_states);
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states);

  OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      n_active_orbitals *
          n_active_orbitals,
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  const int n_alpha_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  const int n_beta_packed_active_pairs =
      infer_n_packed_active_pairs(
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
      opposite_spin_backward_sparse_block_size());
  const int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());

  accumulate_directional_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_packed_active_pairs,
      sparse_block_size,
      dense_batch_size,
      &result.packed_active_two_electron_gradient);

  accumulate_directional_alpha_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  accumulate_directional_beta_overlap_gradient(
      same_spin_pair_cache,
      selected_states,
      directional_selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);

  return result;
}

OppositeSpinMatrixBackwardContribution
build_local_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  validate_matrix_backward_inputs(same_spin_pair_cache, selected_states);

  OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      n_active_orbitals *
          n_active_orbitals,
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  const int n_alpha_packed_active_pairs =
      infer_n_packed_active_pairs(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          "alpha");
  const int n_beta_packed_active_pairs =
      infer_n_packed_active_pairs(
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

  const auto alpha_directional_pair_data =
      build_directional_opposite_spin_pair_data(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha,
          n_active_orbitals,
          active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_packed_active_two_electron_integrals);
  const auto beta_directional_pair_data =
      build_directional_opposite_spin_pair_data(
          same_spin_pair_cache.beta_reuse_table.unique_determinants,
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta,
          n_active_orbitals,
          active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_packed_active_two_electron_integrals);

  const int sparse_block_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_sparse_block_size());
  const int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());

  accumulate_local_opposite_spin_packed_gradient_by_tiles(
      same_spin_pair_cache,
      alpha_directional_pair_data,
      beta_directional_pair_data,
      selected_states,
      n_packed_active_pairs,
      sparse_block_size,
      dense_batch_size,
      &result.packed_active_two_electron_gradient);

  accumulate_local_alpha_overlap_gradient(
      same_spin_pair_cache,
      alpha_directional_pair_data,
      beta_directional_pair_data,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);
  accumulate_local_beta_overlap_gradient(
      same_spin_pair_cache,
      alpha_directional_pair_data,
      beta_directional_pair_data,
      selected_states,
      n_active_orbitals,
      &result.active_orbital_overlap_gradient);

  return result;
}

}  // namespace xmvb::vb
