#include "vb/scf/opposite_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {


constexpr double kContributionTolerance = 1.0e-15;
constexpr int kPackedPairMatrixCacheEntries = 4;
constexpr int kOppositeSpinBackwardSparseBlockSize = 32;
constexpr int kOppositeSpinBackwardDenseBatchSize = 8;
constexpr int kOppositeSpinBackwardOverlapBlockSize = 32;
constexpr std::size_t kOppositeSpinBackwardDenseBatchBytesBudget =
    64ull * 1024ull * 1024ull;
constexpr std::size_t kOppositeSpinBackwardOverlapBlockBytesBudget =
    64ull * 1024ull * 1024ull;

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

std::size_t positive_env_override_size_t(
    const char* env_name,
    std::size_t default_value) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return default_value;
  }
  const std::size_t parsed_value = std::stoull(env_value);
  if (parsed_value == 0) {
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

double max_abs_dense_matrix(const Eigen::MatrixXd& matrix) {
  if (matrix.size() == 0) {
    return 0.0;
  }
  return matrix.cwiseAbs().maxCoeff();
}

std::size_t opposite_spin_backward_dense_batch_bytes_budget() {
  return positive_env_override_size_t(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_DENSE_BATCH_BYTES",
      kOppositeSpinBackwardDenseBatchBytesBudget);
}

std::size_t opposite_spin_backward_overlap_block_bytes_budget() {
  return positive_env_override_size_t(
      "XMVB_CPP_OPPOSITE_SPIN_BACKWARD_OVERLAP_BLOCK_BYTES",
      kOppositeSpinBackwardOverlapBlockBytesBudget);
}

void validate_state_coefficient_matrix(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta) {
  if (state_coefficients.coefficient_matrix.rows() != n_unique_alpha ||
      state_coefficients.coefficient_matrix.cols() != n_unique_beta) {
    throw std::invalid_argument(
        "selected-state coefficient_matrix shape does not match unique-spin dimensions");
  }
}

bool selected_state_has_close_shell_diagonal(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta) {
  return state_coefficients.close_shell_diagonal &&
      n_unique_alpha == n_unique_beta &&
      state_coefficients.diagonal_coefficients.size() ==
          static_cast<std::size_t>(n_unique_alpha);
}

template <typename PairMatrix>
void accumulate_diagonal_pair_matrix(
    const std::vector<double>& diagonal_coefficients,
    const PairMatrix& partner_pair_matrix,
    double scale,
    Eigen::MatrixXd* pair_matrix) {
  if (pair_matrix == nullptr) {
    throw std::invalid_argument("pair_matrix must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      diagonal_coefficients.empty()) {
    return;
  }
  if (partner_pair_matrix.rows() != static_cast<int>(diagonal_coefficients.size()) ||
      partner_pair_matrix.cols() != static_cast<int>(diagonal_coefficients.size())) {
    throw std::invalid_argument(
        "partner_pair_matrix shape does not match close-shell diagonal coefficients");
  }

  for (int column = 0;
       column < static_cast<int>(diagonal_coefficients.size());
       ++column) {
    const double right_coefficient = diagonal_coefficients[column];
    if (std::abs(right_coefficient) <= kContributionTolerance) {
      continue;
    }
    for (int row = 0;
         row < static_cast<int>(diagonal_coefficients.size());
         ++row) {
      const double left_coefficient = diagonal_coefficients[row];
      if (std::abs(left_coefficient) <= kContributionTolerance) {
        continue;
      }
      (*pair_matrix)(row, column) +=
          scale *
          left_coefficient *
          partner_pair_matrix.coeff(row, column) *
          right_coefficient;
    }
  }
}

template <typename PairMatrix>
void accumulate_directional_diagonal_pair_matrix(
    const std::vector<double>& diagonal_coefficients,
    const std::vector<double>& directional_diagonal_coefficients,
    const PairMatrix& partner_pair_matrix,
    double scale,
    Eigen::MatrixXd* pair_matrix) {
  if (pair_matrix == nullptr) {
    throw std::invalid_argument("pair_matrix must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      diagonal_coefficients.empty()) {
    return;
  }
  if (directional_diagonal_coefficients.size() != diagonal_coefficients.size() ||
      partner_pair_matrix.rows() != static_cast<int>(diagonal_coefficients.size()) ||
      partner_pair_matrix.cols() != static_cast<int>(diagonal_coefficients.size())) {
    throw std::invalid_argument(
        "directional close-shell pair-matrix dimensions do not match");
  }

  for (int column = 0;
       column < static_cast<int>(diagonal_coefficients.size());
       ++column) {
    const double right_coefficient = diagonal_coefficients[column];
    const double directional_right =
        directional_diagonal_coefficients[column];
    if (std::abs(right_coefficient) <= kContributionTolerance &&
        std::abs(directional_right) <= kContributionTolerance) {
      continue;
    }
    for (int row = 0;
         row < static_cast<int>(diagonal_coefficients.size());
         ++row) {
      const double left_coefficient = diagonal_coefficients[row];
      const double directional_left =
          directional_diagonal_coefficients[row];
      const double directional_value =
          directional_left * right_coefficient +
          left_coefficient * directional_right;
      if (std::abs(directional_value) <= kContributionTolerance) {
        continue;
      }
      (*pair_matrix)(row, column) +=
          scale *
          partner_pair_matrix.coeff(row, column) *
          directional_value;
    }
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

template <typename DenseMatrixProvider>
Eigen::MatrixXd build_local_packed_pair_dense_image_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    int left_unique_index,
    int right_unique_index,
    DenseMatrixProvider* dense_image_provider) {
  if (dense_image_provider == nullptr) {
    throw std::invalid_argument("dense_image_provider must not be null");
  }

  Eigen::MatrixXd pair_dense_image(
      static_cast<int>(occ_R.size()),
      static_cast<int>(occ_L.size()));
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      pair_dense_image(right_row, left_column) =
          dense_image_provider->get(packed_pair_index)(
              left_unique_index,
              right_unique_index);
    }
  }
  return pair_dense_image;
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

bool gather_sparse_submatrix(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& global_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* local_matrix) {
  if (local_matrix == nullptr) {
    throw std::invalid_argument("local_matrix must not be null");
  }
  local_matrix->setZero(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  bool has_any_nonzero = false;
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator iterator(
             global_matrix,
             column_global);
         iterator;
         ++iterator) {
      const int row_global = iterator.row();
      const auto row_iterator =
          std::lower_bound(row_indices.begin(), row_indices.end(), row_global);
      if (row_iterator == row_indices.end() || *row_iterator != row_global) {
        continue;
      }
      const int row_local =
          static_cast<int>(row_iterator - row_indices.begin());
      (*local_matrix)(row_local, column_local) = iterator.value();
      has_any_nonzero = true;
    }
  }
  return has_any_nonzero;
}

bool gather_pair_submatrix(
    const Eigen::MatrixXd& global_matrix,
    const std::vector<int>& indices,
    Eigen::MatrixXd* local_matrix) {
  gather_dense_submatrix(global_matrix, indices, indices, local_matrix);
  return local_matrix->size() != 0;
}

bool gather_pair_submatrix(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& global_matrix,
    const std::vector<int>& indices,
    Eigen::MatrixXd* local_matrix) {
  return gather_sparse_submatrix(global_matrix, indices, indices, local_matrix);
}

void accumulate_dense_submatrix(
    const Eigen::MatrixXd& local_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    double scale,
    Eigen::MatrixXd* global_matrix) {
  if (global_matrix == nullptr) {
    throw std::invalid_argument("global_matrix must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance || local_matrix.size() == 0) {
    return;
  }
  if (local_matrix.rows() != static_cast<int>(row_indices.size()) ||
      local_matrix.cols() != static_cast<int>(column_indices.size())) {
    throw std::invalid_argument("local_matrix shape does not match support indices");
  }

  for (int column_local = 0;
       column_local < local_matrix.cols();
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (int row_local = 0;
         row_local < local_matrix.rows();
         ++row_local) {
      const int row_global = row_indices[row_local];
      (*global_matrix)(row_global, column_global) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

bool dense_matrix_is_effectively_zero(const Eigen::MatrixXd& matrix) {
  return matrix.size() == 0 || matrix.cwiseAbs().maxCoeff() <= kContributionTolerance;
}

std::size_t dense_square_matrix_storage_bytes(int dimension) {
  return sizeof(double) *
      dimension *
      dimension;
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

std::vector<Eigen::MatrixXd> build_weighted_cofactor_projected_image_block(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair dense-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::MatrixXd> weighted_images;
  weighted_images.resize(block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    weighted_images[local_index] =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
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
      if (projection.projected_pair_values.empty()) {
        continue;
      }

      for (int local_index = 0; local_index < block_size; ++local_index) {
        const int packed_pair_index = packed_pair_begin + local_index;
        if (static_cast<int>(projection.projected_pair_values.size()) <= packed_pair_index) {
          continue;
        }
        weighted_images[local_index](
            left_unique_index,
            right_unique_index) =
            projection.projected_pair_values[packed_pair_index];
      }
    }
  }
  return weighted_images;
}

std::vector<Eigen::MatrixXd> build_directional_weighted_cofactor_projected_image_block(
    const std::vector<DirectionalOppositeSpinPairData>& directional_pair_data,
    int n_unique_determinants,
    int packed_pair_begin,
    int packed_pair_end) {
  if (packed_pair_begin < 0 ||
      packed_pair_end < packed_pair_begin) {
    throw std::invalid_argument(
        "packed-pair dense-image block range must satisfy 0 <= begin <= end");
  }
  const int block_size = packed_pair_end - packed_pair_begin;
  std::vector<Eigen::MatrixXd> weighted_images;
  weighted_images.resize(block_size);
  for (int local_index = 0; local_index < block_size; ++local_index) {
    weighted_images[local_index] =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  }
  if (block_size == 0) {
    return weighted_images;
  }

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

      for (int local_index = 0; local_index < block_size; ++local_index) {
        const int packed_pair_index = packed_pair_begin + local_index;
        const double directional_projected_value =
            static_cast<int>(directional_projection.projected_pair_values.size()) >
                    packed_pair_index
                ? directional_projection.projected_pair_values[
                      packed_pair_index]
                : 0.0;
        weighted_images[local_index](
            left_unique_index,
            right_unique_index) = directional_projected_value;
      }
    }
  }
  return weighted_images;
}

template <typename MatrixType>
class CachedPackedPairBlockProvider {
public:
  CachedPackedPairBlockProvider(
      int n_packed_pairs,
      int block_size,
      std::function<std::vector<MatrixType>(int, int)> block_builder)
      : n_packed_pairs_(n_packed_pairs),
        block_size_(block_size),
        block_builder_(std::move(block_builder)) {
    if (n_packed_pairs_ < 0 || block_size_ <= 0) {
      throw std::invalid_argument(
          "packed-pair block provider requires non-negative size and positive block size");
    }
  }

  const MatrixType& get(int packed_pair_index) {
    if (packed_pair_index < 0 || packed_pair_index >= n_packed_pairs_) {
      throw std::out_of_range("packed_pair_index is outside provider range");
    }
    for (auto& entry : cache_entries_) {
      if (entry.valid &&
          packed_pair_index >= entry.packed_pair_begin &&
          packed_pair_index < entry.packed_pair_end) {
        entry.last_access_stamp = ++access_stamp_;
        return entry.matrices[
            packed_pair_index - entry.packed_pair_begin];
      }
    }

    const int packed_pair_begin = packed_pair_index / block_size_ * block_size_;
    const int packed_pair_end =
        std::min(n_packed_pairs_, packed_pair_begin + block_size_);
    CacheEntry built_entry;
    built_entry.valid = true;
    built_entry.packed_pair_begin = packed_pair_begin;
    built_entry.packed_pair_end = packed_pair_end;
    built_entry.last_access_stamp = ++access_stamp_;
    built_entry.matrices = block_builder_(packed_pair_begin, packed_pair_end);
    if (static_cast<int>(built_entry.matrices.size()) !=
        packed_pair_end - packed_pair_begin) {
      throw std::invalid_argument(
          "packed-pair block builder returned wrong number of matrices");
    }

    if (cache_entries_.size() < kPackedPairMatrixCacheEntries) {
      cache_entries_.push_back(std::move(built_entry));
      auto& entry = cache_entries_.back();
      return entry.matrices[
          packed_pair_index - entry.packed_pair_begin];
    }

    auto victim_iterator = cache_entries_.begin();
    for (auto iterator = cache_entries_.begin();
         iterator != cache_entries_.end();
         ++iterator) {
      if (iterator->last_access_stamp < victim_iterator->last_access_stamp) {
        victim_iterator = iterator;
      }
    }
    *victim_iterator = std::move(built_entry);
    return victim_iterator->matrices[
        packed_pair_index - victim_iterator->packed_pair_begin];
  }

private:
  struct CacheEntry {
    bool valid = false;
    int packed_pair_begin = -1;
    int packed_pair_end = -1;
    std::uint64_t last_access_stamp = 0;
    std::vector<MatrixType> matrices;
  };

  int n_packed_pairs_ = 0;
  int block_size_ = 1;
  std::function<std::vector<MatrixType>(int, int)> block_builder_;
  std::uint64_t access_stamp_ = 0;
  std::vector<CacheEntry> cache_entries_;
};

double contract_sparse_matrix_with_dense_image(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& sparse_matrix,
    const Eigen::MatrixXd& dense_image) {
  if (sparse_matrix.rows() != dense_image.rows() ||
      sparse_matrix.cols() != dense_image.cols()) {
    throw std::invalid_argument(
        "sparse opposite-spin channel shape does not match dense image");
  }

  double contraction = 0.0;
  for (int outer_index = 0; outer_index < sparse_matrix.outerSize(); ++outer_index) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator iterator(
             sparse_matrix,
             outer_index);
         iterator;
         ++iterator) {
      contraction +=
          iterator.value() * dense_image(iterator.row(), iterator.col());
    }
  }
  return contraction;
}

template <typename PairMatrix>
Eigen::MatrixXd accumulate_alpha_pair_matrix(
    const SelectedStateDeterminantMatrices& selected_states,
    const PairMatrix& beta_pair_matrix) {
  if (beta_pair_matrix.rows() != selected_states.n_unique_beta ||
      beta_pair_matrix.cols() != selected_states.n_unique_beta) {
    throw std::invalid_argument(
        "beta channel matrix shape does not match selected-state dimensions");
  }

  Eigen::MatrixXd alpha_pair_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  for (const auto& state_coefficients : selected_states.states) {
    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta)) {
      accumulate_diagonal_pair_matrix(
          state_coefficients.diagonal_coefficients,
          beta_pair_matrix,
          state_coefficients.normalized_state_weight,
          &alpha_pair_matrix);
      continue;
    }
    const Eigen::MatrixXd& coefficient_matrix =
        state_coefficients.coefficient_matrix;
    alpha_pair_matrix.noalias() +=
        state_coefficients.normalized_state_weight *
        (coefficient_matrix * beta_pair_matrix) *
        coefficient_matrix.transpose();
  }
  return alpha_pair_matrix;
}

template <typename PairMatrix>
Eigen::MatrixXd accumulate_beta_pair_matrix(
    const SelectedStateDeterminantMatrices& selected_states,
    const PairMatrix& alpha_pair_matrix) {
  if (alpha_pair_matrix.rows() != selected_states.n_unique_alpha ||
      alpha_pair_matrix.cols() != selected_states.n_unique_alpha) {
    throw std::invalid_argument(
        "alpha channel matrix shape does not match selected-state dimensions");
  }

  Eigen::MatrixXd beta_pair_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  for (const auto& state_coefficients : selected_states.states) {
    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta)) {
      accumulate_diagonal_pair_matrix(
          state_coefficients.diagonal_coefficients,
          alpha_pair_matrix,
          state_coefficients.normalized_state_weight,
          &beta_pair_matrix);
      continue;
    }
    const Eigen::MatrixXd& coefficient_matrix =
        state_coefficients.coefficient_matrix;
    beta_pair_matrix.noalias() +=
        state_coefficients.normalized_state_weight *
        coefficient_matrix.transpose() *
        alpha_pair_matrix *
        coefficient_matrix;
  }
  return beta_pair_matrix;
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

template <typename PairMatrix>
Eigen::MatrixXd accumulate_directional_alpha_pair_matrix(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const PairMatrix& beta_pair_matrix) {
  if (beta_pair_matrix.rows() != selected_states.n_unique_beta ||
      beta_pair_matrix.cols() != selected_states.n_unique_beta) {
    throw std::invalid_argument(
        "beta channel matrix shape does not match selected-state dimensions");
  }
  Eigen::MatrixXd alpha_pair_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd pair_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_pair_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_image(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    validate_state_coefficient_matrix(
        directional_state_coefficients,
        directional_selected_states.n_unique_alpha,
        directional_selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta) &&
        selected_state_has_close_shell_diagonal(
            directional_state_coefficients,
            directional_selected_states.n_unique_alpha,
            directional_selected_states.n_unique_beta)) {
      accumulate_directional_diagonal_pair_matrix(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          beta_pair_matrix,
          state_coefficients.normalized_state_weight,
          &alpha_pair_matrix);
      continue;
    }
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;
    directional_coefficient_matrix_dense =
        directional_state_coefficients.coefficient_matrix;
    // This contraction is linear in `delta C`, so we can evaluate it on a
    // scaled directional matrix and restore the exact amplitude afterward.
    const double directional_scale =
        std::max(
            1.0,
            max_abs_dense_matrix(directional_coefficient_matrix_dense));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_dense /= directional_scale;
    }

    pair_push.noalias() = coefficient_matrix_dense * beta_pair_matrix;
    directional_pair_push.noalias() =
        directional_coefficient_matrix_dense * beta_pair_matrix;
    // Product rule at fixed cache kernel B:
    // d(C B C^T) = dC B C^T + C B dC^T.
    directional_image.noalias() =
        directional_pair_push * coefficient_matrix_dense.transpose();
    directional_image.noalias() +=
        pair_push * directional_coefficient_matrix_dense.transpose();
    alpha_pair_matrix.noalias() +=
        state_coefficients.normalized_state_weight *
        directional_scale *
        directional_image;
  }
  return alpha_pair_matrix;
}

template <typename PairMatrix>
Eigen::MatrixXd accumulate_directional_beta_pair_matrix(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const PairMatrix& alpha_pair_matrix) {
  if (alpha_pair_matrix.rows() != selected_states.n_unique_alpha ||
      alpha_pair_matrix.cols() != selected_states.n_unique_alpha) {
    throw std::invalid_argument(
        "alpha channel matrix shape does not match selected-state dimensions");
  }
  Eigen::MatrixXd beta_pair_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd pair_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_pair_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_image(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    validate_state_coefficient_matrix(
        directional_state_coefficients,
        directional_selected_states.n_unique_alpha,
        directional_selected_states.n_unique_beta);
    if (selected_state_has_close_shell_diagonal(
            state_coefficients,
            selected_states.n_unique_alpha,
            selected_states.n_unique_beta) &&
        selected_state_has_close_shell_diagonal(
            directional_state_coefficients,
            directional_selected_states.n_unique_alpha,
            directional_selected_states.n_unique_beta)) {
      accumulate_directional_diagonal_pair_matrix(
          state_coefficients.diagonal_coefficients,
          directional_state_coefficients.diagonal_coefficients,
          alpha_pair_matrix,
          state_coefficients.normalized_state_weight,
          &beta_pair_matrix);
      continue;
    }
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;
    directional_coefficient_matrix_dense =
        directional_state_coefficients.coefficient_matrix;
    // This contraction is linear in `delta C`, so we can evaluate it on a
    // scaled directional matrix and restore the exact amplitude afterward.
    const double directional_scale =
        std::max(
            1.0,
            max_abs_dense_matrix(directional_coefficient_matrix_dense));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_dense /= directional_scale;
    }

    pair_push.noalias() = alpha_pair_matrix * coefficient_matrix_dense;
    directional_pair_push.noalias() =
        alpha_pair_matrix * directional_coefficient_matrix_dense;
    // Product rule at fixed cache kernel A:
    // d(C^T A C) = dC^T A C + C^T A dC.
    directional_image.noalias() =
        directional_coefficient_matrix_dense.transpose() * pair_push;
    directional_image.noalias() +=
        coefficient_matrix_dense.transpose() * directional_pair_push;
    beta_pair_matrix.noalias() +=
        state_coefficients.normalized_state_weight *
        directional_scale *
        directional_image;
  }
  return beta_pair_matrix;
}

template <typename DenseMatrixProvider>
void build_inverse_overlap_gradient_from_dense_image_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    int left_unique_index,
    int right_unique_index,
    DenseMatrixProvider* dense_image_provider,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (dense_image_provider == nullptr || inverse_overlap_gradient == nullptr) {
    throw std::invalid_argument(
        "dense-image inverse overlap gradient inputs must not be null");
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient->setZero(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      (*inverse_overlap_gradient)(left_column, right_row) =
          dense_image_provider->get(packed_pair_index)(
              left_unique_index,
              right_unique_index);
    }
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> alpha_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto beta_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.beta_pair_cache_ref(),
                selected_states.n_unique_beta,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> alpha_pair_dense_images;
        alpha_pair_dense_images.reserve(beta_weighted_image_block.size());
        for (auto& beta_weighted_image : beta_weighted_image_block) {
          alpha_pair_dense_images.push_back(
              accumulate_alpha_pair_matrix(
                  selected_states,
                  beta_weighted_image));
        }
        return alpha_pair_dense_images;
      });
  for (int alpha_left_id = 0;
       alpha_left_id < selected_states.n_unique_alpha;
       ++alpha_left_id) {
    for (int alpha_right_id = 0;
         alpha_right_id < selected_states.n_unique_alpha;
         ++alpha_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          alpha_left_id,
          alpha_right_id,
          selected_states.n_unique_alpha);
      const auto& alpha_pair_evaluation =
          same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
      if (alpha_pair_evaluation.overlap_result.nullity == 1) {
        const Eigen::MatrixXd pair_dense_image =
            build_local_packed_pair_dense_image_matrix(
                unique_alpha_determinants[alpha_left_id],
                unique_alpha_determinants[alpha_right_id],
                alpha_left_id,
                alpha_right_id,
                &alpha_pair_dense_image_provider);
        accumulate_singular_spin_overlap_gradient_from_dense_image_local(
            unique_alpha_determinants[alpha_left_id],
            unique_alpha_determinants[alpha_right_id],
            alpha_pair_evaluation.overlap_result,
            pair_dense_image,
            n_active_orbitals,
            active_orbital_overlap_gradient);
        continue;
      }
      const auto& inverse_projection =
          alpha_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
        const double projected_value =
            alpha_pair_dense_image_provider.get(packed_pair_index)(
                alpha_left_id,
                alpha_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            projected_value;
      }
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_alpha_determinants[alpha_left_id],
          unique_alpha_determinants[alpha_right_id],
          alpha_left_id,
          alpha_right_id,
          &alpha_pair_dense_image_provider,
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_beta),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> beta_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto alpha_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.alpha_pair_cache_ref(),
                selected_states.n_unique_alpha,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> beta_pair_dense_images;
        beta_pair_dense_images.reserve(alpha_weighted_image_block.size());
        for (auto& alpha_weighted_image : alpha_weighted_image_block) {
          beta_pair_dense_images.push_back(
              accumulate_beta_pair_matrix(
                  selected_states,
                  alpha_weighted_image));
        }
        return beta_pair_dense_images;
      });
  for (int beta_left_id = 0;
       beta_left_id < selected_states.n_unique_beta;
       ++beta_left_id) {
    for (int beta_right_id = 0;
         beta_right_id < selected_states.n_unique_beta;
         ++beta_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          beta_left_id,
          beta_right_id,
          selected_states.n_unique_beta);
      const auto& beta_pair_evaluation =
          same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
      if (beta_pair_evaluation.overlap_result.nullity == 1) {
        const Eigen::MatrixXd pair_dense_image =
            build_local_packed_pair_dense_image_matrix(
                unique_beta_determinants[beta_left_id],
                unique_beta_determinants[beta_right_id],
                beta_left_id,
                beta_right_id,
                &beta_pair_dense_image_provider);
        accumulate_singular_spin_overlap_gradient_from_dense_image_local(
            unique_beta_determinants[beta_left_id],
            unique_beta_determinants[beta_right_id],
            beta_pair_evaluation.overlap_result,
            pair_dense_image,
            n_active_orbitals,
            active_orbital_overlap_gradient);
        continue;
      }
      const auto& inverse_projection =
          beta_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
        const double projected_value =
            beta_pair_dense_image_provider.get(packed_pair_index)(
                beta_left_id,
                beta_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            projected_value;
      }
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_beta_determinants[beta_left_id],
          unique_beta_determinants[beta_right_id],
          beta_left_id,
          beta_right_id,
          &beta_pair_dense_image_provider,
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> alpha_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto beta_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.beta_pair_cache_ref(),
                selected_states.n_unique_beta,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> alpha_pair_dense_images;
        alpha_pair_dense_images.reserve(beta_weighted_image_block.size());
        for (auto& beta_weighted_image : beta_weighted_image_block) {
          alpha_pair_dense_images.push_back(
              accumulate_directional_alpha_pair_matrix(
                  selected_states,
                  directional_selected_states,
                  beta_weighted_image));
        }
        return alpha_pair_dense_images;
      });
  for (int alpha_left_id = 0;
       alpha_left_id < selected_states.n_unique_alpha;
       ++alpha_left_id) {
    for (int alpha_right_id = 0;
         alpha_right_id < selected_states.n_unique_alpha;
         ++alpha_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          alpha_left_id,
          alpha_right_id,
          selected_states.n_unique_alpha);
      const auto& alpha_pair_evaluation =
          same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
      const auto& inverse_projection =
          alpha_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
        const double projected_value =
            alpha_pair_dense_image_provider.get(packed_pair_index)(
                alpha_left_id,
                alpha_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            projected_value;
      }
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_alpha_determinants[alpha_left_id],
          unique_alpha_determinants[alpha_right_id],
          alpha_left_id,
          alpha_right_id,
          &alpha_pair_dense_image_provider,
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_beta),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> beta_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto alpha_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.alpha_pair_cache_ref(),
                selected_states.n_unique_alpha,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> beta_pair_dense_images;
        beta_pair_dense_images.reserve(alpha_weighted_image_block.size());
        for (auto& alpha_weighted_image : alpha_weighted_image_block) {
          beta_pair_dense_images.push_back(
              accumulate_directional_beta_pair_matrix(
                  selected_states,
                  directional_selected_states,
                  alpha_weighted_image));
        }
        return beta_pair_dense_images;
      });
  for (int beta_left_id = 0;
       beta_left_id < selected_states.n_unique_beta;
       ++beta_left_id) {
    for (int beta_right_id = 0;
         beta_right_id < selected_states.n_unique_beta;
         ++beta_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          beta_left_id,
          beta_right_id,
          selected_states.n_unique_beta);
      const auto& beta_pair_evaluation =
          same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
      const auto& inverse_projection =
          beta_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
        const double projected_value =
            beta_pair_dense_image_provider.get(packed_pair_index)(
                beta_left_id,
                beta_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            projected_value;
      }
      if (std::abs(determinant_overlap_weight) <= kContributionTolerance) {
        continue;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_beta_determinants[beta_left_id],
          unique_beta_determinants[beta_right_id],
          beta_left_id,
          beta_right_id,
          &beta_pair_dense_image_provider,
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> alpha_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto beta_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.beta_pair_cache_ref(),
                selected_states.n_unique_beta,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> alpha_pair_dense_images;
        alpha_pair_dense_images.reserve(beta_weighted_image_block.size());
        for (auto& beta_weighted_image : beta_weighted_image_block) {
          alpha_pair_dense_images.push_back(
              accumulate_alpha_pair_matrix(
                  selected_states,
                  beta_weighted_image));
        }
        return alpha_pair_dense_images;
      });
  CachedPackedPairBlockProvider<Eigen::MatrixXd>
      directional_alpha_pair_dense_image_provider(
          n_packed_active_pairs,
          overlap_block_size,
          [&](int packed_pair_begin, int packed_pair_end) {
            auto beta_directional_weighted_image_block =
                build_directional_weighted_cofactor_projected_image_block(
                    beta_directional_pair_data,
                    selected_states.n_unique_beta,
                    packed_pair_begin,
                    packed_pair_end);
            std::vector<Eigen::MatrixXd> alpha_pair_dense_images;
            alpha_pair_dense_images.reserve(
                beta_directional_weighted_image_block.size());
            for (auto& beta_directional_weighted_image :
                 beta_directional_weighted_image_block) {
              alpha_pair_dense_images.push_back(
                  accumulate_alpha_pair_matrix(
                      selected_states,
                      beta_directional_weighted_image));
            }
            return alpha_pair_dense_images;
          });
  for (int alpha_left_id = 0;
       alpha_left_id < selected_states.n_unique_alpha;
       ++alpha_left_id) {
    for (int alpha_right_id = 0;
         alpha_right_id < selected_states.n_unique_alpha;
         ++alpha_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          alpha_left_id,
          alpha_right_id,
          selected_states.n_unique_alpha);
      const auto& alpha_pair_evaluation =
          same_spin_pair_cache.alpha_pair_cache_ref()[ordered_pair_index];
      if (alpha_pair_evaluation.overlap_result.nullity != 0 ||
          alpha_pair_evaluation.overlap_result.overlap_determinant == 0.0) {
        continue;
      }

      const auto& inverse_projection =
          alpha_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
            alpha_pair_dense_image_provider.get(packed_pair_index)(
                alpha_left_id,
                alpha_right_id);
        const double directional_dense_value =
            directional_alpha_pair_dense_image_provider.get(packed_pair_index)(
                alpha_left_id,
                alpha_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            accepted_dense_value;
        delta_determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            directional_dense_value;
      }
      for (std::size_t entry_index = 0;
           entry_index < directional_inverse_projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index =
            directional_inverse_projection.packed_pair_indices[entry_index];
        const double accepted_dense_value =
            alpha_pair_dense_image_provider.get(packed_pair_index)(
                alpha_left_id,
                alpha_right_id);
        delta_determinant_overlap_weight +=
            directional_inverse_projection.packed_pair_values[entry_index] *
            accepted_dense_value;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_alpha_determinants[alpha_left_id],
          unique_alpha_determinants[alpha_right_id],
          alpha_left_id,
          alpha_right_id,
          &alpha_pair_dense_image_provider,
          &inverse_overlap_gradient);
      Eigen::MatrixXd delta_inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_alpha_determinants[alpha_left_id],
          unique_alpha_determinants[alpha_right_id],
          alpha_left_id,
          alpha_right_id,
          &directional_alpha_pair_dense_image_provider,
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
  const int overlap_block_size = limited_packed_pair_block_size(
      n_packed_active_pairs,
      opposite_spin_backward_overlap_block_size(),
      dense_square_matrix_storage_bytes(selected_states.n_unique_beta),
      opposite_spin_backward_overlap_block_bytes_budget());
  CachedPackedPairBlockProvider<Eigen::MatrixXd> beta_pair_dense_image_provider(
      n_packed_active_pairs,
      overlap_block_size,
      [&](int packed_pair_begin, int packed_pair_end) {
        auto alpha_weighted_image_block =
            build_weighted_cofactor_projected_image_block(
                same_spin_pair_cache.alpha_pair_cache_ref(),
                selected_states.n_unique_alpha,
                packed_pair_begin,
                packed_pair_end);
        std::vector<Eigen::MatrixXd> beta_pair_dense_images;
        beta_pair_dense_images.reserve(alpha_weighted_image_block.size());
        for (auto& alpha_weighted_image : alpha_weighted_image_block) {
          beta_pair_dense_images.push_back(
              accumulate_beta_pair_matrix(
                  selected_states,
                  alpha_weighted_image));
        }
        return beta_pair_dense_images;
      });
  CachedPackedPairBlockProvider<Eigen::MatrixXd>
      directional_beta_pair_dense_image_provider(
          n_packed_active_pairs,
          overlap_block_size,
          [&](int packed_pair_begin, int packed_pair_end) {
            auto alpha_directional_weighted_image_block =
                build_directional_weighted_cofactor_projected_image_block(
                    alpha_directional_pair_data,
                    selected_states.n_unique_alpha,
                    packed_pair_begin,
                    packed_pair_end);
            std::vector<Eigen::MatrixXd> beta_pair_dense_images;
            beta_pair_dense_images.reserve(
                alpha_directional_weighted_image_block.size());
            for (auto& alpha_directional_weighted_image :
                 alpha_directional_weighted_image_block) {
              beta_pair_dense_images.push_back(
                  accumulate_beta_pair_matrix(
                      selected_states,
                      alpha_directional_weighted_image));
            }
            return beta_pair_dense_images;
          });
  for (int beta_left_id = 0;
       beta_left_id < selected_states.n_unique_beta;
       ++beta_left_id) {
    for (int beta_right_id = 0;
         beta_right_id < selected_states.n_unique_beta;
         ++beta_right_id) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          beta_left_id,
          beta_right_id,
          selected_states.n_unique_beta);
      const auto& beta_pair_evaluation =
          same_spin_pair_cache.beta_pair_cache_ref()[ordered_pair_index];
      if (beta_pair_evaluation.overlap_result.nullity != 0 ||
          beta_pair_evaluation.overlap_result.overlap_determinant == 0.0) {
        continue;
      }

      const auto& inverse_projection =
          beta_pair_evaluation.opposite_spin_pair_cache.inverse_overlap_projection;
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
            beta_pair_dense_image_provider.get(packed_pair_index)(
                beta_left_id,
                beta_right_id);
        const double directional_dense_value =
            directional_beta_pair_dense_image_provider.get(packed_pair_index)(
                beta_left_id,
                beta_right_id);
        determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            accepted_dense_value;
        delta_determinant_overlap_weight +=
            inverse_projection.packed_pair_values[entry_index] *
            directional_dense_value;
      }
      for (std::size_t entry_index = 0;
           entry_index < directional_inverse_projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index =
            directional_inverse_projection.packed_pair_indices[entry_index];
        const double accepted_dense_value =
            beta_pair_dense_image_provider.get(packed_pair_index)(
                beta_left_id,
                beta_right_id);
        delta_determinant_overlap_weight +=
            directional_inverse_projection.packed_pair_values[entry_index] *
            accepted_dense_value;
      }

      Eigen::MatrixXd inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_beta_determinants[beta_left_id],
          unique_beta_determinants[beta_right_id],
          beta_left_id,
          beta_right_id,
          &beta_pair_dense_image_provider,
          &inverse_overlap_gradient);
      Eigen::MatrixXd delta_inverse_overlap_gradient;
      build_inverse_overlap_gradient_from_dense_image_local(
          unique_beta_determinants[beta_left_id],
          unique_beta_determinants[beta_right_id],
          beta_left_id,
          beta_right_id,
          &directional_beta_pair_dense_image_provider,
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
  // unique same-spin cache. The small provider cache keeps the working set
  // bounded while preserving the exact matrix-form contractions.
  const int sparse_block_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_sparse_block_size());
  int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());
  const std::size_t per_alpha_weight_matrix_bytes =
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha);
  if (per_alpha_weight_matrix_bytes > 0) {
    const std::size_t dense_batch_bytes_budget =
        opposite_spin_backward_dense_batch_bytes_budget();
    const std::size_t budget_limited_batch_size = std::max<std::size_t>(
        1u,
        dense_batch_bytes_budget / per_alpha_weight_matrix_bytes);
    dense_batch_size = std::min(
        dense_batch_size,
        static_cast<int>(std::min<std::size_t>(
            n_packed_active_pairs,
            budget_limited_batch_size)));
  }

  // Opposite-spin packed 2e adjoint:
  //   dE / dG(Q,P) = sum_n w_n < U_alpha(P), C^(n) U_beta(Q) [C^(n)]^T >.
  //
  // Build packed-pair sparse families in bounded blocks rather than per packed
  // pair. This keeps the memory footprint bounded like the tiled forward path
  // while avoiding the previous "re-scan the whole unique-pair cache once per
  // beta pair and once per alpha pair" rebuild pattern. The dense beta batch
  // is also capped by a byte budget so large unique-spin spaces automatically
  // fall back to a smaller working set.
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

    std::vector<int> active_beta_packed_pair_indices;
    std::vector<Eigen::MatrixXd> alpha_pair_weight_matrices;
    active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
    alpha_pair_weight_matrices.reserve(beta_sparse_batch.size());
    for (int beta_local_index = 0;
         beta_local_index < beta_batch_end - beta_batch_begin;
         ++beta_local_index) {
      Eigen::MatrixXd alpha_pair_weight_matrix =
          accumulate_alpha_pair_matrix(
              selected_states,
              beta_sparse_batch[beta_local_index]);
      if (dense_matrix_is_effectively_zero(alpha_pair_weight_matrix)) {
        continue;
      }
      active_beta_packed_pair_indices.push_back(beta_batch_begin + beta_local_index);
      alpha_pair_weight_matrices.push_back(std::move(alpha_pair_weight_matrix));
    }
    if (active_beta_packed_pair_indices.empty()) {
      continue;
    }

    for (int alpha_block_begin = 0;
         alpha_block_begin < n_packed_active_pairs;
         alpha_block_begin += sparse_block_size) {
      const int alpha_block_end =
          std::min(n_packed_active_pairs, alpha_block_begin + sparse_block_size);
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
        const auto& alpha_pair_weight_matrix =
            alpha_pair_weight_matrices[beta_active_index];
        for (int alpha_local_index = 0;
             alpha_local_index < alpha_block_end - alpha_block_begin;
             ++alpha_local_index) {
          const double packed_gradient_value =
              contract_sparse_matrix_with_dense_image(
                  alpha_sparse_block[alpha_local_index],
                  alpha_pair_weight_matrix);
          if (std::abs(packed_gradient_value) <= kContributionTolerance) {
            continue;
          }

          const int alpha_packed_pair_index =
              alpha_block_begin + alpha_local_index;
          const int packed_pair_of_pairs_index =
              TwoElectronIndexer::packed_pair_of_pairs_index(
                  beta_packed_pair_index,
                  alpha_packed_pair_index);
          result.packed_active_two_electron_gradient[
              packed_pair_of_pairs_index] += packed_gradient_value;
        }
      }
    }
  }

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
  int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());
  const std::size_t per_alpha_weight_matrix_bytes =
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha);
  if (per_alpha_weight_matrix_bytes > 0) {
    const std::size_t dense_batch_bytes_budget =
        opposite_spin_backward_dense_batch_bytes_budget();
    const std::size_t budget_limited_batch_size = std::max<std::size_t>(
        1u,
        dense_batch_bytes_budget / per_alpha_weight_matrix_bytes);
    dense_batch_size = std::min(
        dense_batch_size,
        static_cast<int>(std::min<std::size_t>(
            n_packed_active_pairs,
            budget_limited_batch_size)));
  }

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

    std::vector<int> active_beta_packed_pair_indices;
    std::vector<Eigen::MatrixXd> alpha_pair_weight_matrices;
    active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
    alpha_pair_weight_matrices.reserve(beta_sparse_batch.size());
    for (int beta_local_index = 0;
         beta_local_index < beta_batch_end - beta_batch_begin;
         ++beta_local_index) {
      Eigen::MatrixXd alpha_pair_weight_matrix =
          accumulate_directional_alpha_pair_matrix(
              selected_states,
              directional_selected_states,
              beta_sparse_batch[beta_local_index]);
      if (dense_matrix_is_effectively_zero(alpha_pair_weight_matrix)) {
        continue;
      }
      active_beta_packed_pair_indices.push_back(beta_batch_begin + beta_local_index);
      alpha_pair_weight_matrices.push_back(std::move(alpha_pair_weight_matrix));
    }
    if (active_beta_packed_pair_indices.empty()) {
      continue;
    }

    for (int alpha_block_begin = 0;
         alpha_block_begin < n_packed_active_pairs;
         alpha_block_begin += sparse_block_size) {
      const int alpha_block_end =
          std::min(n_packed_active_pairs, alpha_block_begin + sparse_block_size);
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
        const auto& alpha_pair_weight_matrix =
            alpha_pair_weight_matrices[beta_active_index];
        for (int alpha_local_index = 0;
             alpha_local_index < alpha_block_end - alpha_block_begin;
             ++alpha_local_index) {
          const double packed_gradient_value =
              contract_sparse_matrix_with_dense_image(
                  alpha_sparse_block[alpha_local_index],
                  alpha_pair_weight_matrix);
          if (std::abs(packed_gradient_value) <= kContributionTolerance) {
            continue;
          }

          const int alpha_packed_pair_index =
              alpha_block_begin + alpha_local_index;
          const int packed_pair_of_pairs_index =
              TwoElectronIndexer::packed_pair_of_pairs_index(
                  beta_packed_pair_index,
                  alpha_packed_pair_index);
          result.packed_active_two_electron_gradient[
              packed_pair_of_pairs_index] += packed_gradient_value;
        }
      }
    }
  }

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
  int dense_batch_size = std::min(
      n_packed_active_pairs,
      opposite_spin_backward_dense_batch_size());
  const std::size_t per_alpha_weight_matrix_bytes =
      dense_square_matrix_storage_bytes(selected_states.n_unique_alpha);
  if (per_alpha_weight_matrix_bytes > 0) {
    const std::size_t dense_batch_bytes_budget =
        opposite_spin_backward_dense_batch_bytes_budget();
    const std::size_t budget_limited_batch_size = std::max<std::size_t>(
        1u,
        dense_batch_bytes_budget / per_alpha_weight_matrix_bytes);
    dense_batch_size = std::min(
        dense_batch_size,
        static_cast<int>(std::min<std::size_t>(
            n_packed_active_pairs,
            budget_limited_batch_size)));
  }

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

    std::vector<int> active_beta_packed_pair_indices;
    std::vector<Eigen::MatrixXd> alpha_pair_weight_matrices;
    std::vector<Eigen::MatrixXd> alpha_directional_pair_weight_matrices;
    active_beta_packed_pair_indices.reserve(beta_sparse_batch.size());
    alpha_pair_weight_matrices.reserve(beta_sparse_batch.size());
    alpha_directional_pair_weight_matrices.reserve(beta_sparse_batch.size());
    for (int beta_local_index = 0;
         beta_local_index < beta_batch_end - beta_batch_begin;
         ++beta_local_index) {
      Eigen::MatrixXd alpha_pair_weight_matrix =
          accumulate_alpha_pair_matrix(
              selected_states,
              beta_sparse_batch[beta_local_index]);
      Eigen::MatrixXd alpha_directional_pair_weight_matrix =
          accumulate_alpha_pair_matrix(
              selected_states,
              beta_directional_sparse_batch[beta_local_index]);
      if (dense_matrix_is_effectively_zero(alpha_pair_weight_matrix) &&
          dense_matrix_is_effectively_zero(alpha_directional_pair_weight_matrix)) {
        continue;
      }
      active_beta_packed_pair_indices.push_back(beta_batch_begin + beta_local_index);
      alpha_pair_weight_matrices.push_back(std::move(alpha_pair_weight_matrix));
      alpha_directional_pair_weight_matrices.push_back(
          std::move(alpha_directional_pair_weight_matrix));
    }
    if (active_beta_packed_pair_indices.empty()) {
      continue;
    }

    for (int alpha_block_begin = 0;
         alpha_block_begin < n_packed_active_pairs;
         alpha_block_begin += sparse_block_size) {
      const int alpha_block_end =
          std::min(n_packed_active_pairs, alpha_block_begin + sparse_block_size);
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
        const auto& alpha_pair_weight_matrix =
            alpha_pair_weight_matrices[beta_active_index];
        const auto& alpha_directional_pair_weight_matrix =
            alpha_directional_pair_weight_matrices[beta_active_index];
        for (int alpha_local_index = 0;
             alpha_local_index < alpha_block_end - alpha_block_begin;
             ++alpha_local_index) {
          double packed_gradient_value = 0.0;
          if (!dense_matrix_is_effectively_zero(alpha_pair_weight_matrix)) {
            packed_gradient_value += contract_sparse_matrix_with_dense_image(
                alpha_directional_sparse_block[alpha_local_index],
                alpha_pair_weight_matrix);
          }
          if (!dense_matrix_is_effectively_zero(
                  alpha_directional_pair_weight_matrix)) {
            packed_gradient_value += contract_sparse_matrix_with_dense_image(
                alpha_sparse_block[alpha_local_index],
                alpha_directional_pair_weight_matrix);
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
          result.packed_active_two_electron_gradient[
              packed_pair_of_pairs_index] += packed_gradient_value;
        }
      }
    }
  }

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
