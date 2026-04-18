#include "vb/scf/same_spin_matrix_backward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <cblas.h>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {


constexpr double kContributionTolerance = 1.0e-15;
constexpr int kSameSpinBackwardPairTileSize = 64;

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

double max_abs_dense_matrix(const Eigen::MatrixXd& matrix) {
  if (matrix.size() == 0) {
    return 0.0;
  }
  return matrix.cwiseAbs().maxCoeff();
}

void set_symmetric_matrix_entry(
    Eigen::MatrixXd* matrix,
    int row,
    int column,
    double value) {
  (*matrix)(row, column) = value;
  (*matrix)(column, row) = value;
}

void multiply_left_symmetric(
    const Eigen::MatrixXd& symmetric_left,
    const Eigen::MatrixXd& right,
    Eigen::MatrixXd* output) {
  if (output == nullptr) {
    throw std::invalid_argument("output must not be null");
  }
  if (symmetric_left.rows() != symmetric_left.cols() ||
      symmetric_left.cols() != right.rows()) {
    throw std::invalid_argument("left symmetric multiply shape mismatch");
  }
  output->resize(symmetric_left.rows(), right.cols());
  if (output->size() == 0) {
    output->setZero();
    return;
  }
  cblas_dsymm(
      CblasColMajor,
      CblasLeft,
      CblasUpper,
      symmetric_left.rows(),
      right.cols(),
      1.0,
      symmetric_left.data(),
      symmetric_left.outerStride(),
      right.data(),
      right.outerStride(),
      0.0,
      output->data(),
      output->outerStride());
}

void multiply_right_symmetric(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& symmetric_right,
    Eigen::MatrixXd* output) {
  if (output == nullptr) {
    throw std::invalid_argument("output must not be null");
  }
  if (symmetric_right.rows() != symmetric_right.cols() ||
      left.cols() != symmetric_right.rows()) {
    throw std::invalid_argument("right symmetric multiply shape mismatch");
  }
  output->resize(left.rows(), symmetric_right.cols());
  if (output->size() == 0) {
    output->setZero();
    return;
  }
  cblas_dsymm(
      CblasColMajor,
      CblasRight,
      CblasUpper,
      left.rows(),
      symmetric_right.cols(),
      1.0,
      symmetric_right.data(),
      symmetric_right.outerStride(),
      left.data(),
      left.outerStride(),
      0.0,
      output->data(),
      output->outerStride());
}

struct SameSpinExactWeightMatrices {
  Eigen::MatrixXd alpha_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_overlap_weight_matrix;
  Eigen::MatrixXd alpha_partner_total_transfer_matrix;
  Eigen::MatrixXd alpha_singular_partner_transfer_matrix;
  Eigen::MatrixXd beta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_overlap_weight_matrix;
  Eigen::MatrixXd beta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_singular_partner_transfer_matrix;
};

struct SameSpinPairScalarMatrices {
  Eigen::MatrixXd overlap_determinant_matrix;
  Eigen::MatrixXd regular_total_hamiltonian_matrix;
  Eigen::MatrixXd singular_total_hamiltonian_matrix;
};

struct SingleChannelSameSpinWeightMatrices {
  Eigen::MatrixXd alpha_weight_matrix;
  Eigen::MatrixXd beta_weight_matrix;
};

struct SameSpinLocalResponseWeightMatrices {
  Eigen::MatrixXd alpha_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_delta_overlap_weight_matrix;
  Eigen::MatrixXd alpha_delta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_delta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_delta_overlap_weight_matrix;
  Eigen::MatrixXd beta_delta_partner_total_transfer_matrix;
};

struct SameSpinDirectionalScalarMatrices {
  Eigen::MatrixXd delta_overlap_determinant_matrix;
  Eigen::MatrixXd delta_regular_total_hamiltonian_matrix;
  Eigen::MatrixXd delta_singular_total_hamiltonian_matrix;
};

struct RegularSpinDirectionalData {
  double overlap_determinant = 0.0;
  double delta_overlap_determinant = 0.0;
  Eigen::MatrixXd inverse_overlap_submatrix;
  Eigen::MatrixXd delta_inverse_overlap_submatrix;
  Eigen::MatrixXd cofactor_1st;
  Eigen::MatrixXd delta_cofactor_1st;
  double same_spin_total_phi = 0.0;
  double delta_same_spin_total_phi = 0.0;
  Eigen::MatrixXd same_spin_inverse_overlap_gradient;
  Eigen::MatrixXd delta_same_spin_inverse_overlap_gradient;
};

struct SingularSpinDirectionalData {
  double delta_overlap_determinant = 0.0;
  double delta_total_hamiltonian = 0.0;
  Eigen::MatrixXd cofactor_1st;
  Eigen::MatrixXd delta_cofactor_1st;
  Eigen::MatrixXd delta_same_spin_overlap_hamiltonian_gradient;
};

bool selected_state_has_local_support(
    const SelectedStateDeterminantCoefficients& state_coefficients) {
  return !state_coefficients.alpha_support.empty() &&
      !state_coefficients.beta_support.empty() &&
      state_coefficients.local_coefficient_matrix.size() != 0;
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

void scatter_add_dense_submatrix(
    const Eigen::MatrixXd& local_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    double scale,
    Eigen::MatrixXd* global_matrix) {
  if (global_matrix == nullptr) {
    throw std::invalid_argument("global_matrix must not be null");
  }
  if (local_matrix.rows() != static_cast<int>(row_indices.size()) ||
      local_matrix.cols() != static_cast<int>(column_indices.size())) {
    throw std::invalid_argument(
        "local_matrix shape does not match scatter support dimensions");
  }
  if (std::abs(scale) <= kContributionTolerance) {
    return;
  }
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      (*global_matrix)(row_global, column_global) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

std::vector<int> build_merged_support_indices(
    const std::vector<int>& first,
    const std::vector<int>& second) {
  std::vector<int> merged;
  merged.reserve(first.size() + second.size());
  std::set_union(
      first.begin(),
      first.end(),
      second.begin(),
      second.end(),
      std::back_inserter(merged));
  return merged;
}

void accumulate_selected_state_alpha_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& beta_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_alpha_weight_matrix,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* alpha_image) {
  if (global_alpha_weight_matrix == nullptr ||
      beta_push == nullptr ||
      alpha_image == nullptr) {
    throw std::invalid_argument("alpha image outputs must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      !selected_state_has_local_support(state_coefficients)) {
    return;
  }
  validate_local_state_coefficient_matrix(state_coefficients);
  const auto& local_coefficients = state_coefficients.local_coefficient_matrix;
  if (beta_kernel_subblock.rows() != local_coefficients.cols() ||
      beta_kernel_subblock.cols() != local_coefficients.cols()) {
    throw std::invalid_argument(
        "beta support kernel shape does not match selected-state local coefficients");
  }

  multiply_right_symmetric(
      local_coefficients,
      beta_kernel_subblock,
      beta_push);
  alpha_image->noalias() =
      (*beta_push) * local_coefficients.transpose();
  scatter_add_dense_submatrix(
      *alpha_image,
      state_coefficients.alpha_support,
      state_coefficients.alpha_support,
      scale,
      global_alpha_weight_matrix);
}

void accumulate_selected_state_beta_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& alpha_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_beta_weight_matrix,
    Eigen::MatrixXd* alpha_push,
    Eigen::MatrixXd* beta_image) {
  if (global_beta_weight_matrix == nullptr ||
      alpha_push == nullptr ||
      beta_image == nullptr) {
    throw std::invalid_argument("beta image outputs must not be null");
  }
  if (std::abs(scale) <= kContributionTolerance ||
      !selected_state_has_local_support(state_coefficients)) {
    return;
  }
  validate_local_state_coefficient_matrix(state_coefficients);
  const auto& local_coefficients = state_coefficients.local_coefficient_matrix;
  if (alpha_kernel_subblock.rows() != local_coefficients.rows() ||
      alpha_kernel_subblock.cols() != local_coefficients.rows()) {
    throw std::invalid_argument(
        "alpha support kernel shape does not match selected-state local coefficients");
  }

  multiply_left_symmetric(
      alpha_kernel_subblock,
      local_coefficients,
      alpha_push);
  beta_image->noalias() =
      local_coefficients.transpose() * (*alpha_push);
  scatter_add_dense_submatrix(
      *beta_image,
      state_coefficients.beta_support,
      state_coefficients.beta_support,
      scale,
      global_beta_weight_matrix);
}

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = xmvb::index_at(occ_L, left_column);
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = xmvb::index_at(occ_R, right_row);
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_overlap_block_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& overlap_block_gradient,
    double weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }
  if (overlap_block_gradient.rows() != static_cast<int>(occ_R.size()) ||
      overlap_block_gradient.cols() != static_cast<int>(occ_L.size())) {
    throw std::invalid_argument(
        "overlap_block_gradient dimensions do not match occupied lists");
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = xmvb::index_at(occ_L, left_column);
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = xmvb::index_at(occ_R, right_row);
      (*active_orbital_overlap_gradient)[xmvb::to_size(orbital_index_left) *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          weight * overlap_block_gradient(right_row, left_column);
    }
  }
}

void accumulate_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double overlap_determinant,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double cofactor_scale = weight / overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = xmvb::index_at(occ_L, left_first);
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = xmvb::index_at(occ_R, right_first);
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = xmvb::index_at(occ_L, left_second);
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = xmvb::index_at(occ_R, right_second);
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double second_order_cofactor =
              cofactor_scale * (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21);

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          xmvb::index_at(*packed_active_two_electron_gradient, direct_index) +=
              second_order_cofactor;
          xmvb::index_at(*packed_active_two_electron_gradient, exchange_index) -=
              second_order_cofactor;
        }
      }
    }
  }
}

void accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& overlap_result,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  const DeterminantOverlapResolver overlap_resolver;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = xmvb::index_at(occ_L, left_first);
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = xmvb::index_at(occ_R, right_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = xmvb::index_at(occ_L, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = xmvb::index_at(occ_R, right_second);
          const double second_order_cofactor =
              weight *
              calc_deleted_minor_determinant(
                  overlap_block,
                  {right_first, right_second},
                  {left_first, left_second},
                  overlap_resolver);
          if (std::abs(second_order_cofactor) <= kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          xmvb::index_at(*packed_active_two_electron_gradient, direct_index) +=
              second_order_cofactor;
          xmvb::index_at(*packed_active_two_electron_gradient, exchange_index) -=
              second_order_cofactor;
        }
      }
    }
  }
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
  if (std::abs(scale) <= kContributionTolerance ||
      minor_cofactor.size() == 0) {
    return;
  }
  if (minor_cofactor.rows() != static_cast<int>(retained_rows.size()) ||
      minor_cofactor.cols() != static_cast<int>(retained_cols.size())) {
    throw std::invalid_argument(
        "minor cofactor dimensions do not match retained deleted-minor indices");
  }

  for (int retained_col = 0;
       retained_col < static_cast<int>(retained_cols.size());
       ++retained_col) {
    const int overlap_col = retained_cols[xmvb::to_size(retained_col)];
    for (int retained_row = 0;
         retained_row < static_cast<int>(retained_rows.size());
         ++retained_row) {
      const int overlap_row = retained_rows[xmvb::to_size(retained_row)];
      (*overlap_block_gradient)(overlap_row, overlap_col) +=
          scale * minor_cofactor(retained_row, retained_col);
    }
  }
}

void accumulate_directional_deleted_minor_pullback_to_overlap_block_gradient_local(
    const Eigen::MatrixXd& overlap_block,
    const Eigen::MatrixXd& delta_overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    double coefficient,
    double delta_coefficient,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* delta_overlap_gradient) {
  if (delta_overlap_gradient == nullptr) {
    throw std::invalid_argument("delta_overlap_gradient must not be null");
  }
  if ((std::abs(coefficient) <= kContributionTolerance &&
       std::abs(delta_coefficient) <= kContributionTolerance)) {
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
  const Eigen::MatrixXd delta_minor =
      build_deleted_minor_matrix(
          delta_overlap_block,
          deleted_rows,
          deleted_cols);
  const DeterminantOverlapResult minor_result =
      overlap_resolver.resolve_matrix(minor);
  const Eigen::MatrixXd minor_cofactor =
      calc_cofactor_1st(minor_result);
  const Eigen::MatrixXd delta_minor_cofactor =
      build_directional_first_cofactor_matrix(
          minor,
          delta_minor,
          minor_result,
          overlap_resolver);
  const double deleted_minor_sign = calc_deleted_minor_sign(
      overlap_block.rows(),
      overlap_block.cols(),
      deleted_rows,
      deleted_cols);
  scatter_minor_cofactor_to_overlap_block_gradient_local(
      minor_cofactor,
      retained_rows,
      retained_cols,
      deleted_minor_sign * delta_coefficient,
      delta_overlap_gradient);
  scatter_minor_cofactor_to_overlap_block_gradient_local(
      delta_minor_cofactor,
      retained_rows,
      retained_cols,
      deleted_minor_sign * coefficient,
      delta_overlap_gradient);
}

Eigen::MatrixXd build_spin_one_electron_block_matrix_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = xmvb::index_at(occ_L, left_column);
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = xmvb::index_at(occ_R, right_row);
      one_electron_block(right_row, left_column) =
          h1e_act[xmvb::to_size(orbital_index_left) * n_active_orbitals +
                  orbital_index_right];
    }
  }
  return one_electron_block;
}

SameSpinPhiResult evaluate_same_spin_phi_with_optional_cache_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (pair_evaluation.has_same_spin_phi_cache) {
    if (inverse_overlap_gradient != nullptr) {
      *inverse_overlap_gradient = pair_evaluation.same_spin_inverse_overlap_gradient;
    }
    return {
        pair_evaluation.same_spin_one_electron_phi,
        pair_evaluation.same_spin_total_phi,
    };
  }

  return compute_same_spin_original_phi(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      pair_evaluation.overlap_result,
      inverse_overlap_gradient);
}

Eigen::MatrixXd build_local_overlap_direction_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    int n_active_orbitals) {
  const std::vector<double> delta_overlap_storage =
      build_overlap_submatrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  return Eigen::Map<const Eigen::MatrixXd>(
      delta_overlap_storage.data(),
      static_cast<int>(occ_R.size()),
      static_cast<int>(occ_L.size()));
}

double lookup_directional_active_two_electron_kernel_value(
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    int row_packed_pair_index,
    int column_packed_pair_index) {
  if (delta_packed_active_two_electron_integrals.empty()) {
    return 0.0;
  }
  const int packed_pair_of_pairs_index =
      TwoElectronIndexer::packed_pair_of_pairs_index(
          row_packed_pair_index,
          column_packed_pair_index);
  return delta_packed_active_two_electron_integrals[xmvb::to_size(
      packed_pair_of_pairs_index)];
}

SingularSpinDirectionalData build_singular_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const auto& overlap_result = pair_evaluation.overlap_result;
  if (overlap_result.nullity == 0 &&
      overlap_result.overlap_determinant != 0.0) {
    throw std::runtime_error(
        "singular spin directional data requires a singular determinant pair");
  }

  SingularSpinDirectionalData result;
  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  const DeterminantOverlapResolver overlap_resolver;

  result.cofactor_1st = calc_cofactor_1st(overlap_result);
  result.delta_overlap_determinant =
      (result.cofactor_1st.cwiseProduct(delta_overlap_submatrix)).sum();
  result.delta_cofactor_1st =
      build_directional_first_cofactor_matrix(
          overlap_block,
          delta_overlap_submatrix,
          overlap_result,
          overlap_resolver);

  const Eigen::MatrixXd one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          active_one_electron_matrix,
          n_active_orbitals);
  const Eigen::MatrixXd delta_one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          delta_active_one_electron_matrix,
          n_active_orbitals);
  result.delta_total_hamiltonian =
      (delta_one_electron_block.cwiseProduct(result.cofactor_1st)).sum() +
      (one_electron_block.cwiseProduct(result.delta_cofactor_1st)).sum();
  result.delta_same_spin_overlap_hamiltonian_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(occ_R.size()),
          static_cast<int>(occ_L.size()));

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      accumulate_directional_deleted_minor_pullback_to_overlap_block_gradient_local(
          overlap_block,
          delta_overlap_submatrix,
          {right_row},
          {left_column},
          one_electron_block(right_row, left_column),
          delta_one_electron_block(right_row, left_column),
          overlap_resolver,
          &result.delta_same_spin_overlap_hamiltonian_gradient);
    }
  }

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_electrons = static_cast<int>(occ_L.size());
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[xmvb::to_size(right_first)];
      const int direct_left_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right_first,
          orbital_index_left_first);
      for (int left_second = left_first + 1;
           left_second < n_electrons;
           ++left_second) {
        const int orbital_index_left_second = occ_L[xmvb::to_size(left_second)];
        const int exchange_left_pair_index = TwoElectronIndexer::packed_pair_index(
            orbital_index_right_first,
            orbital_index_left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[xmvb::to_size(right_second)];
          const int direct_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_first);
          const double interaction_value =
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  direct_left_pair_index,
                  direct_right_pair_index,
                  n_active_orbitals) -
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  exchange_left_pair_index,
                  exchange_right_pair_index,
                  n_active_orbitals);
          const double delta_interaction_value =
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  direct_left_pair_index,
                  direct_right_pair_index) -
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  exchange_left_pair_index,
                  exchange_right_pair_index);
          const double second_order_cofactor =
              calc_deleted_minor_determinant(
                  overlap_block,
                  {right_first, right_second},
                  {left_first, left_second},
                  overlap_resolver);
          const double directional_second_order_cofactor =
              calc_directional_deleted_minor_determinant(
                  overlap_block,
                  delta_overlap_submatrix,
                  {right_first, right_second},
                  {left_first, left_second},
                  overlap_resolver);
          result.delta_total_hamiltonian +=
              delta_interaction_value * second_order_cofactor +
              interaction_value * directional_second_order_cofactor;
          accumulate_directional_deleted_minor_pullback_to_overlap_block_gradient_local(
              overlap_block,
              delta_overlap_submatrix,
              {right_first, right_second},
              {left_first, left_second},
              interaction_value,
              delta_interaction_value,
              overlap_resolver,
              &result.delta_same_spin_overlap_hamiltonian_gradient);
        }
      }
    }
  }

  return result;
}

RegularSpinDirectionalData build_regular_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const auto& overlap_result = pair_evaluation.overlap_result;
  if (overlap_result.nullity != 0 || overlap_result.overlap_determinant == 0.0) {
    throw std::runtime_error(
        "regular spin directional data requires a non-singular determinant pair");
  }

  RegularSpinDirectionalData result;
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

  const Eigen::MatrixXd one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          active_one_electron_matrix,
          n_active_orbitals);

  const Eigen::MatrixXd delta_one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          delta_active_one_electron_matrix,
          n_active_orbitals);
          
  const SameSpinPhiResult same_spin_phi_result =
      evaluate_same_spin_phi_with_optional_cache_local(
          occ_L,
          occ_R,
          active_one_electron_matrix,
          n_active_orbitals,
          active_space_two_electron_result,
          pair_evaluation,
          &result.same_spin_inverse_overlap_gradient);
  result.same_spin_total_phi = same_spin_phi_result.total_phi;
  result.delta_same_spin_inverse_overlap_gradient =
      delta_one_electron_block.transpose();
  result.delta_same_spin_total_phi =
      (delta_one_electron_block.cwiseProduct(
           result.inverse_overlap_submatrix.transpose()))
          .sum();
  result.delta_same_spin_total_phi +=
      (one_electron_block.cwiseProduct(
           result.delta_inverse_overlap_submatrix.transpose()))
          .sum();

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_electrons = static_cast<int>(occ_L.size());
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[xmvb::to_size(right_first)];
      const int direct_left_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right_first,
          orbital_index_left_first);
      for (int left_second = left_first + 1;
           left_second < n_electrons;
           ++left_second) {
        const int orbital_index_left_second = occ_L[xmvb::to_size(left_second)];
        const int exchange_left_pair_index = TwoElectronIndexer::packed_pair_index(
            orbital_index_right_first,
            orbital_index_left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[xmvb::to_size(right_second)];
          const int direct_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_first);
          const double interaction_value =
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  direct_left_pair_index,
                  direct_right_pair_index,
                  n_active_orbitals) -
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  exchange_left_pair_index,
                  exchange_right_pair_index,
                  n_active_orbitals);
          const double delta_interaction_value =
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  direct_left_pair_index,
                  direct_right_pair_index) -
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  exchange_left_pair_index,
                  exchange_right_pair_index);

          const double x11 =
              result.inverse_overlap_submatrix(left_first, right_first);
          const double x22 =
              result.inverse_overlap_submatrix(left_second, right_second);
          const double x12 =
              result.inverse_overlap_submatrix(left_second, right_first);
          const double x21 =
              result.inverse_overlap_submatrix(left_first, right_second);
          const double dx11 =
              result.delta_inverse_overlap_submatrix(left_first, right_first);
          const double dx22 =
              result.delta_inverse_overlap_submatrix(left_second, right_second);
          const double dx12 =
              result.delta_inverse_overlap_submatrix(left_second, right_first);
          const double dx21 =
              result.delta_inverse_overlap_submatrix(left_first, right_second);
          const double wedge = x11 * x22 - x12 * x21;
          const double delta_wedge =
              dx11 * x22 + x11 * dx22 - dx12 * x21 - x12 * dx21;
          result.delta_same_spin_total_phi +=
              delta_interaction_value * wedge +
              interaction_value * delta_wedge;

          result.delta_same_spin_inverse_overlap_gradient(
              left_first,
              right_first) +=
              delta_interaction_value * x22 +
              interaction_value * dx22;
          result.delta_same_spin_inverse_overlap_gradient(
              left_second,
              right_second) +=
              delta_interaction_value * x11 +
              interaction_value * dx11;
          result.delta_same_spin_inverse_overlap_gradient(
              left_second,
              right_first) -=
              delta_interaction_value * x21 +
              interaction_value * dx21;
          result.delta_same_spin_inverse_overlap_gradient(
              left_first,
              right_second) -=
              delta_interaction_value * x12 +
              interaction_value * dx12;
        }
      }
    }
  }

  return result;
}

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = xmvb::index_at(occ_L, left_column);
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = xmvb::index_at(occ_R, right_row);
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          delta_weight * cofactor_1st(right_row, left_column) +
          weight * delta_cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_directional_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double overlap_determinant,
    double delta_overlap_determinant,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if ((std::abs(weight) <= kContributionTolerance &&
       std::abs(delta_weight) <= kContributionTolerance) ||
      overlap_determinant == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double inv_overlap_determinant = 1.0 / overlap_determinant;
  const double delta_prefactor =
      delta_weight * inv_overlap_determinant -
      weight * delta_overlap_determinant *
          inv_overlap_determinant * inv_overlap_determinant;
  const double prefactor = weight * inv_overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = xmvb::index_at(occ_L, left_first);
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = xmvb::index_at(occ_R, right_first);
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      const double delta_cofactor_11 =
          delta_cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = xmvb::index_at(occ_L, left_second);
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        const double delta_cofactor_12 =
            delta_cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              xmvb::index_at(occ_R, right_second);
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double delta_cofactor_22 =
              delta_cofactor_1st(right_second, left_second);
          const double delta_cofactor_21 =
              delta_cofactor_1st(right_second, left_first);
          const double wedge =
              cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21;
          const double delta_wedge =
              delta_cofactor_11 * cofactor_22 +
              cofactor_11 * delta_cofactor_22 -
              delta_cofactor_12 * cofactor_21 -
              cofactor_12 * delta_cofactor_21;
          const double directional_second_order_cofactor =
              delta_prefactor * wedge +
              prefactor * delta_wedge;

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          xmvb::index_at(*packed_active_two_electron_gradient, direct_index) +=
              directional_second_order_cofactor;
          xmvb::index_at(*packed_active_two_electron_gradient, exchange_index) -=
              directional_second_order_cofactor;
        }
      }
    }
  }
}

void accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& overlap_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    int n_active_orbitals,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  const DeterminantOverlapResolver overlap_resolver;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = xmvb::index_at(occ_L, left_first);
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = xmvb::index_at(occ_R, right_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = xmvb::index_at(occ_L, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = xmvb::index_at(occ_R, right_second);
          const double second_order_cofactor =
              calc_deleted_minor_determinant(
                  overlap_block,
                  {right_first, right_second},
                  {left_first, left_second},
                  overlap_resolver);
          const double directional_second_order_cofactor =
              calc_directional_deleted_minor_determinant(
                  overlap_block,
                  delta_overlap_submatrix,
                  {right_first, right_second},
                  {left_first, left_second},
                  overlap_resolver);
          const double total_directional_second_order_cofactor =
              delta_weight * second_order_cofactor +
              weight * directional_second_order_cofactor;
          if (std::abs(total_directional_second_order_cofactor) <=
              kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          xmvb::index_at(*packed_active_two_electron_gradient, direct_index) +=
              total_directional_second_order_cofactor;
          xmvb::index_at(*packed_active_two_electron_gradient, exchange_index) -=
              total_directional_second_order_cofactor;
        }
      }
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
    const int orbital_index_left = xmvb::index_at(occ_L, left_column);
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = xmvb::index_at(occ_R, right_row);
      (*active_orbital_overlap_gradient)[xmvb::to_size(orbital_index_left) *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_submatrix_gradient_direction(right_row, left_column);
    }
  }
}

SameSpinPairScalarMatrices build_pair_scalar_matrices(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        "ordered same-spin pair cache size does not match unique-spin dimensions");
  }

  SameSpinPairScalarMatrices matrices;
  matrices.overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  matrices.regular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  matrices.singular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);

  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      set_symmetric_matrix_entry(
          &matrices.overlap_determinant_matrix,
          left_id,
          right_id,
          pair_evaluation.overlap_result.overlap_determinant);
      if (pair_evaluation.overlap_result.nullity == 0) {
        set_symmetric_matrix_entry(
            &matrices.regular_total_hamiltonian_matrix,
            left_id,
            right_id,
            pair_evaluation.total_hamiltonian);
      } else {
        set_symmetric_matrix_entry(
            &matrices.singular_total_hamiltonian_matrix,
            left_id,
            right_id,
            pair_evaluation.total_hamiltonian);
      }
    }
  }

  return matrices;
}

SingleChannelSameSpinWeightMatrices
build_dense_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales) {
  if (per_state_scales.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "per_state_scales must align with selected_states.states");
  }

  SingleChannelSameSpinWeightMatrices weight_matrices;
  weight_matrices.alpha_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  Eigen::MatrixXd beta_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_image(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd beta_image(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const double scale = per_state_scales[state_offset];
    if (std::abs(scale) <= kContributionTolerance) {
      continue;
    }

    validate_state_coefficient_matrix(
        selected_states.states[state_offset],
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    coefficient_matrix_dense =
        selected_states.states[state_offset].coefficient_matrix;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_partner_kernel_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_weight_matrix.noalias() += scale * alpha_image;

    multiply_left_symmetric(
        alpha_partner_kernel_matrix,
        coefficient_matrix_dense,
        &alpha_push);
    beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
    weight_matrices.beta_weight_matrix.noalias() += scale * beta_image;
  }

  return weight_matrices;
}

SingleChannelSameSpinWeightMatrices
build_support_sparse_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales) {
  if (per_state_scales.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "per_state_scales must align with selected_states.states");
  }

  SingleChannelSameSpinWeightMatrices weight_matrices;
  weight_matrices.alpha_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  Eigen::MatrixXd alpha_partner_subblock;
  Eigen::MatrixXd beta_partner_subblock;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd alpha_image;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const double scale = per_state_scales[state_offset];
    if (std::abs(scale) <= kContributionTolerance ||
        !selected_state_has_local_support(state_coefficients)) {
      continue;
    }

    gather_dense_submatrix(
        alpha_partner_kernel_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_partner_subblock);
    gather_dense_submatrix(
        beta_partner_kernel_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_partner_subblock);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_partner_subblock,
        scale,
        &weight_matrices.alpha_weight_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_partner_subblock,
        scale,
        &weight_matrices.beta_weight_matrix,
        &alpha_push,
        &beta_image);
  }

  return weight_matrices;
}

SingleChannelSameSpinWeightMatrices
build_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales) {
  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_same_spin_weight_matrices_from_partner_kernels(
        selected_states,
        alpha_partner_kernel_matrix,
        beta_partner_kernel_matrix,
        per_state_scales);
  }
  return build_dense_same_spin_weight_matrices_from_partner_kernels(
      selected_states,
      alpha_partner_kernel_matrix,
      beta_partner_kernel_matrix,
      per_state_scales);
}

SameSpinExactWeightMatrices build_dense_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  SameSpinExactWeightMatrices weight_matrices;
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  const SameSpinPairScalarMatrices beta_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta);

  // Each selected state already stores the determinant expansion in unique
  // alpha/beta matrix form `C^(n)`. The separable same-spin channels therefore
  // compress exactly to BLAS-3 style products
  //
  //   C^(n) B_beta [C^(n)]^T
  //   [C^(n)]^T A_alpha C^(n)
  //
  // rather than rebuilding full determinant-pair adjoints and scattering them
  // back to the same unique-spin spaces.
  Eigen::MatrixXd beta_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd alpha_image(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd beta_image(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double overlap_weight = -selected_state_energies[state_offset] * state_weight;

    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;

    // Reuse the same dense scratch buffers for every selected state so the hot
    // BLAS-3 contractions avoid repeated allocation churn.
    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.overlap_determinant_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();

    multiply_left_symmetric(
        alpha_scalar_matrices.overlap_determinant_matrix,
        coefficient_matrix_dense,
        &alpha_push);
    beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;

    weight_matrices.alpha_hamiltonian_weight_matrix.noalias() +=
        state_weight * alpha_image;
    weight_matrices.alpha_overlap_weight_matrix.noalias() +=
        overlap_weight * alpha_image;
    weight_matrices.beta_hamiltonian_weight_matrix.noalias() +=
        state_weight * beta_image;
    weight_matrices.beta_overlap_weight_matrix.noalias() +=
        overlap_weight * beta_image;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_partner_total_transfer_matrix.noalias() +=
        state_weight * alpha_image;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &beta_push);
    alpha_image.noalias() = beta_push * coefficient_matrix_dense.transpose();
    weight_matrices.alpha_singular_partner_transfer_matrix.noalias() +=
        state_weight * alpha_image;

    multiply_left_symmetric(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        coefficient_matrix_dense,
        &alpha_push);
    beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
    weight_matrices.beta_partner_total_transfer_matrix.noalias() +=
        state_weight * beta_image;

    multiply_left_symmetric(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        coefficient_matrix_dense,
        &alpha_push);
    beta_image.noalias() = coefficient_matrix_dense.transpose() * alpha_push;
    weight_matrices.beta_singular_partner_transfer_matrix.noalias() +=
        state_weight * beta_image;
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_support_sparse_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  SameSpinExactWeightMatrices weight_matrices;
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  const SameSpinPairScalarMatrices beta_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta);

  // Support-aware exact backward contraction:
  // each selected state contributes only on its trimmed unique-spin block
  // `C^(n)[A_n, B_n]`, so all same-spin adjoint images can be formed from the
  // corresponding local alpha/beta submatrices and scattered back afterwards.
  Eigen::MatrixXd alpha_overlap_subblock;
  Eigen::MatrixXd alpha_regular_total_subblock;
  Eigen::MatrixXd alpha_singular_total_subblock;
  Eigen::MatrixXd beta_overlap_subblock;
  Eigen::MatrixXd beta_regular_total_subblock;
  Eigen::MatrixXd beta_singular_total_subblock;
  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd alpha_push;
  Eigen::MatrixXd alpha_image;
  Eigen::MatrixXd beta_image;
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    if (!selected_state_has_local_support(state_coefficients)) {
      continue;
    }

    const double state_weight = state_coefficients.normalized_state_weight;
    const double overlap_weight =
        -selected_state_energies[state_offset] * state_weight;

    gather_dense_submatrix(
        alpha_scalar_matrices.overlap_determinant_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_overlap_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_regular_total_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        state_coefficients.alpha_support,
        state_coefficients.alpha_support,
        &alpha_singular_total_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.overlap_determinant_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_overlap_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.regular_total_hamiltonian_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_regular_total_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.singular_total_hamiltonian_matrix,
        state_coefficients.beta_support,
        state_coefficients.beta_support,
        &beta_singular_total_subblock);

    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_overlap_subblock,
        state_weight,
        &weight_matrices.alpha_hamiltonian_weight_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_overlap_subblock,
        overlap_weight,
        &weight_matrices.alpha_overlap_weight_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_overlap_subblock,
        state_weight,
        &weight_matrices.beta_hamiltonian_weight_matrix,
        &alpha_push,
        &beta_image);
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_overlap_subblock,
        overlap_weight,
        &weight_matrices.beta_overlap_weight_matrix,
        &alpha_push,
        &beta_image);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_regular_total_subblock,
        state_weight,
        &weight_matrices.alpha_partner_total_transfer_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_alpha_image(
        state_coefficients,
        beta_singular_total_subblock,
        state_weight,
        &weight_matrices.alpha_singular_partner_transfer_matrix,
        &beta_push,
        &alpha_image);
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_regular_total_subblock,
        state_weight,
        &weight_matrices.beta_partner_total_transfer_matrix,
        &alpha_push,
        &beta_image);
    accumulate_selected_state_beta_image(
        state_coefficients,
        alpha_singular_total_subblock,
        state_weight,
        &weight_matrices.beta_singular_partner_transfer_matrix,
        &alpha_push,
        &beta_image);
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (should_use_support_sparse_selected_state_contractions(selected_states)) {
    return build_support_sparse_exact_same_spin_weight_matrices(
        same_spin_pair_cache,
        selected_states,
        selected_state_energies);
  }
  return build_dense_exact_same_spin_weight_matrices(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);
}

void validate_directional_selected_state_inputs(
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  if (selected_states.n_unique_alpha != directional_selected_states.n_unique_alpha ||
      selected_states.n_unique_beta != directional_selected_states.n_unique_beta ||
      selected_states.n_determinants != directional_selected_states.n_determinants ||
      selected_states.selected_state_indices !=
          directional_selected_states.selected_state_indices ||
      selected_states.states.size() != directional_selected_states.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices do not match accepted-point dimensions");
  }
  if (selected_state_energies.size() != selected_states.states.size() ||
      directional_selected_state_energies.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "selected-state energy directions must align with selected_states.states");
  }
}

SameSpinExactWeightMatrices build_dense_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  SameSpinExactWeightMatrices weight_matrices;
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  const SameSpinPairScalarMatrices beta_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta);

  Eigen::MatrixXd coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_coefficient_matrix_dense(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_push(
      selected_states.n_unique_alpha,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_image_overlap(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd directional_image_overlap(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd base_image_transfer(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd directional_image_transfer(
      selected_states.n_unique_alpha,
      selected_states.n_unique_alpha);
  Eigen::MatrixXd base_beta_image_overlap(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_beta_image_overlap(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd base_beta_image_transfer(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);
  Eigen::MatrixXd directional_beta_image_transfer(
      selected_states.n_unique_beta,
      selected_states.n_unique_beta);

  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    validate_state_coefficient_matrix(
        state_coefficients,
        selected_states.n_unique_alpha,
        selected_states.n_unique_beta);
    validate_state_coefficient_matrix(
        directional_state_coefficients,
        directional_selected_states.n_unique_alpha,
        directional_selected_states.n_unique_beta);
    coefficient_matrix_dense = state_coefficients.coefficient_matrix;
    directional_coefficient_matrix_dense =
        directional_state_coefficients.coefficient_matrix;
    // The directional response is linear in both `delta C` and `delta E`.
    // Rescaling them here keeps the BLAS products inside a safe dynamic range
    // without changing the exact directional result.
    const double directional_scale =
        std::max(
            1.0,
            std::max(
                max_abs_dense_matrix(directional_coefficient_matrix_dense),
                std::abs(directional_state_energy)));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_dense /= directional_scale;
    }
    const double scaled_directional_state_energy =
        directional_state_energy / directional_scale;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.overlap_determinant_matrix,
        &base_push);
    base_image_overlap.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        beta_scalar_matrices.overlap_determinant_matrix,
        &directional_push);
    directional_image_overlap.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_overlap.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();

    weight_matrices.alpha_hamiltonian_weight_matrix.noalias() +=
        state_weight * directional_scale * directional_image_overlap;
    weight_matrices.alpha_overlap_weight_matrix.noalias() -=
        state_weight * directional_scale *
        (scaled_directional_state_energy * base_image_overlap +
         state_energy * directional_image_overlap);

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &base_push);
    base_image_transfer.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        beta_scalar_matrices.regular_total_hamiltonian_matrix,
        &directional_push);
    directional_image_transfer.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_transfer.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();
    weight_matrices.alpha_partner_total_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_image_transfer;

    multiply_right_symmetric(
        coefficient_matrix_dense,
        beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &base_push);
    base_image_transfer.noalias() =
        base_push * coefficient_matrix_dense.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_dense,
        beta_scalar_matrices.singular_total_hamiltonian_matrix,
        &directional_push);
    directional_image_transfer.noalias() =
        directional_push * coefficient_matrix_dense.transpose();
    directional_image_transfer.noalias() +=
        base_push * directional_coefficient_matrix_dense.transpose();
    weight_matrices.alpha_singular_partner_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_image_transfer;

    multiply_left_symmetric(
        alpha_scalar_matrices.overlap_determinant_matrix,
        coefficient_matrix_dense,
        &base_push);
    base_beta_image_overlap.noalias() =
        coefficient_matrix_dense.transpose() * base_push;
    multiply_left_symmetric(
        alpha_scalar_matrices.overlap_determinant_matrix,
        directional_coefficient_matrix_dense,
        &directional_push);
    directional_beta_image_overlap.noalias() =
        directional_coefficient_matrix_dense.transpose() * base_push;
    directional_beta_image_overlap.noalias() +=
        coefficient_matrix_dense.transpose() * directional_push;

    weight_matrices.beta_hamiltonian_weight_matrix.noalias() +=
        state_weight * directional_scale * directional_beta_image_overlap;
    weight_matrices.beta_overlap_weight_matrix.noalias() -=
        state_weight * directional_scale *
        (scaled_directional_state_energy * base_beta_image_overlap +
         state_energy * directional_beta_image_overlap);

    multiply_left_symmetric(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        coefficient_matrix_dense,
        &base_push);
    base_beta_image_transfer.noalias() =
        coefficient_matrix_dense.transpose() * base_push;
    multiply_left_symmetric(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        directional_coefficient_matrix_dense,
        &directional_push);
    directional_beta_image_transfer.noalias() =
        directional_coefficient_matrix_dense.transpose() * base_push;
    directional_beta_image_transfer.noalias() +=
        coefficient_matrix_dense.transpose() * directional_push;
    weight_matrices.beta_partner_total_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_beta_image_transfer;

    multiply_left_symmetric(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        coefficient_matrix_dense,
        &base_push);
    base_beta_image_transfer.noalias() =
        coefficient_matrix_dense.transpose() * base_push;
    multiply_left_symmetric(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        directional_coefficient_matrix_dense,
        &directional_push);
    directional_beta_image_transfer.noalias() =
        directional_coefficient_matrix_dense.transpose() * base_push;
    directional_beta_image_transfer.noalias() +=
        coefficient_matrix_dense.transpose() * directional_push;
    weight_matrices.beta_singular_partner_transfer_matrix.noalias() +=
        state_weight * directional_scale * directional_beta_image_transfer;
  }

  return weight_matrices;
}

SameSpinExactWeightMatrices build_support_sparse_directional_exact_same_spin_weight_matrices(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  SameSpinExactWeightMatrices weight_matrices;
  weight_matrices.alpha_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.alpha_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_alpha, selected_states.n_unique_alpha);
  weight_matrices.beta_hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_overlap_weight_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_partner_total_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);
  weight_matrices.beta_singular_partner_transfer_matrix =
      Eigen::MatrixXd::Zero(selected_states.n_unique_beta, selected_states.n_unique_beta);

  const SameSpinPairScalarMatrices alpha_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha);
  const SameSpinPairScalarMatrices beta_scalar_matrices =
      build_pair_scalar_matrices(
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta);

  Eigen::MatrixXd alpha_overlap_subblock;
  Eigen::MatrixXd alpha_regular_total_subblock;
  Eigen::MatrixXd alpha_singular_total_subblock;
  Eigen::MatrixXd beta_overlap_subblock;
  Eigen::MatrixXd beta_regular_total_subblock;
  Eigen::MatrixXd beta_singular_total_subblock;
  Eigen::MatrixXd coefficient_matrix_union;
  Eigen::MatrixXd directional_coefficient_matrix_union;
  Eigen::MatrixXd base_push;
  Eigen::MatrixXd directional_push;
  Eigen::MatrixXd alpha_base_image;
  Eigen::MatrixXd alpha_directional_image;
  Eigen::MatrixXd beta_base_image;
  Eigen::MatrixXd beta_directional_image;

  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_states.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_states.states[state_offset];
    const double state_weight = state_coefficients.normalized_state_weight;
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    const std::vector<int> alpha_union_support =
        build_merged_support_indices(
            state_coefficients.alpha_support,
            directional_state_coefficients.alpha_support);
    const std::vector<int> beta_union_support =
        build_merged_support_indices(
            state_coefficients.beta_support,
            directional_state_coefficients.beta_support);
    if (alpha_union_support.empty() || beta_union_support.empty()) {
      continue;
    }

    // The directional selected-state response can activate rows/columns that
    // are zero at the accepted point, so the sparse contraction must run on the
    // union support rather than only on the accepted-state block.
    gather_dense_submatrix(
        state_coefficients.coefficient_matrix,
        alpha_union_support,
        beta_union_support,
        &coefficient_matrix_union);
    gather_dense_submatrix(
        directional_state_coefficients.coefficient_matrix,
        alpha_union_support,
        beta_union_support,
        &directional_coefficient_matrix_union);

    const double directional_scale =
        std::max(
            1.0,
            std::max(
                max_abs_dense_matrix(directional_coefficient_matrix_union),
                std::abs(directional_state_energy)));
    if (directional_scale != 1.0) {
      directional_coefficient_matrix_union /= directional_scale;
    }
    const double scaled_directional_state_energy =
        directional_state_energy / directional_scale;

    gather_dense_submatrix(
        alpha_scalar_matrices.overlap_determinant_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_overlap_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.regular_total_hamiltonian_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_regular_total_subblock);
    gather_dense_submatrix(
        alpha_scalar_matrices.singular_total_hamiltonian_matrix,
        alpha_union_support,
        alpha_union_support,
        &alpha_singular_total_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.overlap_determinant_matrix,
        beta_union_support,
        beta_union_support,
        &beta_overlap_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.regular_total_hamiltonian_matrix,
        beta_union_support,
        beta_union_support,
        &beta_regular_total_subblock);
    gather_dense_submatrix(
        beta_scalar_matrices.singular_total_hamiltonian_matrix,
        beta_union_support,
        beta_union_support,
        &beta_singular_total_subblock);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_overlap_subblock,
        &base_push);
    alpha_base_image.noalias() =
        base_push * coefficient_matrix_union.transpose();
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_overlap_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();

    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_hamiltonian_weight_matrix);
    scatter_add_dense_submatrix(
        alpha_base_image,
        alpha_union_support,
        alpha_union_support,
        -state_weight * directional_scale * scaled_directional_state_energy,
        &weight_matrices.alpha_overlap_weight_matrix);
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        -state_weight * directional_scale * state_energy,
        &weight_matrices.alpha_overlap_weight_matrix);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_regular_total_subblock,
        &base_push);
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_regular_total_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_partner_total_transfer_matrix);

    multiply_right_symmetric(
        coefficient_matrix_union,
        beta_singular_total_subblock,
        &base_push);
    multiply_right_symmetric(
        directional_coefficient_matrix_union,
        beta_singular_total_subblock,
        &directional_push);
    alpha_directional_image.noalias() =
        directional_push * coefficient_matrix_union.transpose();
    alpha_directional_image.noalias() +=
        base_push * directional_coefficient_matrix_union.transpose();
    scatter_add_dense_submatrix(
        alpha_directional_image,
        alpha_union_support,
        alpha_union_support,
        state_weight * directional_scale,
        &weight_matrices.alpha_singular_partner_transfer_matrix);

    multiply_left_symmetric(
        alpha_overlap_subblock,
        coefficient_matrix_union,
        &base_push);
    beta_base_image.noalias() =
        coefficient_matrix_union.transpose() * base_push;
    multiply_left_symmetric(
        alpha_overlap_subblock,
        directional_coefficient_matrix_union,
        &directional_push);
    beta_directional_image.noalias() =
        directional_coefficient_matrix_union.transpose() * base_push;
    beta_directional_image.noalias() +=
        coefficient_matrix_union.transpose() * directional_push;

    scatter_add_dense_submatrix(
        beta_directional_image,
        beta_union_support,
        beta_union_support,
        state_weight * directional_scale,
        &weight_matrices.beta_hamiltonian_weight_matrix);
    scatter_add_dense_submatrix(
        beta_base_image,
        beta_union_support,
        beta_union_support,
        -state_weight * directional_scale * scaled_directional_state_energy,
        &weight_matrices.beta_overlap_weight_matrix);
    scatter_add_dense_submatrix(
        beta_directional_image,
        beta_union_support,
        beta_union_support,
        -state_weight * directional_scale * state_energy,
        &weight_matrices.beta_overlap_weight_matrix);

    multiply_left_symmetric(
        alpha_regular_total_subblock,
        coefficient_matrix_union,
        &base_push);
    multiply_left_symmetric(
        alpha_regular_total_subblock,
        directional_coefficient_matrix_union,
        &directional_push);
    beta_directional_image.noalias() =
        directional_coefficient_matrix_union.transpose() * base_push;
    beta_directional_image.noalias() +=
        coefficient_matrix_union.transpose() * directional_push;
    scatter_add_dense_submatrix(
        beta_directional_image,
        beta_union_support,
        beta_union_support,
        state_weight * directional_scale,
        &weight_matrices.beta_partner_total_transfer_matrix);

    multiply_left_symmetric(
        alpha_singular_total_subblock,
        coefficient_matrix_union,
        &base_push);
    multiply_left_symmetric(
        alpha_singular_total_subblock,
        directional_coefficient_matrix_union,
        &directional_push);
    beta_directional_image.noalias() =
        directional_coefficient_matrix_union.transpose() * base_push;
    beta_directional_image.noalias() +=
        coefficient_matrix_union.transpose() * directional_push;
    scatter_add_dense_submatrix(
        beta_directional_image,
        beta_union_support,
        beta_union_support,
        state_weight * directional_scale,
        &weight_matrices.beta_singular_partner_transfer_matrix);
  }

  return weight_matrices;
}

void validate_full_matrix_same_spin_inputs(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies) {
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "same-spin matrix backward requires an enabled same-spin cache");
  }
  if (selected_states.states.size() != selected_state_energies.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }
  if (static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()) !=
          selected_states.n_unique_alpha ||
      static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size()) !=
          selected_states.n_unique_beta) {
    throw std::invalid_argument(
        "same-spin cache dimensions do not match selected-state matrices");
  }
}

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const std::vector<double>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  // For the local-response we only need the partner-sector directional scalars
  // `δdet` and `δH_same`. These remain symmetric on the ordered unique-spin
  // space, so the same BLAS-3/sparse contraction used by the accepted-point
  // same-spin weights can compress them back to alpha/beta pair weights.
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        "ordered same-spin pair cache size does not match unique-spin dimensions");
  }

  SameSpinDirectionalScalarMatrices scalar_matrices;
  scalar_matrices.delta_overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalar_matrices.delta_regular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalar_matrices.delta_singular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);

  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const auto& pair_evaluation =
          ordered_pair_cache[ordered_spin_pair_storage_index(
              left_id,
              right_id,
              n_unique_determinants)];
      const auto& occ_L = xmvb::index_at(unique_determinants, left_id);
      const auto& occ_R = xmvb::index_at(unique_determinants, right_id);
      const auto& overlap_result = pair_evaluation.overlap_result;
      if (overlap_result.nullity == 0 &&
          overlap_result.overlap_determinant != 0.0) {
        const RegularSpinDirectionalData directional_data =
            build_regular_spin_directional_data(
                occ_L,
                occ_R,
                active_one_electron_matrix,
                active_space_two_electron_result,
                pair_evaluation,
                n_active_orbitals,
                delta_active_orbital_overlap_matrix,
                delta_active_one_electron_matrix,
                delta_packed_active_two_electron_integrals);
        set_symmetric_matrix_entry(
            &scalar_matrices.delta_overlap_determinant_matrix,
            left_id,
            right_id,
            directional_data.delta_overlap_determinant);
        set_symmetric_matrix_entry(
            &scalar_matrices.delta_regular_total_hamiltonian_matrix,
            left_id,
            right_id,
            directional_data.delta_overlap_determinant *
                    directional_data.same_spin_total_phi +
                directional_data.overlap_determinant *
                    directional_data.delta_same_spin_total_phi);
        continue;
      }

      const SingularSpinDirectionalData directional_data =
          build_singular_spin_directional_data(
              occ_L,
              occ_R,
              active_one_electron_matrix,
              active_space_two_electron_result,
              pair_evaluation,
              n_active_orbitals,
              delta_active_orbital_overlap_matrix,
              delta_active_one_electron_matrix,
              delta_packed_active_two_electron_integrals);
      set_symmetric_matrix_entry(
          &scalar_matrices.delta_overlap_determinant_matrix,
          left_id,
          right_id,
          directional_data.delta_overlap_determinant);
      set_symmetric_matrix_entry(
          &scalar_matrices.delta_singular_total_hamiltonian_matrix,
          left_id,
          right_id,
          directional_data.delta_total_hamiltonian);
    }
  }

  return scalar_matrices;
}

SameSpinLocalResponseWeightMatrices build_local_same_spin_response_weight_matrices(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    const SameSpinDirectionalScalarMatrices& alpha_directional_scalars,
    const SameSpinDirectionalScalarMatrices& beta_directional_scalars) {
  // The accepted selected-state coefficients stay fixed. Only the partner
  // kernels change, so the same unique-spin coefficient matrices `C^(n)` map
  // `δdet_partner` and `δH_same,partner` back to the active same-spin weights.
  if (selected_state_energies.size() != selected_states.states.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_states.states");
  }

  std::vector<double> hamiltonian_scales;
  std::vector<double> overlap_scales;
  hamiltonian_scales.reserve(selected_states.states.size());
  overlap_scales.reserve(selected_states.states.size());
  for (std::size_t state_offset = 0;
       state_offset < selected_states.states.size();
       ++state_offset) {
    const double state_weight =
        selected_states.states[state_offset].normalized_state_weight;
    hamiltonian_scales.push_back(state_weight);
    overlap_scales.push_back(-selected_state_energies[state_offset] * state_weight);
  }

  SameSpinLocalResponseWeightMatrices weight_matrices;
  const SingleChannelSameSpinWeightMatrices delta_hamiltonian_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          hamiltonian_scales);
  weight_matrices.alpha_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_hamiltonian_weight_matrix =
      std::move(delta_hamiltonian_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_overlap_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_overlap_determinant_matrix,
          beta_directional_scalars.delta_overlap_determinant_matrix,
          overlap_scales);
  weight_matrices.alpha_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_overlap_weight_matrix =
      std::move(delta_overlap_weights.beta_weight_matrix);

  const SingleChannelSameSpinWeightMatrices delta_partner_weights =
      build_same_spin_weight_matrices_from_partner_kernels(
          selected_states,
          alpha_directional_scalars.delta_regular_total_hamiltonian_matrix +
              alpha_directional_scalars.delta_singular_total_hamiltonian_matrix,
          beta_directional_scalars.delta_regular_total_hamiltonian_matrix +
              beta_directional_scalars.delta_singular_total_hamiltonian_matrix,
          hamiltonian_scales);
  weight_matrices.alpha_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.alpha_weight_matrix);
  weight_matrices.beta_delta_partner_total_transfer_matrix =
      std::move(delta_partner_weights.beta_weight_matrix);
  return weight_matrices;
}

bool same_spin_local_tile_has_any_weight(
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end) {
  for (int left_id = left_begin; left_id < left_end; ++left_id) {
    for (int right_id = right_begin; right_id < right_end; ++right_id) {
      if (std::abs(hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(partner_total_transfer_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(delta_partner_total_transfer_matrix(left_id, right_id)) >
              kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

bool same_spin_tile_has_any_weight(
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end) {
  for (int left_id = left_begin; left_id < left_end; ++left_id) {
    for (int right_id = right_begin; right_id < right_end; ++right_id) {
      if (std::abs(hamiltonian_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(overlap_weight_matrix(left_id, right_id)) > kContributionTolerance ||
          std::abs(partner_total_transfer_matrix(left_id, right_id)) > kContributionTolerance) {
        return true;
      }
    }
  }
  return false;
}

void accumulate_spin_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("same-spin matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  Eigen::MatrixXd scaled_inverse_overlap_gradient;
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      // Tile the ordered unique-pair sweep so sparse/support-aware same-spin
      // weights can skip large zero regions without touching every cached pair.
      if (!same_spin_tile_has_any_weight(
              hamiltonian_weight_matrix,
              overlap_weight_matrix,
              partner_total_transfer_matrix,
              left_begin,
              left_end,
              right_begin,
              right_end)) {
        continue;
      }

      for (int left_id = left_begin; left_id < left_end; ++left_id) {
        for (int right_id = right_begin; right_id < right_end; ++right_id) {
          const double hamiltonian_weight = hamiltonian_weight_matrix(left_id, right_id);
          const double overlap_weight = overlap_weight_matrix(left_id, right_id);
          const double partner_total = partner_total_transfer_matrix(left_id, right_id);
          if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
              std::abs(overlap_weight) <= kContributionTolerance &&
              std::abs(partner_total) <= kContributionTolerance) {
            continue;
          }

          const auto& pair_evaluation =
              ordered_pair_cache[ordered_spin_pair_storage_index(
                  left_id,
                  right_id,
                  n_unique_determinants)];
          const auto& occ_L = xmvb::index_at(unique_determinants, left_id);
          const auto& occ_R = xmvb::index_at(unique_determinants, right_id);
          const auto& overlap_result = pair_evaluation.overlap_result;
          const bool has_hamiltonian_weight =
              std::abs(hamiltonian_weight) > kContributionTolerance;
          const bool is_regular_pair =
              overlap_result.nullity == 0 &&
              overlap_result.overlap_determinant != 0.0;

          if (is_regular_pair) {
            if (!pair_evaluation.has_same_spin_phi_cache) {
              throw std::runtime_error(
                  "matrix-form same-spin backward requires cached same-spin phi payloads");
            }

            if (has_hamiltonian_weight) {
              const Eigen::MatrixXd cofactor_1st =
                  calc_cofactor_1st(overlap_result);
              accumulate_one_electron_gradient_contribution_local(
                  occ_L,
                  occ_R,
                  cofactor_1st,
                  hamiltonian_weight,
                  active_one_electron_gradient);
              accumulate_same_spin_two_electron_gradient_contribution_local(
                  occ_L,
                  occ_R,
                  cofactor_1st,
                  overlap_result.overlap_determinant,
                  hamiltonian_weight,
                  packed_active_two_electron_gradient);
              scaled_inverse_overlap_gradient =
                  hamiltonian_weight * pair_evaluation.same_spin_inverse_overlap_gradient;
            } else {
              const int n_electrons = static_cast<int>(occ_L.size());
              scaled_inverse_overlap_gradient.setZero(n_electrons, n_electrons);
            }

            const double determinant_overlap_weight =
                overlap_weight +
                partner_total +
                (has_hamiltonian_weight
                     ? hamiltonian_weight * pair_evaluation.same_spin_total_phi
                     : 0.0);
            if (!has_hamiltonian_weight &&
                std::abs(determinant_overlap_weight) <= kContributionTolerance) {
              continue;
            }
            accumulate_spin_overlap_gradient(
                occ_L,
                occ_R,
                overlap_result,
                determinant_overlap_weight,
                scaled_inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
            continue;
          }

          const double determinant_overlap_weight =
              overlap_weight + partner_total;
          const bool need_first_order_cofactor =
              has_hamiltonian_weight ||
              std::abs(determinant_overlap_weight) > kContributionTolerance;
          Eigen::MatrixXd cofactor_1st;
          if (need_first_order_cofactor) {
            cofactor_1st = calc_cofactor_1st(overlap_result);
          }

          if (has_hamiltonian_weight) {
            if (cofactor_1st.size() != 0) {
              accumulate_one_electron_gradient_contribution_local(
                  occ_L,
                  occ_R,
                  cofactor_1st,
                  hamiltonian_weight,
                  active_one_electron_gradient);
            }
            accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
                occ_L,
                occ_R,
                overlap_result,
                hamiltonian_weight,
                packed_active_two_electron_gradient);
            if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                    static_cast<int>(occ_R.size()) ||
                pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                    static_cast<int>(occ_L.size())) {
              throw std::runtime_error(
                  "matrix-form same-spin backward requires cached singular overlap Hamiltonian gradients");
            }
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                pair_evaluation.same_spin_overlap_hamiltonian_gradient,
                hamiltonian_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }

          if (std::abs(determinant_overlap_weight) <= kContributionTolerance ||
              cofactor_1st.size() == 0) {
            continue;
          }
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              cofactor_1st,
              determinant_overlap_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);
        }
      }
    }
  }
}

void accumulate_spin_local_matrix_backward(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    const Eigen::MatrixXd& hamiltonian_weight_matrix,
    const Eigen::MatrixXd& overlap_weight_matrix,
    const Eigen::MatrixXd& partner_total_transfer_matrix,
    const Eigen::MatrixXd& delta_hamiltonian_weight_matrix,
    const Eigen::MatrixXd& delta_overlap_weight_matrix,
    const Eigen::MatrixXd& delta_partner_total_transfer_matrix,
    int n_unique_determinants,
    int n_active_orbitals,
    const std::vector<double>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // The compressed same-spin local weights are symmetric on the unique-spin
  // spaces, but the local-response kernels still depend on the ordered
  // left/right role of each unique-determinant pair through the directional
  // cofactors and overlap-gradient adjoints. Reusing the accepted-state
  // half-pair shortcut here corrupts the SSO/HHO local response. Sweep the
  // full ordered pair grid just like the validated accepted/directional
  // same-spin backward path.
  if (active_one_electron_gradient == nullptr ||
      active_orbital_overlap_gradient == nullptr ||
      packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument(
        "same-spin local matrix backward output buffers must not be null");
  }

  const int pair_tile_size = std::min(
      n_unique_determinants,
      same_spin_backward_pair_tile_size());
  for (int left_begin = 0; left_begin < n_unique_determinants; left_begin += pair_tile_size) {
    const int left_end =
        std::min(n_unique_determinants, left_begin + pair_tile_size);
    for (int right_begin = 0;
         right_begin < n_unique_determinants;
         right_begin += pair_tile_size) {
      const int right_end =
          std::min(n_unique_determinants, right_begin + pair_tile_size);
      if (!same_spin_local_tile_has_any_weight(
              hamiltonian_weight_matrix,
              overlap_weight_matrix,
              partner_total_transfer_matrix,
              delta_hamiltonian_weight_matrix,
              delta_overlap_weight_matrix,
              delta_partner_total_transfer_matrix,
              left_begin,
              left_end,
              right_begin,
              right_end)) {
        continue;
      }

      for (int left_id = left_begin; left_id < left_end; ++left_id) {
        for (int right_id = right_begin; right_id < right_end; ++right_id) {
          const double hamiltonian_weight =
              hamiltonian_weight_matrix(left_id, right_id);
          const double overlap_weight =
              overlap_weight_matrix(left_id, right_id);
          const double partner_total =
              partner_total_transfer_matrix(left_id, right_id);
          const double delta_hamiltonian_weight =
              delta_hamiltonian_weight_matrix(left_id, right_id);
          const double delta_overlap_weight =
              delta_overlap_weight_matrix(left_id, right_id);
          const double delta_partner_total =
              delta_partner_total_transfer_matrix(left_id, right_id);
          if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
              std::abs(overlap_weight) <= kContributionTolerance &&
              std::abs(partner_total) <= kContributionTolerance &&
              std::abs(delta_hamiltonian_weight) <= kContributionTolerance &&
              std::abs(delta_overlap_weight) <= kContributionTolerance &&
              std::abs(delta_partner_total) <= kContributionTolerance) {
            continue;
          }

          const auto& pair_evaluation =
              ordered_pair_cache[ordered_spin_pair_storage_index(
                  left_id,
                  right_id,
                  n_unique_determinants)];
          const auto& occ_L = xmvb::index_at(unique_determinants, left_id);
          const auto& occ_R = xmvb::index_at(unique_determinants, right_id);
          const auto& overlap_result = pair_evaluation.overlap_result;
          const bool is_regular_pair =
              overlap_result.nullity == 0 &&
              overlap_result.overlap_determinant != 0.0;
          if (is_regular_pair) {
            if (!pair_evaluation.has_same_spin_phi_cache) {
              throw std::runtime_error(
                  "same-spin local-response requires cached same-spin phi payloads");
            }

            const RegularSpinDirectionalData directional_data =
                build_regular_spin_directional_data(
                    occ_L,
                    occ_R,
                    active_one_electron_matrix,
                    active_space_two_electron_result,
                    pair_evaluation,
                    n_active_orbitals,
                    delta_active_orbital_overlap_matrix,
                    delta_active_one_electron_matrix,
                    delta_packed_active_two_electron_integrals);

            accumulate_directional_one_electron_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.cofactor_1st,
                directional_data.delta_cofactor_1st,
                hamiltonian_weight,
                delta_hamiltonian_weight,
                active_one_electron_gradient);
            accumulate_directional_same_spin_two_electron_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.cofactor_1st,
                directional_data.delta_cofactor_1st,
                directional_data.overlap_determinant,
                directional_data.delta_overlap_determinant,
                hamiltonian_weight,
                delta_hamiltonian_weight,
                packed_active_two_electron_gradient);

            const double determinant_overlap_weight =
                overlap_weight +
                partner_total +
                hamiltonian_weight * directional_data.same_spin_total_phi;
            const double delta_determinant_overlap_weight =
                delta_overlap_weight +
                delta_partner_total +
                delta_hamiltonian_weight * directional_data.same_spin_total_phi +
                hamiltonian_weight * directional_data.delta_same_spin_total_phi;
            const Eigen::MatrixXd inverse_overlap_gradient =
                hamiltonian_weight *
                directional_data.same_spin_inverse_overlap_gradient;
            const Eigen::MatrixXd delta_inverse_overlap_gradient =
                delta_hamiltonian_weight *
                    directional_data.same_spin_inverse_overlap_gradient +
                hamiltonian_weight *
                    directional_data.delta_same_spin_inverse_overlap_gradient;
            accumulate_regular_spin_overlap_gradient_direction_local(
                occ_L,
                occ_R,
                directional_data.overlap_determinant,
                directional_data.delta_overlap_determinant,
                directional_data.inverse_overlap_submatrix,
                directional_data.delta_inverse_overlap_submatrix,
                determinant_overlap_weight,
                delta_determinant_overlap_weight,
                inverse_overlap_gradient,
                delta_inverse_overlap_gradient,
                n_active_orbitals,
                active_orbital_overlap_gradient);
            continue;
          }

          const SingularSpinDirectionalData directional_data =
              build_singular_spin_directional_data(
                  occ_L,
                  occ_R,
                  active_one_electron_matrix,
                  active_space_two_electron_result,
                  pair_evaluation,
                  n_active_orbitals,
                  delta_active_orbital_overlap_matrix,
                  delta_active_one_electron_matrix,
                  delta_packed_active_two_electron_integrals);

          accumulate_directional_one_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.cofactor_1st,
              directional_data.delta_cofactor_1st,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              active_one_electron_gradient);
          accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
              occ_L,
              occ_R,
              overlap_result,
              delta_active_orbital_overlap_matrix,
              n_active_orbitals,
              hamiltonian_weight,
              delta_hamiltonian_weight,
              packed_active_two_electron_gradient);

          if (pair_evaluation.same_spin_overlap_hamiltonian_gradient.rows() !=
                  static_cast<int>(occ_R.size()) ||
              pair_evaluation.same_spin_overlap_hamiltonian_gradient.cols() !=
                  static_cast<int>(occ_L.size())) {
            throw std::runtime_error(
                "same-spin local-response requires cached singular overlap Hamiltonian gradients");
          }
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              pair_evaluation.same_spin_overlap_hamiltonian_gradient,
              delta_hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);
          accumulate_overlap_block_gradient_contribution_local(
              occ_L,
              occ_R,
              directional_data.delta_same_spin_overlap_hamiltonian_gradient,
              hamiltonian_weight,
              n_active_orbitals,
              active_orbital_overlap_gradient);

          const double determinant_overlap_weight =
              overlap_weight + partner_total;
          const double delta_determinant_overlap_weight =
              delta_overlap_weight + delta_partner_total;
          if (directional_data.cofactor_1st.size() != 0) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.cofactor_1st,
                delta_determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
          if (directional_data.delta_cofactor_1st.size() != 0 &&
              std::abs(determinant_overlap_weight) > kContributionTolerance) {
            accumulate_overlap_block_gradient_contribution_local(
                occ_L,
                occ_R,
                directional_data.delta_cofactor_1st,
                determinant_overlap_weight,
                n_active_orbitals,
                active_orbital_overlap_gradient);
          }
        }
      }
    }
  }
}

}  // namespace

SameSpinMatrixBackwardContribution build_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  // Compress the selected-state determinant coefficients directly onto the
  // unique alpha/beta same-spin channels. This keeps the backward algebra exact
  // while replacing the previous full determinant-pair scatter with dense
  // matrix products on the unique-spin spaces.
  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;
  const Eigen::MatrixXd beta_total_partner_transfer_matrix =
      exact_weight_matrices.beta_partner_total_transfer_matrix +
      exact_weight_matrices.beta_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      exact_weight_matrices.beta_hamiltonian_weight_matrix,
      exact_weight_matrices.beta_overlap_weight_matrix,
      beta_total_partner_transfer_matrix,
      selected_states.n_unique_beta,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_directional_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies,
    int n_active_orbitals) {
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);
  validate_directional_selected_state_inputs(
      selected_states,
      directional_selected_states,
      selected_state_energies,
      directional_selected_state_energies);

  const SameSpinExactWeightMatrices directional_weight_matrices =
      should_use_support_sparse_selected_state_contractions(selected_states)
          ? build_support_sparse_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies)
          : build_dense_directional_exact_same_spin_weight_matrices(
                same_spin_pair_cache,
                selected_states,
                directional_selected_states,
                selected_state_energies,
                directional_selected_state_energies);
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      directional_weight_matrices.alpha_partner_total_transfer_matrix +
      directional_weight_matrices.alpha_singular_partner_transfer_matrix;
  const Eigen::MatrixXd beta_total_partner_transfer_matrix =
      directional_weight_matrices.beta_partner_total_transfer_matrix +
      directional_weight_matrices.beta_singular_partner_transfer_matrix;

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      directional_weight_matrices.alpha_hamiltonian_weight_matrix,
      directional_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  accumulate_spin_matrix_backward(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      directional_weight_matrices.beta_hamiltonian_weight_matrix,
      directional_weight_matrices.beta_overlap_weight_matrix,
      beta_total_partner_transfer_matrix,
      selected_states.n_unique_beta,
      n_active_orbitals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

SameSpinMatrixBackwardContribution
build_local_same_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const std::vector<double>& active_one_electron_matrix,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  // Matrix-form local same-spin HVP:
  // 1. compress accepted determinant/structure adjoints onto unique-spin pair weights,
  // 2. compress partner directional scalars onto `δW`,
  // 3. sweep only the ordered unique-spin pairs to accumulate the exact local
  //    response on `(δS_act, δh_act, δg_act)`.
  validate_full_matrix_same_spin_inputs(
      same_spin_pair_cache,
      selected_states,
      selected_state_energies);

  const SameSpinExactWeightMatrices exact_weight_matrices =
      build_exact_same_spin_weight_matrices(
          same_spin_pair_cache,
          selected_states,
          selected_state_energies);
  const Eigen::MatrixXd alpha_total_partner_transfer_matrix =
      exact_weight_matrices.alpha_partner_total_transfer_matrix +
      exact_weight_matrices.alpha_singular_partner_transfer_matrix;
  const Eigen::MatrixXd beta_total_partner_transfer_matrix =
      exact_weight_matrices.beta_partner_total_transfer_matrix +
      exact_weight_matrices.beta_singular_partner_transfer_matrix;

  const SameSpinDirectionalScalarMatrices alpha_directional_scalars =
      build_directional_pair_scalar_matrices(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.alpha_pair_cache_ref(),
          selected_states.n_unique_alpha,
          n_active_orbitals,
          active_one_electron_matrix,
          active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const SameSpinDirectionalScalarMatrices beta_directional_scalars =
      build_directional_pair_scalar_matrices(
          same_spin_pair_cache.beta_reuse_table.unique_determinants,
          same_spin_pair_cache.beta_pair_cache_ref(),
          selected_states.n_unique_beta,
          n_active_orbitals,
          active_one_electron_matrix,
          active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const SameSpinLocalResponseWeightMatrices local_weight_matrices =
      build_local_same_spin_response_weight_matrices(
          selected_states,
          selected_state_energies,
          alpha_directional_scalars,
          beta_directional_scalars);

  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      square_storage_size(n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_spin_local_matrix_backward(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      exact_weight_matrices.alpha_hamiltonian_weight_matrix,
      exact_weight_matrices.alpha_overlap_weight_matrix,
      alpha_total_partner_transfer_matrix,
      local_weight_matrices.alpha_delta_hamiltonian_weight_matrix,
      local_weight_matrices.alpha_delta_overlap_weight_matrix,
      local_weight_matrices.alpha_delta_partner_total_transfer_matrix,
      selected_states.n_unique_alpha,
      n_active_orbitals,
      active_one_electron_matrix,
      active_space_two_electron_result,
      delta_active_orbital_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);
  accumulate_spin_local_matrix_backward(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      same_spin_pair_cache.beta_pair_cache_ref(),
      exact_weight_matrices.beta_hamiltonian_weight_matrix,
      exact_weight_matrices.beta_overlap_weight_matrix,
      beta_total_partner_transfer_matrix,
      local_weight_matrices.beta_delta_hamiltonian_weight_matrix,
      local_weight_matrices.beta_delta_overlap_weight_matrix,
      local_weight_matrices.beta_delta_partner_total_transfer_matrix,
      selected_states.n_unique_beta,
      n_active_orbitals,
      active_one_electron_matrix,
      active_space_two_electron_result,
      delta_active_orbital_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals,
      &active_one_electron_gradient,
      &result.active_orbital_overlap_gradient,
      &result.packed_active_two_electron_gradient);

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

}  // namespace xmvb::vb
