#include "vb/matrices/spin_pair_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {

double product_of_leading_singular_values(
    const DeterminantOverlapResult& det_ovlp_result,
    int count) {
  double product = 1.0;
  for (int singular_index = 0; singular_index < count; ++singular_index) {
    product *= det_ovlp_result.singular_values(singular_index);
  }
  return product;
}

Eigen::MatrixXd build_spin_one_electron_block_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occ_R[xmvb::to_size(right_row)];
      one_electron_block(right_row, left_column) =
          h1e_act[xmvb::to_size(orbital_index_left) *
                                         n_active_orbitals +
                                     orbital_index_right];
    }
  }

  return one_electron_block;
}

bool has_cached_inverse_overlap_submatrix(
    const DeterminantOverlapResult& det_ovlp_result) {
  return det_ovlp_result.inverse_overlap_submatrix.rows() == det_ovlp_result.n_electrons &&
      det_ovlp_result.inverse_overlap_submatrix.cols() == det_ovlp_result.n_electrons;
}

bool has_cached_first_order_cofactor_matrix(
    const DeterminantOverlapResult& det_ovlp_result) {
  return det_ovlp_result.first_order_cofactor_matrix.rows() ==
          det_ovlp_result.n_electrons &&
      det_ovlp_result.first_order_cofactor_matrix.cols() ==
          det_ovlp_result.n_electrons;
}

void require_svd_overlap_result(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.singular_values.size() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.cols() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.cols() != det_ovlp_result.n_electrons) {
    throw std::invalid_argument("det_ovlp_result SVD dimensions are inconsistent");
  }
}

bool has_sparse_packed_pair_projection(
    const OppositeSpinPackedPairProjection& projection) {
  return projection.packed_pair_indices.size() == projection.packed_pair_values.size() &&
      !projection.packed_pair_indices.empty();
}

bool has_dense_projected_pair_values(
    const OppositeSpinPackedPairProjection& projection,
    int n_packed_active_pairs) {
  return static_cast<int>(projection.projected_pair_values.size()) == n_packed_active_pairs;
}

int deleted_index_sum(const std::vector<int>& deleted_indices) {
  int sum = 0;
  for (const int index : deleted_indices) {
    sum += index;
  }
  return sum;
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

double contract_sparse_projection_with_dense_image(
    const OppositeSpinPackedPairProjection& sparse_projection,
    const std::vector<double>& dense_projected_values) {
  double contraction = 0.0;
  for (std::size_t entry_index = 0;
       entry_index < sparse_projection.packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index =
        sparse_projection.packed_pair_indices[entry_index];
    contraction +=
        sparse_projection.packed_pair_values[entry_index] *
        dense_projected_values[xmvb::to_size(packed_pair_index)];
  }
  return contraction;
}

void build_inverse_overlap_gradient_from_projected_pair_values(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& projected_pair_values,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (inverse_overlap_gradient == nullptr) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient->setZero(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occ_R[xmvb::to_size(right_row)];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      (*inverse_overlap_gradient)(left_column, right_row) =
          projected_pair_values[xmvb::to_size(packed_pair_index)];
    }
  }
}

Eigen::MatrixXd build_first_order_cofactor_matrix(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.n_electrons < 0) {
    throw std::invalid_argument("det_ovlp_result.n_electrons must be non-negative");
  }

  Eigen::MatrixXd cofactor_matrix =
      Eigen::MatrixXd::Zero(det_ovlp_result.n_electrons, det_ovlp_result.n_electrons);
  if (det_ovlp_result.n_electrons == 0) {
    return cofactor_matrix;
  }

  if (det_ovlp_result.nullity == 0) {
    if (has_cached_inverse_overlap_submatrix(det_ovlp_result)) {
      cofactor_matrix.noalias() =
          det_ovlp_result.overlap_determinant *
          det_ovlp_result.inverse_overlap_submatrix.transpose();
      return cofactor_matrix;
    }

    require_svd_overlap_result(det_ovlp_result);
    const Eigen::VectorXd inverse_singular_values =
        det_ovlp_result.singular_values.cwiseInverse();
    cofactor_matrix.noalias() =
        det_ovlp_result.overlap_determinant * det_ovlp_result.matrix_U *
        inverse_singular_values.asDiagonal() * det_ovlp_result.matrix_V.transpose();
    return cofactor_matrix;
  }

  if (det_ovlp_result.nullity == 1) {
    require_svd_overlap_result(det_ovlp_result);
    const int null_index = det_ovlp_result.n_electrons - 1;
    const double prefactor =
        det_ovlp_result.parity *
        product_of_leading_singular_values(det_ovlp_result, det_ovlp_result.n_electrons - 1);
    cofactor_matrix.noalias() =
        prefactor *
        (det_ovlp_result.matrix_U.col(null_index) *
         det_ovlp_result.matrix_V.col(null_index).transpose());
  }

  return cofactor_matrix;
}

template <typename InteractionLookup>
SameSpinPhiResult compute_same_spin_original_phi_impl(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const DeterminantOverlapResult& det_ovlp_result,
    Eigen::MatrixXd* inverse_overlap_gradient,
    InteractionLookup&& lookup_interaction) {
  if (det_ovlp_result.nullity != 0 || det_ovlp_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("original phi requires a non-singular determinant pair");
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }

  const Eigen::MatrixXd inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(det_ovlp_result);

  const Eigen::MatrixXd one_electron_block = build_spin_one_electron_block_matrix(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals);

  SameSpinPhiResult result;
  result.one_electron_phi =
      (one_electron_block.cwiseProduct(inverse_overlap_submatrix.transpose())).sum();
  result.total_phi = result.one_electron_phi;

  if (inverse_overlap_gradient != nullptr) {
    inverse_overlap_gradient->setZero(n_electrons, n_electrons);
    (*inverse_overlap_gradient) += one_electron_block.transpose();
  }

  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first =
        occ_L[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first =
          occ_R[xmvb::to_size(right_first)];
      const int direct_left_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right_first,
          orbital_index_left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second =
            occ_L[xmvb::to_size(left_second)];
        const int exchange_left_pair_index = TwoElectronIndexer::packed_pair_index(
            orbital_index_right_first,
            orbital_index_left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second =
              occ_R[xmvb::to_size(right_second)];
          const int direct_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_first);
          const double interaction_value =
              lookup_interaction(direct_left_pair_index, direct_right_pair_index) -
              lookup_interaction(exchange_left_pair_index, exchange_right_pair_index);
          const double x11 = inverse_overlap_submatrix(left_first, right_first);
          const double x22 = inverse_overlap_submatrix(left_second, right_second);
          const double x12 = inverse_overlap_submatrix(left_second, right_first);
          const double x21 = inverse_overlap_submatrix(left_first, right_second);
          result.total_phi += interaction_value * (x11 * x22 - x12 * x21);

          if (inverse_overlap_gradient != nullptr) {
            (*inverse_overlap_gradient)(left_first, right_first) += interaction_value * x22;
            (*inverse_overlap_gradient)(left_second, right_second) += interaction_value * x11;
            (*inverse_overlap_gradient)(left_second, right_first) -= interaction_value * x21;
            (*inverse_overlap_gradient)(left_first, right_second) -= interaction_value * x12;
          }
        }
      }
    }
  }

  return result;
}

template <typename InteractionLookup>
double compute_opposite_spin_original_phi_impl(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient,
    InteractionLookup&& lookup_interaction) {
  const Eigen::MatrixXd alpha_inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(alpha_overlap_result);
  const Eigen::MatrixXd beta_inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(beta_overlap_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  double phi = 0.0;

  if (alpha_inverse_overlap_gradient != nullptr) {
    alpha_inverse_overlap_gradient->setZero(n_alpha_electrons, n_alpha_electrons);
  }
  if (beta_inverse_overlap_gradient != nullptr) {
    beta_inverse_overlap_gradient->setZero(n_beta_electrons, n_beta_electrons);
  }

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[xmvb::to_size(alpha_right_row)];
      const double alpha_inverse_value =
          alpha_inverse_overlap_submatrix(alpha_left_column, alpha_right_row);
      const int alpha_packed_pair_index = TwoElectronIndexer::packed_pair_index(
          alpha_orbital_right,
          alpha_orbital_left);
      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[xmvb::to_size(beta_right_row)];
          const double beta_inverse_value =
              beta_inverse_overlap_submatrix(beta_left_column, beta_right_row);
          const int beta_packed_pair_index = TwoElectronIndexer::packed_pair_index(
              beta_orbital_right,
              beta_orbital_left);
          const double interaction_value =
              lookup_interaction(beta_packed_pair_index, alpha_packed_pair_index);
          phi += interaction_value * alpha_inverse_value * beta_inverse_value;
          if (alpha_inverse_overlap_gradient != nullptr) {
            (*alpha_inverse_overlap_gradient)(alpha_left_column, alpha_right_row) +=
                interaction_value * beta_inverse_value;
          }
          if (beta_inverse_overlap_gradient != nullptr) {
            (*beta_inverse_overlap_gradient)(beta_left_column, beta_right_row) +=
                interaction_value * alpha_inverse_value;
          }
        }
      }
    }
  }

  return phi;
}

}  // namespace


std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons) {
    throw std::invalid_argument("left and right occupation sizes must match");
  }

  std::vector<double> overlap_submatrix(
      xmvb::to_size(n_electrons) * xmvb::to_size(n_electrons),
      0.0);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[xmvb::to_size(right_row)];
      overlap_submatrix[xmvb::to_size(left_column) * n_electrons + right_row] =
          ovlp_act[xmvb::to_size(orbital_index_left) * n_orbitals +
                   orbital_index_right];
    }
  }

  return overlap_submatrix;
}


Eigen::MatrixXd calc_cofactor_1st(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (has_cached_first_order_cofactor_matrix(det_ovlp_result)) {
    return det_ovlp_result.first_order_cofactor_matrix;
  }
  return build_first_order_cofactor_matrix(det_ovlp_result);
}

bool projection_has_any_payload(
    const OppositeSpinPackedPairProjection& projection) {
  return !projection.packed_pair_indices.empty() ||
      !projection.packed_pair_values.empty() ||
      !projection.projected_pair_values.empty();
}

int infer_n_packed_active_pairs(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    const char* spin_label) {
  int n_packed_active_pairs = 0;
  bool initialized = false;
  for (const auto& pair_evaluation : ordered_spin_pair_cache) {
    const int pair_n_packed =
        pair_evaluation.opposite_spin_pair_cache.n_packed_active_pairs;
    if (pair_n_packed <= 0) {
      continue;
    }
    if (!initialized) {
      n_packed_active_pairs = pair_n_packed;
      initialized = true;
      continue;
    }
    if (pair_n_packed != n_packed_active_pairs) {
      throw std::invalid_argument(
          std::string("inconsistent n_packed_active_pairs across ordered ") +
          spin_label + " pair cache entries");
    }
  }
  return n_packed_active_pairs;
}

void validate_sparse_projection_coefficients(
    const OppositeSpinPackedPairProjection& projection,
    int n_packed_active_pairs,
    const char* projection_label) {
  if (projection.packed_pair_indices.size() != projection.packed_pair_values.size()) {
    throw std::invalid_argument(
        std::string(projection_label) +
        " has mismatched packed_pair_indices / packed_pair_values size");
  }
  for (const int packed_pair_index : projection.packed_pair_indices) {
    if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
      throw std::out_of_range(
          std::string(projection_label) +
          " contains packed_pair_index outside [0, n_packed_active_pairs)");
    }
  }
}

void gather_dense_submatrix(
    const Eigen::MatrixXd& global_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* local_matrix) {
  if (local_matrix == nullptr) {
    throw std::invalid_argument("local_matrix must not be null");
  }
  local_matrix->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[xmvb::to_size(column_local)];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[xmvb::to_size(row_local)];
      (*local_matrix)(row_local, column_local) =
          global_matrix(row_global, column_global);
    }
  }
}

int limited_packed_pair_block_size(
    int n_packed_active_pairs,
    int requested_block_size,
    std::size_t per_matrix_bytes,
    std::size_t bytes_budget) {
  if (n_packed_active_pairs <= 0) {
    return 0;
  }
  int block_size = std::min(n_packed_active_pairs, requested_block_size);
  if (per_matrix_bytes > 0) {
    const std::size_t budget_limited_block_size = std::max<std::size_t>(
        1u,
        bytes_budget / per_matrix_bytes);
    block_size = std::min(
        block_size,
        static_cast<int>(std::min<std::size_t>(
            xmvb::to_size(n_packed_active_pairs),
            budget_limited_block_size)));
  }
  return block_size;
}

void cache_first_order_cofactor(
    DeterminantOverlapResult* det_ovlp_result) {
  if (det_ovlp_result == nullptr) {
    throw std::invalid_argument("det_ovlp_result must not be null");
  }
  if (has_cached_first_order_cofactor_matrix(*det_ovlp_result) ||
      det_ovlp_result->nullity >= 2) {
    return;
  }

  // Determinant-pair preparation is the right place to materialize the
  // deleted-minor matrix once: the same object is then reused by forward
  // same-spin Hamiltonians, opposite-spin combination, and backward adjoints.
  det_ovlp_result->first_order_cofactor_matrix =
      build_first_order_cofactor_matrix(*det_ovlp_result);
}

bool has_opposite_spin_first_order_projection(
    const OppositeSpinPairCache& pair_cache) {
  return pair_cache.n_packed_active_pairs > 0 &&
      has_sparse_packed_pair_projection(pair_cache.first_order_cofactor_projection) &&
      has_dense_projected_pair_values(
          pair_cache.first_order_cofactor_projection,
          pair_cache.n_packed_active_pairs);
}

bool has_opposite_spin_inverse_projection(
    const OppositeSpinPairCache& pair_cache) {
  return pair_cache.n_packed_active_pairs > 0 &&
      has_sparse_packed_pair_projection(pair_cache.inverse_overlap_projection) &&
      has_dense_projected_pair_values(
          pair_cache.inverse_overlap_projection,
          pair_cache.n_packed_active_pairs);
}

double contract_opposite_spin_first_order_projections(
    const OppositeSpinPairCache& alpha_pair_cache,
    const OppositeSpinPairCache& beta_pair_cache) {
  if (!has_opposite_spin_first_order_projection(alpha_pair_cache) ||
      !has_opposite_spin_first_order_projection(beta_pair_cache) ||
      alpha_pair_cache.n_packed_active_pairs != beta_pair_cache.n_packed_active_pairs) {
    throw std::invalid_argument(
        "opposite-spin first-order projection cache is unavailable or inconsistent");
  }
  return contract_sparse_projection_with_dense_image(
      alpha_pair_cache.first_order_cofactor_projection,
      beta_pair_cache.first_order_cofactor_projection.projected_pair_values);
}

double project_sparse_projection_onto_packed_pair(
    const OppositeSpinPackedPairProjection& sparse_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (target_packed_pair_index < 0 || target_packed_pair_index >= n_packed_active_pairs) {
    throw std::invalid_argument("packed active-pair index out of range");
  }

  if (target_packed_pair_index <
      static_cast<int>(sparse_projection.projected_pair_values.size())) {
    return sparse_projection.projected_pair_values[xmvb::to_size(
        target_packed_pair_index)];
  }

  double projected_value = 0.0;
  for (std::size_t entry_index = 0;
       entry_index < sparse_projection.packed_pair_indices.size();
       ++entry_index) {
    projected_value +=
        lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            target_packed_pair_index,
            sparse_projection.packed_pair_indices[entry_index],
            n_active_orbitals) *
        sparse_projection.packed_pair_values[entry_index];
  }
  return projected_value;
}

Eigen::VectorXd gather_projected_values_for_packed_pair_indices(
    const OppositeSpinPackedPairProjection& sparse_projection,
    const std::vector<int>& target_packed_pair_indices,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  if (static_cast<int>(sparse_projection.projected_pair_values.size()) ==
      n_packed_active_pairs) {
    Eigen::VectorXd projected_values =
        Eigen::VectorXd::Zero(static_cast<int>(target_packed_pair_indices.size()));
    for (std::size_t target_index = 0;
         target_index < target_packed_pair_indices.size();
         ++target_index) {
      const int packed_pair_index = target_packed_pair_indices[target_index];
      if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
        throw std::invalid_argument("packed active-pair index out of range");
      }
      projected_values(static_cast<int>(target_index)) =
          sparse_projection.projected_pair_values[xmvb::to_size(
              packed_pair_index)];
    }
    return projected_values;
  }

  return apply_active_space_two_electron_kernel_to_sparse_projection_subset(
      two_electron_view,
      n_active_orbitals,
      sparse_projection.packed_pair_indices,
      sparse_projection.packed_pair_values,
      target_packed_pair_indices);
}

Eigen::MatrixXd build_inverse_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.n_electrons < 0) {
    throw std::invalid_argument("det_ovlp_result.n_electrons must be non-negative");
  }
  if (det_ovlp_result.n_electrons == 0) {
    // Open-shell determinant pairs can carry an empty spin channel. The
    // inverse overlap of the empty block is the empty 0x0 matrix, and the
    // surrounding algebra already treats that case correctly via zero-sized
    // products and loops.
    return Eigen::MatrixXd(0, 0);
  }
  if (det_ovlp_result.nullity != 0 || det_ovlp_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("inverse overlap requires a non-singular determinant pair");
  }
  if (has_cached_inverse_overlap_submatrix(det_ovlp_result)) {
    return det_ovlp_result.inverse_overlap_submatrix;
  }

  require_svd_overlap_result(det_ovlp_result);
  const Eigen::VectorXd inverse_singular_values =
      det_ovlp_result.singular_values.cwiseInverse();

  Eigen::MatrixXd inverse_overlap_submatrix(
      det_ovlp_result.n_electrons,
      det_ovlp_result.n_electrons);

  inverse_overlap_submatrix.noalias() =
      det_ovlp_result.matrix_V * inverse_singular_values.asDiagonal() *
      det_ovlp_result.matrix_U.transpose();
  return inverse_overlap_submatrix;
}

Eigen::MatrixXd build_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.n_electrons < 0) {
    throw std::invalid_argument("det_ovlp_result.n_electrons must be non-negative");
  }
  if (det_ovlp_result.n_electrons == 0) {
    return Eigen::MatrixXd(0, 0);
  }

  require_svd_overlap_result(det_ovlp_result);
  Eigen::MatrixXd overlap_submatrix(
      det_ovlp_result.n_electrons,
      det_ovlp_result.n_electrons);
  overlap_submatrix.noalias() =
      det_ovlp_result.matrix_U *
      det_ovlp_result.singular_values.asDiagonal() *
      det_ovlp_result.matrix_V.transpose();
  return overlap_submatrix;
}

Eigen::MatrixXd build_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols) {
  if (overlap_block.rows() < static_cast<int>(deleted_rows.size()) ||
      overlap_block.cols() < static_cast<int>(deleted_cols.size())) {
    throw std::invalid_argument("deleted-minor request is larger than the source matrix");
  }

  Eigen::MatrixXd minor(
      overlap_block.rows() - static_cast<int>(deleted_rows.size()),
      overlap_block.cols() - static_cast<int>(deleted_cols.size()));
  int minor_row = 0;
  for (int row = 0; row < overlap_block.rows(); ++row) {
    if (std::find(deleted_rows.begin(), deleted_rows.end(), row) != deleted_rows.end()) {
      continue;
    }
    int minor_col = 0;
    for (int col = 0; col < overlap_block.cols(); ++col) {
      if (std::find(deleted_cols.begin(), deleted_cols.end(), col) != deleted_cols.end()) {
        continue;
      }
      minor(minor_row, minor_col) = overlap_block(row, col);
      ++minor_col;
    }
    ++minor_row;
  }
  return minor;
}

double calc_deleted_minor_sign(
    int n_rows,
    int n_cols,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols) {
  if (static_cast<int>(deleted_rows.size()) > n_rows ||
      static_cast<int>(deleted_cols.size()) > n_cols) {
    throw std::invalid_argument("deleted-minor sign request is larger than the source matrix");
  }
  return parity_sign(
      deleted_index_sum(deleted_rows) +
      deleted_index_sum(deleted_cols));
}

double calc_deleted_minor_determinant(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    const DeterminantOverlapResolver& overlap_resolver) {
  const Eigen::MatrixXd minor =
      build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
  const double sign = calc_deleted_minor_sign(
      overlap_block.rows(),
      overlap_block.cols(),
      deleted_rows,
      deleted_cols);
  if (minor.size() == 0) {
    return sign;
  }
  return sign * overlap_resolver.resolve_matrix(minor).overlap_determinant;
}

double calc_directional_deleted_minor_determinant(
    const Eigen::MatrixXd& overlap_block,
    const Eigen::MatrixXd& delta_overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (overlap_block.rows() != delta_overlap_block.rows() ||
      overlap_block.cols() != delta_overlap_block.cols()) {
    throw std::invalid_argument(
        "delta_overlap_block dimensions do not match overlap_block");
  }

  const Eigen::MatrixXd minor =
      build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
  if (minor.size() == 0) {
    return 0.0;
  }
  const Eigen::MatrixXd delta_minor =
      build_deleted_minor_matrix(
          delta_overlap_block,
          deleted_rows,
          deleted_cols);
  const DeterminantOverlapResult minor_result =
      overlap_resolver.resolve_matrix(minor);
  const Eigen::MatrixXd minor_cofactor =
      calc_cofactor_1st(minor_result);
  return calc_deleted_minor_sign(
             overlap_block.rows(),
             overlap_block.cols(),
             deleted_rows,
             deleted_cols) *
      (minor_cofactor.cwiseProduct(delta_minor)).sum();
}

Eigen::MatrixXd build_directional_first_cofactor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const Eigen::MatrixXd& delta_overlap_block,
    const DeterminantOverlapResult& det_ovlp_result,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (overlap_block.rows() != overlap_block.cols() ||
      delta_overlap_block.rows() != overlap_block.rows() ||
      delta_overlap_block.cols() != overlap_block.cols() ||
      det_ovlp_result.n_electrons != overlap_block.rows()) {
    throw std::invalid_argument(
        "directional first cofactor inputs have inconsistent dimensions");
  }

  const int n_electrons = overlap_block.rows();
  Eigen::MatrixXd delta_cofactor =
      Eigen::MatrixXd::Zero(n_electrons, n_electrons);
  if (n_electrons == 0) {
    return delta_cofactor;
  }

  if (det_ovlp_result.nullity == 0 &&
      det_ovlp_result.overlap_determinant != 0.0) {
    const Eigen::MatrixXd inverse_overlap_submatrix =
        build_inverse_overlap_submatrix_from_result(det_ovlp_result);
    const double delta_overlap_determinant =
        (calc_cofactor_1st(det_ovlp_result).cwiseProduct(delta_overlap_block)).sum();
    const Eigen::MatrixXd delta_inverse_overlap_submatrix =
        -inverse_overlap_submatrix *
        delta_overlap_block *
        inverse_overlap_submatrix;
    delta_cofactor =
        delta_overlap_determinant *
        inverse_overlap_submatrix.transpose();
    delta_cofactor.noalias() +=
        det_ovlp_result.overlap_determinant *
        delta_inverse_overlap_submatrix.transpose();
    return delta_cofactor;
  }

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      delta_cofactor(right_row, left_column) =
          calc_directional_deleted_minor_determinant(
              overlap_block,
              delta_overlap_block,
              {right_row},
              {left_column},
              overlap_resolver);
    }
  }
  return delta_cofactor;
}

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  return compute_same_spin_original_phi_impl(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      det_ovlp_result,
      inverse_overlap_gradient,
      [&packed_active_two_electron_integrals](
          int row_packed_pair_index,
          int column_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                column_packed_pair_index);
        return packed_active_two_electron_integrals[xmvb::to_size(
            packed_pair_of_pairs_index)];
      });
}

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const DeterminantOverlapResult& det_ovlp_result,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  return compute_same_spin_original_phi_impl(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      det_ovlp_result,
      inverse_overlap_gradient,
      [&two_electron_view, n_active_orbitals](
          int row_packed_pair_index,
          int column_packed_pair_index) {
        return lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            row_packed_pair_index,
            column_packed_pair_index,
            n_active_orbitals);
      });
}

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const OppositeSpinPairCache* beta_pair_cache,
    const std::vector<double>& packed_active_two_electron_integrals,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient) {
  if (alpha_overlap_result.nullity != 0 || beta_overlap_result.nullity != 0 ||
      alpha_overlap_result.overlap_determinant == 0.0 ||
      beta_overlap_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("original opposite-spin phi requires non-singular determinant pairs");
  }

  if (alpha_pair_cache != nullptr && beta_pair_cache != nullptr &&
      has_opposite_spin_inverse_projection(*alpha_pair_cache) &&
      has_opposite_spin_inverse_projection(*beta_pair_cache) &&
      alpha_pair_cache->n_packed_active_pairs == beta_pair_cache->n_packed_active_pairs) {
    build_inverse_overlap_gradient_from_projected_pair_values(
        alpha_occ_L,
        alpha_occ_R,
        beta_pair_cache->inverse_overlap_projection.projected_pair_values,
        alpha_inverse_overlap_gradient);
    build_inverse_overlap_gradient_from_projected_pair_values(
        beta_occ_L,
        beta_occ_R,
        alpha_pair_cache->inverse_overlap_projection.projected_pair_values,
        beta_inverse_overlap_gradient);
    return contract_sparse_projection_with_dense_image(
        alpha_pair_cache->inverse_overlap_projection,
        beta_pair_cache->inverse_overlap_projection.projected_pair_values);
  }
  return compute_opposite_spin_original_phi_impl(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_result,
      beta_occ_L,
      beta_occ_R,
      beta_overlap_result,
      alpha_inverse_overlap_gradient,
      beta_inverse_overlap_gradient,
      [&packed_active_two_electron_integrals](
          int row_packed_pair_index,
          int column_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                column_packed_pair_index);
        return packed_active_two_electron_integrals[xmvb::to_size(
            packed_pair_of_pairs_index)];
      });
}

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const OppositeSpinPairCache* beta_pair_cache,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient) {
  if (alpha_overlap_result.nullity != 0 || beta_overlap_result.nullity != 0 ||
      alpha_overlap_result.overlap_determinant == 0.0 ||
      beta_overlap_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("original opposite-spin phi requires non-singular determinant pairs");
  }

  if (alpha_pair_cache != nullptr && beta_pair_cache != nullptr &&
      has_opposite_spin_inverse_projection(*alpha_pair_cache) &&
      has_opposite_spin_inverse_projection(*beta_pair_cache) &&
      alpha_pair_cache->n_packed_active_pairs == beta_pair_cache->n_packed_active_pairs) {
    build_inverse_overlap_gradient_from_projected_pair_values(
        alpha_occ_L,
        alpha_occ_R,
        beta_pair_cache->inverse_overlap_projection.projected_pair_values,
        alpha_inverse_overlap_gradient);
    build_inverse_overlap_gradient_from_projected_pair_values(
        beta_occ_L,
        beta_occ_R,
        alpha_pair_cache->inverse_overlap_projection.projected_pair_values,
        beta_inverse_overlap_gradient);
    return contract_sparse_projection_with_dense_image(
        alpha_pair_cache->inverse_overlap_projection,
        beta_pair_cache->inverse_overlap_projection.projected_pair_values);
  }

  int n_active_orbitals = 0;
  if (active_space_two_electron_result.representation ==
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    n_active_orbitals = infer_active_orbital_count_from_packed_pair_count(
        active_space_two_electron_result.ri_active_pair_factors.cols());
  } else {
    n_active_orbitals = infer_active_orbital_count_from_packed_integral_count(
        active_space_two_electron_result.packed_active_two_electron_integrals.size());
  }
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  return compute_opposite_spin_original_phi_impl(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_result,
      beta_occ_L,
      beta_occ_R,
      beta_overlap_result,
      alpha_inverse_overlap_gradient,
      beta_inverse_overlap_gradient,
      [&two_electron_view, n_active_orbitals](
          int row_packed_pair_index,
          int column_packed_pair_index) {
        return lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            row_packed_pair_index,
            column_packed_pair_index,
            n_active_orbitals);
      });
}

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient) {
  return compute_opposite_spin_original_phi(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_result,
      nullptr,
      beta_occ_L,
      beta_occ_R,
      beta_overlap_result,
      nullptr,
      packed_active_two_electron_integrals,
      alpha_inverse_overlap_gradient,
      beta_inverse_overlap_gradient);
}

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient) {
  return compute_opposite_spin_original_phi(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_result,
      nullptr,
      beta_occ_L,
      beta_occ_R,
      beta_overlap_result,
      nullptr,
      active_space_two_electron_result,
      alpha_inverse_overlap_gradient,
      beta_inverse_overlap_gradient);
}

void accumulate_spin_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    double determinant_overlap_weight,
    const Eigen::MatrixXd& inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (det_ovlp_result.nullity != 0) {
    throw std::invalid_argument("analytic overlap gradient requires a non-singular determinant pair");
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons ||
      det_ovlp_result.n_electrons != n_electrons) {
    throw std::invalid_argument("determinant overlap gradient size mismatch");
  }

  const Eigen::MatrixXd inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(det_ovlp_result);
  const Eigen::MatrixXd overlap_submatrix_gradient =
      determinant_overlap_weight * det_ovlp_result.overlap_determinant *
          inverse_overlap_submatrix.transpose() -
      det_ovlp_result.overlap_determinant * inverse_overlap_submatrix.transpose() *
          inverse_overlap_gradient * inverse_overlap_submatrix.transpose();

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occ_R[xmvb::to_size(right_row)];
      (*active_orbital_overlap_gradient)[xmvb::to_size(orbital_index_left) *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_submatrix_gradient(right_row, left_column);
    }
  }
}

}  // namespace xmvb::vb
