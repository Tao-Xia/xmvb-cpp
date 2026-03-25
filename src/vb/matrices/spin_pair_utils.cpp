#include "vb/matrices/spin_pair_utils.hpp"

#include <stdexcept>
#include <vector>

#include "vb/matrices/two_electron_indexer.hpp"

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

Matrix build_spin_one_electron_block_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Matrix one_electron_block(n_electrons, n_electrons);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occ_R[static_cast<std::size_t>(right_row)];
      one_electron_block(right_row, left_column) =
          h1e_act[static_cast<std::size_t>(orbital_index_left) *
                                         n_active_orbitals +
                                     orbital_index_right];
    }
  }

  return one_electron_block;
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
      static_cast<std::size_t>(n_electrons) * static_cast<std::size_t>(n_electrons),
      0.0);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[static_cast<std::size_t>(right_row)];
      overlap_submatrix[static_cast<std::size_t>(left_column) * n_electrons + right_row] =
          ovlp_act[static_cast<std::size_t>(orbital_index_left) * n_orbitals +
                   orbital_index_right];
    }
  }

  return overlap_submatrix;
}


Matrix calc_cofactor_1st(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.n_electrons < 0) {
    throw std::invalid_argument("det_ovlp_result.n_electrons must be non-negative");
  }
  if (det_ovlp_result.singular_values.size() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.cols() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.cols() != det_ovlp_result.n_electrons) {
    throw std::invalid_argument("det_ovlp_result SVD dimensions are inconsistent");
  }

  Matrix cofactor_matrix =
      Matrix::Zero(det_ovlp_result.n_electrons, det_ovlp_result.n_electrons);
  if (det_ovlp_result.n_electrons == 0) {
    return cofactor_matrix;
  }

  if (det_ovlp_result.nullity == 0) {
    const Eigen::VectorXd inverse_singular_values =
        det_ovlp_result.singular_values.cwiseInverse();
    cofactor_matrix.noalias() =
        det_ovlp_result.overlap_determinant * det_ovlp_result.matrix_U *
        inverse_singular_values.asDiagonal() * det_ovlp_result.matrix_V.transpose();
    return cofactor_matrix;
  }

  if (det_ovlp_result.nullity == 1) {
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

Matrix build_inverse_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.n_electrons <= 0) {
    throw std::invalid_argument("det_ovlp_result.n_electrons must be positive");
  }
  if (det_ovlp_result.nullity != 0 || det_ovlp_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("inverse overlap requires a non-singular determinant pair");
  }
  if (det_ovlp_result.singular_values.size() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.cols() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.cols() != det_ovlp_result.n_electrons) {
    throw std::invalid_argument("det_ovlp_result SVD dimensions are inconsistent");
  }

  const Eigen::VectorXd inverse_singular_values =
      det_ovlp_result.singular_values.cwiseInverse();

  Matrix inverse_overlap_submatrix(
      det_ovlp_result.n_electrons,
      det_ovlp_result.n_electrons);
  
  inverse_overlap_submatrix.noalias() =
      det_ovlp_result.matrix_V * inverse_singular_values.asDiagonal() *
      det_ovlp_result.matrix_U.transpose();
  return inverse_overlap_submatrix;
}

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Matrix* inverse_overlap_gradient) {
  if (det_ovlp_result.nullity != 0 || det_ovlp_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("original phi requires a non-singular determinant pair");
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }

  const Matrix inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(det_ovlp_result);

  const Matrix one_electron_block = build_spin_one_electron_block_matrix(
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
        occ_L[static_cast<std::size_t>(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first =
          occ_R[static_cast<std::size_t>(right_first)];
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second =
            occ_L[static_cast<std::size_t>(left_second)];
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second =
              occ_R[static_cast<std::size_t>(right_second)];
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
          const double interaction_value =
              packed_active_two_electron_integrals[static_cast<std::size_t>(direct_index)] -
              packed_active_two_electron_integrals[static_cast<std::size_t>(exchange_index)];
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

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    Matrix* alpha_inverse_overlap_gradient,
    Matrix* beta_inverse_overlap_gradient) {
  if (alpha_overlap_result.nullity != 0 || beta_overlap_result.nullity != 0 ||
      alpha_overlap_result.overlap_determinant == 0.0 ||
      beta_overlap_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("original opposite-spin phi requires non-singular determinant pairs");
  }

  const Matrix alpha_inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(alpha_overlap_result);
  const Matrix beta_inverse_overlap_submatrix =
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
        alpha_occ_L[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[static_cast<std::size_t>(alpha_right_row)];
      const double alpha_inverse_value =
          alpha_inverse_overlap_submatrix(alpha_left_column, alpha_right_row);
      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[static_cast<std::size_t>(beta_right_row)];
          const double beta_inverse_value =
              beta_inverse_overlap_submatrix(beta_left_column, beta_right_row);
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          const double interaction_value =
              packed_active_two_electron_integrals[static_cast<std::size_t>(two_electron_index)];
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

void accumulate_spin_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    double determinant_overlap_weight,
    const Matrix& inverse_overlap_gradient,
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

  const Matrix inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(det_ovlp_result);
  const Matrix overlap_submatrix_gradient =
      determinant_overlap_weight * det_ovlp_result.overlap_determinant *
          inverse_overlap_submatrix.transpose() -
      det_ovlp_result.overlap_determinant * inverse_overlap_submatrix.transpose() *
          inverse_overlap_gradient * inverse_overlap_submatrix.transpose();

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right =
          occ_R[static_cast<std::size_t>(right_row)];
      (*active_orbital_overlap_gradient)[static_cast<std::size_t>(orbital_index_left) *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_submatrix_gradient(right_row, left_column);
    }
  }
}

}  // namespace xmvb::vb
