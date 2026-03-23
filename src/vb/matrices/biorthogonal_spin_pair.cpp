#include "vb/matrices/biorthogonal_spin_pair.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct FourIndexTensor {
  int dim0 = 0;
  int dim1 = 0;
  int dim2 = 0;
  int dim3 = 0;
  std::vector<double> values;

  FourIndexTensor() = default;

  FourIndexTensor(int axis0, int axis1, int axis2, int axis3)
      : dim0(axis0),
        dim1(axis1),
        dim2(axis2),
        dim3(axis3),
        values(
            static_cast<std::size_t>(axis0) * static_cast<std::size_t>(axis1) *
                static_cast<std::size_t>(axis2) * static_cast<std::size_t>(axis3),
            0.0) {}

  double& operator()(int index0, int index1, int index2, int index3) {
    return values[
        ((static_cast<std::size_t>(index0) * dim1 + static_cast<std::size_t>(index1)) * dim2 +
         static_cast<std::size_t>(index2)) *
            dim3 +
        static_cast<std::size_t>(index3)];
  }

  double operator()(int index0, int index1, int index2, int index3) const {
    return values[
        ((static_cast<std::size_t>(index0) * dim1 + static_cast<std::size_t>(index1)) * dim2 +
         static_cast<std::size_t>(index2)) *
            dim3 +
        static_cast<std::size_t>(index3)];
  }
};

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

FourIndexTensor build_same_spin_two_electron_tensor(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& packed_active_two_electron_integrals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  FourIndexTensor tensor(n_electrons, n_electrons, n_electrons, n_electrons);

  for (int right_first = 0; right_first < n_electrons; ++right_first) {
    const int orbital_index_right_first =
        occ_R[static_cast<std::size_t>(right_first)];
    for (int left_first = 0; left_first < n_electrons; ++left_first) {
      const int orbital_index_left_first =
          occ_L[static_cast<std::size_t>(left_first)];
      for (int right_second = 0; right_second < n_electrons; ++right_second) {
        const int orbital_index_right_second =
            occ_R[static_cast<std::size_t>(right_second)];
        for (int left_second = 0; left_second < n_electrons; ++left_second) {
          const int orbital_index_left_second =
              occ_L[static_cast<std::size_t>(left_second)];
          const int tensor_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          tensor(right_first, left_first, right_second, left_second) =
              packed_active_two_electron_integrals[static_cast<std::size_t>(tensor_index)];
        }
      }
    }
  }

  return tensor;
}

FourIndexTensor build_opposite_spin_two_electron_tensor(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& packed_active_two_electron_integrals) {
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  FourIndexTensor tensor(n_beta_electrons, n_beta_electrons, n_alpha_electrons, n_alpha_electrons);

  for (int beta_right = 0; beta_right < n_beta_electrons; ++beta_right) {
    const int beta_orbital_right =
        beta_occ_R[static_cast<std::size_t>(beta_right)];
    for (int beta_left = 0; beta_left < n_beta_electrons; ++beta_left) {
      const int beta_orbital_left =
          beta_occ_L[static_cast<std::size_t>(beta_left)];
      for (int alpha_right = 0; alpha_right < n_alpha_electrons; ++alpha_right) {
        const int alpha_orbital_right =
            alpha_occ_R[static_cast<std::size_t>(alpha_right)];
        for (int alpha_left = 0; alpha_left < n_alpha_electrons; ++alpha_left) {
          const int alpha_orbital_left =
              alpha_occ_L[static_cast<std::size_t>(alpha_left)];
          const int tensor_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          tensor(beta_right, beta_left, alpha_right, alpha_left) =
              packed_active_two_electron_integrals[static_cast<std::size_t>(tensor_index)];
        }
      }
    }
  }

  return tensor;
}


FourIndexTensor transform_tensor_axis(
    const FourIndexTensor& input_tensor,
    int axis,
    const Matrix& transform_matrix) {
    
  FourIndexTensor output_tensor(
      axis == 0 ? transform_matrix.cols() : input_tensor.dim0,
      axis == 1 ? transform_matrix.cols() : input_tensor.dim1,
      axis == 2 ? transform_matrix.cols() : input_tensor.dim2,
      axis == 3 ? transform_matrix.cols() : input_tensor.dim3);

  const double* in_data = input_tensor.values.data();
  double* out_data = output_tensor.values.data();

  // 局部类型别名：仅在这个函数内生效，不污染外部命名空间。
  // 因为 std::vector 默认的一维展开是行优先的 (Row-Major)，
  // 必须使用 Eigen::RowMajor 来正确映射连续的内存块。
  using RowMajorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using ConstRowMajorMap = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Axis 0: 将张量看作 (N0) x (N1*N2*N3) 的大矩阵
  if (axis == 0) {
    const int rows = input_tensor.dim0;
    const int cols = input_tensor.dim1 * input_tensor.dim2 * input_tensor.dim3;
    const int out_rows = transform_matrix.cols();
    
    ConstRowMajorMap in_mat(in_data, rows, cols);
    RowMajorMap out_mat(out_data, out_rows, cols);
    // U^T * T
    out_mat.noalias() = transform_matrix.transpose() * in_mat;
  } 
  // Axis 1: 沿第一维度循环，每次处理一个 (N1) x (N2*N3) 的矩阵
  else if (axis == 1) {
    const int outer_loops = input_tensor.dim0;
    const int rows = input_tensor.dim1;
    const int cols = input_tensor.dim2 * input_tensor.dim3;
    const int out_rows = transform_matrix.cols();
    
    for (int i = 0; i < outer_loops; ++i) {
      ConstRowMajorMap in_mat(in_data + i * rows * cols, rows, cols);
      RowMajorMap out_mat(out_data + i * out_rows * cols, out_rows, cols);
      // V^T * T_slice
      out_mat.noalias() = transform_matrix.transpose() * in_mat;
    }
  } 
  // Axis 2: 沿前两个维度循环，每次处理一个 (N2) x (N3) 的矩阵
  else if (axis == 2) {
    const int outer_loops = input_tensor.dim0 * input_tensor.dim1;
    const int rows = input_tensor.dim2;
    const int cols = input_tensor.dim3;
    const int out_rows = transform_matrix.cols();
    
    for (int i = 0; i < outer_loops; ++i) {
      ConstRowMajorMap in_mat(in_data + i * rows * cols, rows, cols);
      RowMajorMap out_mat(out_data + i * out_rows * cols, out_rows, cols);
      // U^T * T_slice
      out_mat.noalias() = transform_matrix.transpose() * in_mat;
    }
  } 
  // Axis 3: 将张量看作 (N0*N1*N2) x (N3) 的大矩阵
  else if (axis == 3) {
    const int rows = input_tensor.dim0 * input_tensor.dim1 * input_tensor.dim2;
    const int cols = input_tensor.dim3;
    const int out_cols = transform_matrix.cols();
    
    ConstRowMajorMap in_mat(in_data, rows, cols);
    RowMajorMap out_mat(out_data, rows, out_cols);
    // T * V
    out_mat.noalias() = in_mat * transform_matrix;
  } 
  else {
    throw std::invalid_argument("tensor axis must be between 0 and 3");
  }

  return output_tensor;
} 


FourIndexTensor transform_same_spin_two_electron_tensor(
    const FourIndexTensor& source_tensor,
    const DeterminantOverlapResult& det_ovlp_result) {
  FourIndexTensor transformed_tensor = transform_tensor_axis(
      source_tensor,
      0,
      det_ovlp_result.matrix_U);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      1,
      det_ovlp_result.matrix_V);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      2,
      det_ovlp_result.matrix_U);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      3,
      det_ovlp_result.matrix_V);
  return transformed_tensor;
}

FourIndexTensor transform_opposite_spin_two_electron_tensor(
    const FourIndexTensor& source_tensor,
    const DeterminantOverlapResult& alpha_overlap_result,
    const DeterminantOverlapResult& beta_overlap_result) {
  FourIndexTensor transformed_tensor = transform_tensor_axis(
      source_tensor,
      0,
      beta_overlap_result.matrix_U);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      1,
      beta_overlap_result.matrix_V);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      2,
      alpha_overlap_result.matrix_U);
  transformed_tensor = transform_tensor_axis(
      transformed_tensor,
      3,
      alpha_overlap_result.matrix_V);
  return transformed_tensor;
}

double product_of_leading_singular_values(
    const DeterminantOverlapResult& det_ovlp_result,
    int count) {
  double product = 1.0;
  for (int singular_index = 0; singular_index < count; ++singular_index) {
    product *= det_ovlp_result.singular_values(singular_index);
  }
  return product;
}

Eigen::VectorXd build_single_orbital_overlap_weights_internal(
    const DeterminantOverlapResult& det_ovlp_result) {
  const int n_electrons = det_ovlp_result.n_electrons;
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(n_electrons);
  if (n_electrons == 0) {
    return weights;
  }

  if (det_ovlp_result.nullity == 0) {
    for (int orbital_index = 0; orbital_index < n_electrons; ++orbital_index) {
      weights(orbital_index) =
          det_ovlp_result.overlap_determinant / det_ovlp_result.singular_values(orbital_index);
    }
    return weights;
  }

  if (det_ovlp_result.nullity == 1) {
    const int null_index = n_electrons - 1;
    weights(null_index) =
        det_ovlp_result.parity *
        product_of_leading_singular_values(det_ovlp_result, n_electrons - 1);
  }
  return weights;
}

double compute_pair_overlap_weight_internal(
    const DeterminantOverlapResult& det_ovlp_result,
    int first_index,
    int second_index) {
  if (first_index == second_index) {
    return 0.0;
  }
  if (first_index > second_index) {
    std::swap(first_index, second_index);
  }

  const int n_electrons = det_ovlp_result.n_electrons;
  if (det_ovlp_result.nullity == 0) {
    return det_ovlp_result.overlap_determinant /
        (det_ovlp_result.singular_values(first_index) * det_ovlp_result.singular_values(second_index));
  }

  if (det_ovlp_result.nullity == 1) {
    const int null_index = n_electrons - 1;
    if (first_index != null_index && second_index != null_index) {
      return 0.0;
    }
    const int non_null_index = (first_index == null_index) ? second_index : first_index;
    const double single_null_weight =
        det_ovlp_result.parity *
        product_of_leading_singular_values(det_ovlp_result, n_electrons - 1);
    return single_null_weight / det_ovlp_result.singular_values(non_null_index);
  }

  if (det_ovlp_result.nullity == 2) {
    const int first_null_index = n_electrons - 2;
    const int second_null_index = n_electrons - 1;
    if (first_index == first_null_index && second_index == second_null_index) {
      return det_ovlp_result.parity *
          product_of_leading_singular_values(det_ovlp_result, n_electrons - 2);
    }
  }

  return 0.0;
}

void accumulate_same_spin_inverse_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Matrix& inverse_overlap_gradient) {
  const int n_electrons = static_cast<int>(occ_L.size());
  inverse_overlap_gradient.setZero(n_electrons, n_electrons);
  if (n_electrons == 0) {
    return;
  }

  const Matrix one_electron_block = build_spin_one_electron_block_matrix(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals);

  Matrix transformed_inverse_overlap_gradient(n_electrons, n_electrons);
  transformed_inverse_overlap_gradient.noalias() =
      det_ovlp_result.matrix_V.transpose() *
      one_electron_block.transpose() *
      det_ovlp_result.matrix_U;

  const FourIndexTensor source_tensor = build_same_spin_two_electron_tensor(
      occ_L,
      occ_R,
      packed_active_two_electron_integrals);
  const FourIndexTensor transformed_g = transform_same_spin_two_electron_tensor(
      source_tensor, det_ovlp_result);

  for (int left_index = 0; left_index < n_electrons; ++left_index) {
    for (int right_index = 0; right_index < n_electrons; ++right_index) {
      double gradient_entry =
          transformed_inverse_overlap_gradient(left_index, right_index);
      for (int c = 0; c < n_electrons; ++c) {
        if (det_ovlp_result.singular_values(c) == 0.0) {
          continue;
        }
        const double d_c = 1.0 / det_ovlp_result.singular_values(c);
        gradient_entry +=
            (transformed_g(right_index, left_index, c, c) -
             transformed_g(right_index, c, c, left_index)) *
            d_c;
      }
      transformed_inverse_overlap_gradient(left_index, right_index) = gradient_entry;
    }
  }

  inverse_overlap_gradient.noalias() +=
      det_ovlp_result.matrix_V *
      transformed_inverse_overlap_gradient *
      det_ovlp_result.matrix_U.transpose();
}


void accumulate_opposite_spin_inverse_overlap_gradient(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& alpha_overlap_result,
    const DeterminantOverlapResult& beta_overlap_result,
    Matrix* alpha_inverse_overlap_gradient,
    Matrix* beta_inverse_overlap_gradient) {
  const int n_alpha = static_cast<int>(alpha_occ_L.size());
  const int n_beta = static_cast<int>(beta_occ_L.size());

  const FourIndexTensor source_tensor = build_opposite_spin_two_electron_tensor(
      alpha_occ_L,
      alpha_occ_R,
      beta_occ_L,
      beta_occ_R,
      packed_active_two_electron_integrals);
  const FourIndexTensor transformed_g = transform_opposite_spin_two_electron_tensor(
      source_tensor, alpha_overlap_result, beta_overlap_result);

  if (alpha_inverse_overlap_gradient != nullptr) {
    if (alpha_inverse_overlap_gradient->rows() != n_alpha ||
        alpha_inverse_overlap_gradient->cols() != n_alpha) {
      alpha_inverse_overlap_gradient->setZero(n_alpha, n_alpha);
    }
    Matrix transformed_alpha_gradient =
        Matrix::Zero(n_alpha, n_alpha);

    for (int left_alpha = 0; left_alpha < n_alpha; ++left_alpha) {
      for (int right_alpha = 0; right_alpha < n_alpha; ++right_alpha) {
        double gradient_entry = 0.0;
        for (int c_b = 0; c_b < n_beta; ++c_b) {
          if (beta_overlap_result.singular_values(c_b) == 0.0) {
            continue;
          }
          const double d_c_beta = 1.0 / beta_overlap_result.singular_values(c_b);
          gradient_entry +=
              transformed_g(c_b, c_b, right_alpha, left_alpha) * d_c_beta;
        }
        transformed_alpha_gradient(left_alpha, right_alpha) = gradient_entry;
      }
    }

    alpha_inverse_overlap_gradient->noalias() +=
        alpha_overlap_result.matrix_V *
        transformed_alpha_gradient *
        alpha_overlap_result.matrix_U.transpose();
  }

  if (beta_inverse_overlap_gradient != nullptr) {
    if (beta_inverse_overlap_gradient->rows() != n_beta ||
        beta_inverse_overlap_gradient->cols() != n_beta) {
      beta_inverse_overlap_gradient->setZero(n_beta, n_beta);
    }
    Matrix transformed_beta_gradient =
        Matrix::Zero(n_beta, n_beta);

    for (int left_beta = 0; left_beta < n_beta; ++left_beta) {
      for (int right_beta = 0; right_beta < n_beta; ++right_beta) {
        double gradient_entry = 0.0;
        for (int c_a = 0; c_a < n_alpha; ++c_a) {
          if (alpha_overlap_result.singular_values(c_a) == 0.0) {
            continue;
          }
          const double d_c_alpha = 1.0 / alpha_overlap_result.singular_values(c_a);
          gradient_entry +=
              transformed_g(right_beta, left_beta, c_a, c_a) * d_c_alpha;
        }
        transformed_beta_gradient(left_beta, right_beta) = gradient_entry;
      }
    }

    beta_inverse_overlap_gradient->noalias() +=
        beta_overlap_result.matrix_V *
        transformed_beta_gradient *
        beta_overlap_result.matrix_U.transpose();
  }
}

}  // namespace

ConstMatrixMap map_column_major_matrix(
    const std::vector<double>& matrix_storage,
    int dimension) {
  return ConstMatrixMap(matrix_storage.data(), dimension, dimension);
}

std::vector<double> matrix_to_column_major_storage(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

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

SameSpinBiorthogonalHamiltonianResult calc_same_spin_biorthogonal_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result) {
  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (det_ovlp_result.n_electrons != n_electrons) {
    throw std::invalid_argument("det_ovlp_result size does not match electron count");
  }

  SameSpinBiorthogonalHamiltonianResult result;
  if (n_electrons == 0 || det_ovlp_result.nullity >= 3) {
    return result;
  }

  const Eigen::VectorXd single_orbital_weights =
      build_single_orbital_overlap_weights_internal(det_ovlp_result);
  for (int orbital_index = 0; orbital_index < n_electrons; ++orbital_index) {
    const double orbital_weight = single_orbital_weights(orbital_index);
    if (orbital_weight == 0.0) {
      continue;
    }
    double transformed_diagonal = 0.0;
    for (int left_column = 0; left_column < n_electrons; ++left_column) {
      const int orbital_index_left =
          occ_L[static_cast<std::size_t>(left_column)];
      const double left_coefficient =
          det_ovlp_result.matrix_V(left_column, orbital_index);
      if (left_coefficient == 0.0) {
        continue;
      }
      for (int right_row = 0; right_row < n_electrons; ++right_row) {
        const int orbital_index_right =
            occ_R[static_cast<std::size_t>(right_row)];
        const double right_coefficient =
            det_ovlp_result.matrix_U(right_row, orbital_index);
        if (right_coefficient == 0.0) {
          continue;
        }
        transformed_diagonal +=
            right_coefficient *
            h1e_act[static_cast<std::size_t>(orbital_index_left) *
                                           n_active_orbitals +
                                       orbital_index_right] *
            left_coefficient;
      }
    }
    result.one_electron_hamiltonian += orbital_weight * transformed_diagonal;
  }
  result.total_hamiltonian = result.one_electron_hamiltonian;

  if (n_electrons < 2) {
    return result;
  }

  for (int first_index = 0; first_index < n_electrons - 1; ++first_index) {
    for (int second_index = first_index + 1; second_index < n_electrons; ++second_index) {
      const double pair_weight = compute_pair_overlap_weight_internal(
          det_ovlp_result,
          first_index,
          second_index);
      if (pair_weight == 0.0) {
        continue;
      }
      double antisymmetrized_pair_integral = 0.0;
      for (int left_first = 0; left_first < n_electrons; ++left_first) {
        const int orbital_index_left_first =
            occ_L[static_cast<std::size_t>(left_first)];
        const double left_coefficient_first =
            det_ovlp_result.matrix_V(left_first, first_index);
        const double left_coefficient_first_exchange =
            det_ovlp_result.matrix_V(left_first, second_index);
        if (left_coefficient_first == 0.0 &&
            left_coefficient_first_exchange == 0.0) {
          continue;
        }
        for (int left_second = 0; left_second < n_electrons; ++left_second) {
          const int orbital_index_left_second =
              occ_L[static_cast<std::size_t>(left_second)];
          const double left_antisymmetrized_coefficient =
              left_coefficient_first * det_ovlp_result.matrix_V(left_second, second_index) -
              left_coefficient_first_exchange * det_ovlp_result.matrix_V(left_second, first_index);
          if (left_antisymmetrized_coefficient == 0.0) {
            continue;
          }
          for (int right_first = 0; right_first < n_electrons; ++right_first) {
            const int orbital_index_right_first =
                occ_R[static_cast<std::size_t>(right_first)];
            const double right_coefficient_first =
                det_ovlp_result.matrix_U(right_first, first_index);
            if (right_coefficient_first == 0.0) {
              continue;
            }
            for (int right_second = 0; right_second < n_electrons; ++right_second) {
              const int orbital_index_right_second =
                  occ_R[static_cast<std::size_t>(right_second)];
              const double right_coefficient_second =
                  det_ovlp_result.matrix_U(right_second, second_index);
              if (right_coefficient_second == 0.0) {
                continue;
              }
              const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
                  orbital_index_right_first,
                  orbital_index_left_first,
                  orbital_index_right_second,
                  orbital_index_left_second);
              antisymmetrized_pair_integral +=
                  right_coefficient_first *
                  right_coefficient_second *
                  left_antisymmetrized_coefficient *
                  packed_active_two_electron_integrals[static_cast<std::size_t>(two_electron_index)];
            }
          }
        }
      }
      result.total_hamiltonian += pair_weight * antisymmetrized_pair_integral;
    }
  }

  return result;
}

SameSpinBiorthogonalPhiResult compute_same_spin_original_phi(
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

  SameSpinBiorthogonalPhiResult result;
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

SameSpinBiorthogonalPhiResult compute_same_spin_biorthogonal_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Matrix* inverse_overlap_gradient) {
  if (det_ovlp_result.nullity != 0 || det_ovlp_result.overlap_determinant == 0.0) {
    throw std::invalid_argument("biorthogonal phi requires a non-singular determinant pair");
  }

  const auto hamiltonian_result = calc_same_spin_biorthogonal_hamiltonian(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      packed_active_two_electron_integrals,
      det_ovlp_result);

  SameSpinBiorthogonalPhiResult result;
  result.one_electron_phi =
      hamiltonian_result.one_electron_hamiltonian / det_ovlp_result.overlap_determinant;
  result.total_phi =
      hamiltonian_result.total_hamiltonian / det_ovlp_result.overlap_determinant;

  if (inverse_overlap_gradient != nullptr) {
    compute_same_spin_original_phi(
        occ_L,
        occ_R,
        h1e_act,
        n_active_orbitals,
        packed_active_two_electron_integrals,
        det_ovlp_result,
        inverse_overlap_gradient);
  }

  return result;
}

double compute_opposite_spin_biorthogonal_hamiltonian(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals) {
  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return 0.0;
  }
  if (alpha_overlap_result.nullity >= 2 || beta_overlap_result.nullity >= 2) {
    return 0.0;
  }

  const Matrix alpha_cofactor_1st =
      calc_cofactor_1st(alpha_overlap_result);
  const Matrix beta_cofactor_1st =
      calc_cofactor_1st(beta_overlap_result);
  double hamiltonian = 0.0;
  for (int alpha_left_column = 0;
       alpha_left_column < alpha_overlap_result.n_electrons;
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0;
         alpha_right_row < alpha_overlap_result.n_electrons;
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[static_cast<std::size_t>(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      if (alpha_cofactor == 0.0) {
        continue;
      }
      for (int beta_left_column = 0;
           beta_left_column < beta_overlap_result.n_electrons;
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0;
             beta_right_row < beta_overlap_result.n_electrons;
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[static_cast<std::size_t>(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          if (beta_cofactor == 0.0) {
            continue;
          }
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          hamiltonian +=
              alpha_cofactor * beta_cofactor *
              packed_active_two_electron_integrals[static_cast<std::size_t>(two_electron_index)];
        }
      }
    }
  }

  return hamiltonian;
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

double compute_opposite_spin_biorthogonal_phi(
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
    throw std::invalid_argument(
        "opposite-spin biorthogonal phi requires non-singular determinant pairs");
  }

  const double hamiltonian = compute_opposite_spin_biorthogonal_hamiltonian(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_result,
      beta_occ_L,
      beta_occ_R,
      beta_overlap_result,
      packed_active_two_electron_integrals);

  if (alpha_inverse_overlap_gradient != nullptr || beta_inverse_overlap_gradient != nullptr) {
    const bool alpha_can_write_direct =
        alpha_inverse_overlap_gradient != nullptr &&
        (alpha_inverse_overlap_gradient->rows() != alpha_overlap_result.n_electrons ||
         alpha_inverse_overlap_gradient->cols() != alpha_overlap_result.n_electrons ||
         alpha_inverse_overlap_gradient->isZero(0.0));
    const bool beta_can_write_direct =
        beta_inverse_overlap_gradient != nullptr &&
        (beta_inverse_overlap_gradient->rows() != beta_overlap_result.n_electrons ||
         beta_inverse_overlap_gradient->cols() != beta_overlap_result.n_electrons ||
         beta_inverse_overlap_gradient->isZero(0.0));

    Matrix alpha_gradient;
    Matrix beta_gradient;
    compute_opposite_spin_original_phi(
        alpha_occ_L,
        alpha_occ_R,
        alpha_overlap_result,
        beta_occ_L,
        beta_occ_R,
        beta_overlap_result,
        packed_active_two_electron_integrals,
        alpha_can_write_direct ? alpha_inverse_overlap_gradient :
                                 ((alpha_inverse_overlap_gradient != nullptr) ? &alpha_gradient : nullptr),
        beta_can_write_direct ? beta_inverse_overlap_gradient :
                                ((beta_inverse_overlap_gradient != nullptr) ? &beta_gradient : nullptr));

    if (alpha_inverse_overlap_gradient != nullptr) {
      if (!alpha_can_write_direct) {
        if (alpha_inverse_overlap_gradient->rows() != alpha_gradient.rows() ||
            alpha_inverse_overlap_gradient->cols() != alpha_gradient.cols()) {
          *alpha_inverse_overlap_gradient = alpha_gradient;
        } else {
          *alpha_inverse_overlap_gradient += alpha_gradient;
        }
      }
    }
    if (beta_inverse_overlap_gradient != nullptr) {
      if (!beta_can_write_direct) {
        if (beta_inverse_overlap_gradient->rows() != beta_gradient.rows() ||
            beta_inverse_overlap_gradient->cols() != beta_gradient.cols()) {
          *beta_inverse_overlap_gradient = beta_gradient;
        } else {
          *beta_inverse_overlap_gradient += beta_gradient;
        }
      }
    }
  }

  return hamiltonian /
      (alpha_overlap_result.overlap_determinant * beta_overlap_result.overlap_determinant);
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
