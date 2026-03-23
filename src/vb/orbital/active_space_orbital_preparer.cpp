#include "vb/orbital/active_space_orbital_preparer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace xmvb::vb {

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct JacobiDiagonalizationResult {
  Matrix eigenvector_matrix;
  Eigen::VectorXd eigenvalues;
};

void require_finite_matrix(const Matrix& matrix, const char* label) {
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      if (!std::isfinite(matrix(row, column))) {
        throw std::runtime_error(
            std::string(label) + " contains non-finite values at row=" +
            std::to_string(row) + " col=" + std::to_string(column));
      }
    }
  }
}

void require_finite_vector(const std::vector<double>& values, const char* label) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error(
          std::string(label) + " contains non-finite values at index=" +
          std::to_string(index));
    }
  }
}

int get_sparse_coefficient_count(
    const std::vector<int>& orbital_basis_counts,
    const std::vector<int>& orbital_basis_index_table,
    int n_basis_functions,
    int orbital_index) {
  const int explicit_count = orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_basis_index_table[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                                  coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<double> normalize_sparse_orbitals(
    const OrbitalPreparationInput& input,
    const Eigen::Map<const Matrix>& active_orbital_overlap_matrix) {
  std::vector<double> normalized_values = input.orbital_value_table;

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count = get_sparse_coefficient_count(
        input.orbital_basis_counts,
        input.orbital_basis_index_table,
        input.n_basis_functions,
        orbital_index);
    double squared_norm = 0.0;
    for (int left_index = 0; left_index < coefficient_count; ++left_index) {
      const int left_basis_function =
          input.orbital_basis_index_table[static_cast<std::size_t>(orbital_index) *
                                              input.n_basis_functions +
                                          left_index] -
          1;
      if (left_basis_function < 0 || left_basis_function >= input.n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index");
      }
      const double left_value =
          normalized_values[static_cast<std::size_t>(orbital_index) * input.n_basis_functions +
                            left_index];
      for (int right_index = 0; right_index < coefficient_count; ++right_index) {
        const int right_basis_function =
            input.orbital_basis_index_table[static_cast<std::size_t>(orbital_index) *
                                                input.n_basis_functions +
                                            right_index] -
            1;
        const double right_value =
            normalized_values[static_cast<std::size_t>(orbital_index) * input.n_basis_functions +
                              right_index];
        squared_norm +=
            left_value * right_value * active_orbital_overlap_matrix(left_basis_function, right_basis_function);
      }
    }

    if (!std::isfinite(squared_norm) ||
        squared_norm <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "orbital normalization failed for orbital " + std::to_string(orbital_index) +
          " with squared_norm=" + std::to_string(squared_norm));
    }

    const double normalization_factor = std::sqrt(1.0 / squared_norm);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      normalized_values[static_cast<std::size_t>(orbital_index) * input.n_basis_functions +
                        coefficient_index] *= normalization_factor;
    }
  }

  return normalized_values;
}

Matrix expand_sparse_orbitals(
    const OrbitalPreparationInput& input,
    const std::vector<double>& normalized_orbital_values) {
  Matrix orbital_matrix =
      Matrix::Zero(input.n_basis_functions, input.n_orbitals);

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count = get_sparse_coefficient_count(
        input.orbital_basis_counts,
        input.orbital_basis_index_table,
        input.n_basis_functions,
        orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[static_cast<std::size_t>(orbital_index) *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      orbital_matrix(
          basis_function_index,
          orbital_index) =
          normalized_orbital_values[static_cast<std::size_t>(orbital_index) *
                                        input.n_basis_functions +
                                    coefficient_index];
    }
  }

  return orbital_matrix;
}

JacobiDiagonalizationResult diagonalize_symmetric_jacobi(
    const Matrix& symmetric_matrix) {
  if (symmetric_matrix.rows() != symmetric_matrix.cols()) {
    throw std::invalid_argument("Jacobi diagonalization requires a square matrix");
  }

  constexpr double kZero = 0.0;
  constexpr double kOne = 1.0;
  constexpr double kTwo = 2.0;
  constexpr double kFour = 4.0;
  constexpr double kEpsilon = 1.0e-30;

  const int dimension = symmetric_matrix.rows();
  Matrix diagonal_matrix = symmetric_matrix;
  Matrix eigenvector_matrix =
      Matrix::Identity(dimension, dimension);

  double max_off_diagonal = kOne;
  while ((max_off_diagonal - kEpsilon) > kZero) {
    max_off_diagonal = kZero;
    for (int row = 1; row < dimension; ++row) {
      for (int column = 0; column < row; ++column) {
        const double aii = diagonal_matrix(row, row);
        const double ajj = diagonal_matrix(column, column);
        const double aij = diagonal_matrix(row, column);
        const double square = aij * aij;
        if (square > max_off_diagonal) {
          max_off_diagonal = square;
        }
        if (square <= kEpsilon) {
          continue;
        }

        double difference = aii - ajj;
        double sign = kTwo;
        if (difference < kZero) {
          sign = -kTwo;
          difference = -difference;
        }

        const double tangent_denominator =
            difference + std::sqrt(difference * difference + kFour * square);
        const double tangent = sign * aij / tangent_denominator;
        const double cosine = kOne / std::sqrt(kOne + tangent * tangent);
        const double sine = cosine * tangent;

        for (int k = 0; k < dimension; ++k) {
          const double xj =
              cosine * eigenvector_matrix(k, column) -
              sine * eigenvector_matrix(k, row);
          eigenvector_matrix(k, row) =
              sine * eigenvector_matrix(k, column) +
              cosine * eigenvector_matrix(k, row);
          eigenvector_matrix(k, column) = xj;

          if (k != column && k != row) {
            const double rotated_column =
                cosine * diagonal_matrix(k, column) -
                sine * diagonal_matrix(k, row);
            diagonal_matrix(k, row) =
                sine * diagonal_matrix(k, column) +
                cosine * diagonal_matrix(k, row);
            diagonal_matrix(k, column) = rotated_column;
            diagonal_matrix(row, k) = diagonal_matrix(k, row);
            diagonal_matrix(column, k) = diagonal_matrix(k, column);
          }
        }

        diagonal_matrix(row, row) =
            cosine * cosine * aii + sine * sine * ajj + kTwo * sine * cosine * aij;
        diagonal_matrix(row, column) = kZero;
        diagonal_matrix(column, row) = kZero;
        diagonal_matrix(column, column) =
            cosine * cosine * ajj + sine * sine * aii - kTwo * sine * cosine * aij;
      }
    }
  }

  JacobiDiagonalizationResult result;
  result.eigenvector_matrix = std::move(eigenvector_matrix);
  result.eigenvalues = diagonal_matrix.diagonal();
  return result;
}

}  // namespace

OrbitalPreparationResult ActiveSpaceOrbitalPreparer::prepare(
    const OrbitalPreparationInput& input) const {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 || input.n_active_orbitals <= 0) {
    throw std::invalid_argument("orbital preparation input dimensions must be positive");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_occupied_orbitals = n_inactive_doubly_occupied_orbitals + input.n_active_orbitals;
  const int n_virtual_orbitals = input.n_basis_functions - n_occupied_orbitals;
  if (n_inactive_doubly_occupied_orbitals < 0 || n_virtual_orbitals < 0) {
    throw std::invalid_argument("invalid occupied/virtual partition");
  }

  const Eigen::Map<const Matrix> basis_overlap_matrix(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const std::vector<double> normalized_orbital_values =
      normalize_sparse_orbitals(input, basis_overlap_matrix);
  require_finite_vector(normalized_orbital_values, "normalized_orbital_values");
  Matrix original_orbital_matrix =
      expand_sparse_orbitals(input, normalized_orbital_values);
  require_finite_matrix(original_orbital_matrix, "original_orbital_matrix");

  Matrix auxiliary_orbital_matrix =
      Matrix::Zero(input.n_basis_functions, input.n_basis_functions);
  Matrix inactive_auxiliary_transform =
      Matrix::Zero(input.n_basis_functions, input.n_basis_functions);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    auxiliary_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
  }

  Matrix inactive_density_matrix =
      Matrix::Zero(input.n_basis_functions, input.n_basis_functions);
  Matrix inactive_overlap_inverse =
      Matrix::Zero(n_inactive_doubly_occupied_orbitals,
                                n_inactive_doubly_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const auto inactive_orbitals =
        auxiliary_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
    const Matrix inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap_matrix * inactive_orbitals;
    require_finite_matrix(inactive_overlap, "inactive_overlap");
    inactive_overlap_inverse = inactive_overlap.inverse();
    require_finite_matrix(inactive_overlap_inverse, "inactive_overlap_inverse");
    const Matrix auxiliary_inactive_orbitals =
        inactive_orbitals * inactive_overlap_inverse;
    inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals) =
        auxiliary_inactive_orbitals;
    inactive_density_matrix = auxiliary_inactive_orbitals * inactive_orbitals.transpose();
    require_finite_matrix(inactive_density_matrix, "inactive_density_matrix");
  }

  const auto active_orbitals = original_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);
  Matrix occupied_space_projector =
      Matrix::Identity(input.n_basis_functions, input.n_basis_functions);
  occupied_space_projector -= inactive_density_matrix * basis_overlap_matrix;
  auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals) = occupied_space_projector * active_orbitals;
  require_finite_matrix(auxiliary_orbital_matrix, "auxiliary_orbital_matrix");

  const Matrix active_auxiliary_orbitals = auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);
  const Matrix active_overlap_matrix =
      active_auxiliary_orbitals.transpose() * basis_overlap_matrix * active_auxiliary_orbitals;
  require_finite_matrix(active_overlap_matrix, "active_orbital_overlap_matrix");

  std::vector<int> active_sparse_row_offsets(
      static_cast<std::size_t>(input.n_basis_functions) + 1,
      0);
  std::vector<int> active_sparse_orbital_indices;
  std::vector<double> active_sparse_values;
  active_sparse_orbital_indices.reserve(
      static_cast<std::size_t>(input.n_basis_functions) * input.n_active_orbitals);
  active_sparse_values.reserve(
      static_cast<std::size_t>(input.n_basis_functions) * input.n_active_orbitals);
  for (int basis_function_index = 0;
       basis_function_index < input.n_basis_functions;
       ++basis_function_index) {
    active_sparse_row_offsets[static_cast<std::size_t>(basis_function_index)] =
        static_cast<int>(active_sparse_values.size());
    for (int active_orbital_index = 0;
         active_orbital_index < input.n_active_orbitals;
         ++active_orbital_index) {
      const double coefficient =
          active_auxiliary_orbitals(basis_function_index, active_orbital_index);
      if (coefficient == 0.0) {
        continue;
      }
      active_sparse_orbital_indices.push_back(active_orbital_index);
      active_sparse_values.push_back(coefficient);
    }
  }
  active_sparse_row_offsets[static_cast<std::size_t>(input.n_basis_functions)] =
      static_cast<int>(active_sparse_values.size());
  if (input.n_active_orbitals > 0) {
    const Matrix active_orbital_overlap_inverse =
        active_overlap_matrix.inverse();
    require_finite_matrix(active_orbital_overlap_inverse, "active_orbital_overlap_inverse");

    Matrix occupied_overlap_inverse =
        Matrix::Zero(n_occupied_orbitals, n_occupied_orbitals);
    if (n_inactive_doubly_occupied_orbitals > 0) {
      occupied_overlap_inverse.topLeftCorner(
          n_inactive_doubly_occupied_orbitals,
          n_inactive_doubly_occupied_orbitals) = inactive_overlap_inverse;
    }
    occupied_overlap_inverse.bottomRightCorner(
        input.n_active_orbitals,
        input.n_active_orbitals) = active_orbital_overlap_inverse;

    if (n_virtual_orbitals > 0) {
      const auto occupied_auxiliary_orbitals =
          auxiliary_orbital_matrix.leftCols(n_occupied_orbitals);
      const Matrix tmp_occ =
          occupied_auxiliary_orbitals.transpose() * basis_overlap_matrix;
      const Matrix tmp2 = occupied_overlap_inverse * tmp_occ;
      Matrix complementary_projector =
          Matrix::Identity(input.n_basis_functions, input.n_basis_functions) -
          occupied_auxiliary_orbitals * tmp2;
      const Matrix virtual_overlap =
          complementary_projector.transpose() * basis_overlap_matrix * complementary_projector;
      const auto diagonalization = diagonalize_symmetric_jacobi(virtual_overlap);

      int virtual_column = 0;
      for (int eigen_index = 0; eigen_index < diagonalization.eigenvalues.size(); ++eigen_index) {
        const double eigenvalue = diagonalization.eigenvalues(eigen_index);
        if (eigenvalue <= 1.0e-10) {
          continue;
        }
        if (virtual_column >= n_virtual_orbitals) {
          break;
        }

        Eigen::VectorXd virtual_orbital =
            complementary_projector * diagonalization.eigenvector_matrix.col(eigen_index);
        const double norm =
            virtual_orbital.transpose() * basis_overlap_matrix * virtual_orbital;
        if (norm <= std::numeric_limits<double>::epsilon()) {
          continue;
        }
        auxiliary_orbital_matrix.col(n_occupied_orbitals + virtual_column) =
            virtual_orbital / std::sqrt(norm);
        ++virtual_column;
      }

      if (virtual_column != n_virtual_orbitals) {
        throw std::runtime_error("failed to construct the full virtual auxiliary orbital space");
      }
    }
  }

  const Matrix active_overlap_source =
      basis_overlap_matrix * active_orbitals;
  const Matrix inactive_active_overlap_matrix_full =
      inactive_auxiliary_transform.transpose() * active_overlap_source;
  const Matrix inactive_active_overlap_matrix =
      inactive_active_overlap_matrix_full.topRows(n_inactive_doubly_occupied_orbitals);
  const Matrix projected_active_overlap_matrix =
      occupied_space_projector.transpose() * active_overlap_source;
  const Matrix auxiliary_orbital_inverse =
      auxiliary_orbital_matrix.inverse();
  require_finite_matrix(inactive_active_overlap_matrix, "inactive_active_overlap_matrix");
  require_finite_matrix(projected_active_overlap_matrix, "projected_active_overlap_matrix");
  require_finite_matrix(auxiliary_orbital_inverse, "auxiliary_orbital_inverse");

  OrbitalPreparationResult result;
  result.auxiliary_orbital_matrix.assign(
      auxiliary_orbital_matrix.data(),
      auxiliary_orbital_matrix.data() + auxiliary_orbital_matrix.size());
  result.active_sparse_row_offsets = std::move(active_sparse_row_offsets);
  result.active_sparse_orbital_indices = std::move(active_sparse_orbital_indices);
  result.active_sparse_values = std::move(active_sparse_values);
  result.active_orbital_overlap_matrix.assign(
      active_overlap_matrix.data(),
      active_overlap_matrix.data() + active_overlap_matrix.size());
  result.inactive_density_matrix.assign(
      inactive_density_matrix.data(),
      inactive_density_matrix.data() + inactive_density_matrix.size());
  result.occupied_space_projector.assign(
      occupied_space_projector.data(),
      occupied_space_projector.data() + occupied_space_projector.size());
  result.inactive_auxiliary_transform.assign(
      inactive_auxiliary_transform.data(),
      inactive_auxiliary_transform.data() + inactive_auxiliary_transform.size());
  result.inactive_active_overlap_matrix.assign(
      inactive_active_overlap_matrix.data(),
      inactive_active_overlap_matrix.data() + inactive_active_overlap_matrix.size());
  result.projected_active_overlap_matrix.assign(
      projected_active_overlap_matrix.data(),
      projected_active_overlap_matrix.data() + projected_active_overlap_matrix.size());
  result.auxiliary_orbital_inverse_matrix.assign(
      auxiliary_orbital_inverse.data(),
      auxiliary_orbital_inverse.data() + auxiliary_orbital_inverse.size());
  return result;
}

}  // namespace xmvb::vb
