#include "vb/orbital/active_space_orbital_backpropagator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

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
    const Eigen::Map<const ColumnMajorMatrixXd>& basis_overlap_matrix,
    std::vector<double>* squared_norms_out) {
  std::vector<double> normalized_values = input.orbital_value_table;
  squared_norms_out->assign(static_cast<std::size_t>(input.n_orbitals), 0.0);

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
            left_value * right_value * basis_overlap_matrix(left_basis_function, right_basis_function);
      }
    }

    if (!std::isfinite(squared_norm) ||
        squared_norm <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("orbital normalization failed during backpropagation");
    }

    (*squared_norms_out)[static_cast<std::size_t>(orbital_index)] = squared_norm;
    const double normalization_factor = std::sqrt(1.0 / squared_norm);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      normalized_values[static_cast<std::size_t>(orbital_index) * input.n_basis_functions +
                        coefficient_index] *= normalization_factor;
    }
  }

  return normalized_values;
}

ColumnMajorMatrixXd expand_sparse_orbitals(
    const OrbitalPreparationInput& input,
    const std::vector<double>& normalized_orbital_values) {
  ColumnMajorMatrixXd orbital_matrix =
      ColumnMajorMatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

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
      orbital_matrix(basis_function_index, orbital_index) =
          normalized_orbital_values[static_cast<std::size_t>(orbital_index) *
                                        input.n_basis_functions +
                                    coefficient_index];
    }
  }

  return orbital_matrix;
}

}  // namespace

ActiveSpaceOrbitalBackpropagationResult ActiveSpaceOrbitalBackpropagator::backpropagate(
    const std::vector<double>& auxiliary_orbital_gradient,
    const std::vector<double>& active_orbital_overlap_gradient,
    const std::vector<double>& inactive_density_gradient,
    const OrbitalPreparationInput& input) const {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 || input.n_active_orbitals <= 0) {
    throw std::invalid_argument("orbital preparation input dimensions must be positive");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const std::size_t auxiliary_matrix_size =
      static_cast<std::size_t>(input.n_basis_functions) * input.n_basis_functions;
  const std::size_t active_matrix_size =
      static_cast<std::size_t>(input.n_active_orbitals) * input.n_active_orbitals;
  if (auxiliary_orbital_gradient.size() != auxiliary_matrix_size) {
    throw std::invalid_argument("auxiliary_orbital_gradient size mismatch");
  }
  if (active_orbital_overlap_gradient.size() != active_matrix_size) {
    throw std::invalid_argument("active_orbital_overlap_gradient size mismatch");
  }
  if (inactive_density_gradient.size() != auxiliary_matrix_size) {
    throw std::invalid_argument("inactive_density_gradient size mismatch");
  }

  const Eigen::Map<const ColumnMajorMatrixXd> basis_overlap_matrix(
      input.basis_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> auxiliary_gradient_matrix(
      auxiliary_orbital_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> active_overlap_gradient_matrix(
      active_orbital_overlap_gradient.data(),
      input.n_active_orbitals,
      input.n_active_orbitals);
  const Eigen::Map<const ColumnMajorMatrixXd> inactive_density_gradient_matrix(
      inactive_density_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  std::vector<double> squared_norms;
  const std::vector<double> normalized_orbital_values =
      normalize_sparse_orbitals(input, basis_overlap_matrix, &squared_norms);
  const ColumnMajorMatrixXd original_orbital_matrix =
      expand_sparse_orbitals(input, normalized_orbital_values);
  const auto inactive_orbitals = original_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
  const auto active_orbitals = original_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);

  ColumnMajorMatrixXd inactive_density_matrix =
      ColumnMajorMatrixXd::Zero(input.n_basis_functions, input.n_basis_functions);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const ColumnMajorMatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap_matrix * inactive_orbitals;
    const ColumnMajorMatrixXd inactive_overlap_inverse = inactive_overlap.inverse();
    inactive_density_matrix =
        inactive_orbitals * inactive_overlap_inverse * inactive_orbitals.transpose();
  }
  const ColumnMajorMatrixXd occupied_space_projector =
      ColumnMajorMatrixXd::Identity(input.n_basis_functions, input.n_basis_functions) -
      inactive_density_matrix * basis_overlap_matrix;
  const ColumnMajorMatrixXd active_auxiliary_orbitals =
      occupied_space_projector * active_orbitals;

  const ColumnMajorMatrixXd active_overlap_gradient_symmetric =
      active_overlap_gradient_matrix + active_overlap_gradient_matrix.transpose();
  const ColumnMajorMatrixXd active_auxiliary_gradient =
      auxiliary_gradient_matrix.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals) +
      basis_overlap_matrix * active_auxiliary_orbitals * active_overlap_gradient_symmetric;

  ColumnMajorMatrixXd original_orbital_gradient =
      ColumnMajorMatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = active_auxiliary_gradient;
  } else {
    const ColumnMajorMatrixXd bs_active = basis_overlap_matrix * active_orbitals;
    const ColumnMajorMatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap_matrix * inactive_orbitals;
    const ColumnMajorMatrixXd inactive_overlap_inverse = inactive_overlap.inverse();

    const ColumnMajorMatrixXd original_active_gradient =
        active_auxiliary_gradient -
        basis_overlap_matrix * inactive_density_matrix * active_auxiliary_gradient;
    const ColumnMajorMatrixXd total_inactive_density_gradient =
        inactive_density_gradient_matrix -
        active_auxiliary_gradient * bs_active.transpose();
    const ColumnMajorMatrixXd inactive_density_gradient_symmetric =
        total_inactive_density_gradient + total_inactive_density_gradient.transpose();
    const ColumnMajorMatrixXd inactive_overlap_inverse_gradient =
        inactive_orbitals.transpose() * total_inactive_density_gradient * inactive_orbitals;
    const ColumnMajorMatrixXd inactive_overlap_gradient =
        -inactive_overlap_inverse *
        inactive_overlap_inverse_gradient *
        inactive_overlap_inverse;
    const ColumnMajorMatrixXd original_inactive_gradient =
        inactive_density_gradient_symmetric * inactive_orbitals * inactive_overlap_inverse +
        basis_overlap_matrix * inactive_orbitals *
            (inactive_overlap_gradient + inactive_overlap_gradient.transpose());

    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = original_active_gradient;
  }

  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count = get_sparse_coefficient_count(
        input.orbital_basis_counts,
        input.orbital_basis_index_table,
        input.n_basis_functions,
        orbital_index);

    std::vector<int> basis_function_indices(static_cast<std::size_t>(coefficient_count), 0);
    Eigen::VectorXd normalized_vector = Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd dense_gradient = Eigen::VectorXd::Zero(coefficient_count);

    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[static_cast<std::size_t>(orbital_index) *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      basis_function_indices[static_cast<std::size_t>(coefficient_index)] = basis_function_index;
      normalized_vector(coefficient_index) =
          normalized_orbital_values[static_cast<std::size_t>(orbital_index) *
                                        input.n_basis_functions +
                                    coefficient_index];
      dense_gradient(coefficient_index) =
          original_orbital_gradient(basis_function_index, orbital_index);
    }

    Eigen::MatrixXd overlap_submatrix =
        Eigen::MatrixXd::Zero(coefficient_count, coefficient_count);
    for (int row = 0; row < coefficient_count; ++row) {
      for (int column = 0; column < coefficient_count; ++column) {
        overlap_submatrix(row, column) = basis_overlap_matrix(
            basis_function_indices[static_cast<std::size_t>(row)],
            basis_function_indices[static_cast<std::size_t>(column)]);
      }
    }

    const double normalization_factor =
        std::sqrt(1.0 / squared_norms[static_cast<std::size_t>(orbital_index)]);
    const double scalar_term =
        dense_gradient.dot(normalized_vector);
    const Eigen::VectorXd raw_gradient =
        normalization_factor * dense_gradient -
        normalization_factor * scalar_term * (overlap_submatrix * normalized_vector);

    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      orbital_value_gradient[static_cast<std::size_t>(orbital_index) * input.n_basis_functions +
                             coefficient_index] = raw_gradient(coefficient_index);
    }
  }

  ActiveSpaceOrbitalBackpropagationResult result;
  result.orbital_value_gradient = std::move(orbital_value_gradient);
  return result;
}

}  // namespace xmvb::vb
