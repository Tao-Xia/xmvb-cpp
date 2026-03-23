#include "vb/orbital/legacy_style_orbital_gradient_projector.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}  // namespace

LegacyStyleOrbitalGradientProjectionResult LegacyStyleOrbitalGradientProjector::project(
    const std::vector<double>& active_active_gradient_matrix,
    const std::vector<double>& active_virtual_gradient_matrix,
    const std::vector<double>& active_space_coulomb_exchange_matrix,
    const std::vector<double>& overlap_response_matrix,
    const std::vector<double>& active_density_matrix,
    const std::vector<double>& basis_overlap_matrix,
    const std::vector<double>& ao_effective_one_electron_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_inactive_doubly_occupied_orbitals,
    int n_virtual_orbitals,
    double weight) const {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_active_orbitals = orbital_preparation_input.n_active_orbitals;
  if (n_basis_functions <= 0 || n_orbitals <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("invalid orbital dimensions");
  }

  const std::size_t ao_matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  const std::size_t active_matrix_size =
      static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals;
  if (active_active_gradient_matrix.size() != active_matrix_size ||
      active_virtual_gradient_matrix.size() != ao_matrix_size ||
      active_space_coulomb_exchange_matrix.size() != ao_matrix_size ||
      overlap_response_matrix.size() != ao_matrix_size ||
      active_density_matrix.size() != ao_matrix_size ||
      basis_overlap_matrix.size() != ao_matrix_size ||
      ao_effective_one_electron_matrix.size() != ao_matrix_size ||
      orbital_preparation_result.occupied_space_projector.size() != ao_matrix_size ||
      orbital_preparation_result.auxiliary_orbital_inverse_matrix.size() != ao_matrix_size) {
    throw std::invalid_argument("legacy-style projector matrix size mismatch");
  }

  const Eigen::Map<const ColumnMajorMatrixXd> grda(
      active_active_gradient_matrix.data(), n_active_orbitals, n_active_orbitals);
  const Eigen::Map<const ColumnMajorMatrixXd> grdv(
      active_virtual_gradient_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> g22(
      active_space_coulomb_exchange_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> q22(
      overlap_response_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> p22(
      active_density_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> ssf(
      basis_overlap_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> f11(
      ao_effective_one_electron_matrix.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> a1(
      orbital_preparation_result.occupied_space_projector.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> a5(
      orbital_preparation_result.auxiliary_orbital_inverse_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  const Eigen::Map<const ColumnMajorMatrixXd> a2(
      orbital_preparation_result.inactive_active_overlap_matrix.data(),
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Eigen::Map<const ColumnMajorMatrixXd> a3(
      orbital_preparation_result.inactive_auxiliary_transform.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const ColumnMajorMatrixXd> a4(
      orbital_preparation_result.projected_active_overlap_matrix.data(),
      n_basis_functions,
      n_active_orbitals);

  ColumnMajorMatrixXd gradient_auxiliary =
      ColumnMajorMatrixXd::Zero(n_basis_functions, n_active_orbitals);
  for (int active_orbital = 0; active_orbital < n_active_orbitals; ++active_orbital) {
    for (int basis_row = 0; basis_row < n_basis_functions; ++basis_row) {
      for (int active_source = 0; active_source < n_active_orbitals; ++active_source) {
        gradient_auxiliary(basis_row, active_orbital) +=
            grda(active_source, active_orbital) *
            a5(active_source + n_inactive_doubly_occupied_orbitals, basis_row);
      }
      for (int virtual_source = 0; virtual_source < n_virtual_orbitals; ++virtual_source) {
        gradient_auxiliary(basis_row, active_orbital) +=
            grdv(virtual_source, active_orbital) *
            a5(virtual_source + n_orbitals, basis_row);
      }
    }
  }

  ColumnMajorMatrixXd slot_gradient_matrix =
      ColumnMajorMatrixXd::Zero(n_basis_functions, n_orbitals);

  for (int active_orbital = 0; active_orbital < n_active_orbitals; ++active_orbital) {
    const int orbital_index = active_orbital + n_inactive_doubly_occupied_orbitals;
    const int coefficient_count =
        orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
    if (coefficient_count == 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const int basis_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_slot] -
          1;
      for (int basis_row = 0; basis_row < n_basis_functions; ++basis_row) {
        slot_gradient_matrix(coefficient_slot, orbital_index) +=
            gradient_auxiliary(basis_row, active_orbital) * a1(basis_row, basis_index);
      }
    }
  }

  ColumnMajorMatrixXd symmetrized_g22 = g22;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      symmetrized_g22(row, column) += f11(row, column);
      symmetrized_g22(row, column) += symmetrized_g22(row, column);
      symmetrized_g22(column, row) = symmetrized_g22(row, column);
    }
  }
  const ColumnMajorMatrixXd tmp1 = symmetrized_g22 * a3.leftCols(n_inactive_doubly_occupied_orbitals);
  const ColumnMajorMatrixXd tmp2 = a1.transpose() * tmp1;
  for (int inactive_orbital = 0; inactive_orbital < n_inactive_doubly_occupied_orbitals; ++inactive_orbital) {
    const int coefficient_count =
        orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(inactive_orbital)];
    if (coefficient_count == 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const int basis_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(inactive_orbital) * n_basis_functions + coefficient_slot] -
          1;
      slot_gradient_matrix(coefficient_slot, inactive_orbital) =
          tmp2(basis_index, inactive_orbital) + tmp2(basis_index, inactive_orbital);
    }
  }

  const ColumnMajorMatrixXd inactive_correction_from_density =
      ssf * p22 * f11 * a3.leftCols(n_inactive_doubly_occupied_orbitals);
  const ColumnMajorMatrixXd inactive_correction_from_overlap =
      ssf * q22 * a3.leftCols(n_inactive_doubly_occupied_orbitals);
  for (int inactive_orbital = 0; inactive_orbital < n_inactive_doubly_occupied_orbitals; ++inactive_orbital) {
    const int coefficient_count =
        orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(inactive_orbital)];
    if (coefficient_count == 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const int basis_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(inactive_orbital) * n_basis_functions + coefficient_slot] -
          1;
      slot_gradient_matrix(coefficient_slot, inactive_orbital) -=
          2.0 * inactive_correction_from_density(basis_index, inactive_orbital);
      slot_gradient_matrix(coefficient_slot, inactive_orbital) -=
          2.0 * inactive_correction_from_overlap(basis_index, inactive_orbital);
    }
  }

  for (int inactive_orbital = 0; inactive_orbital < n_inactive_doubly_occupied_orbitals; ++inactive_orbital) {
    const int coefficient_count =
        orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(inactive_orbital)];
    if (coefficient_count == 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const int basis_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(inactive_orbital) * n_basis_functions + coefficient_slot] -
          1;
      for (int active_orbital = 0; active_orbital < n_active_orbitals; ++active_orbital) {
        for (int basis_row = 0; basis_row < n_basis_functions; ++basis_row) {
          const double coupling =
              a1(basis_row, basis_index) * a2(inactive_orbital, active_orbital) +
              a3(basis_row, inactive_orbital) * a4(basis_index, active_orbital);
          slot_gradient_matrix(coefficient_slot, inactive_orbital) -=
              gradient_auxiliary(basis_row, active_orbital) * coupling;
        }
      }
    }
  }

  std::vector<double> parameter_gradient;
  for (int orbital_index = 0;
       orbital_index < n_inactive_doubly_occupied_orbitals + n_active_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
    if (coefficient_count == 0 || coefficient_count == 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      parameter_gradient.push_back(weight * slot_gradient_matrix(coefficient_slot, orbital_index));
    }
  }

  LegacyStyleOrbitalGradientProjectionResult result;
  result.parameter_gradient = std::move(parameter_gradient);
  result.slot_gradient_matrix.assign(
      slot_gradient_matrix.data(),
      slot_gradient_matrix.data() + slot_gradient_matrix.size());
  return result;
}

}  // namespace xmvb::vb
