#include "vb/orbital/ao_effective_one_electron_builder.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}  // namespace

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("AO matrix sizes do not match n_basis_functions");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  Eigen::Map<const ColumnMajorMatrixXd> p11(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  Eigen::Map<const ColumnMajorMatrixXd> hhf(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  ColumnMajorMatrixXd g11 =
      ColumnMajorMatrixXd::Zero(n_basis_functions, n_basis_functions);

  for (std::size_t integral_index = 0;
       integral_index < ao_two_electron_integral_values.size();
       ++integral_index) {
    double two_electron_value = ao_two_electron_integral_values[integral_index];
    const int i = ao_two_electron_integral_indices[integral_index * 4];
    const int j = ao_two_electron_integral_indices[integral_index * 4 + 1];
    const int k = ao_two_electron_integral_indices[integral_index * 4 + 2];
    const int l = ao_two_electron_integral_indices[integral_index * 4 + 3];

    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      throw std::invalid_argument("AO two-electron index out of range");
    }

    if (i == j) {
      two_electron_value *= 0.5;
    }
    if (k == l) {
      two_electron_value *= 0.5;
    }
    if (i == k && j == l) {
      two_electron_value *= 0.5;
    }

    const double a0 = p11(i, j) * two_electron_value * 4.0;
    const double a1 = p11(k, l) * two_electron_value * 4.0;
    g11(i, j) += a1;
    g11(k, l) += a0;
    g11(i, k) -= p11(l, j) * two_electron_value;
    g11(j, l) -= p11(k, i) * two_electron_value;
    g11(i, l) -= p11(k, j) * two_electron_value;
    g11(j, k) -= p11(l, i) * two_electron_value;
  }

  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column <= row; ++column) {
      g11(row, column) = g11(row, column) + g11(column, row);
      g11(column, row) = g11(row, column);
    }
  }

  const ColumnMajorMatrixXd f11 = g11 + hhf;

  AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix.assign(
      g11.data(),
      g11.data() + g11.size());
  result.ao_effective_one_electron_matrix.assign(
      f11.data(),
      f11.data() + f11.size());
  return result;
}

}  // namespace xmvb::vb
