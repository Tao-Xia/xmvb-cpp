#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"

#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

using ColumnMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}  // namespace

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& ao_effective_one_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
  if (ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument("ao_effective_one_electron_gradient size mismatch");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  const Eigen::Map<const ColumnMajorMatrixXd> ao_effective_gradient(
      ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  const ColumnMajorMatrixXd unsymmetrized_g11_gradient =
      ao_effective_gradient + ao_effective_gradient.transpose();
  ColumnMajorMatrixXd inactive_density_gradient =
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

    inactive_density_gradient(i, j) +=
        unsymmetrized_g11_gradient(k, l) * two_electron_value * 4.0;
    inactive_density_gradient(k, l) +=
        unsymmetrized_g11_gradient(i, j) * two_electron_value * 4.0;
    inactive_density_gradient(l, j) -=
        unsymmetrized_g11_gradient(i, k) * two_electron_value;
    inactive_density_gradient(k, i) -=
        unsymmetrized_g11_gradient(j, l) * two_electron_value;
    inactive_density_gradient(k, j) -=
        unsymmetrized_g11_gradient(i, l) * two_electron_value;
    inactive_density_gradient(l, i) -=
        unsymmetrized_g11_gradient(j, k) * two_electron_value;
  }

  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient.assign(
      inactive_density_gradient.data(),
      inactive_density_gradient.data() + inactive_density_gradient.size());
  return result;
}

}  // namespace xmvb::vb
