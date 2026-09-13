#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/builder.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  constexpr int n_active = 11;

  xmvb::vb::OrbitalPreparationResult orbitals;
  orbitals.active_sparse_row_offsets.resize(n_active + 1);
  orbitals.active_sparse_orbital_indices.resize(n_active);
  orbitals.active_sparse_values.assign(n_active, 1.0);
  for (int index = 0; index < n_active; ++index) {
    orbitals.active_sparse_row_offsets[index] = index;
    orbitals.active_sparse_orbital_indices[index] = index;
  }
  orbitals.active_sparse_row_offsets[n_active] = n_active;

  xmvb::vb::AoIntegralInput integrals;
  integrals.n_basis_functions = n_active;
  integrals.ao_two_electron_integral_values = {2.0};
  integrals.ao_two_electron_integral_indices = {0, 0, 0, 0};

  const xmvb::vb::ActiveSpaceTwoElectronBuilder builder;
  const auto result = builder.build(integrals, orbitals, n_active);

  require(
      result.dense_active_coefficients.rows() == n_active &&
          result.dense_active_coefficients.cols() == n_active,
      "sparse active-ERI path omitted dense active coefficients");
  require(
      result.dense_active_coefficients.isApprox(
          Eigen::MatrixXd::Identity(n_active, n_active), 0.0),
      "sparse active-ERI path changed active coefficients");
  require(
      !result.packed_active_two_electron_integrals.empty() &&
          std::abs(result.packed_active_two_electron_integrals.front() - 2.0) <
              1.0e-14,
      "sparse active-ERI contraction returned the wrong integral");
  require(
      result.dense_ao_pair_products.size() == 0,
      "memory-bounded active-ERI path retained dense AO-pair products");
  return 0;
}
