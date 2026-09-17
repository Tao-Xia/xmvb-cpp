#include <array>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/builder.hpp"
#include "vbscf/integrals/active/two_electron/transformation/ao_pair_operator.hpp"
#include "vbscf/integrals/ao/pairs/two_electron_index.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  constexpr int n_active = 11;

  const xmvb::vb::AoPairGraph ordered_graph =
      xmvb::vb::build_ao_pair_graph(
          {2, 1, 2}, {0, 1, 2}, {1.25, -0.5, 3.0}, n_active);
  constexpr std::array<int, 3> expected_left = {2, 1, 2};
  constexpr std::array<int, 3> expected_right = {0, 1, 2};
  constexpr std::array<double, 3> expected_values = {1.25, -0.5, 3.0};
  require(ordered_graph.integral_count() == 3, "AO-pair graph lost an integral");
  for (std::size_t index = 0; index < ordered_graph.integral_count(); ++index) {
    const xmvb::vb::AoPairIntegral integral = ordered_graph.integral(index);
    require(
        integral.left_pair == expected_left[index] &&
            integral.right_pair == expected_right[index] &&
            integral.value == expected_values[index],
        "AO-pair graph changed unique-integral order");
  }

  // The large-AO representation keeps the canonical unique stream and omits
  // the duplicated symmetric CSR edges.  Its block action must remain
  // numerically identical to the fast CSR representation.
  constexpr int compact_n_bf = 2;
  const xmvb::vb::AoPairGraph csr_graph =
      xmvb::vb::build_ao_pair_graph(
          {0, 1, 2}, {0, 0, 1}, {2.0, -0.25, 0.75}, compact_n_bf);
  xmvb::vb::AoPairGraph compact_graph;
  compact_graph.integral_rows = {0, 1, 2};
  compact_graph.integral_columns = {0, 0, 1};
  compact_graph.integral_values = {2.0, -0.25, 0.75};
  compact_graph.pair_first = csr_graph.pair_first;
  compact_graph.pair_second = csr_graph.pair_second;

  xmvb::vb::ExactCtxPairMatrix coefficients(3, 2);
  coefficients << 0.5, -1.0,
                  2.0, 0.25,
                 -0.5, 1.5;
  xmvb::vb::AoIntegralInput csr_input;
  csr_input.n_basis_functions = compact_n_bf;
  csr_input.pair_graph = csr_graph;
  xmvb::vb::AoIntegralInput compact_input;
  compact_input.n_basis_functions = compact_n_bf;
  compact_input.pair_graph = compact_graph;
  xmvb::vb::ExactCtxPairMatrix csr_products;
  xmvb::vb::ExactCtxPairMatrix compact_products;
  xmvb::vb::detail::apply_exact_ao_pair_kernel(
      csr_input, coefficients, compact_n_bf, 2, &csr_products);
  xmvb::vb::detail::apply_exact_ao_pair_kernel(
      compact_input, coefficients, compact_n_bf, 2, &compact_products);
  require(
      compact_products.isApprox(csr_products, 1.0e-14),
      "compact unique-integral action differs from symmetric CSR action");

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
  integrals.pair_graph =
      xmvb::vb::build_ao_pair_graph({0}, {0}, {2.0}, n_active);

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
