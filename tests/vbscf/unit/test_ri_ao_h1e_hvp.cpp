#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/ao/one_electron/backpropagator.hpp"
#include "vbscf/integrals/ao/one_electron/builder.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"
#include "vbscf/integrals/ao/pairs/two_electron_index.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using xmvb::vb::RiAoFactorization;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::Ref<const Matrix>& actual,
    const Eigen::Ref<const Matrix>& expected,
    double tolerance,
    const std::string& message) {
  require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
          message + ": shape mismatch");
  const double error = (actual - expected).cwiseAbs().maxCoeff();
  require(error <= tolerance, message + ": max error = " + std::to_string(error));
}

RiAoFactorization make_ri_factorization() {
  RiAoFactorization ri;
  ri.n_basis_functions = 3;
  ri.n_auxiliary_functions = 2;
  ri.n_packed_ao_pairs = 6;
  ri.metric_whitened_ao_pair_factors.resize(2, 6);
  ri.metric_whitened_ao_pair_factors <<
      0.7, -0.2, 0.5, 0.1, 0.3, -0.4,
      -0.1, 0.6, 0.2, -0.5, 0.4, 0.8;
  return ri;
}

Matrix unpack_factor(const RiAoFactorization& ri, int auxiliary_index) {
  Matrix factor = Matrix::Zero(
      ri.n_basis_functions, ri.n_basis_functions);
  int packed_index = 0;
  for (int column = 0; column < ri.n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          ri.metric_whitened_ao_pair_factors(auxiliary_index, packed_index++);
      factor(row, column) = value;
      factor(column, row) = value;
    }
  }
  return factor;
}

Matrix reference_ri_action(
    const Eigen::Ref<const Matrix>& input,
    const RiAoFactorization& ri) {
  Matrix result = Matrix::Zero(input.rows(), input.cols());
  for (int auxiliary = 0;
       auxiliary < ri.n_auxiliary_functions;
       ++auxiliary) {
    const Matrix factor = unpack_factor(ri, auxiliary);
    const double projection = (factor.array() * input.array()).sum();
    result.noalias() += 2.0 * projection * factor;
    result.noalias() -= factor * input * factor;
  }
  return result;
}

Matrix map_square(const std::vector<double>& storage, int n) {
  require(storage.size() == static_cast<std::size_t>(n * n),
          "square storage size mismatch");
  return Eigen::Map<const Matrix>(storage.data(), n, n);
}

void check_raw_and_fused_operator() {
  const RiAoFactorization ri = make_ri_factorization();
  Matrix source(3, 3);
  source <<
      0.9, -0.3, 0.2,
      -0.3, -0.4, 0.7,
      0.2, 0.7, 0.5;
  Matrix adjoint(3, 3);
  adjoint <<
      -0.2, 0.1, 0.6,
      0.1, 0.8, -0.5,
      0.6, -0.5, 0.3;

  const Matrix reference_source = reference_ri_action(source, ri);
  const Matrix reference_adjoint = reference_ri_action(adjoint, ri);
  const std::vector<double> source_storage(
      source.data(), source.data() + source.size());
  const Matrix raw = map_square(
      xmvb::vb::apply_ao_effective_one_electron_ri_operator(
          source_storage,
          ri,
          3,
          {.attempt_spectral_factorization = false}),
      3);
  require_close(raw, reference_source, 2.0e-13,
                "raw RI operator is not the complete physical matrix");

  xmvb::vb::AoEffectiveOneElectronRiFusedWorkspace workspace;
  Matrix forward;
  Matrix transpose;
  xmvb::vb::apply_ao_effective_one_electron_ri_operator_fused(
      source, adjoint, ri, &workspace, &forward, &transpose);
  require_close(forward, reference_source, 2.0e-13,
                "fused RI forward action mismatch");
  require_close(transpose, reference_adjoint, 2.0e-13,
                "fused RI transpose action mismatch");
  require_close(forward, forward.transpose(), 2.0e-13,
                "fused RI forward result is not symmetric");

  const double adjoint_error = std::abs(
      (adjoint.array() * forward.array()).sum() -
      (transpose.array() * source.array()).sum());
  require(adjoint_error <= 5.0e-13,
          "RI operator is not self-adjoint on symmetric AO matrices");

  const Matrix second_source = -0.4 * source + 0.2 * adjoint;
  const Matrix second_adjoint = source - 0.7 * adjoint;
  xmvb::vb::apply_ao_effective_one_electron_ri_operator_fused(
      second_source, second_adjoint, ri, &workspace, &forward, &transpose);
  require_close(forward, reference_ri_action(second_source, ri), 2.0e-13,
                "reused fused RI workspace retained forward data");
  require_close(transpose, reference_ri_action(second_adjoint, ri), 2.0e-13,
                "reused fused RI workspace retained adjoint data");
}

void check_ri_builder_and_directional_action() {
  const RiAoFactorization ri = make_ri_factorization();
  Matrix density(3, 3);
  density <<
      0.8, 0.2, -0.1,
      0.2, 0.4, 0.3,
      -0.1, 0.3, 0.6;
  Matrix direction(3, 3);
  direction <<
      -0.3, 0.5, 0.2,
      0.5, 0.1, -0.4,
      0.2, -0.4, 0.7;
  Matrix core(3, 3);
  core <<
      -1.1, 0.2, 0.0,
      0.2, -0.7, -0.1,
      0.0, -0.1, -0.4;

  xmvb::vb::AoEffectiveOneElectronBuilder builder;
  const auto result = builder.build(density, core, ri, 3);
  const Matrix reference = reference_ri_action(density, ri);
  require_close(result.ao_coulomb_exchange_matrix, reference, 2.0e-13,
                "RI builder doubled the complete raw RI matrix");
  require_close(result.ao_effective_h1e, core + reference, 2.0e-13,
                "RI builder effective matrix mismatch");

  constexpr double step = 1.0e-6;
  const Matrix plus = builder.build(
      density + step * direction, core, ri, 3).ao_effective_h1e;
  const Matrix minus = builder.build(
      density - step * direction, core, ri, 3).ao_effective_h1e;
  require_close((plus - minus) / (2.0 * step),
                reference_ri_action(direction, ri),
                2.0e-9,
                "RI builder directional action mismatch");
}

void check_ri_backpropagation_adjoint_identity() {
  const RiAoFactorization ri = make_ri_factorization();
  Matrix direction(3, 3);
  direction <<
      0.4, -0.2, 0.5,
      -0.2, -0.6, 0.1,
      0.5, 0.1, 0.3;
  Matrix gradient(3, 3);
  gradient <<
      0.7, -0.4, 0.2,
      0.6, -0.1, 0.8,
      -0.3, 0.5, 0.9;

  xmvb::vb::AoEffectiveOneElectronBackpropagator backpropagator;
  const auto backward = backpropagator.backpropagate(gradient, ri, 3);
  const Eigen::Map<const Matrix> density_gradient(
      backward.inactive_density_gradient.data(), 3, 3);
  const double forward_pairing =
      (gradient.array() * reference_ri_action(direction, ri).array()).sum();
  const double reverse_pairing =
      (density_gradient.array() * direction.array()).sum();
  require(std::abs(forward_pairing - reverse_pairing) <= 5.0e-13,
          "RI backpropagator fails the builder adjoint identity");
}

void check_exact_builder_triangular_regression() {
  xmvb::vb::AoIntegralInput ao;
  ao.n_basis_functions = 2;
  ao.ao_core_hamiltonian_matrix.resize(2, 2);
  ao.ao_core_hamiltonian_matrix << -1.0, 0.15, 0.15, -0.6;
  ao.pair_graph = xmvb::vb::build_ao_pair_graph(
      {0, 0, 0, 1, 1, 2},
      {0, 1, 2, 1, 2, 2},
      {0.7, -0.2, 0.1, 0.5, 0.3, 0.9},
      2);
  Matrix density(2, 2);
  density << 0.8, -0.3, -0.3, 0.4;

  const std::vector<double> raw =
      xmvb::vb::apply_ao_h1e(density.data(), ao, 1);
  Matrix expected = Eigen::Map<const Matrix>(raw.data(), 2, 2);
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column <= row; ++column) {
      expected(row, column) += expected(column, row);
      expected(column, row) = expected(row, column);
    }
  }

  xmvb::vb::AoEffectiveOneElectronBuilder builder;
  const auto result = builder.build(density, ao);
  require_close(result.ao_coulomb_exchange_matrix, expected, 2.0e-13,
                "exact builder triangular finalization changed");
  require_close(result.ao_effective_h1e,
                ao.ao_core_hamiltonian_matrix + expected,
                2.0e-13,
                "exact builder effective matrix changed");
}

}  // namespace

int main() {
  try {
    check_raw_and_fused_operator();
    check_ri_builder_and_directional_action();
    check_ri_backpropagation_adjoint_identity();
    check_exact_builder_triangular_regression();
    std::cout << "RI AO-H1E fused HVP primitives: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
