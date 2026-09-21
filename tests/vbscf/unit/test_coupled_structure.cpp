#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct StructureFixture {
  xmvb::vb::StructureAction action;
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
};

StructureFixture make_fixture() {
  constexpr int n_orbitals = 2;
  constexpr int n_structures = 4;
  const std::vector<std::vector<int>> alpha_strings{{0}, {1}};
  const std::vector<std::vector<int>> beta_strings{{0}, {1}};

  std::vector<std::vector<int>> determinant_alpha;
  std::vector<std::vector<int>> determinant_beta;
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> expansion(
      n_structures);
  for (int alpha = 0; alpha < 2; ++alpha) {
    for (int beta = 0; beta < 2; ++beta) {
      const int determinant = 2 * alpha + beta;
      determinant_alpha.push_back(alpha_strings[alpha]);
      determinant_beta.push_back(beta_strings[beta]);
      expansion[determinant].push_back({determinant, 1.0});
    }
  }
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      determinant_alpha, determinant_beta, n_orbitals);

  Eigen::Matrix2d active_overlap;
  active_overlap << 1.0, 0.18,
                    0.18, 1.24;
  const std::vector<double> packed_overlap(
      active_overlap.data(), active_overlap.data() + active_overlap.size());
  Eigen::Matrix2d one_electron;
  one_electron << -1.1, 0.16,
                   0.16, -0.45;

  const int n_pairs = xmvb::vb::packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd pair_factor(3, n_pairs);
  pair_factor << 0.31, 0.07, -0.02,
                 0.04, 0.26,  0.08,
                -0.03, 0.05,  0.22;
  const Eigen::MatrixXd pair_kernel = pair_factor.transpose() * pair_factor;
  xmvb::vb::ActiveSpaceTwoElectronResult two_electron;
  two_electron.packed_active_two_electron_integrals.resize(
      xmvb::vb::packed_active_two_electron_integral_count(n_orbitals));
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row <= column; ++row) {
      two_electron.packed_active_two_electron_integrals[
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              row, column)] = pair_kernel(row, column);
    }
  }

  xmvb::vb::StructureDiagonal diagonal;
  diagonal.hamiltonian = Eigen::VectorXd::Zero(n_structures);
  diagonal.overlap = Eigen::VectorXd::Ones(n_structures);
  xmvb::vb::StructureAction action(
      expansion,
      n_structures,
      topology,
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals,
      &diagonal);

  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(n_structures, n_structures);
  const xmvb::vb::StructureActionResult matrices = action.apply(identity);
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      matrices.hamiltonian, matrices.overlap);
  require(eigensolver.info() == Eigen::Success,
          "fixture generalized eigensolve failed");
  return StructureFixture{
      std::move(action),
      matrices.hamiltonian,
      matrices.overlap,
      eigensolver.eigenvalues(),
      eigensolver.eigenvectors()};
}

std::shared_ptr<xmvb::vb::AcceptedPointContext> accepted_point(
    const StructureFixture& fixture,
    int n_states,
    std::vector<double> weights) {
  auto accepted = std::make_shared<xmvb::vb::AcceptedPointContext>();
  accepted->n_structures = static_cast<int>(fixture.hamiltonian.rows());
  accepted->selected_state_indices.resize(n_states);
  accepted->selected_state_energies.resize(n_states);
  for (int state = 0; state < n_states; ++state) {
    accepted->selected_state_indices[state] = state;
    accepted->selected_state_energies[state] = fixture.eigenvalues[state];
  }
  accepted->normalized_state_weights = std::move(weights);
  accepted->selected_state_eigenvectors =
      fixture.eigenvectors.leftCols(n_states);
  return accepted;
}

double block_dot(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right) {
  return (left.array() * right.array()).sum();
}

void check_single_state(const StructureFixture& fixture) {
  const auto accepted = accepted_point(fixture, 1, {1.0});
  const xmvb::vb::StructureTangentOperator structure(
      accepted, fixture.action);
  require(structure.n_structures() == 4 && structure.n_states() == 1,
          "single-state coupled structure dimensions are wrong");

  Eigen::Vector4d raw;
  raw << 0.7, -0.4, 0.25, 0.8;
  const xmvb::vb::StructureTangent tangent = structure.project(raw);
  const Eigen::MatrixXd overlap_tangent =
      fixture.overlap * tangent.scaled_coefficients;
  require((accepted->selected_state_eigenvectors.transpose() *
           overlap_tangent).norm() < 2.0e-13,
          "projected structure tangent is not horizontal");

  const Eigen::MatrixXd coefficient_response = 0.6 * raw;
  const xmvb::vb::StructureTangent scaled =
      structure.from_coefficient_response(coefficient_response);
  const Eigen::MatrixXd expected_scaled =
      structure.project(std::sqrt(2.0) * coefficient_response)
          .scaled_coefficients;
  require((scaled.scaled_coefficients - expected_scaled).norm() < 2.0e-13,
          "coefficient response was not scaled by sqrt(2w)");
  const Eigen::MatrixXd round_trip = structure.coefficient_response(scaled);
  const Eigen::MatrixXd expected_response =
      structure.project(coefficient_response).scaled_coefficients;
  require((round_trip - expected_response).norm() < 2.0e-13,
          "scaled structure coordinate round trip is inconsistent");

  const double expected_norm = block_dot(
      tangent.scaled_coefficients,
      fixture.overlap * tangent.scaled_coefficients);
  require(std::abs(structure.squared_norm(tangent) - expected_norm) < 2.0e-13,
          "structure physical norm differs from z^T S z");
  require(expected_norm > 0.0, "test structure tangent has zero physical norm");
}

void check_equal_weight_multistate(const StructureFixture& fixture) {
  const auto accepted = accepted_point(fixture, 2, {0.5, 0.5});
  const xmvb::vb::StructureTangentOperator structure(
      accepted, fixture.action);
  Eigen::Matrix<double, 4, 2> x;
  x << 0.2, -0.5,
       0.7,  0.1,
      -0.3,  0.8,
       0.6, -0.2;
  Eigen::Matrix<double, 4, 2> y;
  y << -0.4,  0.3,
        0.5, -0.7,
        0.9,  0.2,
       -0.1,  0.6;
  const auto horizontal_x = structure.project(x);
  const auto horizontal_y = structure.project(y);
  require((accepted->selected_state_eigenvectors.transpose() *
           fixture.overlap * horizontal_x.scaled_coefficients).norm() < 3.0e-13,
          "multistate tangent retains a selected-subspace component");

  const auto hx = structure.apply_hessian(horizontal_x);
  const auto hy = structure.apply_hessian(horizontal_y);
  const double h_xy = block_dot(horizontal_x.scaled_coefficients,
                                hy.scaled_coefficients);
  const double h_yx = block_dot(hx.scaled_coefficients,
                                horizontal_y.scaled_coefficients);
  require(std::abs(h_xy - h_yx) <
              2.0e-12 * std::max({1.0, std::abs(h_xy), std::abs(h_yx)}),
          "coupled structure Hessian is not symmetric");

  const auto mx = structure.apply_metric(horizontal_x);
  const auto my = structure.apply_metric(horizontal_y);
  const double m_xy = block_dot(horizontal_x.scaled_coefficients,
                                my.scaled_coefficients);
  const double m_yx = block_dot(mx.scaled_coefficients,
                                horizontal_y.scaled_coefficients);
  require(std::abs(m_xy - m_yx) <
              2.0e-12 * std::max({1.0, std::abs(m_xy), std::abs(m_yx)}),
          "coupled structure metric is not symmetric");
  require(std::abs(structure.squared_norm(horizontal_x) -
                   block_dot(horizontal_x.scaled_coefficients,
                             fixture.overlap *
                                 horizontal_x.scaled_coefficients)) < 3.0e-13,
          "multistate physical norm is inconsistent");
}

void check_unequal_weights_rejected(const StructureFixture& fixture) {
  const auto accepted = accepted_point(fixture, 2, {0.6, 0.4});
  bool rejected = false;
  try {
    (void)xmvb::vb::StructureTangentOperator(accepted, fixture.action);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unequal multistate structure tangent was accepted");
}

}  // namespace

int main() {
  try {
    const StructureFixture fixture = make_fixture();
    check_single_state(fixture);
    check_equal_weight_multistate(fixture);
    check_unequal_weights_rejected(fixture);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
