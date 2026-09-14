#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"

namespace {

using xmvb::vb::OrbitalPreparationInput;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

OrbitalPreparationInput make_input(const Eigen::MatrixXd& orbitals) {
  OrbitalPreparationInput input;
  input.n_basis_functions = 3;
  input.n_orbitals = 2;
  input.n_active_orbitals = 0;
  input.n_total_electrons = 4;
  input.n_active_electrons = 0;
  input.orbital_type = 1;
  input.ao_overlap_matrix = Eigen::MatrixXd::Identity(3, 3);
  input.orbital_basis_counts = {2, 2};
  input.orbital_basis_index_table = {1, 2, 0, 1, 2, 0};
  input.orbital_value_table.assign(6, 0.0);
  for (int orbital = 0; orbital < 2; ++orbital) {
    input.orbital_value_table[3 * orbital] = orbitals(0, orbital);
    input.orbital_value_table[3 * orbital + 1] = orbitals(1, orbital);
  }
  return input;
}

Eigen::MatrixXd dense_inactive(const OrbitalPreparationInput& input) {
  Eigen::MatrixXd orbitals = Eigen::MatrixXd::Zero(3, 2);
  for (int orbital = 0; orbital < 2; ++orbital) {
    for (int coefficient = 0; coefficient < 2; ++coefficient) {
      const int slot = 3 * orbital + coefficient;
      orbitals(input.orbital_basis_index_table[slot] - 1, orbital) =
          input.orbital_value_table[slot];
    }
  }
  return orbitals;
}

double metric_condition(const Eigen::MatrixXd& orbitals) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      orbitals.transpose() * orbitals);
  require(solver.info() == Eigen::Success, "failed to diagonalize test metric");
  return solver.eigenvalues().maxCoeff() / solver.eigenvalues().minCoeff();
}

void check_ill_conditioned_sparse_gauge() {
  Eigen::MatrixXd orbitals = Eigen::MatrixXd::Zero(3, 2);
  orbitals.col(0) << 1.0, 0.0, 0.0;
  orbitals.col(1) << 1.0, 1.0e-5, 0.0;
  OrbitalPreparationInput input = make_input(orbitals);
  const Eigen::MatrixXd before = dense_inactive(input);
  const double condition_before = metric_condition(before);

  const bool chart_changed =
      xmvb::vb::apply_support_preserving_inactive_gauge(&input);
  require(chart_changed, "ill-conditioned gauge was not balanced");
  require(input.maintain_inactive_gauge,
          "selected sparse gauge section was not retained");

  const Eigen::MatrixXd after = dense_inactive(input);
  require(after.row(2).isZero(0.0),
          "gauge balancing introduced a forbidden AO coefficient");
  require(metric_condition(after) < 1.0e-8 * condition_before,
          "gauge balancing did not remove avoidable ill-conditioning");

  const Eigen::MatrixXd before_basis =
      before.householderQr().householderQ() *
      Eigen::MatrixXd::Identity(3, 2);
  const Eigen::MatrixXd after_basis =
      after.householderQr().householderQ() *
      Eigen::MatrixXd::Identity(3, 2);
  require((before_basis * before_basis.transpose() -
           after_basis * after_basis.transpose()).norm() < 1.0e-10,
          "gauge balancing changed the inactive occupied subspace");

  // A later accepted point can be well-conditioned yet leave the selected
  // section. Once enabled, canonicalization must remain part of the chart.
  input.orbital_value_table[3] += 0.2 * input.orbital_value_table[0];
  input.orbital_value_table[4] += 0.2 * input.orbital_value_table[1];
  const bool continued =
      xmvb::vb::apply_support_preserving_inactive_gauge(&input);
  require(continued,
          "accepted point was allowed to drift off the selected gauge section");
  require(metric_condition(dense_inactive(input)) < 2.0,
          "maintained gauge section is poorly conditioned");
}

void check_well_conditioned_gauge_is_unchanged() {
  Eigen::MatrixXd orbitals = Eigen::MatrixXd::Zero(3, 2);
  orbitals(0, 0) = 1.0;
  orbitals(1, 1) = 1.0;
  OrbitalPreparationInput input = make_input(orbitals);
  const std::vector<double> values_before = input.orbital_value_table;
  const bool chart_changed =
      xmvb::vb::apply_support_preserving_inactive_gauge(&input);
  require(!chart_changed,
          "well-conditioned sparse gauge was changed unnecessarily");
  require(input.orbital_value_table == values_before,
          "well-conditioned orbital coefficients were modified");
}

}  // namespace

int main() {
  try {
    check_ill_conditioned_sparse_gauge();
    check_well_conditioned_gauge_is_unchanged();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
