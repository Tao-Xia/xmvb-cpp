#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include "input/deck/keywords.hpp"
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
  input.orbital_type = xmvb::vb::kOrbitalTypeHao;
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

OrbitalPreparationInput make_active_gauge_input() {
  OrbitalPreparationInput input;
  input.n_basis_functions = 4;
  input.n_orbitals = 3;
  input.n_active_orbitals = 1;
  input.n_total_electrons = 6;
  input.n_active_electrons = 2;
  input.orbital_type = xmvb::vb::kOrbitalTypeHao;
  input.ao_overlap_matrix = Eigen::MatrixXd::Identity(4, 4);
  input.orbital_basis_counts = {1, 1, 3};
  input.orbital_basis_index_table = {
      1, 0, 0, 0,
      2, 0, 0, 0,
      1, 2, 3, 0};
  input.orbital_value_table = {
      1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0,
      1.0, 2.0, 1.0e-3, 0.0};
  return input;
}

void check_active_sparse_gauge_balance() {
  OrbitalPreparationInput input = make_active_gauge_input();
  require(xmvb::vb::balance_active_gauge(&input),
          "active gauge was not balanced");
  require(input.orbital_value_table[8] == 0.0 &&
              input.orbital_value_table[9] == 0.0 &&
              std::abs(input.orbital_value_table[10] - 1.0) < 1.0e-12 &&
              input.orbital_value_table[11] == 0.0,
          "active balancing changed the projected ray or strict support");
  require(!xmvb::vb::balance_active_gauge(&input),
          "active gauge balance is not idempotent");

  input = make_active_gauge_input();
  input.original_orbital_basis_counts = {1, 1, 2};
  input.orbital_value_table[10] = 0.5;  // Stored but fixed active coefficient.
  require(xmvb::vb::balance_active_gauge(&input),
          "active gauge with a fixed coefficient was not balanced");
  require(input.orbital_value_table[8] == 0.0 &&
              input.orbital_value_table[9] == 0.0 &&
              input.orbital_value_table[10] == 0.5,
          "active balancing rescaled a fixed physical coefficient");

  input = make_active_gauge_input();
  input.orbital_type = xmvb::vb::kOrbitalTypeOeo;
  const std::vector<double> before = input.orbital_value_table;
  require(!xmvb::vb::balance_active_gauge(&input) &&
              input.orbital_value_table == before,
          "per-orbital balancing was applied to the OEO subspace");
}

void check_nonorthogonal_active_ray() {
  OrbitalPreparationInput input = make_active_gauge_input();
  input.ao_overlap_matrix(0, 2) = 0.2;
  input.ao_overlap_matrix(2, 0) = 0.2;
  input.ao_overlap_matrix(1, 2) = -0.1;
  input.ao_overlap_matrix(2, 1) = -0.1;
  const Eigen::MatrixXd& S = input.ao_overlap_matrix;
  Eigen::MatrixXd inactive = Eigen::MatrixXd::Zero(4, 2);
  inactive(0, 0) = 1.0;
  inactive(1, 1) = 1.0;
  const Eigen::MatrixXd complement =
      Eigen::MatrixXd::Identity(4, 4) -
      inactive * (inactive.transpose() * S * inactive).inverse() *
          inactive.transpose() * S;
  auto projected_ray = [&](const OrbitalPreparationInput& state) {
    Eigen::VectorXd active = Eigen::VectorXd::Zero(4);
    for (int slot = 0; slot < 3; ++slot) {
      const int ao = state.orbital_basis_index_table[8 + slot] - 1;
      active[ao] = state.orbital_value_table[8 + slot];
    }
    Eigen::VectorXd projected = complement * active;
    return (projected / std::sqrt(projected.dot(S * projected))).eval();
  };
  const Eigen::VectorXd before = projected_ray(input);
  require(xmvb::vb::balance_active_gauge(&input),
          "nonorthogonal active gauge was not balanced");
  require((projected_ray(input) - before).norm() < 1.0e-10,
          "nonorthogonal projected active ray changed under gauge balance");
}

}  // namespace

int main() {
  try {
    check_ill_conditioned_sparse_gauge();
    check_well_conditioned_gauge_is_unchanged();
    check_active_sparse_gauge_balance();
    check_nonorthogonal_active_ray();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
