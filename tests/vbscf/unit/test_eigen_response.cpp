#include "core/eigen_response.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace {

Eigen::MatrixXd symmetric_test_matrix(int n, double diagonal_shift) {
  Eigen::MatrixXd matrix(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          std::sin(0.37 * (row + 1) * (column + 2)) /
          static_cast<double>(1 + row + column);
      matrix(row, column) = value;
      matrix(column, row) = value;
    }
    matrix(column, column) += diagonal_shift;
  }
  return matrix;
}

}  // namespace

int main() {
  constexpr int n = 28;
  const Eigen::MatrixXd overlap = symmetric_test_matrix(n, 4.0);
  const Eigen::MatrixXd hamiltonian = symmetric_test_matrix(n, 0.0);
  const Eigen::MatrixXd delta_hamiltonian =
      symmetric_test_matrix(n, 0.17);
  const Eigen::MatrixXd delta_overlap =
      0.03 * symmetric_test_matrix(n, 0.0);
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      hamiltonian, overlap);
  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "failed to construct reference eigensystem\n";
    return 1;
  }

  const std::vector<int> roots{0, 2, 5};
  Eigen::VectorXd eigenvalues(roots.size());
  Eigen::MatrixXd eigenvectors(n, roots.size());
  for (std::size_t state = 0; state < roots.size(); ++state) {
    eigenvalues[state] = eigensolver.eigenvalues()[roots[state]];
    eigenvectors.col(state) = eigensolver.eigenvectors().col(roots[state]);
  }
  const Eigen::MatrixXd delta_hamiltonian_selected =
      delta_hamiltonian * eigenvectors;
  const Eigen::MatrixXd delta_overlap_selected =
      delta_overlap * eigenvectors;

  int widest_action = 0;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        widest_action = std::max(widest_action, static_cast<int>(vectors.cols()));
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors,
            overlap * vectors};
      };
  const xmvb::core::EigenResponseOptions options{
      n + 1,
      1.0e-8};
  const auto response = xmvb::core::solve_generalized_eigen_response(
      action,
      hamiltonian.diagonal(),
      overlap.diagonal(),
      eigenvalues,
      eigenvectors,
      delta_hamiltonian_selected,
      delta_overlap_selected,
      options);

  double max_vector_error = 0.0;
  double max_energy_error = 0.0;
  double max_gauge_error = 0.0;
  for (std::size_t state = 0; state < roots.size(); ++state) {
    Eigen::MatrixXd bordered = Eigen::MatrixXd::Zero(n + 1, n + 1);
    bordered.topLeftCorner(n, n) =
        hamiltonian - eigenvalues[state] * overlap;
    const Eigen::VectorXd overlap_vector = overlap * eigenvectors.col(state);
    bordered.col(n).head(n) = overlap_vector;
    bordered.row(n).head(n) = overlap_vector.transpose();
    Eigen::VectorXd rhs(n + 1);
    const Eigen::VectorXd forcing =
        delta_hamiltonian_selected.col(state) -
        eigenvalues[state] * delta_overlap_selected.col(state);
    rhs.head(n) = -forcing;
    rhs[n] = -0.5 * eigenvectors.col(state).dot(
        delta_overlap_selected.col(state));
    const Eigen::VectorXd reference = bordered.fullPivLu().solve(rhs);
    max_vector_error = std::max(
        max_vector_error,
        (response.eigenvector_response.col(state) - reference.head(n))
            .cwiseAbs()
            .maxCoeff());
    const double reference_energy = eigenvectors.col(state).dot(forcing);
    max_energy_error = std::max(
        max_energy_error,
        std::abs(response.eigenvalue_response[state] - reference_energy));
    max_gauge_error = std::max(
        max_gauge_error,
        std::abs(
            eigenvectors.col(state).dot(
                overlap * response.eigenvector_response.col(state)) +
            0.5 * eigenvectors.col(state).dot(
                delta_overlap_selected.col(state))));
  }

  const double max_residual = response.relative_residual_norms.maxCoeff();
  const bool passed =
      widest_action == static_cast<int>(roots.size()) &&
      max_vector_error <= 1.0e-8 &&
      max_energy_error <= 1.0e-12 &&
      max_gauge_error <= 1.0e-8 &&
      max_residual <= options.relative_residual_tolerance;
  std::cout << "block_width=" << widest_action
            << " actions=" << response.block_actions
            << " vector_error=" << max_vector_error
            << " energy_error=" << max_energy_error
            << " gauge_error=" << max_gauge_error
            << " residual=" << max_residual
            << (passed ? " PASS\n" : " FAIL\n");
  return passed ? 0 : 1;
}
