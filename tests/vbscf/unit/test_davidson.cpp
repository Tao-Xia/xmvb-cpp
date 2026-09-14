#include "core/eigensolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace {

struct DenseProblem {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
};

DenseProblem make_tridiagonal_problem(int dimension) {
  DenseProblem problem{
      Eigen::MatrixXd::Zero(dimension, dimension),
      Eigen::MatrixXd::Identity(dimension, dimension)};
  for (int row = 0; row < dimension; ++row) {
    problem.hamiltonian(row, row) = 2.0;
    if (row > 0) {
      problem.hamiltonian(row, row - 1) = -1.0;
      problem.hamiltonian(row - 1, row) = -1.0;
    }
  }
  return problem;
}

DenseProblem make_nonorthogonal_problem(int dimension) {
  const DenseProblem orthogonal_problem =
      make_tridiagonal_problem(dimension);
  Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(dimension, dimension);
  for (int row = 0; row < dimension; ++row) {
    factor(row, row) = 1.0 + 0.003 * row;
    if (row > 0) {
      factor(row, row - 1) = 0.2;
    }
  }
  return DenseProblem{
      factor * orthogonal_problem.hamiltonian * factor.transpose(),
      factor * factor.transpose()};
}

std::vector<double> flatten(const Eigen::MatrixXd& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

double max_eigenvalue_error(
    const std::vector<double>& reference,
    const std::vector<double>& computed,
    int n_roots) {
  double error = 0.0;
  for (int root = 0; root < n_roots; ++root) {
    error = std::max(error, std::abs(reference[root] - computed[root]));
  }
  return error;
}

double max_eigenvector_error(
    const std::vector<double>& reference,
    const std::vector<double>& computed,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    int n_roots) {
  const int dimension = static_cast<int>(overlap.rows());
  const Eigen::Map<const Eigen::MatrixXd> reference_vectors(
      reference.data(), dimension, dimension);
  const Eigen::Map<const Eigen::MatrixXd> computed_vectors(
      computed.data(), dimension, n_roots);
  double error = 0.0;
  for (int root = 0; root < n_roots; ++root) {
    const double alignment = std::abs(
        reference_vectors.col(root).dot(
            overlap * computed_vectors.col(root)));
    error = std::max(error, 1.0 - alignment);
  }
  return error;
}

double s_orthonormality_error(
    const std::vector<double>& vectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    int n_roots) {
  const int dimension = static_cast<int>(overlap.rows());
  const Eigen::Map<const Eigen::MatrixXd> roots(
      vectors.data(), dimension, n_roots);
  const Eigen::MatrixXd error =
      roots.transpose() * overlap * roots -
      Eigen::MatrixXd::Identity(n_roots, n_roots);
  return error.cwiseAbs().maxCoeff();
}

bool run_case(
    const std::string& label,
    const DenseProblem& problem,
    int n_roots,
    double eigenvector_tolerance) {
  const int dimension = static_cast<int>(problem.hamiltonian.rows());
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        xmvb::core::GeneralizedEigenActionResult images;
        images.hamiltonian.noalias() = problem.hamiltonian * vectors;
        images.overlap.noalias() = problem.overlap * vectors;
        return images;
      };
  const xmvb::core::DavidsonOptions options{
      n_roots,
      4 * dimension,
      std::min(dimension, std::max(128, 24 * n_roots)),
      1.0,
      2.0e-8};

  const xmvb::core::GeneralizedEigensolver solver;
  const auto reference = solver.solve_dense(
      flatten(problem.hamiltonian),
      flatten(problem.overlap),
      dimension);
  const auto start = std::chrono::steady_clock::now();
  const auto result = solver.solve_davidson(
      action,
      problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(),
      options);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();

  const double eigenvalue_error = max_eigenvalue_error(
      reference.eigenvalues,
      result.eigenpairs.eigenvalues,
      n_roots);
  const double eigenvector_error = max_eigenvector_error(
      reference.eigenvector_matrix,
      result.eigenpairs.eigenvector_matrix,
      problem.overlap,
      n_roots);
  const double orthonormality_error = s_orthonormality_error(
      result.eigenpairs.eigenvector_matrix,
      problem.overlap,
      n_roots);
  const Eigen::Map<const Eigen::MatrixXd> computed_roots(
      result.eigenpairs.eigenvector_matrix.data(), dimension, n_roots);
  const double overlap_image_error =
      (result.overlap_eigenvectors - problem.overlap * computed_roots)
          .cwiseAbs()
          .maxCoeff();
  const double max_residual = *std::max_element(
      result.relative_residual_norms.begin(),
      result.relative_residual_norms.end());
  const bool passed =
      eigenvalue_error <= 1.0e-8 &&
      eigenvector_error <= eigenvector_tolerance &&
      orthonormality_error <= 1.0e-10 &&
      overlap_image_error <= 1.0e-12 &&
      max_residual <= options.residual_tolerance;

  std::cout << label
            << " roots=" << n_roots
            << " iterations=" << result.iterations
            << " actions=" << result.block_actions
            << " subspace=" << result.peak_subspace_dimension
            << " seconds=" << seconds
            << " eigenvalue_error=" << eigenvalue_error
            << " eigenvector_error=" << eigenvector_error
            << " s_orthonormality_error=" << orthonormality_error
            << " overlap_image_error=" << overlap_image_error
            << " residual=" << max_residual
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_recycled_case() {
  constexpr int dimension = 160;
  constexpr int n_roots = 4;
  const DenseProblem problem = make_nonorthogonal_problem(dimension);
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        xmvb::core::GeneralizedEigenActionResult images;
        images.hamiltonian.noalias() = problem.hamiltonian * vectors;
        images.overlap.noalias() = problem.overlap * vectors;
        return images;
      };
  const xmvb::core::DavidsonOptions options{
      n_roots,
      4 * dimension,
      128,
      1.0,
      2.0e-8};
  const xmvb::core::GeneralizedEigensolver solver;
  const auto cold = solver.solve_davidson(
      action,
      problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(),
      options);
  const Eigen::Map<const Eigen::MatrixXd> recycled_vectors(
      cold.eigenpairs.eigenvector_matrix.data(),
      dimension,
      n_roots);
  const auto recycled = solver.solve_davidson(
      action,
      problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(),
      recycled_vectors,
      options);
  const double max_residual = *std::max_element(
      recycled.relative_residual_norms.begin(),
      recycled.relative_residual_norms.end());
  const bool passed =
      recycled.iterations == 1 &&
      recycled.block_actions == 1 &&
      max_residual <= options.residual_tolerance &&
      max_eigenvalue_error(
          cold.eigenpairs.eigenvalues,
          recycled.eigenpairs.eigenvalues,
          n_roots) <= 1.0e-12;
  std::cout << "recycled nonorthogonal n=" << dimension
            << " roots=" << n_roots
            << " iterations=" << recycled.iterations
            << " actions=" << recycled.block_actions
            << " residual=" << max_residual
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_default_option_cases() {
  const auto small =
      xmvb::core::make_davidson_options(3, 2, 1.0e-7, 1.0e-3);
  const auto large =
      xmvb::core::make_davidson_options(10000, 4, 1.0e-7, 1.0e-3);
  const bool passed =
      small.n_roots == 2 &&
      small.max_subspace_dimension == 3 &&
      small.max_iterations >= 3 &&
      small.energy_tolerance == 1.0e-7 &&
      small.residual_tolerance == 1.0e-3 &&
      large.n_roots == 4 &&
      large.max_subspace_dimension < 10000 &&
      large.max_subspace_dimension >= 2 * large.n_roots;
  std::cout << "default Davidson budgets"
            << " small_subspace=" << small.max_subspace_dimension
            << " large_subspace=" << large.max_subspace_dimension
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

}  // namespace

int main() {
  bool passed = true;
  passed = run_default_option_cases() && passed;
  for (const int dimension : {80, 120, 200, 400, 600}) {
    for (const int n_roots : {1, 3, 5, 8}) {
      passed = run_case(
                   "orthogonal n=" + std::to_string(dimension),
                   make_tridiagonal_problem(dimension),
                   n_roots,
                   1.0e-5) &&
          passed;
    }
  }
  passed = run_case(
               "nonorthogonal n=160",
               make_nonorthogonal_problem(160),
               4,
               2.0e-5) &&
      passed;
  passed = run_recycled_case() && passed;
  std::cout << (passed ? "ALL PASSED\n" : "SOME FAILED\n");
  return passed ? 0 : 1;
}
