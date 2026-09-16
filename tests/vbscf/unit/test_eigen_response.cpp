#include "core/eigen_response.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
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

Eigen::MatrixXd directional_test_matrix(int n) {
  Eigen::MatrixXd matrix(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          std::cos(0.41 * (row + 3) * (column + 1)) /
          static_cast<double>(1 + std::abs(row - column));
      matrix(row, column) = value;
      matrix(column, row) = value;
    }
  }
  return matrix;
}

/** @brief Certifies the first MINRES Givens residual in the preconditioner metric. */
bool test_preconditioned_minres_residual_scale() {
  for (const double curvature : {100.0, -100.0}) {
    Eigen::Matrix2d operator_matrix;
    operator_matrix << curvature, 10.0, 10.0, 1.5;
    const Eigen::Vector2d rhs(1.0, 1.0);
    const Eigen::Vector2d inverse_diagonal(
        1.0 / std::abs(curvature), 1.0 / 1.5);
    const double beta = std::sqrt(
        rhs.dot((inverse_diagonal.array() * rhs.array()).matrix()));
    const Eigen::Vector2d v = rhs / beta;
    const Eigen::Vector2d w =
        (inverse_diagonal.array() * v.array()).matrix();
    const Eigen::Vector2d image = operator_matrix * w;
    const double alpha = image.dot(w);
    const Eigen::Vector2d next = image - alpha * v;
    const double next_beta = std::sqrt(std::max(0.0,
        next.dot((inverse_diagonal.array() * next.array()).matrix())));
    const double rotation = std::hypot(alpha, next_beta);
    const double sine = next_beta / rotation;
    const Eigen::Vector2d solution =
        beta * (alpha / rotation) * w / rotation;
    const Eigen::Vector2d true_residual =
        rhs - operator_matrix * solution;
    const double true_preconditioned_norm = std::sqrt(std::max(0.0,
        true_residual.dot(
            (inverse_diagonal.array() * true_residual.array()).matrix())));
    const double correct_estimate = beta * std::abs(sine);
    const double old_estimate = rhs.norm() * std::abs(sine);
    std::cout << "minres_scale curvature=" << curvature
              << " true_preconditioned=" << true_preconditioned_norm
              << " correct_estimate=" << correct_estimate
              << " old_estimate=" << old_estimate << '\n';
    if (std::abs(correct_estimate - true_preconditioned_norm) >
            1.0e-12 * std::max(1.0, true_preconditioned_norm) ||
        std::abs(old_estimate - true_preconditioned_norm) <
            0.01 * true_preconditioned_norm) {
      return false;
    }
  }
  return true;
}

/** @brief Reproduces the unresolved noncommuting, soft-spectrum response. */
bool test_ill_conditioned_selected_roots() {
  constexpr int n = 96;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> mixer(
      symmetric_test_matrix(n, 0.0));
  if (mixer.info() != Eigen::Success) {
    return false;
  }
  const Eigen::MatrixXd& basis = mixer.eigenvectors();
  Eigen::VectorXd metric(n);
  Eigen::VectorXd energies(n);
  for (int state = 0; state < n; ++state) {
    metric[state] = std::pow(1.0e-8,
        static_cast<double>(n - 1 - state) / (n - 1));
    energies[state] = -2.0 + 0.1 * state;
  }
  metric[0] = 1.0;
  energies[1] = energies[0] + 1.0e-4;
  const Eigen::MatrixXd overlap =
      basis * metric.asDiagonal() * basis.transpose();
  const Eigen::MatrixXd hamiltonian =
      basis * (metric.array() * energies.array()).matrix().asDiagonal() *
      basis.transpose();
  const std::vector<int> roots{0, 95};
  Eigen::VectorXd selected_energies(roots.size());
  Eigen::MatrixXd selected_vectors(n, roots.size());
  for (std::size_t state = 0; state < roots.size(); ++state) {
    selected_energies[state] = energies[roots[state]];
    selected_vectors.col(state) =
        basis.col(roots[state]) / std::sqrt(metric[roots[state]]);
  }
  const Eigen::MatrixXd metric_sqrt =
      basis * metric.cwiseSqrt().asDiagonal() * basis.transpose();
  const Eigen::MatrixXd delta_hamiltonian =
      0.003 * metric_sqrt * directional_test_matrix(n) * metric_sqrt;
  const Eigen::MatrixXd delta_overlap =
      0.0001 * metric_sqrt * directional_test_matrix(n) * metric_sqrt;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  const Eigen::MatrixXd complete_vectors =
      basis * metric.cwiseSqrt().cwiseInverse().asDiagonal();
  Eigen::MatrixXd spectral_vectors(n, roots.size());
  for (std::size_t column = 0; column < roots.size(); ++column) {
    const int root = roots[column];
    const Eigen::VectorXd forcing =
        delta_hamiltonian * selected_vectors.col(column) -
        selected_energies[column] *
            delta_overlap * selected_vectors.col(column);
    Eigen::VectorXd coefficients = complete_vectors.transpose() * forcing;
    for (int other = 0; other < n; ++other) {
      coefficients[other] = other == root
          ? -0.5 * selected_vectors.col(column).dot(
                delta_overlap * selected_vectors.col(column))
          : coefficients[other] /
                (energies[root] - energies[other]);
    }
    spectral_vectors.col(column) = complete_vectors * coefficients;
  }
  const Eigen::VectorXd forcing =
      delta_hamiltonian * selected_vectors.col(0) -
      selected_energies[0] * delta_overlap * selected_vectors.col(0);
  const Eigen::VectorXd selected_overlap =
      overlap * selected_vectors.col(0);
  const double energy_response = selected_vectors.col(0).dot(forcing);
  const Eigen::MatrixXd shifted =
      hamiltonian - selected_energies[0] * overlap;
  Eigen::VectorXd reference_rhs(n + 1);
  reference_rhs.head(n) = -forcing;
  reference_rhs[n] = -0.5 * selected_vectors.col(0).dot(
      delta_overlap * selected_vectors.col(0));
  Eigen::VectorXd reference_image(n + 1);
  reference_image.head(n) = shifted * spectral_vectors.col(0) -
      energy_response * selected_overlap;
  reference_image[n] = selected_overlap.dot(spectral_vectors.col(0));
  const double reference_bordered_relative_residual =
      (reference_rhs - reference_image).norm() / reference_rhs.norm();
  std::cout << "soft_spectrum_reference_bordered_residual="
            << reference_bordered_relative_residual << '\n';
  try {
    const auto response = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energies, selected_vectors, overlap * selected_vectors,
        delta_hamiltonian * selected_vectors,
        delta_overlap * selected_vectors,
        {n + 1, 1.0e-6});
    std::cout << "unresolved_soft_spectrum actions="
              << response.block_actions << " iterations="
              << response.iterations[0] << ',' << response.iterations[1]
              << " bordered_residual="
              << response.relative_residual_norms.maxCoeff() << '\n';
    return response.relative_residual_norms.maxCoeff() <= 1.0e-6;
  } catch (const std::exception& error) {
    std::cout << "unresolved_soft_spectrum failed=" << error.what() << '\n';
    return false;
  }
}

bool test_inexact_ritz_root() {
  constexpr int n = 28;
  const Eigen::MatrixXd overlap = symmetric_test_matrix(n, 4.0);
  const Eigen::MatrixXd hamiltonian = symmetric_test_matrix(n, 0.0);
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      hamiltonian, overlap);
  if (eigensolver.info() != Eigen::Success) {
    return false;
  }
  Eigen::VectorXd root = eigensolver.eigenvectors().col(0) +
      1.0e-8 * eigensolver.eigenvectors().col(2);
  root /= std::sqrt(root.dot(overlap * root));
  Eigen::VectorXd selected_energy(1);
  selected_energy[0] = root.dot(hamiltonian * root);
  Eigen::MatrixXd selected_root(n, 1);
  selected_root.col(0) = root;
  const Eigen::MatrixXd delta_hamiltonian =
      symmetric_test_matrix(n, 0.17);
  const Eigen::MatrixXd delta_overlap =
      0.03 * symmetric_test_matrix(n, 0.0);
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  try {
    const auto response = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energy, selected_root, overlap * selected_root,
        delta_hamiltonian * selected_root,
        delta_overlap * selected_root,
        {n + 1, 1.0e-7});
    const double ritz_residual =
        (hamiltonian * root - selected_energy[0] * overlap * root).norm();
    const double gauge_error = std::abs(
        root.dot(overlap * response.eigenvector_response.col(0)) +
        0.5 * root.dot(delta_overlap * root));
    Eigen::VectorXd exact_energy(1);
    exact_energy[0] = eigensolver.eigenvalues()[0];
    Eigen::MatrixXd exact_root(n, 1);
    exact_root.col(0) = eigensolver.eigenvectors().col(0);
    const auto exact_response =
        xmvb::core::solve_generalized_eigen_response_from_full_spectrum(
            action, eigensolver.eigenvalues(), eigensolver.eigenvectors(),
            {0}, exact_energy, exact_root, overlap * exact_root,
            delta_hamiltonian * exact_root, delta_overlap * exact_root,
            1.0e-7);
    const double derivative_error =
        (response.eigenvector_response - exact_response.eigenvector_response)
            .cwiseAbs().maxCoeff();
    std::cout << "inexact_ritz_root ritz_residual=" << ritz_residual
              << " response_residual="
              << response.relative_residual_norms[0]
              << " gauge_error=" << gauge_error
              << " derivative_error_vs_exact_root=" << derivative_error
              << " iterations=" << response.iterations[0] << '\n';
    return response.relative_residual_norms[0] <= 1.0e-7 &&
        gauge_error <= 1.0e-12;
  } catch (const std::exception& error) {
    std::cout << "inexact_ritz_root failed=" << error.what() << '\n';
    return false;
  }
}

/** @brief Tests a matrix-free response with nonorthogonal, ill-conditioned S. */
bool test_solvable_ill_conditioned_overlap() {
  constexpr int n = 8;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> mixer(
      symmetric_test_matrix(n, 0.0));
  if (mixer.info() != Eigen::Success) return false;
  const Eigen::MatrixXd basis = mixer.eigenvectors();
  Eigen::VectorXd metric(n);
  Eigen::VectorXd energies(n);
  for (int index = 0; index < n; ++index) {
    metric[index] = std::pow(1.0e-4,
        static_cast<double>(n - 1 - index) / (n - 1));
    energies[index] = -2.0 + 0.2 * index;
  }
  metric[0] = 1.0;
  const Eigen::MatrixXd overlap =
      basis * metric.asDiagonal() * basis.transpose();
  const Eigen::MatrixXd hamiltonian =
      basis * (metric.array() * energies.array()).matrix().asDiagonal() *
      basis.transpose();
  const Eigen::MatrixXd root = basis.col(0);
  const Eigen::VectorXd selected_energy = energies.head(1);
  const Eigen::MatrixXd metric_sqrt =
      basis * metric.cwiseSqrt().asDiagonal() * basis.transpose();
  const Eigen::MatrixXd delta_hamiltonian =
      0.003 * metric_sqrt * directional_test_matrix(n) * metric_sqrt;
  const Eigen::MatrixXd delta_overlap =
      0.0001 * metric_sqrt * directional_test_matrix(n) * metric_sqrt;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  try {
    const auto response = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energy, root, overlap * root,
        delta_hamiltonian * root, delta_overlap * root,
        {n + 1, 1.0e-6});
    std::cout << "solvable_ill_overlap condition=1e4 iterations="
              << response.iterations[0]
              << " bordered_residual="
              << response.relative_residual_norms[0] << '\n';
    return response.relative_residual_norms[0] <= 1.0e-6;
  } catch (const std::exception& error) {
    std::cout << "solvable_ill_overlap failed=" << error.what() << '\n';
    return false;
  }
}

bool test_repeated_root_bordered_candidate() {
  constexpr int n = 3;
  Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(n, n);
  overlap(0, 1) = overlap(1, 0) = 0.9;
  Eigen::MatrixXd shifted = Eigen::MatrixXd::Zero(n, n);
  shifted(1, 1) = 1.0;
  shifted(1, 2) = shifted(2, 1) = 0.4;
  shifted(2, 2) = 3.0;
  const Eigen::MatrixXd hamiltonian = -2.0 * overlap + shifted;
  Eigen::VectorXd selected_energy = Eigen::VectorXd::Constant(2, -2.0);
  Eigen::MatrixXd selected_root = Eigen::MatrixXd::Zero(n, 2);
  selected_root.row(0).setOnes();
  Eigen::MatrixXd delta_hamiltonian(n, 2);
  delta_hamiltonian.col(0) << 1.0, -1.0, 1.0;
  delta_hamiltonian.col(1) << 0.0, 0.0, 1.0;
  const Eigen::MatrixXd delta_overlap = Eigen::MatrixXd::Zero(n, 2);
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  constexpr double tolerance = 0.38;
  const auto first_candidate = xmvb::core::solve_generalized_eigen_response(
      action, hamiltonian.diagonal(), overlap.diagonal(),
      selected_energy, selected_root, overlap * selected_root,
      delta_hamiltonian, delta_overlap, {n + 1, 0.50});
  Eigen::VectorXd projected_rhs = -delta_hamiltonian.col(0) +
      overlap.col(0);
  projected_rhs[0] = 0.0;
  Eigen::VectorXd projected_solution =
      first_candidate.eigenvector_response.col(0);
  projected_solution[0] = 0.0;
  const double projected_relative_residual =
      (projected_rhs - shifted * projected_solution).norm() /
      projected_rhs.norm();
  const double bordered_relative_residual =
      first_candidate.relative_residual_norms[0];
  std::cout << "first_repeated_root_candidate projected_residual="
            << projected_relative_residual
            << " bordered_residual=" << bordered_relative_residual
            << " iterations=" << first_candidate.iterations[0] << '\n';
  if (first_candidate.iterations[0] != 1 ||
      !(projected_relative_residual < tolerance) ||
      !(bordered_relative_residual > tolerance)) {
    return false;
  }
  try {
    const auto response = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energy, selected_root, overlap * selected_root,
        delta_hamiltonian, delta_overlap, {n + 1, tolerance});
    std::cout << "repeated_root_bordered_candidate iterations="
              << response.iterations[0] << ',' << response.iterations[1]
              << " residual=" << response.relative_residual_norms.maxCoeff()
              << " actions=" << response.block_actions << '\n';
    return response.relative_residual_norms.maxCoeff() <= tolerance;
  } catch (const std::exception& error) {
    std::cout << "repeated_root_bordered_candidate failed="
              << error.what() << '\n';
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--stress") {
    // Explicit unresolved soft-spectrum diagnostic: intentionally reports
    // failure until a matrix-free selected-root solver meets the true residual.
    return test_ill_conditioned_selected_roots() ? 0 : 1;
  }
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
      overlap * eigenvectors,
      delta_hamiltonian_selected,
      delta_overlap_selected,
      options);
  const auto spectral_response =
      xmvb::core::solve_generalized_eigen_response_from_full_spectrum(
          action,
          eigensolver.eigenvalues(),
          eigensolver.eigenvectors(),
          roots,
          eigenvalues,
          eigenvectors,
          overlap * eigenvectors,
          delta_hamiltonian_selected,
          delta_overlap_selected,
          options.relative_residual_tolerance);

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
  const double spectral_difference =
      (spectral_response.eigenvector_response -
       response.eigenvector_response)
          .cwiseAbs().maxCoeff();
  const bool spectral_passed =
      spectral_response.block_actions == 1 &&
      spectral_response.relative_residual_norms.maxCoeff() <=
          options.relative_residual_tolerance &&
      spectral_difference <= 1.0e-8;
  std::cout << "block_width=" << widest_action
            << " actions=" << response.block_actions
            << " vector_error=" << max_vector_error
            << " energy_error=" << max_energy_error
            << " gauge_error=" << max_gauge_error
            << " residual=" << max_residual
            << " full_spectrum_difference=" << spectral_difference
            << (passed && spectral_passed ? " PASS\n" : " FAIL\n");
  const bool minres_scale_passed = test_preconditioned_minres_residual_scale();
  const bool solvable_ill_passed = test_solvable_ill_conditioned_overlap();
  const bool inexact_ritz_passed = test_inexact_ritz_root();
  const bool repeated_root_passed = test_repeated_root_bordered_candidate();
  return passed && spectral_passed && solvable_ill_passed &&
      minres_scale_passed &&
      inexact_ritz_passed && repeated_root_passed ? 0 : 1;
}
