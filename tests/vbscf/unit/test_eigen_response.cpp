#include "core/eigen_response.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

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
  constexpr double tolerance = 1.0e-11;
  try {
    std::vector<xmvb::core::EigenResponseRecycleSpace> storage(2);
    std::vector<xmvb::core::EigenResponseRecycleSpace*> spaces{
        &storage[0], &storage[1]};
    const auto response = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energy, selected_root, overlap * selected_root,
        delta_hamiltonian, delta_overlap, {n + 1, tolerance}, spaces);
    const auto recycled = xmvb::core::solve_generalized_eigen_response(
        action, hamiltonian.diagonal(), overlap.diagonal(),
        selected_energy, selected_root, overlap * selected_root,
        delta_hamiltonian, delta_overlap, {n + 1, tolerance}, spaces);
    double reference_error = 0.0;
    double gauge_error = 0.0;
    for (int state = 0; state < 2; ++state) {
      Eigen::MatrixXd bordered = Eigen::MatrixXd::Zero(n + 1, n + 1);
      bordered.topLeftCorner(n, n) = shifted;
      const Eigen::VectorXd constraint = overlap * selected_root.col(state);
      bordered.col(n).head(n) = constraint;
      bordered.row(n).head(n) = constraint.transpose();
      Eigen::VectorXd rhs(n + 1);
      rhs.head(n) = -delta_hamiltonian.col(state);
      rhs[n] = 0.0;
      const Eigen::VectorXd reference = bordered.fullPivLu().solve(rhs);
      reference_error = std::max(reference_error,
          (response.eigenvector_response.col(state) - reference.head(n))
              .norm());
      reference_error = std::max(reference_error,
          std::abs(response.eigenvalue_response[state] + reference[n]));
      gauge_error = std::max(gauge_error,
          std::abs(constraint.dot(response.eigenvector_response.col(state))));
    }
    const bool recycled_without_iteration = std::all_of(
        recycled.iterations.begin(), recycled.iterations.end(),
        [](int iterations) { return iterations == 0; });
    std::cout << "repeated_root_bordered_candidate iterations="
              << response.iterations[0] << ',' << response.iterations[1]
              << " residual=" << response.relative_residual_norms.maxCoeff()
              << " reference_error=" << reference_error
              << " gauge_error=" << gauge_error
              << " recycled_iterations=" << recycled.iterations[0] << ','
              << recycled.iterations[1]
              << " actions=" << response.block_actions << '\n';
    return response.relative_residual_norms.maxCoeff() <= tolerance &&
        recycled.relative_residual_norms.maxCoeff() <= tolerance &&
        reference_error <= 1.0e-10 && gauge_error <= 1.0e-11 &&
        recycled_without_iteration &&
        (recycled.eigenvector_response - response.eigenvector_response)
                .norm() <= 1.0e-11;
  } catch (const std::exception& error) {
    std::cout << "repeated_root_bordered_candidate failed="
              << error.what() << '\n';
    return false;
  }
}

/** @brief Verifies equal-weight selected-selected cancellation and Hessian. */
bool test_equal_weight_subspace_response() {
  constexpr int n = 12;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> mixer(
      symmetric_test_matrix(n, 0.0));
  if (mixer.info() != Eigen::Success) return false;
  const Eigen::MatrixXd basis = mixer.eigenvectors();
  Eigen::VectorXd metric(n);
  Eigen::VectorXd energies(n);
  for (int index = 0; index < n; ++index) {
    metric[index] = 0.35 + 0.11 * index;
    energies[index] = -2.0 + 0.3 * index;
  }
  // The individual derivatives are ill-conditioned in this small gap, while
  // the equally weighted two-state projector has no selected-selected pole.
  energies[2] = energies[1] + 1.0e-6;
  const Eigen::MatrixXd metric_sqrt =
      basis * metric.cwiseSqrt().asDiagonal() * basis.transpose();
  const Eigen::MatrixXd metric_inverse_sqrt =
      basis * metric.cwiseSqrt().cwiseInverse().asDiagonal() *
      basis.transpose();
  const Eigen::MatrixXd overlap = metric_sqrt * metric_sqrt;
  // Deliberately keep the generalized eigenvectors out of the ordinary
  // eigenspace of S.  Then span(C) != span(S C), which detects projecting a
  // generalized-eigen response against C instead of its metric image S C.
  const Eigen::MatrixXd hamiltonian =
      metric_sqrt * energies.asDiagonal() * metric_sqrt;
  const Eigen::MatrixXd full_vectors = metric_inverse_sqrt;
  const std::vector<int> roots{1, 2};
  Eigen::VectorXd selected_energies(2);
  Eigen::MatrixXd selected_vectors(n, 2);
  for (int state = 0; state < 2; ++state) {
    selected_energies[state] = energies[roots[state]];
    selected_vectors.col(state) = full_vectors.col(roots[state]);
  }
  const Eigen::MatrixXd delta_hamiltonian =
      0.007 * metric_sqrt * directional_test_matrix(n) * metric_sqrt;
  const Eigen::MatrixXd delta_overlap =
      0.002 * metric_sqrt * symmetric_test_matrix(n, 0.0) * metric_sqrt;
  const Eigen::MatrixXd delta_hamiltonian_selected =
      delta_hamiltonian * selected_vectors;
  const Eigen::MatrixXd delta_overlap_selected =
      delta_overlap * selected_vectors;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  const xmvb::core::EigenResponseOptions options{n + 1, 1.0e-9};
  try {
    std::vector<xmvb::core::EigenResponseRecycleSpace> recycle_storage(2);
    std::vector<xmvb::core::EigenResponseRecycleSpace*> recycle_spaces{
        &recycle_storage[0], &recycle_storage[1]};
    const auto response =
        xmvb::core::solve_equal_weight_generalized_eigen_subspace_response(
            action, hamiltonian.diagonal(), overlap.diagonal(),
            selected_energies, selected_vectors,
            overlap * selected_vectors, delta_hamiltonian_selected,
            delta_overlap_selected, options, recycle_spaces);
    const auto recycled_response =
        xmvb::core::solve_equal_weight_generalized_eigen_subspace_response(
            action, hamiltonian.diagonal(), overlap.diagonal(),
            selected_energies, selected_vectors,
            overlap * selected_vectors, delta_hamiltonian_selected,
            delta_overlap_selected, options, recycle_spaces);
    const auto spectral =
        xmvb::core::
            solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
                action, energies, full_vectors, roots, selected_energies,
                selected_vectors, overlap * selected_vectors,
                delta_hamiltonian_selected, delta_overlap_selected,
                options.relative_residual_tolerance);
    const auto isolated =
        xmvb::core::solve_generalized_eigen_response_from_full_spectrum(
            action, energies, full_vectors, roots, selected_energies,
            selected_vectors, overlap * selected_vectors,
            delta_hamiltonian_selected, delta_overlap_selected,
            options.relative_residual_tolerance);

    const Eigen::MatrixXd projector_response =
        response.eigenvector_response * selected_vectors.transpose() +
        selected_vectors * response.eigenvector_response.transpose();
    const Eigen::MatrixXd spectral_projector_response =
        spectral.eigenvector_response * selected_vectors.transpose() +
        selected_vectors * spectral.eigenvector_response.transpose();
    const Eigen::MatrixXd isolated_projector_response =
        isolated.eigenvector_response * selected_vectors.transpose() +
        selected_vectors * isolated.eigenvector_response.transpose();
    const double spectral_error =
        (projector_response - spectral_projector_response).norm();
    const double cancellation_error =
        (projector_response - isolated_projector_response).norm();
    const Eigen::MatrixXd gauge = selected_vectors.transpose() *
        overlap * response.eigenvector_response;
    const Eigen::MatrixXd gauge_target = -0.5 *
        selected_vectors.transpose() * delta_overlap_selected;
    const double gauge_error = (gauge - gauge_target).norm();

    constexpr double step = 2.0e-5;
    Eigen::MatrixXd finite_difference_projector;
    double finite_difference_curvature = 0.0;
    double central_energy = selected_energies.sum();
    Eigen::MatrixXd projectors[2];
    double energy_sums[2];
    for (int side = 0; side < 2; ++side) {
      const double sign = side == 0 ? -1.0 : 1.0;
      const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> perturbed(
          hamiltonian + sign * step * delta_hamiltonian,
          overlap + sign * step * delta_overlap);
      if (perturbed.info() != Eigen::Success) return false;
      const Eigen::MatrixXd perturbed_selected =
          perturbed.eigenvectors().middleCols(roots.front(), 2);
      projectors[side] =
          perturbed_selected * perturbed_selected.transpose();
      energy_sums[side] =
          perturbed.eigenvalues().segment(roots.front(), 2).sum();
    }
    finite_difference_projector =
        (projectors[1] - projectors[0]) / (2.0 * step);
    finite_difference_curvature =
        (energy_sums[1] - 2.0 * central_energy + energy_sums[0]) /
        (step * step);
    const double projector_fd_error =
        (projector_response - finite_difference_projector).norm();

    const Eigen::MatrixXd forcing = delta_hamiltonian_selected -
        delta_overlap_selected * selected_energies.asDiagonal();
    double analytic_curvature = 0.0;
    for (int state = 0; state < 2; ++state) {
      analytic_curvature +=
          2.0 * response.eigenvector_response.col(state).dot(
              forcing.col(state)) -
          response.selected_matrix_response(state, state) *
              selected_vectors.col(state).dot(
                  delta_overlap_selected.col(state));
    }
    const double curvature_error =
        std::abs(analytic_curvature - finite_difference_curvature);

    Eigen::VectorXd degenerate_energies = energies;
    degenerate_energies[2] = degenerate_energies[1];
    const Eigen::MatrixXd degenerate_hamiltonian =
        metric_sqrt * degenerate_energies.asDiagonal() * metric_sqrt;
    const Eigen::VectorXd degenerate_selected_energies =
        degenerate_energies.segment(1, 2);
    const xmvb::core::GeneralizedEigenAction degenerate_action =
        [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
          return xmvb::core::GeneralizedEigenActionResult{
              degenerate_hamiltonian * vectors, overlap * vectors};
        };
    const auto degenerate_response =
        xmvb::core::solve_equal_weight_generalized_eigen_subspace_response(
            degenerate_action, degenerate_hamiltonian.diagonal(),
            overlap.diagonal(), degenerate_selected_energies,
            selected_vectors, overlap * selected_vectors,
            delta_hamiltonian_selected, delta_overlap_selected, options);
    const auto degenerate_spectral =
        xmvb::core::
            solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
                degenerate_action, degenerate_energies, full_vectors, roots,
                degenerate_selected_energies, selected_vectors,
                overlap * selected_vectors, delta_hamiltonian_selected,
                delta_overlap_selected,
                options.relative_residual_tolerance);
    const Eigen::MatrixXd degenerate_projector_response =
        degenerate_response.eigenvector_response *
            selected_vectors.transpose() +
        selected_vectors *
            degenerate_response.eigenvector_response.transpose();
    const Eigen::MatrixXd degenerate_spectral_projector_response =
        degenerate_spectral.eigenvector_response *
            selected_vectors.transpose() +
        selected_vectors *
            degenerate_spectral.eigenvector_response.transpose();
    const double degeneracy_error =
        (degenerate_projector_response -
         degenerate_spectral_projector_response).norm();
    std::cout << "equal_weight_subspace residual="
              << response.relative_residual_norms.maxCoeff()
              << " spectral_error=" << spectral_error
              << " selected_cancellation_error=" << cancellation_error
              << " gauge_error=" << gauge_error
              << " projector_fd_error=" << projector_fd_error
              << " curvature_error=" << curvature_error
              << " degeneracy_error=" << degeneracy_error
              << " recycled_actions=" << recycled_response.block_actions
              << " recycled_iterations="
              << recycled_response.iterations[0] << ','
              << recycled_response.iterations[1]
              << " isolated_norm=" << isolated.eigenvector_response.norm()
              << " subspace_norm=" << response.eigenvector_response.norm()
              << '\n';
    return response.relative_residual_norms.maxCoeff() <= 1.0e-9 &&
        spectral_error <= 1.0e-8 && cancellation_error <= 1.0e-8 &&
        gauge_error <= 1.0e-10 && projector_fd_error <= 1.0e-6 &&
        curvature_error <= 1.0e-5 && degeneracy_error <= 1.0e-8 &&
        recycled_response.block_actions == 3 &&
        std::all_of(
            recycled_response.iterations.begin(),
            recycled_response.iterations.end(),
            [](int iterations) { return iterations == 0; }) &&
        (recycled_response.eigenvector_response -
         response.eigenvector_response).norm() <= 1.0e-10 &&
        degenerate_response.relative_residual_norms.maxCoeff() <= 1.0e-9 &&
        isolated.eigenvector_response.norm() >
            1.0e2 * response.eigenvector_response.norm();
  } catch (const std::exception& error) {
    std::cout << "equal_weight_subspace failed=" << error.what() << '\n';
    return false;
  }
}

bool test_unconditional_indefinite_galerkin_application() {
  xmvb::core::EigenResponseRecycleSpace space;
  const Eigen::Vector2d direction(1.0, 0.0);
  const Eigen::Vector2d image(-0.1, 10.0);
  if (space.revision() != 0 || !space.append(direction, image) ||
      space.revision() != 1) {
    return false;
  }
  const Eigen::Vector2d rhs(1.0, 0.0);
  const auto application = space.galerkin_apply(rhs);
  const bool linear_application_passed =
      application.available &&
      (application.solution - Eigen::Vector2d(-10.0, 0.0)).norm() <=
          1.0e-12 &&
      (application.operator_image - Eigen::Vector2d(1.0, -100.0)).norm() <=
          1.0e-12 &&
      (rhs - application.operator_image).norm() > rhs.norm();
  const bool dependent_append = space.append(direction, image);
  const std::uint64_t revision_before_clear = space.revision();
  space.clear();
  return linear_application_passed && !dependent_append &&
      revision_before_clear == 1 && space.revision() == 2 &&
      space.size() == 0;
}

/** @brief Checks the exact finite-Ritz bordered reduction and reciprocity. */
bool test_finite_ritz_frozen_isolated_response() {
  constexpr int n = 4;
  Eigen::Matrix4d overlap;
  overlap << 1.4, 0.2, 0.0, 0.1,
             0.2, 1.1, 0.15, 0.0,
             0.0, 0.15, 0.9, 0.05,
             0.1, 0.0, 0.05, 1.3;
  Eigen::Matrix4d hamiltonian;
  hamiltonian << -1.2, 0.3, 0.1, 0.0,
                  0.3, 0.4, 0.25, 0.05,
                  0.1, 0.25, 1.5, -0.2,
                  0.0, 0.05, -0.2, 2.1;
  Eigen::Vector4d root(0.8, -0.3, 0.25, 0.1);
  root /= std::sqrt(root.dot(overlap * root));
  const double energy = root.dot(hamiltonian * root);
  const Eigen::Vector4d overlap_root = overlap * root;
  const Eigen::Vector4d ritz_residual =
      hamiltonian * root - energy * overlap_root;
  const Eigen::Matrix4d shifted = hamiltonian - energy * overlap;

  const std::array<Eigen::Matrix4d, 2> delta_hamiltonians{
      0.07 * directional_test_matrix(n),
      0.04 * symmetric_test_matrix(n, 0.23)};
  const std::array<Eigen::Matrix4d, 2> delta_overlaps{
      0.012 * symmetric_test_matrix(n, 0.0),
      0.009 * directional_test_matrix(n)};
  std::array<Eigen::Vector4d, 3> forcing;
  std::array<double, 3> gauge;
  std::array<Eigen::Vector4d, 3> reference_dc;
  std::array<double, 3> reference_de;
  Eigen::Matrix<double, n + 1, n + 1> bordered =
      Eigen::Matrix<double, n + 1, n + 1>::Zero();
  bordered.topLeftCorner<n, n>() = shifted;
  bordered.topRightCorner<n, 1>() = overlap_root;
  bordered.bottomLeftCorner<1, n>() = overlap_root.transpose();
  for (int direction = 0; direction < 3; ++direction) {
    const Eigen::Matrix4d delta_h = direction < 2
        ? delta_hamiltonians[direction]
        : delta_hamiltonians[0] + delta_hamiltonians[1];
    const Eigen::Matrix4d delta_s = direction < 2
        ? delta_overlaps[direction]
        : delta_overlaps[0] + delta_overlaps[1];
    forcing[direction] = delta_h * root - energy * delta_s * root;
    gauge[direction] = -0.5 * root.dot(delta_s * root);
    Eigen::Matrix<double, n + 1, 1> rhs;
    rhs.head<n>() = -forcing[direction];
    rhs[n] = gauge[direction];
    const Eigen::Matrix<double, n + 1, 1> reference =
        bordered.fullPivLu().solve(rhs);
    reference_dc[direction] = reference.head<n>();
    reference_de[direction] = -reference[n];
  }

  const Eigen::Matrix4d projector = Eigen::Matrix4d::Identity() -
      overlap_root * overlap_root.transpose() / overlap_root.squaredNorm();
  xmvb::core::EigenResponseRecycleSpace space;
  for (int direction = 0; direction < 2; ++direction) {
    const Eigen::Vector4d particular =
        root * gauge[direction];  // c^T S c = 1.
    const Eigen::Vector4d external =
        reference_dc[direction] - particular;
    if (!space.append(external, projector * shifted * external)) {
      return false;
    }
  }

  Eigen::VectorXd energies = Eigen::VectorXd::Constant(3, energy);
  Eigen::MatrixXd roots = root.replicate(1, 3);
  Eigen::MatrixXd overlap_roots = overlap_root.replicate(1, 3);
  Eigen::MatrixXd residuals = ritz_residual.replicate(1, 3);
  Eigen::MatrixXd delta_h_selected(n, 3);
  Eigen::MatrixXd delta_s_selected(n, 3);
  for (int direction = 0; direction < 3; ++direction) {
    const Eigen::Matrix4d delta_h = direction < 2
        ? delta_hamiltonians[direction]
        : delta_hamiltonians[0] + delta_hamiltonians[1];
    const Eigen::Matrix4d delta_s = direction < 2
        ? delta_overlaps[direction]
        : delta_overlaps[0] + delta_overlaps[1];
    delta_h_selected.col(direction) = delta_h * root;
    delta_s_selected.col(direction) = delta_s * root;
  }
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  std::vector<xmvb::core::EigenResponseRecycleSpace> iterative_storage(3);
  std::vector<xmvb::core::EigenResponseRecycleSpace*> iterative_spaces{
      &iterative_storage[0], &iterative_storage[1], &iterative_storage[2]};
  const auto iterative = xmvb::core::solve_generalized_eigen_response(
      action, hamiltonian.diagonal(), overlap.diagonal(), energies, roots,
      overlap_roots, delta_h_selected, delta_s_selected, {2 * n, 1.0e-11},
      iterative_spaces, &residuals);
  const auto iterative_recycled =
      xmvb::core::solve_generalized_eigen_response(
          action, hamiltonian.diagonal(), overlap.diagonal(), energies, roots,
          overlap_roots, delta_h_selected, delta_s_selected,
          {2 * n, 1.0e-11}, iterative_spaces, &residuals);
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces(
      3, &space);
  const auto response =
      xmvb::core::evaluate_frozen_generalized_eigen_response(
          action, energies, roots, overlap_roots,
          delta_h_selected, delta_s_selected, spaces, &residuals);

  xmvb::core::EigenResponseRecycleSpace incomplete_space;
  const Eigen::Vector4d first_particular = root * gauge[0];
  const Eigen::Vector4d first_external =
      reference_dc[0] - first_particular;
  if (!incomplete_space.append(
          first_external, projector * shifted * first_external)) {
    return false;
  }
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*>
      incomplete_spaces(3, &incomplete_space);
  const auto incomplete =
      xmvb::core::evaluate_frozen_generalized_eigen_response(
          action, energies, roots, overlap_roots,
          delta_h_selected, delta_s_selected, incomplete_spaces,
          &residuals);

  double reference_error = 0.0;
  double iterative_reference_error = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    reference_error = std::max(reference_error,
        (response.eigenvector_response.col(direction) -
         reference_dc[direction]).norm());
    reference_error = std::max(reference_error,
        std::abs(response.eigenvalue_response[direction] -
                 reference_de[direction]));
    iterative_reference_error = std::max(iterative_reference_error,
        (iterative.eigenvector_response.col(direction) -
         reference_dc[direction]).norm());
    iterative_reference_error = std::max(iterative_reference_error,
        std::abs(iterative.eigenvalue_response[direction] -
                 reference_de[direction]));
  }
  const double uv = 2.0 * forcing[0].dot(reference_dc[1]) +
      2.0 * gauge[0] * reference_de[1];
  const double vu = 2.0 * forcing[1].dot(reference_dc[0]) +
      2.0 * gauge[1] * reference_de[0];
  const double incomplete_uv =
      2.0 * forcing[0].dot(incomplete.eigenvector_response.col(1)) +
      2.0 * gauge[0] * incomplete.eigenvalue_response[1];
  const double incomplete_vu =
      2.0 * forcing[1].dot(incomplete.eigenvector_response.col(0)) +
      2.0 * gauge[1] * incomplete.eigenvalue_response[0];
  const double old_de = root.dot(forcing[0]);
  const double old_bordered_residual =
      (shifted * reference_dc[0] - old_de * overlap_root +
       forcing[0]).norm();
  const double linearity_error =
      (response.eigenvector_response.col(2) -
       response.eigenvector_response.col(0) -
       response.eigenvector_response.col(1)).norm() +
      std::abs(response.eigenvalue_response[2] -
               response.eigenvalue_response[0] -
               response.eigenvalue_response[1]);
  const bool iterative_recycled_without_iteration = std::all_of(
      iterative_recycled.iterations.begin(),
      iterative_recycled.iterations.end(),
      [](int iterations) { return iterations == 0; });
  std::cout << "finite_ritz_isolated frozen_error=" << reference_error
            << " iterative_error=" << iterative_reference_error
            << " frozen_residual="
            << response.relative_residual_norms.maxCoeff()
            << " iterative_residual="
            << iterative.relative_residual_norms.maxCoeff()
            << " recycled_iterations=" << iterative_recycled.iterations[0]
            << ',' << iterative_recycled.iterations[1] << ','
            << iterative_recycled.iterations[2] << '\n';
  return ritz_residual.norm() > 1.0e-3 &&
      old_bordered_residual > 1.0e-6 && response.block_actions == 1 &&
      response.relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      iterative.relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      iterative_recycled.relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      reference_error <= 1.0e-11 && iterative_reference_error <= 1.0e-10 &&
      iterative_recycled_without_iteration &&
      (iterative_recycled.eigenvector_response -
       iterative.eigenvector_response).norm() <= 1.0e-10 &&
      linearity_error <= 1.0e-11 &&
      std::abs(uv - vu) <= 1.0e-11 * std::max({1.0, std::abs(uv),
                                               std::abs(vu)}) &&
      incomplete.relative_residual_norms[1] > 1.0e-6 &&
      std::abs(incomplete_uv - incomplete_vu) <=
          1.0e-11 * std::max({1.0, std::abs(incomplete_uv),
                              std::abs(incomplete_vu)});
}

bool test_frozen_isolated_root_response() {
  constexpr int n = 3;
  Eigen::Matrix3d overlap = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d hamiltonian = Eigen::Matrix3d::Zero();
  hamiltonian.diagonal() << -1.0, 1.0, 3.0;
  Eigen::VectorXd energies(1);
  energies << -1.0;
  Eigen::MatrixXd roots = Eigen::MatrixXd::Zero(n, 1);
  roots(0, 0) = 1.0;
  Eigen::MatrixXd delta_overlap_selected = Eigen::MatrixXd::Zero(n, 1);
  delta_overlap_selected(0, 0) = 0.2;
  Eigen::MatrixXd delta_hamiltonian_selected(n, 1);
  delta_hamiltonian_selected.col(0) << 0.3, 0.4, -0.2;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };

  xmvb::core::EigenResponseRecycleSpace space;
  Eigen::Vector3d external_response(0.0, -0.2, 0.05);
  Eigen::Vector3d projected_rhs(0.0, -0.4, 0.2);
  if (!space.append(external_response, projected_rhs)) return false;
  const std::uint64_t frozen_revision = space.revision();
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces{
      &space};
  const Eigen::MatrixXd selected_residuals = Eigen::MatrixXd::Zero(n, 1);
  const auto response =
      xmvb::core::evaluate_frozen_generalized_eigen_response(
          action, energies, roots, overlap * roots,
          delta_hamiltonian_selected, delta_overlap_selected, spaces,
          &selected_residuals);
  const Eigen::Vector3d reference(-0.1, -0.2, 0.05);

  xmvb::core::EigenResponseRecycleSpace empty_space;
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*>
      empty_spaces{&empty_space};
  const auto unresolved =
      xmvb::core::evaluate_frozen_generalized_eigen_response(
          action, energies, roots, overlap * roots,
          delta_hamiltonian_selected, delta_overlap_selected, empty_spaces,
          &selected_residuals);
  return space.revision() == frozen_revision &&
      response.block_actions == 1 && response.iterations[0] == 0 &&
      (response.eigenvector_response.col(0) - reference).norm() <= 1.0e-12 &&
      std::abs(response.eigenvalue_response[0] - 0.5) <= 1.0e-12 &&
      response.relative_residual_norms[0] <= 1.0e-12 &&
      unresolved.relative_residual_norms[0] > 1.0e-2;
}

bool test_frozen_equal_weight_response() {
  constexpr int n = 4;
  Eigen::Matrix4d overlap = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d hamiltonian = Eigen::Matrix4d::Zero();
  hamiltonian.diagonal() << -1.0, -0.5, 1.0, 2.0;
  Eigen::Vector2d energies(-1.0, -0.5);
  Eigen::MatrixXd roots = Eigen::MatrixXd::Zero(n, 2);
  roots(0, 0) = 1.0;
  roots(1, 1) = 1.0;
  Eigen::Matrix4d delta_overlap = Eigen::Matrix4d::Zero();
  delta_overlap(0, 0) = 0.2;
  delta_overlap(0, 1) = delta_overlap(1, 0) = 0.03;
  delta_overlap(1, 1) = -0.1;
  Eigen::Matrix4d delta_hamiltonian = Eigen::Matrix4d::Zero();
  delta_hamiltonian(0, 0) = 0.4;
  delta_hamiltonian(0, 1) = delta_hamiltonian(1, 0) = 0.07;
  delta_hamiltonian(1, 1) = 0.3;
  delta_hamiltonian(0, 2) = delta_hamiltonian(2, 0) = 0.2;
  delta_hamiltonian(0, 3) = delta_hamiltonian(3, 0) = -0.1;
  delta_hamiltonian(1, 2) = delta_hamiltonian(2, 1) = -0.15;
  delta_hamiltonian(1, 3) = delta_hamiltonian(3, 1) = 0.25;
  const Eigen::MatrixXd delta_hamiltonian_selected =
      delta_hamiltonian * roots;
  const Eigen::MatrixXd delta_overlap_selected = delta_overlap * roots;
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };

  std::vector<xmvb::core::EigenResponseRecycleSpace> storage(2);
  Eigen::Vector4d response0(0.0, 0.0, -0.1, 1.0 / 30.0);
  Eigen::Vector4d rhs0(0.0, 0.0, -0.2, 0.1);
  Eigen::Vector4d response1(0.0, 0.0, 0.1, -0.1);
  Eigen::Vector4d rhs1(0.0, 0.0, 0.15, -0.25);
  if (!storage[0].append(response0, rhs0) ||
      !storage[1].append(response1, rhs1)) {
    return false;
  }
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces{
      &storage[0], &storage[1]};
  const Eigen::MatrixXd selected_residuals = Eigen::MatrixXd::Zero(n, 2);
  const auto response = xmvb::core::
      evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
          action, energies, roots, overlap * roots,
          delta_hamiltonian_selected, delta_overlap_selected, spaces,
          &selected_residuals);
  const Eigen::Matrix2d gauge_target = -0.5 *
      roots.transpose() * delta_overlap_selected;
  const double gauge_error =
      (roots.transpose() * overlap * response.eigenvector_response -
       gauge_target).norm();

  std::vector<xmvb::core::EigenResponseRecycleSpace> empty_storage(2);
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*>
      empty_spaces{&empty_storage[0], &empty_storage[1]};
  const auto unresolved = xmvb::core::
      evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
          action, energies, roots, overlap * roots,
          delta_hamiltonian_selected, delta_overlap_selected, empty_spaces,
          &selected_residuals);
  return response.block_actions == 2 &&
      response.iterations == std::vector<int>({0, 0}) &&
      response.relative_residual_norms.maxCoeff() <= 1.0e-12 &&
      gauge_error <= 1.0e-12 &&
      response.selected_matrix_response.rows() == 2 &&
      response.selected_matrix_response.cols() == 2 &&
      storage[0].revision() == 1 && storage[1].revision() == 1 &&
      unresolved.relative_residual_norms.maxCoeff() > 1.0e-2;
}

/** @brief Checks finite-Ritz gauge forcing for an equal-weight cluster. */
bool test_finite_ritz_frozen_equal_weight_response() {
  constexpr int n = 5;
  constexpr int states = 2;
  const Eigen::MatrixXd overlap = symmetric_test_matrix(n, 4.0);
  const Eigen::MatrixXd hamiltonian = symmetric_test_matrix(n, 0.2);
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      hamiltonian, overlap);
  if (eigensolver.info() != Eigen::Success) return false;
  const Eigen::MatrixXd full_roots = eigensolver.eigenvectors();
  Eigen::MatrixXd roots(n, states);
  roots.col(0) = std::cos(0.08) * full_roots.col(0) +
      std::sin(0.08) * full_roots.col(3);
  roots.col(1) = std::cos(-0.06) * full_roots.col(1) +
      std::sin(-0.06) * full_roots.col(4);
  Eigen::Vector2d energies;
  for (int state = 0; state < states; ++state) {
    energies[state] = roots.col(state).dot(hamiltonian * roots.col(state));
  }
  const Eigen::MatrixXd overlap_roots = overlap * roots;
  const Eigen::MatrixXd ritz_residuals = hamiltonian * roots -
      overlap_roots * energies.asDiagonal();
  const std::array<Eigen::MatrixXd, 2> delta_hamiltonians{
      0.05 * directional_test_matrix(n),
      0.035 * symmetric_test_matrix(n, 0.31)};
  const std::array<Eigen::MatrixXd, 2> delta_overlaps{
      0.008 * symmetric_test_matrix(n, 0.0),
      0.006 * directional_test_matrix(n)};
  const Eigen::Matrix2d metric = roots.transpose() * overlap_roots;

  std::array<Eigen::MatrixXd, 2> forcing;
  std::array<Eigen::Matrix2d, 2> gauge;
  std::array<Eigen::MatrixXd, 2> reference_dc;
  std::array<Eigen::Matrix2d, 2> reference_matrix_response;
  for (int direction = 0; direction < 2; ++direction) {
    const Eigen::MatrixXd delta_h_selected =
        delta_hamiltonians[direction] * roots;
    const Eigen::MatrixXd delta_s_selected =
        delta_overlaps[direction] * roots;
    forcing[direction] = delta_h_selected -
        delta_s_selected * energies.asDiagonal();
    gauge[direction] = -0.5 *
        (roots.transpose() * delta_s_selected);
    gauge[direction] =
        0.5 * (gauge[direction] + gauge[direction].transpose()).eval();
    reference_dc[direction].resize(n, states);
    reference_matrix_response[direction].resize(states, states);
    for (int state = 0; state < states; ++state) {
      Eigen::MatrixXd bordered = Eigen::MatrixXd::Zero(
          n + states, n + states);
      bordered.topLeftCorner(n, n) =
          hamiltonian - energies[state] * overlap;
      bordered.topRightCorner(n, states) = overlap_roots;
      bordered.bottomLeftCorner(states, n) = overlap_roots.transpose();
      Eigen::VectorXd rhs(n + states);
      rhs.head(n) = -forcing[direction].col(state);
      rhs.tail(states) = gauge[direction].col(state);
      const Eigen::VectorXd solution = bordered.fullPivLu().solve(rhs);
      reference_dc[direction].col(state) = solution.head(n);
      reference_matrix_response[direction].col(state) =
          -solution.tail(states);
    }
  }

  const Eigen::MatrixXd selected_units =
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(overlap_roots)
          .householderQ() *
      Eigen::MatrixXd::Identity(n, states);
  const Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(n, n) -
      selected_units * selected_units.transpose();
  std::vector<xmvb::core::EigenResponseRecycleSpace> storage(states);
  for (int state = 0; state < states; ++state) {
    for (int direction = 0; direction < 2; ++direction) {
      const Eigen::Matrix2d internal =
          metric.partialPivLu().solve(gauge[direction]);
      const Eigen::VectorXd external =
          reference_dc[direction].col(state) -
          roots * internal.col(state);
      const Eigen::VectorXd image = projector *
          (hamiltonian - energies[state] * overlap) * external;
      if (!storage[state].append(external, image)) return false;
    }
  }
  const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces{
      &storage[0], &storage[1]};
  const xmvb::core::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::core::GeneralizedEigenActionResult{
            hamiltonian * vectors, overlap * vectors};
      };
  std::vector<xmvb::core::EigenResponseRecycleSpace> iterative_storage(
      states);
  std::vector<xmvb::core::EigenResponseRecycleSpace*> iterative_spaces{
      &iterative_storage[0], &iterative_storage[1]};
  const auto iterative = xmvb::core::
      solve_equal_weight_generalized_eigen_subspace_response(
          action, hamiltonian.diagonal(), overlap.diagonal(), energies, roots,
          overlap_roots, delta_hamiltonians[0] * roots,
          delta_overlaps[0] * roots, {2 * n, 1.0e-11}, iterative_spaces,
          &ritz_residuals);
  const auto iterative_recycled = xmvb::core::
      solve_equal_weight_generalized_eigen_subspace_response(
          action, hamiltonian.diagonal(), overlap.diagonal(), energies, roots,
          overlap_roots, delta_hamiltonians[0] * roots,
          delta_overlaps[0] * roots, {2 * n, 1.0e-11}, iterative_spaces,
          &ritz_residuals);
  std::array<xmvb::core::EigenSubspaceResponseResult, 2> response;
  for (int direction = 0; direction < 2; ++direction) {
    response[direction] = xmvb::core::
        evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
            action, energies, roots, overlap_roots,
            delta_hamiltonians[direction] * roots,
            delta_overlaps[direction] * roots, spaces, &ritz_residuals);
  }
  double reference_error = 0.0;
  for (int direction = 0; direction < 2; ++direction) {
    reference_error = std::max(reference_error,
        (response[direction].eigenvector_response -
         reference_dc[direction]).norm());
    reference_error = std::max(reference_error,
        (response[direction].selected_matrix_response -
         reference_matrix_response[direction]).norm());
  }
  const double uv =
      2.0 * (forcing[0].array() * reference_dc[1].array()).sum() +
      2.0 * (gauge[0].array() *
             reference_matrix_response[1].array()).sum();
  const double vu =
      2.0 * (forcing[1].array() * reference_dc[0].array()).sum() +
      2.0 * (gauge[1].array() *
             reference_matrix_response[0].array()).sum();
  const double iterative_reference_error = std::max(
      (iterative.eigenvector_response - reference_dc[0]).norm(),
      (iterative.selected_matrix_response -
       reference_matrix_response[0]).norm());
  const bool iterative_recycled_without_iteration = std::all_of(
      iterative_recycled.iterations.begin(),
      iterative_recycled.iterations.end(),
      [](int iterations) { return iterations == 0; });
  std::cout << "finite_ritz_equal_weight frozen_error=" << reference_error
            << " iterative_error=" << iterative_reference_error
            << " frozen_residual="
            << std::max(response[0].relative_residual_norms.maxCoeff(),
                        response[1].relative_residual_norms.maxCoeff())
            << " iterative_residual="
            << iterative.relative_residual_norms.maxCoeff()
            << " recycled_iterations=" << iterative_recycled.iterations[0]
            << ',' << iterative_recycled.iterations[1] << '\n';
  return ritz_residuals.norm() > 1.0e-3 &&
      response[0].block_actions == 2 && response[1].block_actions == 2 &&
      response[0].relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      response[1].relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      iterative.relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      iterative_recycled.relative_residual_norms.maxCoeff() <= 1.0e-11 &&
      reference_error <= 1.0e-11 && iterative_reference_error <= 1.0e-10 &&
      iterative_recycled_without_iteration &&
      (iterative_recycled.eigenvector_response -
       iterative.eigenvector_response).norm() <= 1.0e-10 &&
      (iterative_recycled.selected_matrix_response -
       iterative.selected_matrix_response).norm() <= 1.0e-10 &&
      std::abs(uv - vu) <= 1.0e-11 * std::max({1.0, std::abs(uv),
                                               std::abs(vu)});
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
  // A noncommuting perturbation must train a nonzero response space. A shift
  // of the same test matrix changes only the metric gauge of these roots.
  const Eigen::MatrixXd delta_hamiltonian =
      directional_test_matrix(n);
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
  std::vector<xmvb::core::EigenResponseRecycleSpace> recycle_storage(
      roots.size());
  std::vector<xmvb::core::EigenResponseRecycleSpace*> recycle_spaces;
  recycle_spaces.reserve(roots.size());
  for (auto& space : recycle_storage) recycle_spaces.push_back(&space);
  const auto cache_seed = xmvb::core::solve_generalized_eigen_response(
      action, hamiltonian.diagonal(), overlap.diagonal(), eigenvalues,
      eigenvectors, overlap * eigenvectors, delta_hamiltonian_selected,
      delta_overlap_selected, options, recycle_spaces);
  const auto recycled_response = xmvb::core::solve_generalized_eigen_response(
      action, hamiltonian.diagonal(), overlap.diagonal(), eigenvalues,
      eigenvectors, overlap * eigenvectors, delta_hamiltonian_selected,
      delta_overlap_selected, options, recycle_spaces);
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
  // An inexact seed may need legitimate Galerkin refinement; certify the
  // equation and independent reference, not a particular iteration count.
  const bool recycling_passed =
      cache_seed.relative_residual_norms.maxCoeff() <=
          options.relative_residual_tolerance &&
      std::all_of(
          recycle_storage.begin(), recycle_storage.end(),
          [n](const auto& space) {
            return space.dimension() == n && space.size() > 0;
          }) &&
      (spectral_response.eigenvector_response -
       recycled_response.eigenvector_response).cwiseAbs().maxCoeff() <= 1.0e-7 &&
      recycled_response.relative_residual_norms.maxCoeff() <=
          options.relative_residual_tolerance;
  std::cout << "block_width=" << widest_action
            << " actions=" << response.block_actions
            << " vector_error=" << max_vector_error
            << " energy_error=" << max_energy_error
            << " gauge_error=" << max_gauge_error
            << " residual=" << max_residual
            << " full_spectrum_difference=" << spectral_difference
            << " recycled_actions=" << recycled_response.block_actions
            << " recycled_iterations="
            << recycled_response.iterations[0] << ','
            << recycled_response.iterations[1] << ','
            << recycled_response.iterations[2]
            << (passed && spectral_passed && recycling_passed
                    ? " PASS\n" : " FAIL\n");
  const bool minres_scale_passed = test_preconditioned_minres_residual_scale();
  const bool solvable_ill_passed = test_solvable_ill_conditioned_overlap();
  const bool inexact_ritz_passed = test_inexact_ritz_root();
  const bool repeated_root_passed = test_repeated_root_bordered_candidate();
  const bool equal_weight_subspace_passed =
      test_equal_weight_subspace_response();
  const bool galerkin_application_passed =
      test_unconditional_indefinite_galerkin_application();
  const bool frozen_isolated_passed = test_frozen_isolated_root_response();
  const bool frozen_equal_weight_passed = test_frozen_equal_weight_response();
  const bool finite_ritz_isolated_passed =
      test_finite_ritz_frozen_isolated_response();
  const bool finite_ritz_equal_weight_passed =
      test_finite_ritz_frozen_equal_weight_response();
  std::cout << "test_status"
            << " base=" << passed
            << " spectral=" << spectral_passed
            << " recycling=" << recycling_passed
            << " minres_scale=" << minres_scale_passed
            << " ill_overlap=" << solvable_ill_passed
            << " inexact_ritz=" << inexact_ritz_passed
            << " repeated_root=" << repeated_root_passed
            << " equal_weight=" << equal_weight_subspace_passed
            << " galerkin=" << galerkin_application_passed
            << " frozen_isolated=" << frozen_isolated_passed
            << " frozen_equal_weight=" << frozen_equal_weight_passed
            << " finite_ritz_isolated=" << finite_ritz_isolated_passed
            << " finite_ritz_equal_weight="
            << finite_ritz_equal_weight_passed << '\n';
  return passed && spectral_passed && recycling_passed &&
      solvable_ill_passed &&
      minres_scale_passed &&
      inexact_ritz_passed && repeated_root_passed &&
      equal_weight_subspace_passed && galerkin_application_passed &&
      frozen_isolated_passed && frozen_equal_weight_passed &&
      finite_ritz_isolated_passed && finite_ritz_equal_weight_passed ? 0 : 1;
}
