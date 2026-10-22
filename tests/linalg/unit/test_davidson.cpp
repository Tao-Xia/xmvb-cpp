#include "linalg/eigensolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
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

DenseProblem make_scaled_nonorthogonal_problem(int dimension) {
  const DenseProblem orthogonal_problem =
      make_tridiagonal_problem(dimension);
  Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(dimension, dimension);
  for (int row = 0; row < dimension; ++row) {
    const double scale = std::pow(
        2.0,
        static_cast<double>(row) /
            static_cast<double>(dimension - 1));
    factor(row, row) = scale;
    if (row > 0) {
      factor(row, row - 1) = 0.15 * scale;
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
    double eigenvector_tolerance,
    bool identity_overlap = false) {
  const int dimension = static_cast<int>(problem.hamiltonian.rows());
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        xmvb::linalg::GeneralizedEigenActionResult images;
        images.hamiltonian.noalias() = problem.hamiltonian * vectors;
        if (!identity_overlap) images.overlap.noalias() = problem.overlap * vectors;
        return images;
      };
  xmvb::linalg::DavidsonOptions options{
      n_roots,
      4 * dimension,
      std::min(dimension, std::max(128, 24 * n_roots)),
      1.0,
      2.0e-8,
      2.0e-8};
  options.identity_overlap=identity_overlap;

  const xmvb::linalg::GeneralizedEigensolver solver;
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
  const double hamiltonian_image_error =
      (result.hamiltonian_eigenvectors - problem.hamiltonian * computed_roots)
          .cwiseAbs().maxCoeff();
  const double max_residual = *std::max_element(
      result.relative_residual_norms.begin(),
      result.relative_residual_norms.end());
  const bool passed =
      eigenvalue_error <= 1.0e-8 &&
      eigenvector_error <= eigenvector_tolerance &&
      orthonormality_error <= 1.0e-10 &&
      overlap_image_error <= 1.0e-12 &&
      hamiltonian_image_error <= 1.0e-12 &&
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
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        xmvb::linalg::GeneralizedEigenActionResult images;
        images.hamiltonian.noalias() = problem.hamiltonian * vectors;
        images.overlap.noalias() = problem.overlap * vectors;
        return images;
      };
  const xmvb::linalg::DavidsonOptions options{
      n_roots,
      4 * dimension,
      128,
      1.0,
      2.0e-8,
      2.0e-8};
  const xmvb::linalg::GeneralizedEigensolver solver;
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

bool run_residual_preconditioner_case() {
  constexpr int dimension = 160;
  constexpr int n_roots = 4;
  const DenseProblem problem = make_nonorthogonal_problem(dimension);
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::linalg::GeneralizedEigenActionResult{
            problem.hamiltonian * vectors,
            problem.overlap * vectors};
      };
  const xmvb::linalg::GeneralizedEigensolver solver;
  const auto reference = solver.solve_dense(
      flatten(problem.hamiltonian), flatten(problem.overlap), dimension);
  xmvb::linalg::DavidsonOptions options{
      n_roots, 4 * dimension, 128, 1.0, 2.0e-8, 2.0e-8};
  const auto diagonal = solver.solve_davidson(
      action, problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(), options);

  int inverse_calls = 0;
  int observed_iterations = 0;
  bool observations_valid = true;
  double last_energy = 0.0;
  double last_relative_residual = 0.0;
  double last_stationarity_residual = 0.0;
  options.residual_preconditioner =
      [&](const Eigen::VectorXd& residual, double energy) {
        ++inverse_calls;
        Eigen::VectorXd inverse_image(residual.size());
        for (Eigen::Index row = 0; row < residual.size(); ++row) {
          double denominator =
              problem.hamiltonian(row, row) -
              energy * problem.overlap(row, row);
          const double floor =
              std::sqrt(std::numeric_limits<double>::epsilon()) *
              std::max({1.0, std::abs(problem.hamiltonian(row, row)),
                        std::abs(energy * problem.overlap(row, row))});
          if (std::abs(denominator) < floor) {
            denominator = std::copysign(
                floor, denominator == 0.0 ? 1.0 : denominator);
          }
          inverse_image[row] = residual[row] / denominator;
        }
        return inverse_image;
      };
  options.observe_iteration =
      [&](int iteration, int active_subspace, double lowest_energy,
          double largest_relative_residual, double largest_stationarity_residual) {
        observations_valid = observations_valid &&
            iteration == ++observed_iterations &&
            active_subspace >= n_roots &&
            active_subspace <= options.max_subspace_dimension &&
            std::isfinite(lowest_energy) &&
            std::isfinite(largest_relative_residual) &&
            largest_relative_residual >= 0.0 &&
            std::isfinite(largest_stationarity_residual) && largest_stationarity_residual >= 0.0;
        last_energy = lowest_energy;
        last_relative_residual = largest_relative_residual;
        last_stationarity_residual = largest_stationarity_residual;
      };
  const auto preconditioned = solver.solve_davidson(
      action, problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(), options);
  const Eigen::Map<const Eigen::MatrixXd> roots(
      preconditioned.eigenpairs.eigenvector_matrix.data(),
      dimension, n_roots);
  double max_explicit_relative_residual = 0.0;
  double max_reported_residual_error = 0.0;
  for (int root = 0; root < n_roots; ++root) {
    const double energy = preconditioned.eigenpairs.eigenvalues[root];
    const Eigen::VectorXd h_image = problem.hamiltonian * roots.col(root);
    const Eigen::VectorXd s_image = problem.overlap * roots.col(root);
    const double relative_residual =
        (h_image - energy * s_image).norm() /
        std::max(1.0, h_image.norm() + std::abs(energy) * s_image.norm());
    max_explicit_relative_residual =
        std::max(max_explicit_relative_residual, relative_residual);
    max_reported_residual_error = std::max(
        max_reported_residual_error,
        std::abs(relative_residual -
                 preconditioned.relative_residual_norms[root]));
  }
  const double reported_max = *std::max_element(
      preconditioned.relative_residual_norms.begin(),
      preconditioned.relative_residual_norms.end());
  const bool passed =
      inverse_calls > 0 && observations_valid &&
      observed_iterations == preconditioned.iterations &&
      last_energy == preconditioned.eigenpairs.eigenvalues[0] &&
      last_relative_residual == reported_max &&
      last_stationarity_residual == *std::max_element(preconditioned.stationarity_residual_norms.begin(),
          preconditioned.stationarity_residual_norms.end()) &&
      max_eigenvalue_error(
          reference.eigenvalues, preconditioned.eigenpairs.eigenvalues,
          n_roots) <= 1.0e-8 &&
      max_eigenvalue_error(
          diagonal.eigenpairs.eigenvalues,
          preconditioned.eigenpairs.eigenvalues, n_roots) <= 1.0e-10 &&
      s_orthonormality_error(
          preconditioned.eigenpairs.eigenvector_matrix,
          problem.overlap, n_roots) <= 1.0e-10 &&
      max_explicit_relative_residual <= options.residual_tolerance &&
      max_reported_residual_error <= 1.0e-12;
  std::cout << "nonorthogonal callback inverse"
            << " calls=" << inverse_calls
            << " observations=" << observed_iterations
            << " iterations=" << preconditioned.iterations
            << " residual=" << max_explicit_relative_residual
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_invalid_residual_preconditioner_cases() {
  constexpr int dimension = 8;
  const DenseProblem problem = make_tridiagonal_problem(dimension);
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::linalg::GeneralizedEigenActionResult{
            problem.hamiltonian * vectors,
            problem.overlap * vectors};
      };
  Eigen::MatrixXd initial = Eigen::MatrixXd::Zero(dimension, 2);
  initial(0, 0) = 1.0;
  initial(1, 1) = 1.0;
  auto options = xmvb::linalg::make_davidson_options(
      dimension, 1, 1.0e-7, 1.0e-8, 1.0e-8);
  const xmvb::linalg::GeneralizedEigensolver solver;
  auto rejects = [&](auto inverse) {
    options.residual_preconditioner = inverse;
    try {
      solver.solve_davidson(
          action, problem.hamiltonian.diagonal(),
          problem.overlap.diagonal(), initial, options);
    } catch (const std::runtime_error& error) {
      return std::string(error.what()).find(
          "residual preconditioner returned an invalid vector") !=
          std::string::npos;
    }
    return false;
  };
  const bool wrong_size = rejects([](const Eigen::VectorXd&, double) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dimension - 1);
    return result;
  });
  const bool nonfinite = rejects([](const Eigen::VectorXd&, double) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dimension);
    result[0] = std::numeric_limits<double>::quiet_NaN();
    return result;
  });
  // A zero correction cannot extend the basis. Preserve its residual and
  // rank scalars in the exception rather than reporting only "dependent".
  options.residual_preconditioner=[](const Eigen::VectorXd&,double) {
    return Eigen::VectorXd(Eigen::VectorXd::Zero(dimension));
  };
  bool diagnosed=false;
  try {
    solver.solve_davidson(action,problem.hamiltonian.diagonal(),
        problem.overlap.diagonal(),initial,options);
  } catch (const std::runtime_error& error) {
    const std::string message=error.what();
    const std::string metric_key="projected metric norm squared=";
    const auto metric_begin=message.find(metric_key);
    diagnosed=message.find("linearly dependent before convergence")!=std::string::npos &&
        message.find("max stationarity residual=")!=std::string::npos &&
        metric_begin!=std::string::npos &&
        std::stod(message.substr(metric_begin+metric_key.size()))==0.0 &&
        message.find("dependence threshold=")!=std::string::npos;
    if (!diagnosed) std::cerr<<"dependent correction diagnostic: "<<message<<'\n';
  }
  const bool passed = wrong_size && nonfinite && diagnosed;
  std::cout << "invalid callback inverse"
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_physical_residual_case() {
  constexpr double scale = 1.0e-4;
  const Eigen::Vector3d energies(1.0, 2.0, 4.0);
  const Eigen::Matrix3d h = scale * scale * energies.asDiagonal();
  const Eigen::Matrix3d m = scale * scale * Eigen::Matrix3d::Identity();
  const xmvb::linalg::GeneralizedEigenAction action =
      [h, m](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::linalg::GeneralizedEigenActionResult{h * vectors, m * vectors};
      };
  Eigen::Matrix<double, 3, 2> initial;
  initial << 1.0 / scale, 0.0,
             1.0e-4 / scale, 0.0,
             0.0, 1.0 / scale;
  auto options = xmvb::linalg::make_davidson_options(3, 1, 1e-7, 1e-3, 1e-7);
  const auto physical_residual = [energies, scale](const Eigen::VectorXd& coefficients, double energy) {
    const Eigen::VectorXd state = scale * coefficients;
    return (energies.array() * state.array() - energy * state.array()).matrix().norm() /
        state.norm();
  };
  const xmvb::linalg::GeneralizedEigensolver solver;
  const auto algebraic = solver.solve_davidson(action, h.diagonal(), m.diagonal(), initial, options);
  const Eigen::Map<const Eigen::VectorXd> old_root(algebraic.eigenpairs.eigenvector_matrix.data(), 3);
  const double old_error = physical_residual(old_root, algebraic.eigenpairs.eigenvalues[0]);
  int certificates = 0;
  options.physical_residual_norm = [&](const Eigen::VectorXd& coefficients, double energy) {
    ++certificates;
    return physical_residual(coefficients, energy);
  };
  const auto certified = solver.solve_davidson(action, h.diagonal(), m.diagonal(), initial, options);
  const Eigen::Map<const Eigen::VectorXd> new_root(certified.eigenpairs.eigenvector_matrix.data(), 3);
  const double new_error = physical_residual(new_root, certified.eigenpairs.eigenvalues[0]);
  const bool passed = algebraic.iterations == 1 && old_error > options.stationarity_tolerance &&
      certified.iterations > 1 && certificates > 0 && new_error <= options.stationarity_tolerance;
  std::cout << "physical residual old=" << old_error << " certified=" << new_error
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_outer_accuracy_case() {
  constexpr int dimension = 400;
  constexpr int n_roots = 2;
  const DenseProblem problem = make_tridiagonal_problem(dimension);
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::linalg::GeneralizedEigenActionResult{
            problem.hamiltonian * vectors,
            problem.overlap * vectors};
      };
  const xmvb::linalg::DavidsonOptions options{
      n_roots,
      2 * dimension,
      128,
      1.0e-7,
      1.0e-3,
      1.0e-7};
  const xmvb::linalg::GeneralizedEigensolver solver;
  const auto reference = solver.solve_dense(
      flatten(problem.hamiltonian),
      flatten(problem.overlap),
      dimension);
  const auto result = solver.solve_davidson(
      action,
      problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(),
      options);
  const double eigenvalue_error = max_eigenvalue_error(
      reference.eigenvalues,
      result.eigenpairs.eigenvalues,
      n_roots);
  const double residual = *std::max_element(
      result.relative_residual_norms.begin(),
      result.relative_residual_norms.end());
  const bool passed =
      eigenvalue_error <= options.energy_tolerance &&
      residual <= options.residual_tolerance &&
      *std::max_element(
          result.stationarity_residual_norms.begin(),
          result.stationarity_residual_norms.end()) <=
          options.stationarity_tolerance;
  std::cout << "outer-accuracy Davidson"
            << " iterations=" << result.iterations
            << " eigenvalue_error=" << eigenvalue_error
            << " residual=" << residual
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_bounded_multiroot_restart_case(bool identity_overlap = false) {
  constexpr int dimension = 600;
  constexpr int n_roots = 3;
  const DenseProblem problem = make_tridiagonal_problem(dimension);
  const xmvb::linalg::GeneralizedEigenAction action =
      [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        return xmvb::linalg::GeneralizedEigenActionResult{
            problem.hamiltonian * vectors,
            identity_overlap ? Eigen::MatrixXd{} : Eigen::MatrixXd(problem.overlap * vectors)};
      };
  xmvb::linalg::DavidsonOptions options{
      n_roots,
      300,
      128,
      1.0,
      2.0e-8,
      2.0e-8};
  options.identity_overlap=identity_overlap;
  const xmvb::linalg::GeneralizedEigensolver solver;
  const auto reference = solver.solve_dense(
      flatten(problem.hamiltonian),
      flatten(problem.overlap),
      dimension);
  const auto result = solver.solve_davidson(
      action,
      problem.hamiltonian.diagonal(),
      problem.overlap.diagonal(),
      options);
  const double eigenvalue_error = max_eigenvalue_error(
      reference.eigenvalues,
      result.eigenpairs.eigenvalues,
      n_roots);
  const bool passed =
      result.iterations <= options.max_iterations &&
      result.peak_subspace_dimension == options.max_subspace_dimension - 1 &&
      eigenvalue_error <= 1.0e-8;
  std::cout << "bounded multiroot restart identity=" << identity_overlap
            << " iterations=" << result.iterations
            << " actions=" << result.block_actions
            << " subspace=" << result.peak_subspace_dimension
            << " eigenvalue_error=" << eigenvalue_error
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

bool run_default_option_cases() {
  const auto small =
      xmvb::linalg::make_davidson_options(3, 2, 1.0e-7, 1.0e-3, 1.0e-7);
  const auto large =
      xmvb::linalg::make_davidson_options(10000, 4, 1.0e-7, 1.0e-3, 1.0e-7);
  const bool passed =
      small.n_roots == 2 &&
      small.max_subspace_dimension == 3 &&
      small.max_iterations >= 3 &&
      small.energy_tolerance == 1.0e-7 &&
      small.residual_tolerance == 1.0e-3 &&
      small.stationarity_tolerance == 1.0e-7 &&
      large.n_roots == 4 &&
      large.max_subspace_dimension < 10000 &&
      large.max_subspace_dimension >= 2 * large.n_roots;
  std::cout << "default Davidson budgets"
            << " small_subspace=" << small.max_subspace_dimension
            << " large_subspace=" << large.max_subspace_dimension
            << (passed ? " PASS\n" : " FAIL\n");
  return passed;
}

/** @brief A large energy offset must not conceal a first-order CI defect. */
bool run_stationarity_case() {
  constexpr int dimension = 4;
  Eigen::MatrixXd initial = Eigen::MatrixXd::Zero(dimension, 2);
  initial(0, 0) = 1.0;
  initial(1, 0) = 1.5e-6;
  initial(2, 1) = 1.0;
  Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(dimension, dimension);
  derivative(0, 1) = derivative(1, 0) = 1000.0;
  const auto options = xmvb::linalg::make_davidson_options(
      dimension, 1, 1.0e-7, 1.0e-3, 1.0e-7);
  const xmvb::linalg::GeneralizedEigensolver solver;
  bool passed = true;
  for (double shift : {0.0, -2000.0}) {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dimension, dimension);
    h.diagonal() << shift, shift + 0.1, shift + 1.0, shift + 2.0;
    const auto result = solver.solve_davidson(
        [&h](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
          return xmvb::linalg::GeneralizedEigenActionResult{h * vectors, vectors};
        },
        h.diagonal(), Eigen::VectorXd::Ones(dimension), initial, options);
    const Eigen::Map<const Eigen::VectorXd> c(
        result.eigenpairs.eigenvector_matrix.data(), dimension);
    const double gradient_error = std::abs(c.dot(derivative * c));
    const bool case_passed =
        result.stationarity_residual_norms[0] <= options.stationarity_tolerance &&
        gradient_error < 1.0e-8 && result.iterations > 1;
    passed = case_passed && passed;
    std::cout << "CI stationarity shift=" << shift
              << " iterations=" << result.iterations
              << " gradient_error=" << gradient_error
              << " stationarity=" << result.stationarity_residual_norms[0]
              << (case_passed ? " PASS\n" : " FAIL\n");
  }
  return passed;
}

}  // namespace

int main() {
  bool passed = true;
  passed = run_default_option_cases() && passed;
  passed = run_stationarity_case() && passed;
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
  passed = run_case(
               "scaled nonorthogonal n=128",
               make_scaled_nonorthogonal_problem(128),
               1,
               2.0e-5) &&
      passed;
  passed = run_recycled_case() && passed;
  for (int roots : {1,3})
    passed = run_case("identity metric n=200",make_tridiagonal_problem(200),
        roots,1.0e-5,true) && passed;
  passed = run_residual_preconditioner_case() && passed;
  passed = run_invalid_residual_preconditioner_cases() && passed;
  passed = run_outer_accuracy_case() && passed;
  passed = run_physical_residual_case() && passed;
  passed = run_bounded_multiroot_restart_case() && passed;
  passed = run_bounded_multiroot_restart_case(true) && passed;
  std::cout << (passed ? "ALL PASSED\n" : "SOME FAILED\n");
  return passed ? 0 : 1;
}
