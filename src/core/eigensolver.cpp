#include "core/eigensolver.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include "lapacke.h"

namespace xmvb::core {

namespace {

void validate_generalized_eigenproblem_inputs(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  const std::size_t expected_size =
      static_cast<std::size_t>(dimension) * dimension;
  if (hamiltonian_matrix.size() != expected_size ||
      overlap_matrix.size() != expected_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix and overlap_matrix must be square matrices");
  }
}

void check_generalized_eigensolver_info(lapack_int info) {
  if (info < 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: illegal argument " + std::to_string(-info));
  }
  if (info > 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: overlap_matrix is not positive definite"
        " (info=" + std::to_string(info) + ")");
  }
}

struct ProjectedEigenpairs {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
};

ProjectedEigenpairs solve_lowest_projected_eigenpairs(
    const Eigen::Ref<const Eigen::MatrixXd>& hamiltonian,
    int n_roots) {
  const int dimension = static_cast<int>(hamiltonian.rows());
  Eigen::MatrixXd matrix = hamiltonian;
  ProjectedEigenpairs result{
      Eigen::VectorXd(n_roots),
      Eigen::MatrixXd(dimension, n_roots)};
  std::vector<lapack_int> support(2 * n_roots);
  lapack_int n_converged = 0;
  const lapack_int info = LAPACKE_dsyevr(
      LAPACK_COL_MAJOR,
      'V',
      'I',
      'U',
      dimension,
      matrix.data(),
      dimension,
      0.0,
      0.0,
      1,
      n_roots,
      0.0,
      &n_converged,
      result.eigenvalues.data(),
      result.eigenvectors.data(),
      dimension,
      support.data());
  if (info != 0 || n_converged != n_roots) {
    throw std::runtime_error(
        "Davidson projected eigensolve failed (info=" +
        std::to_string(info) + ")");
  }
  return result;
}

Eigen::VectorXd precondition_residual(
    const Eigen::Ref<const Eigen::VectorXd>& diag_H,
    const Eigen::Ref<const Eigen::VectorXd>& diag_S,
    double lambda,
    const Eigen::Ref<const Eigen::VectorXd>& vector) {
  Eigen::VectorXd result(vector.size());
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    double denom = diag_H[i] - lambda * diag_S[i];
    const double denominator_floor =
        std::sqrt(std::numeric_limits<double>::epsilon()) *
        std::max({1.0, std::abs(diag_H[i]), std::abs(lambda * diag_S[i])});
    if (std::abs(denom) < denominator_floor) {
      denom = std::copysign(denominator_floor, denom == 0.0 ? 1.0 : denom);
    }
    result[i] = vector[i] / denom;
  }
  return -result;
}

void validate_davidson_options(
    int dimension,
    const DavidsonOptions& options) {
  if (dimension <= 0 || options.n_roots <= 0 ||
      options.n_roots > dimension) {
    throw std::invalid_argument("invalid Davidson eigenproblem dimensions");
  }
  if (options.max_iterations <= 0 ||
      options.max_subspace_dimension <
          std::min(dimension, 2 * options.n_roots) ||
      options.max_subspace_dimension > dimension) {
    throw std::invalid_argument("invalid Davidson iteration or subspace budget");
  }
  if (options.complete_spectrum &&
      options.max_subspace_dimension != dimension) {
    throw std::invalid_argument(
        "complete Davidson spectrum exceeds the basis workspace budget");
  }
  if (!std::isfinite(options.residual_tolerance) ||
      options.residual_tolerance <= 0.0) {
    throw std::invalid_argument("Davidson residual tolerance must be positive");
  }
  if (!std::isfinite(options.energy_tolerance) ||
      options.energy_tolerance <= 0.0) {
    throw std::invalid_argument("Davidson energy tolerance must be positive");
  }
}

void validate_davidson_diagonals(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal) {
  if (overlap_diagonal.size() != hamiltonian_diagonal.size() ||
      !hamiltonian_diagonal.allFinite() ||
      !overlap_diagonal.allFinite() ||
      (overlap_diagonal.array() <= 0.0).any()) {
    throw std::invalid_argument("invalid generalized eigenproblem diagonals");
  }
}

void validate_action_result(
    const GeneralizedEigenActionResult& result,
    int dimension,
    int block_width) {
  if (result.hamiltonian.rows() != dimension ||
      result.hamiltonian.cols() != block_width ||
      result.overlap.rows() != dimension ||
      result.overlap.cols() != block_width ||
      !result.hamiltonian.allFinite() ||
      !result.overlap.allFinite()) {
    throw std::runtime_error(
        "generalized eigenvalue action returned invalid block images");
  }
}

GeneralizedEigenActionResult apply_checked(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int dimension,
    int* block_actions) {
  if (!action) {
    throw std::invalid_argument("generalized eigenvalue action is empty");
  }
  GeneralizedEigenActionResult result = action(vectors);
  validate_action_result(result, dimension, static_cast<int>(vectors.cols()));
  ++(*block_actions);
  return result;
}

int append_s_orthonormal_block(
    Eigen::MatrixXd candidates,
    Eigen::MatrixXd hamiltonian_candidates,
    Eigen::MatrixXd overlap_candidates,
    Eigen::MatrixXd* basis,
    Eigen::MatrixXd* hamiltonian_basis,
    Eigen::MatrixXd* overlap_basis,
    int active_dimension) {
  const double dependence_floor =
      64.0 * std::numeric_limits<double>::epsilon();
  for (int candidate = 0;
       candidate < candidates.cols() && active_dimension < basis->cols();
       ++candidate) {
    Eigen::VectorXd vector = candidates.col(candidate);
    Eigen::VectorXd hamiltonian_image =
        hamiltonian_candidates.col(candidate);
    Eigen::VectorXd overlap_image = overlap_candidates.col(candidate);
    const double initial_metric_scale = std::max(
        std::abs(vector.dot(overlap_image)),
        vector.norm() * overlap_image.norm());

    for (int pass = 0; pass < 2; ++pass) {
      if (active_dimension == 0) {
        break;
      }
      const Eigen::VectorXd coefficients =
          basis->leftCols(active_dimension).transpose() * overlap_image;
      vector.noalias() -=
          basis->leftCols(active_dimension) * coefficients;
      hamiltonian_image.noalias() -=
          hamiltonian_basis->leftCols(active_dimension) * coefficients;
      overlap_image.noalias() -=
          overlap_basis->leftCols(active_dimension) * coefficients;
    }

    const double metric_norm_squared = vector.dot(overlap_image);
    const double scale = vector.norm() * overlap_image.norm();
    if (!std::isfinite(metric_norm_squared) ||
        metric_norm_squared <=
            dependence_floor *
                std::max(
                    initial_metric_scale,
                    std::numeric_limits<double>::min()) ||
        metric_norm_squared <= 0.0 || !std::isfinite(scale)) {
      continue;
    }
    const double inverse_metric_norm = 1.0 / std::sqrt(metric_norm_squared);
    basis->col(active_dimension) = vector * inverse_metric_norm;
    hamiltonian_basis->col(active_dimension) =
        hamiltonian_image * inverse_metric_norm;
    overlap_basis->col(active_dimension) =
        overlap_image * inverse_metric_norm;
    ++active_dimension;
  }
  return active_dimension;
}

Eigen::MatrixXd build_initial_vectors(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    int n_vectors) {
  const int dimension = static_cast<int>(hamiltonian_diagonal.size());
  std::vector<int> diagonal_order(dimension);
  for (int index = 0; index < dimension; ++index) {
    diagonal_order[index] = index;
  }
  std::partial_sort(
      diagonal_order.begin(),
      diagonal_order.begin() + n_vectors / 2,
      diagonal_order.end(),
      [&](int left, int right) {
        return hamiltonian_diagonal[left] / overlap_diagonal[left] <
            hamiltonian_diagonal[right] / overlap_diagonal[right];
      });

  Eigen::MatrixXd vectors = Eigen::MatrixXd::Zero(dimension, n_vectors);
  const int n_diagonal_vectors = n_vectors / 2;
  for (int vector = 0; vector < n_diagonal_vectors; ++vector) {
    vectors(diagonal_order[vector], vector) = 1.0;
  }

  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (int vector = n_diagonal_vectors; vector < n_vectors; ++vector) {
    for (int row = 0; row < dimension; ++row) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      vectors(row, vector) = (state & 1ULL) == 0ULL ? -1.0 : 1.0;
    }
  }
  return vectors;
}

void update_projected_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& hamiltonian_basis,
    int first_new_column,
    int active_dimension,
    Eigen::MatrixXd* projected_hamiltonian) {
  const int n_new_columns = active_dimension - first_new_column;
  if (n_new_columns <= 0) {
    return;
  }
  const Eigen::MatrixXd new_columns =
      basis.leftCols(active_dimension).transpose() *
      hamiltonian_basis.middleCols(first_new_column, n_new_columns);
  projected_hamiltonian->block(
      0,
      first_new_column,
      active_dimension,
      n_new_columns) = new_columns;
  if (first_new_column > 0) {
    projected_hamiltonian->block(
        first_new_column,
        0,
        n_new_columns,
        first_new_column) =
        new_columns.topRows(first_new_column).transpose();
  }
  auto diagonal_block = projected_hamiltonian->block(
      first_new_column,
      first_new_column,
      n_new_columns,
      n_new_columns);
  diagonal_block =
      (0.5 * (diagonal_block + diagonal_block.transpose())).eval();
}

/** @brief Completes an action-generated S-orthonormal Davidson basis. */
DavidsonResult complete_davidson_spectrum(
    const GeneralizedEigenAction& action,
    const DavidsonOptions& options,
    Eigen::MatrixXd* basis,
    Eigen::MatrixXd* hamiltonian_basis,
    Eigen::MatrixXd* overlap_basis,
    Eigen::MatrixXd* projected_hamiltonian,
    int active_dimension,
    DavidsonResult result) {
  const int dimension = static_cast<int>(basis->rows());
  // Coordinate vectors span the structure space without assembling H or S.
  constexpr std::size_t kImageBlockBytes = 1ULL << 20;
  const int block_width = std::max(1, std::min(
      dimension,
      static_cast<int>(kImageBlockBytes /
                       (3ULL * sizeof(double) * dimension))));
  for (int first = 0;
       first < dimension && active_dimension < dimension;
       first += block_width) {
    const int count = std::min(block_width, dimension - first);
    Eigen::MatrixXd coordinates =
        Eigen::MatrixXd::Zero(dimension, count);
    for (int column = 0; column < count; ++column) {
      coordinates(first + column, column) = 1.0;
    }
    auto images = apply_checked(
        action, coordinates, dimension, &result.block_actions);
    const int previous_dimension = active_dimension;
    active_dimension = append_s_orthonormal_block(
        std::move(coordinates),
        std::move(images.hamiltonian),
        std::move(images.overlap),
        basis,
        hamiltonian_basis,
        overlap_basis,
        active_dimension);
    update_projected_hamiltonian(
        *basis,
        *hamiltonian_basis,
        previous_dimension,
        active_dimension,
        projected_hamiltonian);
  }
  if (active_dimension != dimension) {
    throw std::runtime_error(
        "Davidson action basis could not span the overlap metric");
  }

  result.peak_subspace_dimension = dimension;
  const ProjectedEigenpairs complete =
      solve_lowest_projected_eigenpairs(
          *projected_hamiltonian, dimension);
  const Eigen::MatrixXd complete_vectors =
      *basis * complete.eigenvectors;
  const Eigen::MatrixXd root_coefficients =
      complete.eigenvectors.leftCols(options.n_roots);
  const Eigen::MatrixXd hamiltonian_roots =
      *hamiltonian_basis * root_coefficients;
  const Eigen::MatrixXd overlap_roots =
      *overlap_basis * root_coefficients;
  for (int root = 0; root < options.n_roots; ++root) {
    const Eigen::VectorXd residual =
        hamiltonian_roots.col(root) -
        complete.eigenvalues[root] * overlap_roots.col(root);
    const double scale = std::max(
        1.0,
        hamiltonian_roots.col(root).norm() +
            std::abs(complete.eigenvalues[root]) *
                overlap_roots.col(root).norm());
    result.relative_residual_norms[root] = residual.norm() / scale;
    if (result.relative_residual_norms[root] > std::min(
            options.residual_tolerance,
            options.energy_tolerance /
                std::max(1.0, std::abs(complete.eigenvalues[root])))) {
      throw std::runtime_error(
          "complete Davidson spectrum disagrees with H/S actions");
    }
  }
  result.eigenpairs.eigenvalues.assign(
      complete.eigenvalues.data(),
      complete.eigenvalues.data() + dimension);
  result.eigenpairs.eigenvector_matrix.assign(
      complete_vectors.data(),
      complete_vectors.data() + complete_vectors.size());
  result.overlap_eigenvectors = overlap_roots;
  return result;
}

}  // namespace

DavidsonOptions make_davidson_options(
    int dimension,
    int n_roots,
    double energy_tolerance,
    double relative_residual_tolerance) {
  if (dimension <= 0 || n_roots <= 0 || n_roots > dimension) {
    throw std::invalid_argument("invalid Davidson eigenproblem dimensions");
  }
  if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
      !std::isfinite(relative_residual_tolerance) ||
      relative_residual_tolerance <= 0.0) {
    throw std::invalid_argument(
        "Davidson accuracy tolerances must be positive");
  }
  // The stored basis and its H/S images use three dimension-by-subspace
  // matrices; the projected Hamiltonian uses one subspace square. Cap this
  // workspace by bytes, not by a molecular or structure-count heuristic.
  constexpr long double kBasisBudgetDoubles =
      static_cast<long double>(64ULL * 1024ULL * 1024ULL) / sizeof(double);
  const long double n = static_cast<long double>(dimension);
  const long double capacity =
      2.0L * kBasisBudgetDoubles /
      (std::sqrt(9.0L * n * n + 4.0L * kBasisBudgetDoubles) + 3.0L * n);
  const int max_subspace = std::min(
      dimension,
      static_cast<int>(std::floor(capacity)));
  if (max_subspace < std::min(dimension, 2 * n_roots)) {
    throw std::runtime_error(
        "Davidson selected-root block exceeds the basis workspace budget");
  }
  return DavidsonOptions{
      n_roots,
      2 * dimension,
      max_subspace,
      energy_tolerance,
      relative_residual_tolerance};
}

GeneralizedEigenResult GeneralizedEigensolver::solve_dense(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  GeneralizedEigenResult result;
  std::vector<double> H_copy = hamiltonian_matrix;
  std::vector<double> S_copy = overlap_matrix;
  result.eigenvalues.resize(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'V', 'U', dimension,
      H_copy.data(), dimension, S_copy.data(), dimension,
      result.eigenvalues.data());
  check_generalized_eigensolver_info(info);
  result.eigenvector_matrix = std::move(H_copy);
  return result;
}

std::vector<double> GeneralizedEigensolver::solve_dense_eigenvalues(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  std::vector<double> H_copy = hamiltonian_matrix;
  std::vector<double> S_copy = overlap_matrix;
  std::vector<double> eigenvalues(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'N', 'U', dimension,
      H_copy.data(), dimension, S_copy.data(), dimension,
      eigenvalues.data());
  check_generalized_eigensolver_info(info);
  return eigenvalues;
}

DavidsonResult GeneralizedEigensolver::solve_davidson(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const DavidsonOptions& options) const {
  const int dimension = static_cast<int>(hamiltonian_diagonal.size());
  validate_davidson_diagonals(
      hamiltonian_diagonal,
      overlap_diagonal);
  validate_davidson_options(dimension, options);
  const int n_initial = std::min(
      options.max_subspace_dimension,
      2 * options.n_roots);
  const Eigen::MatrixXd initial_vectors = build_initial_vectors(
      hamiltonian_diagonal,
      overlap_diagonal,
      n_initial);
  return solve_davidson(
      action,
      hamiltonian_diagonal,
      overlap_diagonal,
      initial_vectors,
      options);
}

DavidsonResult GeneralizedEigensolver::solve_davidson(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
    const DavidsonOptions& options) const {
  const int dimension = static_cast<int>(hamiltonian_diagonal.size());
  validate_davidson_diagonals(
      hamiltonian_diagonal,
      overlap_diagonal);
  validate_davidson_options(dimension, options);
  if (initial_vectors.rows() != dimension ||
      initial_vectors.cols() < options.n_roots ||
      initial_vectors.cols() > options.max_subspace_dimension ||
      !initial_vectors.allFinite()) {
    throw std::invalid_argument("invalid Davidson initial vector block");
  }

  const int max_subspace = options.max_subspace_dimension;
  Eigen::MatrixXd basis(dimension, max_subspace);
  Eigen::MatrixXd hamiltonian_basis(dimension, max_subspace);
  Eigen::MatrixXd overlap_basis(dimension, max_subspace);
  DavidsonResult result;

  auto initial_images = apply_checked(
      action,
      initial_vectors,
      dimension,
      &result.block_actions);
  int active_dimension = append_s_orthonormal_block(
      initial_vectors,
      std::move(initial_images.hamiltonian),
      std::move(initial_images.overlap),
      &basis,
      &hamiltonian_basis,
      &overlap_basis,
      0);
  if (active_dimension < options.n_roots) {
    throw std::runtime_error(
        "Davidson initial vectors are linearly dependent in the overlap metric");
  }
  Eigen::MatrixXd projected_hamiltonian =
      Eigen::MatrixXd::Zero(max_subspace, max_subspace);
  update_projected_hamiltonian(
      basis,
      hamiltonian_basis,
      0,
      active_dimension,
      &projected_hamiltonian);

  for (int iteration = 1;
       iteration <= options.max_iterations;
       ++iteration) {
    result.iterations = iteration;
    result.peak_subspace_dimension =
        std::max(result.peak_subspace_dimension, active_dimension);
    const auto active_basis = basis.leftCols(active_dimension);
    const auto active_hamiltonian_basis =
        hamiltonian_basis.leftCols(active_dimension);
    const auto active_overlap_basis =
        overlap_basis.leftCols(active_dimension);
    const Eigen::MatrixXd active_projected_hamiltonian =
        projected_hamiltonian.topLeftCorner(
            active_dimension,
            active_dimension);
    const int n_projected_roots =
        active_dimension + options.n_roots > max_subspace
            ? active_dimension
            : options.n_roots;
    const ProjectedEigenpairs projected =
        solve_lowest_projected_eigenpairs(
            active_projected_hamiltonian,
            n_projected_roots);
    const Eigen::VectorXd& projected_eigenvalues = projected.eigenvalues;
    const Eigen::MatrixXd& projected_eigenvectors = projected.eigenvectors;
    const Eigen::MatrixXd root_coefficients =
        projected_eigenvectors.leftCols(options.n_roots);
    const Eigen::MatrixXd root_vectors = active_basis * root_coefficients;
    const Eigen::MatrixXd hamiltonian_root_vectors =
        active_hamiltonian_basis * root_coefficients;
    const Eigen::MatrixXd overlap_root_vectors =
        active_overlap_basis * root_coefficients;

    result.relative_residual_norms.assign(options.n_roots, 0.0);
    bool converged = true;
    Eigen::MatrixXd corrections(dimension, options.n_roots);
    int n_corrections = 0;
    for (int root = 0; root < options.n_roots; ++root) {
      const double eigenvalue = projected_eigenvalues[root];
      const Eigen::VectorXd residual =
          hamiltonian_root_vectors.col(root) -
          eigenvalue * overlap_root_vectors.col(root);
      const double residual_scale = std::max(
          1.0,
          hamiltonian_root_vectors.col(root).norm() +
              std::abs(eigenvalue) *
                  overlap_root_vectors.col(root).norm());
      const double relative_residual = residual.norm() / residual_scale;
      result.relative_residual_norms[root] = relative_residual;
      const double energy_scaled_tolerance =
          options.energy_tolerance / std::max(1.0, std::abs(eigenvalue));
      const double required_residual =
          std::min(options.residual_tolerance, energy_scaled_tolerance);
      if (relative_residual <= required_residual) {
        continue;
      }
      converged = false;
      corrections.col(n_corrections) = precondition_residual(
          hamiltonian_diagonal,
          overlap_diagonal,
          eigenvalue,
          residual);
      ++n_corrections;
    }

    if (converged) {
      if (options.complete_spectrum) {
        return complete_davidson_spectrum(
            action,
            options,
            &basis,
            &hamiltonian_basis,
            &overlap_basis,
            &projected_hamiltonian,
            active_dimension,
            std::move(result));
      }
      result.eigenpairs.eigenvalues.assign(
          projected_eigenvalues.data(),
          projected_eigenvalues.data() + options.n_roots);
      result.eigenpairs.eigenvector_matrix.assign(
          root_vectors.data(),
          root_vectors.data() + root_vectors.size());
      result.overlap_eigenvectors = overlap_root_vectors;
      return result;
    }

    // A complete S-orthonormal basis leaves no independent correction
    // direction. Returning an inaccurate root here would violate the outer
    // derivative contract, so report the finite-precision limit explicitly.
    if (active_dimension == dimension) {
      std::ostringstream message;
      message << "complete Davidson basis cannot meet the requested Ritz residual: "
              << *std::max_element(
                     result.relative_residual_norms.begin(),
                     result.relative_residual_norms.end());
      throw std::runtime_error(message.str());
    }

    corrections.conservativeResize(Eigen::NoChange, n_corrections);
    auto correction_images = apply_checked(
        action,
        corrections,
        dimension,
        &result.block_actions);

    if (active_dimension + n_corrections > max_subspace) {
      const int available_after_restart = max_subspace - n_corrections;
      // A truncated single-root solve releases half of the saturated Ritz
      // space, preserving useful spectral information without replacing only
      // one direction at a time. Block solves retain the largest compatible
      // space because simultaneous corrections need neighboring Ritz vectors.
      const int n_keep = max_subspace < dimension && options.n_roots == 1
          ? std::max(options.n_roots, max_subspace / 2)
          : std::min(
                active_dimension,
                std::max(options.n_roots, available_after_restart));
      const Eigen::MatrixXd keep_coefficients =
          projected_eigenvectors.leftCols(n_keep);
      const Eigen::MatrixXd restarted_basis =
          active_basis * keep_coefficients;
      const Eigen::MatrixXd restarted_hamiltonian_basis =
          active_hamiltonian_basis * keep_coefficients;
      const Eigen::MatrixXd restarted_overlap_basis =
          active_overlap_basis * keep_coefficients;
      basis.leftCols(n_keep) = restarted_basis;
      hamiltonian_basis.leftCols(n_keep) =
          restarted_hamiltonian_basis;
      overlap_basis.leftCols(n_keep) = restarted_overlap_basis;
      projected_hamiltonian.topLeftCorner(n_keep, n_keep) =
          projected_eigenvalues.head(n_keep).asDiagonal();
      active_dimension = n_keep;
    }

    const int previous_dimension = active_dimension;
    active_dimension = append_s_orthonormal_block(
        std::move(corrections),
        std::move(correction_images.hamiltonian),
        std::move(correction_images.overlap),
        &basis,
        &hamiltonian_basis,
        &overlap_basis,
        active_dimension);
    update_projected_hamiltonian(
        basis,
        hamiltonian_basis,
        previous_dimension,
        active_dimension,
        &projected_hamiltonian);
    if (active_dimension == previous_dimension) {
      throw std::runtime_error(
          "Davidson correction space became linearly dependent before convergence");
    }
  }

  std::ostringstream message;
  message << "Davidson did not converge in " << options.max_iterations
          << " iterations; largest relative residual = "
          << *std::max_element(
                 result.relative_residual_norms.begin(),
                 result.relative_residual_norms.end());
  throw std::runtime_error(message.str());
}

}  // namespace xmvb::core
