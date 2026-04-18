#include "vb/biorthogonal_vbscf/biorthogonal_projected_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

using ComplexScalar = std::complex<double>;
using ComplexMatrix = Eigen::Matrix<
    ComplexScalar,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::ColMajor>;
using ComplexVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

struct SortedMatchedEigenpairs {
  std::vector<int> right_indices;
  std::vector<int> left_indices;
};

SortedMatchedEigenpairs sort_and_match_eigenpairs(
    const Eigen::VectorXcd& right_eigenvalues,
    const Eigen::VectorXcd& left_eigenvalues) {
  if (right_eigenvalues.size() != left_eigenvalues.size()) {
    throw std::invalid_argument("left/right eigenvalue counts are inconsistent");
  }

  SortedMatchedEigenpairs result;
  result.right_indices.resize(xmvb::to_size(right_eigenvalues.size()));
  std::iota(result.right_indices.begin(), result.right_indices.end(), 0);
  std::stable_sort(
      result.right_indices.begin(),
      result.right_indices.end(),
      [&right_eigenvalues](int left_index, int right_index) {
        const ComplexScalar left_value = right_eigenvalues(left_index);
        const ComplexScalar right_value = right_eigenvalues(right_index);
        if (left_value.real() != right_value.real()) {
          return left_value.real() < right_value.real();
        }
        return left_value.imag() < right_value.imag();
      });

  result.left_indices.assign(result.right_indices.size(), -1);
  std::vector<bool> used_left_indices(result.right_indices.size(), false);
  for (std::size_t state_index = 0; state_index < result.right_indices.size(); ++state_index) {
    const int right_index = result.right_indices[state_index];
    double best_distance = std::numeric_limits<double>::infinity();
    int best_left_index = -1;
    for (int left_index = 0; left_index < left_eigenvalues.size(); ++left_index) {
      if (used_left_indices[xmvb::to_size(left_index)]) {
        continue;
      }
      const double distance =
          std::abs(left_eigenvalues(left_index) - right_eigenvalues(right_index));
      if (distance < best_distance) {
        best_distance = distance;
        best_left_index = left_index;
      }
    }
    if (best_left_index < 0) {
      throw std::runtime_error("failed to match a left eigenpair");
    }
    used_left_indices[xmvb::to_size(best_left_index)] = true;
    result.left_indices[state_index] = best_left_index;
  }

  return result;
}

double max_imaginary_magnitude(const ComplexVector& vector) {
  double max_magnitude = 0.0;
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    max_magnitude = std::max(max_magnitude, std::abs(vector(index).imag()));
  }
  return max_magnitude;
}

Eigen::MatrixXd require_real_columns(
    const ComplexMatrix& complex_matrix,
    const std::vector<int>& column_indices,
    double imaginary_tolerance,
    double* max_imaginary_component) {
  if (max_imaginary_component == nullptr) {
    throw std::invalid_argument("max_imaginary_component must not be null");
  }

  Eigen::MatrixXd real_matrix(
      complex_matrix.rows(),
      static_cast<int>(column_indices.size()));
  for (std::size_t column_offset = 0; column_offset < column_indices.size(); ++column_offset) {
    const ComplexVector complex_vector = complex_matrix.col(column_indices[column_offset]);
    const double vector_max_imaginary = max_imaginary_magnitude(complex_vector);
    *max_imaginary_component = std::max(*max_imaginary_component, vector_max_imaginary);
    if (vector_max_imaginary > imaginary_tolerance) {
      throw std::runtime_error(
          "non-negligible imaginary component detected in biorthogonal eigenvector");
    }
    real_matrix.col(static_cast<int>(column_offset)) = complex_vector.real();
  }
  return real_matrix;
}

}  // namespace

BiorthogonalProjectedStructureSolveResult solve_biorthogonal_projected_structure_problem(
    const BiorthogonalProjectedStructureProblem& projected_problem,
    const BiorthogonalSelectedStructureSpace& structure_space,
    double imaginary_tolerance) {
  if (imaginary_tolerance < 0.0) {
    throw std::invalid_argument("imaginary_tolerance must be non-negative");
  }

  validate_biorthogonal_selected_structure_space(structure_space, 0.0);
  validate_biorthogonal_projected_structure_problem(
      projected_problem,
      structure_space.structure_to_determinant.cols(),
      1.0e-10);

  Eigen::EigenSolver<Eigen::MatrixXd> right_solver(
      projected_problem.orthogonalized_structure_hamiltonian,
      true);
  if (right_solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to solve right biorthogonal structure eigenproblem");
  }

  Eigen::EigenSolver<Eigen::MatrixXd> left_solver(
      projected_problem.orthogonalized_structure_hamiltonian.transpose(),
      true);
  if (left_solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to solve left biorthogonal structure eigenproblem");
  }

  const SortedMatchedEigenpairs matched_pairs = sort_and_match_eigenpairs(
      right_solver.eigenvalues(),
      left_solver.eigenvalues());

  BiorthogonalProjectedStructureSolveResult result;
  const int n_structures = structure_space.structure_to_determinant.cols();
  result.eigenvalues.resize(xmvb::to_size(n_structures), 0.0);
  result.right_orthogonalized_coefficient_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  result.left_orthogonalized_coefficient_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  result.right_structure_coefficient_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  result.left_structure_coefficient_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  result.right_residual_norms.resize(xmvb::to_size(n_structures), 0.0);
  result.left_residual_norms.resize(xmvb::to_size(n_structures), 0.0);

  const ComplexMatrix right_eigenvectors = right_solver.eigenvectors();
  const ComplexMatrix left_eigenvectors = left_solver.eigenvectors();

  for (int state_index = 0; state_index < n_structures; ++state_index) {
    const int right_index = matched_pairs.right_indices[xmvb::to_size(state_index)];
    const int left_index = matched_pairs.left_indices[xmvb::to_size(state_index)];

    const ComplexScalar right_eigenvalue = right_solver.eigenvalues()(right_index);
    const ComplexScalar left_eigenvalue = left_solver.eigenvalues()(left_index);
    result.max_eigenvalue_imaginary_magnitude = std::max(
        result.max_eigenvalue_imaginary_magnitude,
        std::max(std::abs(right_eigenvalue.imag()), std::abs(left_eigenvalue.imag())));
    if (std::abs(right_eigenvalue.imag()) > imaginary_tolerance ||
        std::abs(left_eigenvalue.imag()) > imaginary_tolerance) {
      throw std::runtime_error(
          "complex biorthogonal eigenvalues exceed the allowed tolerance");
    }
    if (std::abs(right_eigenvalue - left_eigenvalue) > imaginary_tolerance) {
      throw std::runtime_error("left/right eigenvalue pairing is inconsistent");
    }

    const ComplexVector right_complex_vector = right_eigenvectors.col(right_index);
    const ComplexVector left_complex_vector = left_eigenvectors.col(left_index);
    const double right_vector_max_imaginary = max_imaginary_magnitude(right_complex_vector);
    const double left_vector_max_imaginary = max_imaginary_magnitude(left_complex_vector);
    result.max_right_vector_imaginary_magnitude = std::max(
        result.max_right_vector_imaginary_magnitude,
        right_vector_max_imaginary);
    result.max_left_vector_imaginary_magnitude = std::max(
        result.max_left_vector_imaginary_magnitude,
        left_vector_max_imaginary);
    if (right_vector_max_imaginary > imaginary_tolerance ||
        left_vector_max_imaginary > imaginary_tolerance) {
      throw std::runtime_error(
          "complex biorthogonal eigenvectors exceed the allowed tolerance");
    }

    Eigen::MatrixXd::ColXpr right_vector =
        result.right_orthogonalized_coefficient_matrix.col(state_index);
    Eigen::MatrixXd::ColXpr left_vector =
        result.left_orthogonalized_coefficient_matrix.col(state_index);
    right_vector = right_complex_vector.real();
    left_vector = left_complex_vector.real();

    const double right_norm = right_vector.norm();
    const double left_norm = left_vector.norm();
    if (!(right_norm > 0.0) || !(left_norm > 0.0)) {
      throw std::runtime_error("encountered a zero-norm biorthogonal eigenvector");
    }
    right_vector /= right_norm;
    left_vector /= left_norm;

    const double overlap = left_vector.dot(right_vector);
    if (std::abs(overlap) <= imaginary_tolerance) {
      throw std::runtime_error("left/right eigenvectors are nearly orthogonal");
    }
    left_vector /= overlap;

    result.eigenvalues[xmvb::to_size(state_index)] = right_eigenvalue.real();
    result.right_structure_coefficient_matrix.col(state_index) =
        structure_space.fixed_metric_factor_inverse.transpose() * right_vector;
    result.left_structure_coefficient_matrix.col(state_index) =
        structure_space.fixed_metric_factor_inverse.transpose() * left_vector;

    const Eigen::MatrixXd::ColXpr right_structure_vector =
        result.right_structure_coefficient_matrix.col(state_index);
    const Eigen::MatrixXd::ColXpr left_structure_vector =
        result.left_structure_coefficient_matrix.col(state_index);
    result.right_residual_norms[xmvb::to_size(state_index)] =
        (projected_problem.selected_structure_hamiltonian * right_structure_vector -
         result.eigenvalues[xmvb::to_size(state_index)] *
             structure_space.fixed_metric * right_structure_vector)
            .norm();
    result.left_residual_norms[xmvb::to_size(state_index)] =
        (projected_problem.selected_structure_hamiltonian.transpose() *
             left_structure_vector -
         result.eigenvalues[xmvb::to_size(state_index)] *
             structure_space.fixed_metric * left_structure_vector)
            .norm();
  }

  result.euclidean_biorthogonality_matrix =
      result.left_orthogonalized_coefficient_matrix.transpose() *
      result.right_orthogonalized_coefficient_matrix;
  result.metric_biorthogonality_matrix =
      result.left_structure_coefficient_matrix.transpose() *
      structure_space.fixed_metric *
      result.right_structure_coefficient_matrix;
  throw_if_nonfinite(
      result.right_orthogonalized_coefficient_matrix,
      "right_orthogonalized_coefficient_matrix");
  throw_if_nonfinite(
      result.left_orthogonalized_coefficient_matrix,
      "left_orthogonalized_coefficient_matrix");
  throw_if_nonfinite(
      result.right_structure_coefficient_matrix,
      "right_structure_coefficient_matrix");
  throw_if_nonfinite(
      result.left_structure_coefficient_matrix,
      "left_structure_coefficient_matrix");
  throw_if_nonfinite(
      result.euclidean_biorthogonality_matrix,
      "euclidean_biorthogonality_matrix");
  throw_if_nonfinite(
      result.metric_biorthogonality_matrix,
      "metric_biorthogonality_matrix");
  return result;
}

void validate_biorthogonal_projected_structure_solve_result(
    const BiorthogonalProjectedStructureSolveResult& solve_result,
    int expected_structure_count,
    double residual_tolerance,
    double biorthogonality_tolerance) {
  if (expected_structure_count <= 0) {
    throw std::invalid_argument("expected_structure_count must be positive");
  }
  if (residual_tolerance < 0.0 || biorthogonality_tolerance < 0.0) {
    throw std::invalid_argument("validation tolerances must be non-negative");
  }
  if (static_cast<int>(solve_result.eigenvalues.size()) != expected_structure_count) {
    throw std::invalid_argument("eigenvalue count is inconsistent");
  }
  if (solve_result.right_orthogonalized_coefficient_matrix.rows() !=
          expected_structure_count ||
      solve_result.right_orthogonalized_coefficient_matrix.cols() !=
          expected_structure_count ||
      solve_result.left_orthogonalized_coefficient_matrix.rows() !=
          expected_structure_count ||
      solve_result.left_orthogonalized_coefficient_matrix.cols() !=
          expected_structure_count ||
      solve_result.right_structure_coefficient_matrix.rows() !=
          expected_structure_count ||
      solve_result.right_structure_coefficient_matrix.cols() !=
          expected_structure_count ||
      solve_result.left_structure_coefficient_matrix.rows() !=
          expected_structure_count ||
      solve_result.left_structure_coefficient_matrix.cols() !=
          expected_structure_count ||
      solve_result.euclidean_biorthogonality_matrix.rows() !=
          expected_structure_count ||
      solve_result.euclidean_biorthogonality_matrix.cols() !=
          expected_structure_count ||
      solve_result.metric_biorthogonality_matrix.rows() !=
          expected_structure_count ||
      solve_result.metric_biorthogonality_matrix.cols() !=
          expected_structure_count) {
    throw std::invalid_argument("solve-result matrix dimensions are inconsistent");
  }
  if (static_cast<int>(solve_result.right_residual_norms.size()) !=
          expected_structure_count ||
      static_cast<int>(solve_result.left_residual_norms.size()) !=
          expected_structure_count) {
    throw std::invalid_argument("solve-result residual counts are inconsistent");
  }
  if (!std::isfinite(solve_result.max_eigenvalue_imaginary_magnitude) ||
      !std::isfinite(solve_result.max_right_vector_imaginary_magnitude) ||
      !std::isfinite(solve_result.max_left_vector_imaginary_magnitude)) {
    throw std::invalid_argument("solve-result imaginary diagnostics must be finite");
  }
  throw_if_nonfinite(
      solve_result.right_orthogonalized_coefficient_matrix,
      "right_orthogonalized_coefficient_matrix");
  throw_if_nonfinite(
      solve_result.left_orthogonalized_coefficient_matrix,
      "left_orthogonalized_coefficient_matrix");
  throw_if_nonfinite(
      solve_result.right_structure_coefficient_matrix,
      "right_structure_coefficient_matrix");
  throw_if_nonfinite(
      solve_result.left_structure_coefficient_matrix,
      "left_structure_coefficient_matrix");
  throw_if_nonfinite(
      solve_result.euclidean_biorthogonality_matrix,
      "euclidean_biorthogonality_matrix");
  throw_if_nonfinite(
      solve_result.metric_biorthogonality_matrix,
      "metric_biorthogonality_matrix");

  for (int state_index = 0; state_index < expected_structure_count; ++state_index) {
    const double right_residual = solve_result.right_residual_norms[xmvb::to_size(state_index)];
    const double left_residual = solve_result.left_residual_norms[xmvb::to_size(state_index)];
    if (!std::isfinite(right_residual) || !std::isfinite(left_residual)) {
      throw std::invalid_argument("solve-result residual norms must be finite");
    }
    if (right_residual > residual_tolerance || left_residual > residual_tolerance) {
      throw std::runtime_error("biorthogonal projected solve residual exceeds tolerance");
    }
  }

  const Eigen::MatrixXd euclidean_identity =
      Eigen::MatrixXd::Identity(expected_structure_count, expected_structure_count);
  const Eigen::MatrixXd metric_identity =
      Eigen::MatrixXd::Identity(expected_structure_count, expected_structure_count);
  if ((solve_result.euclidean_biorthogonality_matrix - euclidean_identity).norm() >
          biorthogonality_tolerance ||
      (solve_result.metric_biorthogonality_matrix - metric_identity).norm() >
          biorthogonality_tolerance) {
    throw std::runtime_error("biorthogonality validation failed");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
