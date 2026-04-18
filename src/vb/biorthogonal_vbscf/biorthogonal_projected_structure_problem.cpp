#include "vb/biorthogonal_vbscf/biorthogonal_projected_structure_problem.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

}  // namespace

BiorthogonalProjectedStructureProblem
build_biorthogonal_projected_structure_problem_from_selected_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& selected_structure_hamiltonian,
    const BiorthogonalSelectedStructureSpace& structure_space) {
  if (selected_structure_hamiltonian.rows() <= 0 ||
      selected_structure_hamiltonian.cols() <= 0) {
    throw std::invalid_argument("selected_structure_hamiltonian must be non-empty");
  }
  if (selected_structure_hamiltonian.rows() != selected_structure_hamiltonian.cols()) {
    throw std::invalid_argument("selected_structure_hamiltonian must be square");
  }
  if (selected_structure_hamiltonian.rows() != structure_space.fixed_metric.rows()) {
    throw std::invalid_argument(
        "selected_structure_hamiltonian size must match fixed metric");
  }
  throw_if_nonfinite(
      selected_structure_hamiltonian,
      "selected_structure_hamiltonian");

  BiorthogonalProjectedStructureProblem projected_problem;
  projected_problem.selected_structure_hamiltonian = selected_structure_hamiltonian;

  // Because the selected metric is fixed, the orthogonalized operator can be
  // formed by one reusable inverse Cholesky factor. This is the linear-algebra
  // bridge from the fixed-metric generalized problem to a standard
  // non-Hermitian eigenproblem in the same frozen selected space.
  projected_problem.orthogonalized_structure_hamiltonian =
      structure_space.fixed_metric_factor_inverse *
      projected_problem.selected_structure_hamiltonian *
      structure_space.fixed_metric_factor_inverse.transpose();
  throw_if_nonfinite(
      projected_problem.orthogonalized_structure_hamiltonian,
      "orthogonalized_structure_hamiltonian");

  const Eigen::MatrixXd reconstruction =
      structure_space.fixed_metric_cholesky_factor *
      projected_problem.orthogonalized_structure_hamiltonian *
      structure_space.fixed_metric_cholesky_factor.transpose();
  projected_problem.orthogonalization_reconstruction_residual_frobenius_norm =
      (reconstruction - projected_problem.selected_structure_hamiltonian).norm();
  if (!std::isfinite(
          projected_problem
              .orthogonalization_reconstruction_residual_frobenius_norm)) {
    throw std::runtime_error(
        "projected-structure orthogonalization residual is not finite");
  }

  return projected_problem;
}

BiorthogonalProjectedStructureProblem build_biorthogonal_projected_structure_problem(
    const Eigen::Ref<const Eigen::MatrixXd>& determinant_hamiltonian,
    const BiorthogonalSelectedStructureSpace& structure_space) {
  if (determinant_hamiltonian.rows() <= 0 || determinant_hamiltonian.cols() <= 0) {
    throw std::invalid_argument("determinant_hamiltonian must be non-empty");
  }
  if (determinant_hamiltonian.rows() != determinant_hamiltonian.cols()) {
    throw std::invalid_argument("determinant_hamiltonian must be square");
  }
  if (structure_space.structure_to_determinant.rows() != determinant_hamiltonian.rows()) {
    throw std::invalid_argument(
        "determinant_hamiltonian size must match structure_to_determinant rows");
  }
  throw_if_nonfinite(determinant_hamiltonian, "determinant_hamiltonian");

  return build_biorthogonal_projected_structure_problem_from_selected_hamiltonian(
      structure_space.structure_to_determinant.transpose() *
          determinant_hamiltonian *
          structure_space.structure_to_determinant,
      structure_space);
}

void validate_biorthogonal_projected_structure_problem(
    const BiorthogonalProjectedStructureProblem& projected_problem,
    int expected_structure_count,
    double reconstruction_tolerance) {
  if (expected_structure_count <= 0) {
    throw std::invalid_argument("expected_structure_count must be positive");
  }
  if (reconstruction_tolerance < 0.0) {
    throw std::invalid_argument("reconstruction_tolerance must be non-negative");
  }
  if (projected_problem.selected_structure_hamiltonian.rows() !=
          expected_structure_count ||
      projected_problem.selected_structure_hamiltonian.cols() !=
          expected_structure_count) {
    throw std::invalid_argument(
        "selected_structure_hamiltonian dimensions are inconsistent");
  }
  if (projected_problem.orthogonalized_structure_hamiltonian.rows() !=
          expected_structure_count ||
      projected_problem.orthogonalized_structure_hamiltonian.cols() !=
          expected_structure_count) {
    throw std::invalid_argument(
        "orthogonalized_structure_hamiltonian dimensions are inconsistent");
  }
  throw_if_nonfinite(
      projected_problem.selected_structure_hamiltonian,
      "selected_structure_hamiltonian");
  throw_if_nonfinite(
      projected_problem.orthogonalized_structure_hamiltonian,
      "orthogonalized_structure_hamiltonian");
  if (!std::isfinite(
          projected_problem
              .orthogonalization_reconstruction_residual_frobenius_norm)) {
    throw std::invalid_argument("orthogonalization residual must be finite");
  }
  if (projected_problem.orthogonalization_reconstruction_residual_frobenius_norm >
      reconstruction_tolerance) {
    throw std::runtime_error("orthogonalized structure problem reconstruction failed");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
