#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_projected_structure_problem.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Left/right eigenpairs of the fixed-metric biorthogonal structure problem.
 *
 * `eigenvalues` are sorted in ascending order. The orthogonalized coefficient
 * matrices store the right eigenvectors `y_n` and left eigenvectors `z_n` of
 * the non-Hermitian orthogonalized operator
 *
 * `\widehat H = L^{-1} H^{(bi)} L^{-T}`.
 *
 * The structure-space coefficient matrices store the corresponding
 * generalized-eigenproblem vectors
 *
 * `r_n = L^{-T} y_n`,
 * `l_n = L^{-T} z_n`.
 *
 * `euclidean_biorthogonality_matrix` stores `Z^T Y`, while
 * `metric_biorthogonality_matrix` stores `L^T M^(0) R`.
 */
struct BiorthogonalProjectedStructureSolveResult {
  std::vector<double> eigenvalues;
  Eigen::MatrixXd right_orthogonalized_coefficient_matrix;
  Eigen::MatrixXd left_orthogonalized_coefficient_matrix;
  Eigen::MatrixXd right_structure_coefficient_matrix;
  Eigen::MatrixXd left_structure_coefficient_matrix;
  Eigen::MatrixXd euclidean_biorthogonality_matrix;
  Eigen::MatrixXd metric_biorthogonality_matrix;
  std::vector<double> right_residual_norms;
  std::vector<double> left_residual_norms;
  double max_eigenvalue_imaginary_magnitude = 0.0;
  double max_right_vector_imaginary_magnitude = 0.0;
  double max_left_vector_imaginary_magnitude = 0.0;
};

/**
 * @brief Solves the fixed-metric biorthogonal selected-space coefficient problem.
 *
 * The solver diagonalizes the orthogonalized non-Hermitian operator
 * `\widehat H`, computes both right and left eigenvectors, and then maps those
 * vectors back to the selected-structure coefficient basis.
 *
 * `imaginary_tolerance` bounds the allowed imaginary part of eigenvalues and
 * eigenvectors. The first prototype expects a real spectrum and real
 * left/right coefficient vectors; if complex roots appear above this threshold,
 * the routine throws instead of silently discarding them.
 */
BiorthogonalProjectedStructureSolveResult solve_biorthogonal_projected_structure_problem(
    const BiorthogonalProjectedStructureProblem& projected_problem,
    const BiorthogonalSelectedStructureSpace& structure_space,
    double imaginary_tolerance = 1.0e-8);

/**
 * @brief Validates dimensions, residuals, and left/right biorthogonality.
 */
void validate_biorthogonal_projected_structure_solve_result(
    const BiorthogonalProjectedStructureSolveResult& solve_result,
    int expected_structure_count,
    double residual_tolerance,
    double biorthogonality_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
