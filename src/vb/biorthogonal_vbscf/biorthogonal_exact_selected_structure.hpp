#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Memory-reduced exact selected-space matrix build result.
 *
 * This object only retains the physical selected-space overlap and
 * Hamiltonian matrices
 *
 * `S_str = T_sel^T S_det^(full) T_sel`,
 * `H_str = T_sel^T H_det^(full) T_sel`,
 *
 * together with the symmetry diagnostics needed by benchmarks and higher-level
 * matrix builders. Unlike the full evaluation result, it does not keep the
 * dense determinant-column actions `T_sel`, `U_sel`, or `Y_sel`, so it is the
 * preferred entry point when callers only need one selected-space matrix build
 * step.
 */
struct BiorthogonalExactSelectedStructureMatrixBuildResult {
  int n_active_orbitals = 0;
  int n_full_structures = 0;
  int n_selected_structures = 0;
  int n_determinants = 0;
  std::vector<int> selected_structure_indices;
  Eigen::MatrixXd physical_structure_overlap;
  Eigen::MatrixXd physical_structure_hamiltonian;
  double structure_overlap_symmetry_residual_max_abs = 0.0;
  double structure_hamiltonian_symmetry_residual_max_abs = 0.0;
};

/**
 * @brief Exact biorthogonal selected-structure projection built from the full determinant space.
 *
 * `selected_structure_to_determinant` stores the selected right-structure
 * columns `T_sel` with dimensions `(n_determinants, n_selected_structures)`.
 * `overlap_action_on_selected_columns` stores the full-space dual left
 * projection
 *
 * `U_sel = S_det^(full) T_sel`
 *
 * with the same dimensions.
 *
 * `biorthogonal_hamiltonian_action_on_selected_columns` stores the full-space
 * action
 *
 * `Y_sel = h_det^(bi,full) T_sel`.
 *
 * The physical selected-space matrices are then
 *
 * `S_str = U_sel^T T_sel`,
 * `H_str = U_sel^T Y_sel`.
 *
 * `structure_coefficient_matrix` stores the generalized eigenvectors of
 * `H_str C = S_str C E` in the selected right-structure basis.
 *
 * The selected-state backward can reconstruct
 *
 * `r^(n) = T_sel c_n`,
 * `l^(n) = U_sel c_n`,
 *
 * directly from `T_sel`, `U_sel`, and the selected generalized eigenvector
 * `c_n`, so the full dense determinant coefficient matrices are optional and
 * may be left empty to avoid the much more expensive all-state contractions
 * `T_sel C` and `U_sel C`.
 */
struct BiorthogonalExactSelectedStructureEvaluationResult {
  int n_active_orbitals = 0;
  int n_full_structures = 0;
  int n_selected_structures = 0;
  int n_determinants = 0;
  std::vector<int> selected_structure_indices;
  Eigen::MatrixXd selected_structure_to_determinant;
  Eigen::MatrixXd overlap_action_on_selected_columns;
  Eigen::MatrixXd biorthogonal_hamiltonian_action_on_selected_columns;
  Eigen::MatrixXd physical_structure_overlap;
  Eigen::MatrixXd physical_structure_hamiltonian;
  std::vector<double> eigenvalues;
  Eigen::MatrixXd structure_coefficient_matrix;
  Eigen::MatrixXd right_determinant_coefficient_matrix;
  Eigen::MatrixXd left_determinant_coefficient_matrix;
  double structure_overlap_symmetry_residual_max_abs = 0.0;
  double structure_hamiltonian_symmetry_residual_max_abs = 0.0;
};

/**
 * @brief Builds only the exact selected-space overlap and Hamiltonian matrices.
 *
 * This routine streams the exact projected matrices while storing only one
 * determinant-indexed action family, so its peak memory is substantially lower
 * than the full selected-space evaluation path.
 */
BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload for callers that already prepared the active-space tensors once.
 */
BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload that prepares the active-space data from one top-level input bundle.
 */
BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Evaluates one exact biorthogonal selected-structure subspace.
 *
 * Unlike the first fixed-metric prototype, this routine keeps the full
 * determinant space intact, forms the exact actions `S_det^(full) T_sel` and
 * `h_det^(bi,full) T_sel` directly without materializing the full dense
 * determinant matrices, and therefore reproduces the original nonorthogonal
 * VB generalized eigenproblem for the selected structure subset.
 */
BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload for callers that already prepared the active-space tensors once.
 *
 * This is the preferred entry point when multiple selected-structure subspaces
 * are evaluated on the same orbitals, because the expensive active-space
 * preparation can be reused across all subspaces.
 */
BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload that prepares the active-space data from one top-level input bundle.
 */
BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Validates dimensions, finiteness, and symmetry residuals of the exact result.
 */
void validate_biorthogonal_exact_selected_structure_evaluation_result(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    double symmetry_tolerance);

/**
 * @brief Validates dimensions, finiteness, and symmetry residuals of the matrix-only result.
 */
void validate_biorthogonal_exact_selected_structure_matrix_build_result(
    const BiorthogonalExactSelectedStructureMatrixBuildResult& matrix_result,
    double symmetry_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
