#pragma once

#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_projected_solver.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_hamiltonian_builder.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_expansion.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief End-to-end fixed-metric biorthogonal forward evaluation output.
 *
 * This object closes the first purely forward biorthogonal chain:
 *
 * `FullDeterminantStructureData -> (D, T) -> h^(bi) -> H^(bi) -> (E_n, l_n, r_n)`.
 *
 * The returned determinant coefficient matrices are
 *
 * `C_R = T R`,
 * `C_L = T L`,
 *
 * so they can be interpreted directly as left/right determinant amplitudes for
 * the selected-space states produced by the fixed-metric solver.
 */
struct BiorthogonalForwardEvaluationResult {
  int n_active_orbitals = 0;
  int n_structures = 0;
  int n_determinants = 0;
  BiorthogonalStructureExpansion structure_expansion;
  BiorthogonalOrbitalIntegrals orbital_integrals;
  Eigen::MatrixXd determinant_hamiltonian;
  BiorthogonalStructureHamiltonianBuildResult structure_hamiltonian_build_result;
  BiorthogonalSelectedStructureSpace structure_space;
  BiorthogonalProjectedStructureProblem projected_problem;
  BiorthogonalProjectedStructureSolveResult solve_result;
  Eigen::MatrixXd right_determinant_coefficient_matrix;
  Eigen::MatrixXd left_determinant_coefficient_matrix;
};

/**
 * @brief Evaluates the first end-to-end biorthogonal forward model.
 *
 * The input may be the full determinant expansion or any selected-structure
 * subspace already filtered through `StructureSubspaceBuilder`; the forward
 * evaluator treats both identically because it consumes only the explicit
 * determinant expansion and active-space integral buffers. For a filtered
 * selected-structure subspace this fixed-metric path is not the exact
 * projected nonorthogonal VB problem, because the omitted determinant rows do
 * not contribute to the left dual projector. Use
 * `evaluate_biorthogonal_exact_selected_structure_subspace(...)` when the goal
 * is to reproduce the original nonorthogonal selected-subspace pencil exactly.
 */
BiorthogonalForwardEvaluationResult evaluate_biorthogonal_forward(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    double imaginary_tolerance = 1.0e-8);

/**
 * @brief Convenience overload that consumes the existing top-level input bundle.
 */
BiorthogonalForwardEvaluationResult evaluate_biorthogonal_forward(
    const xmvb::vb::CppVbInput& input,
    double imaginary_tolerance = 1.0e-8);

/**
 * @brief Validates the dimensions and residuals of the forward result.
 */
void validate_biorthogonal_forward_evaluation_result(
    const BiorthogonalForwardEvaluationResult& evaluation_result,
    double metric_min_diagonal_tolerance,
    double orthogonalization_reconstruction_tolerance,
    double residual_tolerance,
    double biorthogonality_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
