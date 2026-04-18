#pragma once

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_selected_structure_space.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Selected-space Hamiltonian and metric-orthogonalized operator.
 *
 * `selected_structure_hamiltonian` has dimensions `(n_structures, n_structures)`
 * and stores
 *
 * `H^(bi) = T^T h^(bi) T`.
 *
 * `orthogonalized_structure_hamiltonian` stores
 *
 * `\widehat H = L^{-1} H^(bi) L^{-T}`
 *
 * where `L` is the Cholesky factor of the fixed metric `M^(0)`.
 */
struct BiorthogonalProjectedStructureProblem {
  Eigen::MatrixXd selected_structure_hamiltonian;
  Eigen::MatrixXd orthogonalized_structure_hamiltonian;
  double orthogonalization_reconstruction_residual_frobenius_norm = 0.0;
};

/**
 * @brief Builds the projected selected-space Hamiltonian for one determinant matrix.
 *
 * The input `determinant_hamiltonian` must be the square left/right determinant
 * Hamiltonian matrix `h^(bi)` over the same determinant ordering that appears in
 * `structure_space.structure_to_determinant`.
 */
BiorthogonalProjectedStructureProblem build_biorthogonal_projected_structure_problem(
    const Eigen::Ref<const Eigen::MatrixXd>& determinant_hamiltonian,
    const BiorthogonalSelectedStructureSpace& structure_space);

/**
 * @brief Orthogonalizes one already-projected selected-space Hamiltonian.
 *
 * This overload is used by the unique-spin / block-contraction forward path,
 * which builds `H^(bi)` directly without materializing the full determinant
 * matrix `h^(bi)`.
 */
BiorthogonalProjectedStructureProblem
build_biorthogonal_projected_structure_problem_from_selected_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& selected_structure_hamiltonian,
    const BiorthogonalSelectedStructureSpace& structure_space);

/**
 * @brief Validates the projected Hamiltonian dimensions and reconstruction error.
 */
void validate_biorthogonal_projected_structure_problem(
    const BiorthogonalProjectedStructureProblem& projected_problem,
    int expected_structure_count,
    double reconstruction_tolerance);

}  // namespace xmvb::vb::biorthogonal_vbscf
