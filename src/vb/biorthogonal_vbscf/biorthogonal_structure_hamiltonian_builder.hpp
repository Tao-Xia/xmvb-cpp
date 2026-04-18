#pragma once

#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Block-contracted selected-structure Hamiltonian in the biorthogonal basis.
 *
 * The first prototype does not need to materialize the full determinant matrix
 *
 * `h^{(bi)}_{JI} = <\widetilde D_J|H|D_I>`
 *
 * to reach the selected-space coefficient problem. Instead it reuses the same
 * unique-spin-string plus local block-contraction organization as the
 * nonorthogonal structure builder and assembles the projected matrices
 * directly:
 *
 * `S^{(bi)}_{KL} = <\widetilde \Phi_K | \Phi_L>`,
 * `H^{(bi)}_{KL} = <\widetilde \Phi_K | H | \Phi_L>`.
 */
struct BiorthogonalStructureHamiltonianBuildResult {
  Eigen::MatrixXd selected_structure_overlap;
  Eigen::MatrixXd selected_structure_one_electron_hamiltonian;
  Eigen::MatrixXd selected_structure_hamiltonian;
  int n_unique_alpha = 0;
  int n_unique_beta = 0;
};

/**
 * @brief Builds the ordered selected-space biorthogonal matrices by block contraction.
 *
 * The returned overlap matrix is the exact fixed metric `T^T T` implied by the
 * determinant expansion, while `selected_structure_hamiltonian` is the ordered
 * non-Hermitian projected Hamiltonian `T^T h^{(bi)} T`.
 */
BiorthogonalStructureHamiltonianBuildResult
build_biorthogonal_structure_hamiltonian(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Convenience overload from one forward active-space 2e result.
 */
BiorthogonalStructureHamiltonianBuildResult
build_biorthogonal_structure_hamiltonian(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result);

/**
 * @brief Validates dimensions and finiteness of the projected matrices.
 */
void validate_biorthogonal_structure_hamiltonian_build_result(
    const BiorthogonalStructureHamiltonianBuildResult& build_result,
    int expected_structure_count);

}  // namespace xmvb::vb::biorthogonal_vbscf
