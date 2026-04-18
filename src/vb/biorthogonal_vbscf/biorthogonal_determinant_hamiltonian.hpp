#pragma once

#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Canonical alpha/beta occupied-orbital lists for one determinant label.
 *
 * Both occupation lists must be strictly increasing and use zero-based active
 * orbital labels. This is the same canonical representation already used by
 * the rest of the C++ VB determinant kernels.
 */
struct BiorthogonalDeterminant {
  std::vector<int> alpha_occupied_orbitals;
  std::vector<int> beta_occupied_orbitals;
};

/**
 * @brief Determinant-pair matrix element in the biorthogonal determinant basis.
 *
 * `overlap` is exactly `1.0` only when the left and right determinant labels
 * are identical and `0.0` otherwise. `one_electron_hamiltonian` stores only
 * the one-electron part of the matrix element, while `total_hamiltonian`
 * includes same-spin and opposite-spin two-electron contributions.
 */
struct BiorthogonalDeterminantHamiltonianEntry {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  int alpha_excitation_rank = 0;
  int beta_excitation_rank = 0;
};

/**
 * @brief Validates one determinant label against the active-orbital dimension.
 */
void validate_biorthogonal_determinant(
    const BiorthogonalDeterminant& determinant,
    int n_orbitals);

/**
 * @brief Evaluates one determinant-pair matrix element `<\widetilde D_J|H|D_I>`.
 *
 * The returned matrix element follows ordinary Slater-Condon sparsity because
 * the determinant metric is identity in the biorthogonal basis. Only pairs
 * differing by up to a double excitation can contribute.
 */
BiorthogonalDeterminantHamiltonianEntry
evaluate_biorthogonal_determinant_hamiltonian(
    const BiorthogonalDeterminant& left_determinant,
    const BiorthogonalDeterminant& right_determinant,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Evaluates one determinant-pair matrix element from a forward 2e result.
 */
BiorthogonalDeterminantHamiltonianEntry
evaluate_biorthogonal_determinant_hamiltonian(
    const BiorthogonalDeterminant& left_determinant,
    const BiorthogonalDeterminant& right_determinant,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result);

/**
 * @brief Builds one square determinant Hamiltonian matrix over a shared label list.
 *
 * The returned matrix has rows indexed by left determinants and columns indexed
 * by right determinants, so it is the direct matrix representation of
 *
 * `h^{bi}_{JI} = <\widetilde D_J|H|D_I>`.
 */
Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Builds one square determinant Hamiltonian matrix from a forward 2e result.
 */
Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result);

/**
 * @brief Builds one rectangular determinant Hamiltonian block.
 */
Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& left_determinants,
    const std::vector<BiorthogonalDeterminant>& right_determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Builds one rectangular determinant Hamiltonian block from a forward 2e result.
 */
Eigen::MatrixXd build_biorthogonal_determinant_hamiltonian_matrix(
    const std::vector<BiorthogonalDeterminant>& left_determinants,
    const std::vector<BiorthogonalDeterminant>& right_determinants,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result);

/**
 * @brief Builds the square biorthogonal determinant metric over one label list.
 *
 * With matched left/right determinant labels the metric is identity exactly.
 */
Eigen::MatrixXd build_biorthogonal_determinant_metric_matrix(
    const std::vector<BiorthogonalDeterminant>& determinants,
    int n_orbitals);

}  // namespace xmvb::vb::biorthogonal_vbscf
