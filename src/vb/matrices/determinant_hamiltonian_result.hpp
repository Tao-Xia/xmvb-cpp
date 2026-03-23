#pragma once

namespace xmvb::vb {

/**
 * @brief Determinant-level Hamiltonian and overlap quantities.
 */
struct DeterminantHamiltonianResult {
  /**
   * @brief Determinant overlap between the two determinants.
   */
  double overlap_determinant = 0.0;

  /**
   * @brief One-electron Hamiltonian matrix element.
   */
  double one_electron_hamiltonian = 0.0;

  /**
   * @brief Total Hamiltonian matrix element including one- and two-electron parts.
   */
  double total_hamiltonian = 0.0;

  /**
   * @brief Nullity of the determinant overlap submatrix.
   */
  int nullity = 0;
};

}  // namespace xmvb::vb
