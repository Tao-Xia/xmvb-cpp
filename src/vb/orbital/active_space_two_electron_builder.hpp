#pragma once

#include <vector>

#include "vb/orbital/orbital_preparation_result.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"

namespace xmvb::vb {

/**
 * @brief Transforms AO two-electron integrals into packed active-space `GGO`.
 *
 * The current implementation is correctness-oriented: it reconstructs the
 * symmetry-equivalent AO integral permutations on the fly and contracts them
 * directly with the active auxiliary orbitals. This keeps the code compact and
 * easy to verify before later optimization.
 */
class ActiveSpaceTwoElectronBuilder {
public:
  /**
   * @brief Builds packed active-space two-electron integrals.
   *
   * @param ao_two_electron_integral_values Sparse AO eri_act values `ggf`.
   * @param ao_two_electron_integral_indices Flattened AO index table `g2eidx`.
   * @param auxiliary_orbital_matrix Column-major auxiliary orbital matrix.
   * @param n_basis_functions Number of AO basis functions.
   * @param n_inactive_doubly_occupied_orbitals Number of inactive doubly occupied orbitals.
   * @param n_active_orbitals Number of active orbitals.
   * @return ActiveSpaceTwoElectronResult Packed active-space `GGO`.
   */
  ActiveSpaceTwoElectronResult build(
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      const std::vector<double>& auxiliary_orbital_matrix,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;

  /**
   * @brief Builds packed active-space two-electron integrals using precomputed sparse active coefficients.
   */
  ActiveSpaceTwoElectronResult build(
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_basis_functions,
      int n_active_orbitals) const;
};

}  // namespace xmvb::vb
