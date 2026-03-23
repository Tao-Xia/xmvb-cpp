#pragma once

#include <vector>

#include "vb/orbital/orbital_preparation_result.hpp"
#include "vb/orbital/active_space_two_electron_backpropagation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Reverse-mode backpropagator for the active-space two-electron builder.
 *
 * Given the objective gradient with respect to the packed active-space
 * two-electron integrals `GGO`, this class computes the induced gradient with
 * respect to the full auxiliary orbital matrix.
 */
class ActiveSpaceTwoElectronBackpropagator {
public:
  /**
   * @brief Backpropagates the packed active-space two-electron adjoint.
   *
   * @param packed_active_two_electron_gradient Gradient with respect to packed `GGO`.
   * @param ao_two_electron_integral_values Sparse AO two-electron integral values.
   * @param ao_two_electron_integral_indices Flattened AO two-electron index table.
   * @param auxiliary_orbital_matrix Column-major full auxiliary orbital matrix.
   * @param n_basis_functions Number of AO basis functions.
   * @param n_inactive_doubly_occupied_orbitals Number of inactive doubly occupied orbitals.
   * @param n_active_orbitals Number of active orbitals.
   * @return ActiveSpaceTwoElectronBackpropagationResult Reverse-mode derivative.
   */
  ActiveSpaceTwoElectronBackpropagationResult backpropagate(
      const std::vector<double>& packed_active_two_electron_gradient,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      const std::vector<double>& auxiliary_orbital_matrix,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;

  /**
   * @brief Backpropagates the packed active-space two-electron adjoint using precomputed sparse active coefficients.
   */
  ActiveSpaceTwoElectronBackpropagationResult backpropagate(
      const std::vector<double>& packed_active_two_electron_gradient,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;
};

}  // namespace xmvb::vb
