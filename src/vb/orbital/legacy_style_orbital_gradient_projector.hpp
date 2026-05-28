#pragma once

#include <vector>

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Result of the legacy-style orbital-gradient projection.
 */
struct LegacyStyleOrbitalGradientProjectionResult {
  /**
   * @brief Packed orbital-parameter gradient in legacy `X` ordering.
   */
  std::vector<double> parameter_gradient;

  /**
   * @brief Slot-based basis-gradient buffer indexed as `(slot, orbital)`.
   *
   * This mirrors the legacy `Grdbas(J, I)` storage, where `J` is the sparse
   * coefficient slot within orbital `I`, not the absolute AO basis index.
   */
  std::vector<double> slot_gradient_matrix;
};

/**
 * @brief Reproduces the legacy `Grdori` orbital-gradient projection.
 *
 * This class maps active-space gradient intermediates onto the packed VBSCF
 * orbital parameter vector using the same algebraic structure as the Fortran
 * `Grdori` implementation.
 */
class LegacyStyleOrbitalGradientProjector {
public:
  /**
   * @brief Projects active-space gradient intermediates into parameter space.
   *
   * @param active_active_gradient_matrix Legacy-style `Grda`, dimension `(nao, nao)`.
   * @param active_virtual_gradient_matrix Legacy-style `Grdv`, dimension `(nb, nb)`.
   * @param active_space_coulomb_exchange_matrix Legacy-style `G22`, dimension `(nb, nb)`.
   * @param overlap_response_matrix Legacy-style `Q22`, dimension `(nb, nb)`.
   * @param active_density_matrix Legacy-style `P22`, dimension `(nb, nb)`.
   * @param ao_overlap_matrix AO overlap matrix `SSF`, dimension `(nb, nb)`.
   * @param ao_effective_h1e AO effective one-electron matrix `F11`, dimension `(nb, nb)`.
   * @param orbital_preparation_input Sparse-orbital indexing and original `ma0` counts.
   * @param orbital_preparation_result Legacy-style orbital-preparation intermediates.
   * @param n_inactive_doubly_occupied_orbitals Number of inactive doubly occupied orbitals.
   * @param n_virtual_orbitals Number of virtual auxiliary orbitals.
   * @param weight State-average weight applied to the projected gradient.
   * @return LegacyStyleOrbitalGradientProjectionResult Packed and slot-resolved gradient.
   */
  LegacyStyleOrbitalGradientProjectionResult project(
      const std::vector<double>& active_active_gradient_matrix,
      const std::vector<double>& active_virtual_gradient_matrix,
      const std::vector<double>& active_space_coulomb_exchange_matrix,
      const std::vector<double>& overlap_response_matrix,
      const std::vector<double>& active_density_matrix,
      const std::vector<double>& ao_overlap_matrix,
      const std::vector<double>& ao_effective_h1e,
      const OrbitalPreparationInput& orbital_preparation_input,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_inactive_doubly_occupied_orbitals,
      int n_virtual_orbitals,
      double weight) const;
};

}  // namespace xmvb::vb
