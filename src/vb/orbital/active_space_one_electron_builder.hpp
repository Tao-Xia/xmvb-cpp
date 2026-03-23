#pragma once

#include <vector>

#include "vb/orbital/active_space_one_electron_result.hpp"

namespace xmvb::vb {

/**
 * @brief Projects the AO effective one-electron matrix into the active space.
 *
 * This class implements the `HHO = T_a^T F11 T_a` part of the legacy VBSCF
 * preparation in pure C++. All matrices use column-major storage.
 */
class ActiveSpaceOneElectronBuilder {
public:
  /**
   * @brief Builds the active-space effective one-electron matrix.
   *
   * @param ao_effective_h1e Column-major AO `F11` matrix.
   * @param auxiliary_orbital_matrix Column-major auxiliary orbital matrix.
   * @param n_basis_functions Number of AO basis functions.
   * @param n_inactive_doubly_occupied_orbitals Number of inactive doubly occupied orbitals.
   * @param n_active_orbitals Number of active orbitals.
   * @return ActiveSpaceOneElectronResult Active-space `HHO` matrix.
   */
  ActiveSpaceOneElectronResult build(
      const std::vector<double>& ao_effective_h1e,
      const std::vector<double>& auxiliary_orbital_matrix,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals,
      int n_active_orbitals) const;
};

}  // namespace xmvb::vb
