#pragma once

#include <vector>

#include "vb/orbital/ao_effective_one_electron_result.hpp"

namespace xmvb::vb {

/**
 * @brief Builds AO effective one-electron matrices from the inactive density.
 *
 * This class is the pure C++ replacement for the `Cal_G11` and `Cal_F11`
 * stages in the legacy VBSCF implementation for the standard non-RI path.
 */
class AoEffectiveOneElectronBuilder {
public:
  /**
   * @brief Evaluates `G11` and `F11 = HHF + G11`.
   *
   * @param inactive_density_matrix Column-major inactive density matrix `P11`.
   * @param ao_core_hamiltonian_matrix Column-major AO core Hamiltonian `HHF`.
   * @param ao_two_electron_integral_values Sparse AO two-electron values `ggf`.
   * @param ao_two_electron_integral_indices Flattened AO index table `g2eidx`.
   * @param n_basis_functions Number of AO basis functions.
   * @return AoEffectiveOneElectronResult AO `G11` and `F11`.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const std::vector<double>& ao_core_hamiltonian_matrix,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      int n_basis_functions) const;
};

}  // namespace xmvb::vb
