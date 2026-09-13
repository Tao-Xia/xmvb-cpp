#pragma once

#include "vbscf/integrals/ao/contracts/input.hpp"
#include "vbscf/integrals/active/two_electron/construction/result.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

/**
 * @brief Transforms AO two-electron integrals into packed active-space `GGO`.
 *
 * Dense active-pair contractions use the canonical AO-pair graph. Large,
 * strictly sparse active spaces contract the original unique ERI stream with
 * sparse AO-pair coefficients, avoiding a dense active transformation tensor.
 */
class ActiveSpaceTwoElectronBuilder {
public:
  /**
   * @brief Builds packed active-space two-electron integrals.
   *
   * @param ao_integral_input AO integrals and their canonical pair graph.
   * @param orbital_preparation_result Prepared active-orbital coefficients.
   * @param n_active_orbitals Number of active orbitals.
   * @return Packed active-space two-electron integrals.
   */
  ActiveSpaceTwoElectronResult build(
      const AoIntegralInput& ao_integral_input,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_active_orbitals) const;
};

}  // namespace xmvb::vb
