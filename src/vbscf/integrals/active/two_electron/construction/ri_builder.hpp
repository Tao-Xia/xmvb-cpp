#pragma once

#include "vbscf/integrals/active/two_electron/construction/result.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

/**
 * @brief Transforms molecule-static AO-side RI factors into active-space RI factors.
 *
 * The result remains factorized. Reference tools that need four-index values
 * must reconstruct them explicitly through the kernel utility; production
 * code cannot request a hidden `O(n_active^4)` allocation from this builder.
 */
class RiActiveSpaceTwoElectronBuilder {
public:
  ActiveSpaceTwoElectronResult build(
      const RiAoFactorization& ao_ri_result,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_basis_functions,
      int n_active_orbitals) const;
};

}  // namespace xmvb::vb
