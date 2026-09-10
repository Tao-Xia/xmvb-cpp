#pragma once

#include "vbscf/integrals/active/active_space_two_electron_result.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

struct RiActiveSpaceTwoElectronBuilderOptions {
  bool reconstruct_packed_integrals = false;
};

/**
 * @brief Transforms molecule-static AO-side RI factors into active-space RI factors.
 */
class RiActiveSpaceTwoElectronBuilder {
public:
  ActiveSpaceTwoElectronResult build(
      const RiAoFactorization& ao_ri_result,
      const OrbitalPreparationResult& orbital_preparation_result,
      int n_basis_functions,
      int n_active_orbitals,
      const RiActiveSpaceTwoElectronBuilderOptions& options = {}) const;
};

}  // namespace xmvb::vb
