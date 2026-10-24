#pragma once

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief End-to-end evaluator for VB structure Hamiltonian/overlap matrices.
 *
 * This class stitches together the full C++ matrix path:
 * 1. Rebuild auxiliary orbitals and inactive density from sparse orbital input.
 * 2. Build AO `G11/F11` from AO integrals and inactive density.
 * 3. Project AO effective one-electron integrals into active-space `HHO`.
 * 4. Transform AO two-electron integrals into active-space `GGO`.
 * 5. Contract spin-string contributions into structure Hamiltonian/overlap matrices.
 */
class StructureMatrixEvaluator {
  public:
    StructureMatrixEvaluator() = default;

    /**
   * @brief Evaluates structure matrices from prebuilt active-space intermediates.
   */
    StructureAccumulationResult
    evaluate(const VbScfInput& input,
             const PreparedActiveSpaceContext& prepared_active_space) const;

    /** @brief Builds the reusable unique-spin pair cache for a structure action. */
    SameSpinPairCacheContext
    build_pair_cache(const VbScfInput& input,
                     const PreparedActiveSpaceContext& prepared_active_space,
                     SameSpinPairCacheBuildOptions options) const;

  private:
    FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
};

} // namespace xmvb::vb
