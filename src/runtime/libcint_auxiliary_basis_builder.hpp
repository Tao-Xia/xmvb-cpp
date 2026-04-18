#pragma once

#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

struct LibcintAuxiliaryBasisBuilderOptions {
  /**
   * @brief GEN-A_n level from the generated auxiliary-basis construction.
   *
   * The current implementation supports `2 <= level <= 4`.
   */
  int level = 2;

  /**
   * @brief Whether to build the starred `GEN-A_n*` variant.
   *
   * The production RI path defaults to the starred variant because the legacy
   * Coulomb/exchange energy evaluation uses the richer `GEN-A_n*` auxiliary
   * basis for its final energy build. `false` keeps only the `s + spd` blocks,
   * while `true` also appends the `spdf` block on non-hydrogen/helium atoms.
   */
  bool use_star = true;
};

/**
 * @brief Builds a generated Coulomb-fitting auxiliary basis from a primary basis.
 *
 * This is a pure C++ implementation of the generated `GEN-A_n` / `GEN-A_n*`
 * auxiliary-basis policy used by the forthcoming RI path. It intentionally
 * does not depend on the legacy runtime's hidden basis state.
 */
class LibcintAuxiliaryBasisBuilder {
public:
  LibcintInput build(
      const LibcintInput& primary_input,
      const LibcintAuxiliaryBasisBuilderOptions& options = {}) const;
};

}  // namespace xmvb::vb
