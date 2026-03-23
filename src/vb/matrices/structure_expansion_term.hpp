#pragma once

namespace xmvb::vb {

/**
 * @brief One determinant-to-structure expansion term.
 *
 * In the legacy VBSCF code each determinant can contribute to one or more
 * structures with a sign factor. This structure makes that mapping explicit.
 */
struct StructureExpansionTerm {
  /**
   * @brief Zero-based structure index.
   */
  int structure_index = 0;

  /**
   * @brief Signed coefficient contributed by the determinant to the structure.
   *
   * The current legacy mapping uses `+1` or `-1`, but this representation
   * intentionally allows any real coefficient.
   */
  double coefficient = 0.0;
};

}  // namespace xmvb::vb
