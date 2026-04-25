#pragma once

#include "runtime/input_deck_metadata.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Returns whether a `$CTRL STR=...` structure class can be generated in C++.
 *
 * This builder covers the legacy `genstr.c` semantics for the standalone
 * structure-wavefunction path: `STR=FULL`, `STR=COV`, `STR=ION`, and
 * `STR=ION(...)` combinations over determinant-type VB functions. The caller
 * supplies the total electron count because that quantity is resolved upstream
 * from the input deck geometry/charge metadata.
 */
bool can_build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons) noexcept;

/**
 * @brief Rebuilds the legacy generated raw structures directly in C++.
 *
 * The output uses the same one-based orbital labels and packed structure-by-
 * structure storage expected by the existing determinant expander. No legacy C
 * `genstr/getstr` fallback is involved.
 */
RawStructureData build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons);

}  // namespace xmvb::vb
