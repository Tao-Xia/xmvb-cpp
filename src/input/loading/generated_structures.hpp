#pragma once

#include "input/deck/metadata.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Returns whether a `$CTRL STR=...` structure class can be generated in C++.
 *
 * This builder covers the structure-class input semantics: `STR=FULL`,
 * `STR=COV`, `STR=ION`, and
 * `STR=ION(...)` combinations over determinant-type VB functions. The caller
 * supplies the total electron count because that quantity is resolved upstream
 * from the input deck geometry/charge metadata.
 */
bool can_build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons) noexcept;

/**
 * @brief Builds raw structures from the input structure class.
 *
 * The output uses the same one-based orbital labels and packed structure-by-
 * structure storage expected by the determinant expander.
 */
RawStructureData build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons);

}  // namespace xmvb::vb
