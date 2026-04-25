#pragma once

#include <string>

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Returns whether the input deck contains an explicit `$STR` block.
 *
 * Legacy decks can define VB structures either explicitly through `$STR` or
 * implicitly through `$CTRL` structure-class keywords such as `STR=FULL`.
 * The standalone loader uses this predicate to decide whether it should parse
 * raw structures directly from the input deck or keep using the legacy
 * generated snapshot for the implicit path.
 */
bool input_deck_contains_raw_structure_block(
    const std::string& input_file_path);

/**
 * @brief Parses the `$STR` block directly into the raw legacy structure layout.
 *
 * The returned `raw_structure_orbitals` keeps the original one-based orbital
 * labels expected by the existing determinant expander. This parser mirrors the
 * lightweight token expansion semantics used by legacy `readstr.c` without
 * pulling the rest of the C runtime into the C++ loader.
 */
RawStructureData parse_input_deck_raw_structures(
    const std::string& input_file_path,
    int n_total_electrons,
    int n_active_electrons,
    int spin_multiplicity,
    int wavefunction_type,
    int vb_function_type);

}  // namespace xmvb::vb
