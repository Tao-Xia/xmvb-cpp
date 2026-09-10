#pragma once

#include <vector>

#include "vbscf/orbitals/orbital_preparation_input.hpp"

namespace xmvb::vb {

/**
 * @brief Partitions orbitals by identical explicit AO support.
 *
 * stored block metadata is authoritative when present. Otherwise blocks are
 * inferred from the explicit sparse support table. The returned orbital
 * indices are zero based.
 */
std::vector<std::vector<int>> detect_orbital_blocks(
    const OrbitalPreparationInput& orbital_preparation_input);

}  // namespace xmvb::vb
