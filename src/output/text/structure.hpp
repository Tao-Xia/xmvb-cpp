#pragma once

#include <iosfwd>

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::output {

/**
 * @brief Print one VB structure without changing its input pair topology.
 *
 * Closed-shell active pairs are printed as `i-j` for a covalent pair and
 * `i i` for an ionic pair. Open-shell orbitals follow the pairs. Long
 * structures wrap after every ten active pairs and align below the first
 * pair field.
 *
 * @param output Destination stream.
 * @param structures Raw one-based structure definitions.
 * @param structure_index Zero-based structure index.
 * @param prefix_width Number of columns already printed on the first line.
 */
void print_structure(
    std::ostream& output,
    const vb::RawStructureData& structures,
    int structure_index,
    int prefix_width);

}  // namespace xmvb::output
