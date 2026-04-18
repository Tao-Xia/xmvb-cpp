#pragma once

#include <vector>

#include "vb/exact_separator/component_data.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Returns `(-1)^parity`.
 */
double parity_sign(int parity);

/**
 * @brief Counts the parity of sorting an occupied-orbital list into ascending order.
 *
 * The input list may be in component/block order. The returned parity is the
 * number of pair inversions modulo two.
 */
int canonicalization_parity(const std::vector<int>& occupied_orbitals);

/**
 * @brief Enumerates and merges the exact global determinant terms for one side.
 *
 * The routine takes the Cartesian product of all component-local orientation
 * terms on the requested side, applies the exact inter-component
 * canonicalization sign, canonicalizes the occupied-orbital lists, and merges
 * duplicate global determinants by summing their coefficients.
 */
std::vector<GlobalOrientationTerm> build_global_orientation_terms(
    const std::vector<ComponentData>& ordered_components,
    bool use_left_terms);

}  // namespace xmvb::vb::exact_separator
