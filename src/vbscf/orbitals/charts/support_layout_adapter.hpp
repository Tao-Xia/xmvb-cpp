#pragma once

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::vb {

/**
 * @brief Returns whether the sparse orbital layout contains partially overlapping blocks.
 *
 * Partial-overlap support patterns are harmless for the legacy/runtime guess
 * reader, but they are a bad fit for any optimizer path that wants an exact
 * sparse support per overlap-connected orbital component.
 */
bool orbital_input_requires_partial_overlap_support_expansion(
    const OrbitalPreparationInput& orbital_preparation_input);

/**
 * @brief Expands overlap-connected sparse orbital components to their union support.
 *
 * This transformation does not change the represented orbitals. It only
 * rewrites the sparse slot layout so every orbital inside one
 * overlap-connected component shares the same explicit AO support. Downstream
 * code can then rebuild exact-support blocks without seeing stale legacy
 * partial-overlap metadata.
 */
OrbitalPreparationInput build_partial_overlap_support_expanded_input(
    const OrbitalPreparationInput& orbital_preparation_input);

/**
 * @brief Expands the sparse support of a selected orbital subset.
 *
 * The overlap graph and union support are built from the full orbital layout,
 * but only orbitals with a nonzero entry in `expand_orbital_mask` are rewritten
 * to the component union support. Afterwards the exact-support block metadata is
 * rebuilt from the adapted sparse layout and `block_partial_overlap` is updated
 * to reflect whether any partial overlaps remain anywhere in the system.
 */
OrbitalPreparationInput build_partial_overlap_support_expanded_input(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<char>& expand_orbital_mask);

/**
 * @brief Returns the nonredundant optimizer input without changing the legacy sparse chart.
 *
 * The reference VBSCF implementation keeps exact-support orbital blocks and
 * treats partial support overlap only as metadata. For `guess=mo`, preserving
 * that original block partition is necessary to match the legacy variational
 * manifold and converged energy, so this helper currently forwards the input
 * unchanged.
 */
CppVbInput build_nonredundant_optimizer_input(
    const CppVbInput& input);

}  // namespace xmvb::vb
