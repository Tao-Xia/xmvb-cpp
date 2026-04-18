#pragma once

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Biorthogonal-ready bundle with both prepared active-space tensors and determinant topology.
 *
 * The runtime `CppVbInput` stores the determinant/structure expansion and the
 * AO/orbital data separately. Biorthogonal determinant and selected-space
 * evaluators repeatedly need both:
 * 1. the prepared active-space tensors `X`, `HHO`, and active-space `GGO`
 * 2. the explicit full-determinant structure topology
 *
 * This bundle centralizes that conversion so repeated selected-subspace
 * evaluations can prepare the active-space layer once and reuse it.
 */
struct PreparedBiorthogonalInput {
  xmvb::vb::PreparedActiveSpaceContext prepared_active_space;
  xmvb::vb::FullDeterminantStructureData structure_data;
};

/**
 * @brief Merges one prepared active-space context back into explicit full determinant data.
 *
 * The returned `FullDeterminantStructureData` carries the full determinant
 * topology together with the prepared active-space overlap, one-electron, and
 * packed two-electron tensors needed by the biorthogonal matrix builders.
 */
xmvb::vb::FullDeterminantStructureData build_biorthogonal_full_structure_data(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space);

/**
 * @brief Prepares the active-space tensors and returns the merged biorthogonal input bundle.
 */
PreparedBiorthogonalInput prepare_biorthogonal_input(
    const xmvb::vb::CppVbInput& input);

}  // namespace xmvb::vb::biorthogonal_vbscf
