#pragma once

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Builds compact selected-structure subspaces from full expanded inputs.
 *
 * This helper filters determinant expansion terms to a chosen structure subset
 * and remaps kept structure indices into a dense local index space.
 */
class StructureSubspaceBuilder {
public:
  /**
   * @brief Filters an expanded determinant dataset down to a selected structure subset.
   *
   * @param full_data Full determinant/structure expansion data.
   * @param selected_structure_indices Zero-based structure indices to keep.
   * @return FullDeterminantStructureData Filtered determinant data with remapped indices.
   */
  FullDeterminantStructureData build(
      const FullDeterminantStructureData& full_data,
      const std::vector<int>& selected_structure_indices) const;

  /**
   * @brief Filters a complete C++ VB input bundle to a selected structure subset.
   *
   * Orbital and AO integral inputs are forwarded unchanged. Only the structure
   * expansion data is filtered and reindexed.
   *
   * @param input Full C++ VB input bundle.
   * @param selected_structure_indices Zero-based structure indices to keep.
   * @return CppVbInput Filtered input bundle.
   */
  CppVbInput build(
      const CppVbInput& input,
      const std::vector<int>& selected_structure_indices) const;
};

}  // namespace xmvb::vb
