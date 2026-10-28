#pragma once

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Expands raw VB structures into unique spin-string pairs.
 *
 * This class replaces the pair/structure bookkeeping part of
 * `rdm_vbscf.F90`. It currently supports the determinant expansion path used by
 * the existing full-VBSCF real-case regression, namely:
 * - `vb_function_type == VbFunctionType::Determinant`
 * - `wavefunction_type != WavefunctionType::Determinant`
 *
 * The implementation produces unique spin strings, pair IDs, and signed
 * structure expansion terms consumed by the Hamiltonian
 * builder.
 */
class StructureExpander {
  public:
    /**
   * @brief Expands raw structure definitions into unique spin-string pairs.
   *
   * The returned object only fills spin-string occupations and signed
   * structure-expansion terms. Integral matrices remain empty and must be
   * populated by a separate input-preparation stage.
   *
   * @param raw_structure_data Raw structure definitions.
   * @return StructureExpansion Spin-string pairs and expansion terms.
   */
    StructureExpansion expand(const RawStructureData& raw_structure_data) const;

    /**
   * @brief Expands only a selected subset of raw structures.
   *
   * The selected structures are reindexed into a compact local structure space
   * with indices `0..selected_structure_indices.size()-1`.
   *
   * @param raw_structure_data Raw structure definitions.
   * @param selected_structure_indices Zero-based structure indices to keep.
   * @return StructureExpansion Spin-string pairs and remapped terms.
   */
    StructureExpansion
    expand_subset(const RawStructureData& raw_structure_data,
                  const std::vector<std::size_t>& selected_structure_indices) const;
};

} // namespace xmvb::vb
