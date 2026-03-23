#pragma once

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Expands raw VB structures into unique full determinants.
 *
 * This class replaces the determinant/structure bookkeeping part of
 * `rdm_vbscf.F90`. It currently supports the determinant expansion path used by
 * the existing full-VBSCF real-case regression, namely:
 * - `vb_function_type == VbFunctionType::Determinant`
 * - `wavefunction_type != WavefunctionType::Determinant`
 *
 * The implementation is fully in C++ and produces the determinant occupations
 * and signed structure expansion terms consumed by the pure C++ Hamiltonian
 * builder.
 */
class FullDeterminantStructureExpander {
public:
  /**
   * @brief Expands raw structure definitions into unique determinants.
   *
   * The returned object only fills determinant occupations and signed
   * structure-expansion terms. Integral matrices remain empty and must be
   * populated by a separate input-preparation stage.
   *
   * @param raw_structure_data Raw structure definitions.
   * @return FullDeterminantStructureData Determinants and expansion terms.
   */
  FullDeterminantStructureData expand(
      const RawStructureData& raw_structure_data) const;

  /**
   * @brief Expands only a selected subset of raw structures.
   *
   * The selected structures are reindexed into a compact local structure space
   * with indices `0..selected_structure_indices.size()-1`.
   *
   * @param raw_structure_data Raw structure definitions.
   * @param selected_structure_indices Zero-based structure indices to keep.
   * @return FullDeterminantStructureData Determinants and remapped expansion terms.
   */
  FullDeterminantStructureData expand_subset(
      const RawStructureData& raw_structure_data,
      const std::vector<int>& selected_structure_indices) const;
};

}  // namespace xmvb::vb
