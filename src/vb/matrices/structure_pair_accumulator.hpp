#pragma once

#include <vector>

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Accumulates determinant-pair contributions into structure-level matrices.
 *
 * This class is the new C++ equivalent of the legacy `Hovstr1` kernel. It does
 * not decide which determinant pairs should be visited; it only performs the
 * numerical accumulation once a determinant-pair contribution is known.
 */
class StructurePairAccumulator {
public:
  /**
   * @brief Creates empty structure-level matrices.
   *
   * @param n_structures Number of structures.
   * @param n_determinants Number of determinants for the overlap cache.
   * @return StructureAccumulationResult Zero-initialized accumulation result.
   */
  StructureAccumulationResult create_result(int n_structures, int n_determinants) const;

  /**
   * @brief Accumulates one determinant-pair contribution.
   *
   * @param determinant_index_left Zero-based determinant index on the left.
   * @param determinant_index_right Zero-based determinant index on the right.
   * @param determinant_to_structures_left Structure expansion of the left determinant.
   * @param determinant_to_structures_right Structure expansion of the right determinant.
   * @param overlap_determinant Determinant-pair overlap.
   * @param total_hamiltonian Determinant-pair total Hamiltonian.
   * @param one_electron_hamiltonian Determinant-pair one-electron Hamiltonian.
   * @param accumulation_result Structure matrices updated in place.
   */
  void accumulate(
      int determinant_index_left,
      int determinant_index_right,
      const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
      const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
      double overlap_determinant,
      double total_hamiltonian,
      double one_electron_hamiltonian,
      StructureAccumulationResult& accumulation_result) const;
};

}  // namespace xmvb::vb
