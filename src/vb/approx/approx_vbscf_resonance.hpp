#pragma once

#include <vector>

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

struct ApproxVbScfResonanceDecompositionInput {
  int n_active_orbitals = 0;
  const RawStructureData* raw_structure_data = nullptr;
  const std::vector<double>* structure_hamiltonian_matrix = nullptr;
  const std::vector<double>* structure_overlap_matrix = nullptr;
  const std::vector<double>* eigenvector_matrix = nullptr;
  int state_index = 0;
};

struct ApproxVbScfResonanceDecomposition {
  int n_structures = 0;
  int diagonal_terms = 0;
  int same_pattern_terms = 0;
  int resonance_terms = 0;
  double denominator = 0.0;
  double diagonal_hamiltonian = 0.0;
  double same_pattern_hamiltonian = 0.0;
  double resonance_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  double diagonal_energy = 0.0;
  double same_pattern_energy = 0.0;
  double resonance_energy = 0.0;
  double active_energy = 0.0;
};

/**
 * @brief Decomposes one exact VBSCF state by pair-pattern coherence.
 *
 * The diagonal block is `I == J`.  The same-pattern block has distinct
 * structures with identical active-pair patterns.  The resonance block contains
 * all remaining off-diagonal structure couplings and is the exact-label target
 * for future compact resonance descriptors.
 */
ApproxVbScfResonanceDecomposition decompose_approx_vbscf_resonance(
    const ApproxVbScfResonanceDecompositionInput& input);

}  // namespace xmvb::vb
