#include "vb/approx/approx_vbscf_resonance.hpp"

#include <algorithm>
#include <vector>

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

std::vector<int> build_structure_pair_pattern(
    const RawStructureData& raw_structure_data,
    int structure_index) {
  const int active_start =
      raw_structure_data.n_total_electrons -
      raw_structure_data.n_active_electrons;
  const int n_inactive_doubly_occupied_orbitals = active_start / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_pairs =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  const int* orbitals =
      raw_structure_data.structure_orbitals_data(structure_index);

  std::vector<int> pair_pattern;
  pair_pattern.reserve(n_active_pairs);
  for (int pair_index = 0; pair_index < n_active_pairs; ++pair_index) {
    const int first_orbital =
        orbitals[active_start + 2 * pair_index] -
        n_inactive_doubly_occupied_orbitals - 1;
    const int second_orbital =
        orbitals[active_start + 2 * pair_index + 1] -
        n_inactive_doubly_occupied_orbitals - 1;
    pair_pattern.push_back(
        TwoElectronIndexer::packed_pair_index(first_orbital, second_orbital));
  }
  std::sort(pair_pattern.begin(), pair_pattern.end());
  return pair_pattern;
}

std::vector<std::vector<int>> build_structure_pair_patterns(
    const RawStructureData& raw_structure_data) {
  std::vector<std::vector<int>> patterns(raw_structure_data.n_structures);
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    patterns[structure_index] =
        build_structure_pair_pattern(raw_structure_data, structure_index);
  }
  return patterns;
}

}  // namespace

ApproxVbScfResonanceDecomposition decompose_approx_vbscf_resonance(
    const ApproxVbScfResonanceDecompositionInput& input) {
  const RawStructureData& raw_structure_data = *input.raw_structure_data;
  const int n_structures = raw_structure_data.n_structures;
  const double* coefficients =
      input.eigenvector_matrix->data() +
      input.state_index * n_structures;
  const std::vector<std::vector<int>> pair_patterns =
      build_structure_pair_patterns(raw_structure_data);

  ApproxVbScfResonanceDecomposition result;
  result.n_structures = n_structures;

  for (int column = 0; column < n_structures; ++column) {
    for (int row = 0; row < n_structures; ++row) {
      const int matrix_index = column * n_structures + row;
      const double pair_coefficient =
          coefficients[row] * coefficients[column];
      const double h_contribution =
          pair_coefficient *
          (*input.structure_hamiltonian_matrix)[matrix_index];
      const double s_contribution =
          pair_coefficient *
          (*input.structure_overlap_matrix)[matrix_index];
      result.denominator += s_contribution;
      result.total_hamiltonian += h_contribution;

      if (row == column) {
        result.diagonal_hamiltonian += h_contribution;
        ++result.diagonal_terms;
      } else if (pair_patterns[row] == pair_patterns[column]) {
        result.same_pattern_hamiltonian += h_contribution;
        ++result.same_pattern_terms;
      } else {
        result.resonance_hamiltonian += h_contribution;
        ++result.resonance_terms;
      }
    }
  }

  result.diagonal_energy =
      result.diagonal_hamiltonian / result.denominator;
  result.same_pattern_energy =
      result.same_pattern_hamiltonian / result.denominator;
  result.resonance_energy =
      result.resonance_hamiltonian / result.denominator;
  result.active_energy =
      result.total_hamiltonian / result.denominator;
  return result;
}

}  // namespace xmvb::vb
