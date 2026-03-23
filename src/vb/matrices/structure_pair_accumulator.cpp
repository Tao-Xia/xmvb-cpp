#include "vb/matrices/structure_pair_accumulator.hpp"

#include <stdexcept>

namespace xmvb::vb {

StructureAccumulationResult StructurePairAccumulator::create_result(
    int n_structures,
    int n_determinants) const {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (n_determinants <= 0) {
    throw std::invalid_argument("n_determinants must be positive");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_structures) * static_cast<std::size_t>(n_structures);

  StructureAccumulationResult result;
  result.n_structures = n_structures;
  result.overlap_matrix.assign(matrix_size, 0.0);
  result.hamiltonian_matrix.assign(matrix_size, 0.0);
  result.one_electron_hamiltonian_matrix.assign(matrix_size, 0.0);
  result.determinant_overlap_cache.assign(static_cast<std::size_t>(n_determinants), 0.0);
  return result;
}

void StructurePairAccumulator::accumulate(
    int determinant_index_left,
    int determinant_index_right,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
    double overlap_determinant,
    double total_hamiltonian,
    double one_electron_hamiltonian,
    StructureAccumulationResult& accumulation_result) const {
  if (determinant_index_left < 0 || determinant_index_right < 0) {
    throw std::invalid_argument("determinant indices must be non-negative");
  }
  if (static_cast<std::size_t>(determinant_index_left) >=
          accumulation_result.determinant_overlap_cache.size() ||
      static_cast<std::size_t>(determinant_index_right) >=
          accumulation_result.determinant_overlap_cache.size()) {
    throw std::invalid_argument("determinant index out of cache range");
  }

  if (determinant_index_left == determinant_index_right) {
    accumulation_result.determinant_overlap_cache[static_cast<std::size_t>(determinant_index_right)] =
        overlap_determinant;
  }

  for (const auto& left_term : determinant_to_structures_left) {
    if (left_term.structure_index < 0 || left_term.structure_index >= accumulation_result.n_structures) {
      throw std::invalid_argument("left structure index out of range");
    }

    for (const auto& right_term : determinant_to_structures_right) {
      if (right_term.structure_index < 0 ||
          right_term.structure_index >= accumulation_result.n_structures) {
        throw std::invalid_argument("right structure index out of range");
      }

      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }

      const double combined_coefficient = left_term.coefficient * right_term.coefficient;
      // Matrices in the VB C++ kernels are stored in column-major order to
      // match the legacy Fortran data layout and simplify regression checks.
      const std::size_t linear_index =
          static_cast<std::size_t>(right_term.structure_index) * accumulation_result.n_structures +
          left_term.structure_index;

      accumulation_result.overlap_matrix[linear_index] += overlap_determinant * combined_coefficient;
      accumulation_result.hamiltonian_matrix[linear_index] += total_hamiltonian * combined_coefficient;
      accumulation_result.one_electron_hamiltonian_matrix[linear_index] +=
          one_electron_hamiltonian * combined_coefficient;
    }
  }
}

}  // namespace xmvb::vb
