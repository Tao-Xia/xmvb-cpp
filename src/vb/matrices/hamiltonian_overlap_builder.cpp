#include "vb/matrices/hamiltonian_overlap_builder.hpp"

#include <stdexcept>

namespace xmvb::vb {

HamiltonianOverlapMatrices HamiltonianOverlapBuilder::build(
    const OrbitalSpace& orbital_space,
    VbWavefunctionData& wavefunction_data) const {
  const int dimension = wavefunction_data.dims.n_structures;
  if (dimension <= 0) {
    throw std::invalid_argument("dims.n_structures must be positive");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension);
  if (!wavefunction_data.hamiltonian_matrix.empty() &&
      wavefunction_data.hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix size does not match dims.n_structures");
  }
  if (!wavefunction_data.overlap_matrix.empty() &&
      wavefunction_data.overlap_matrix.size() != matrix_size) {
    throw std::invalid_argument(
        "overlap_matrix size does not match dims.n_structures");
  }

  HamiltonianOverlapMatrices matrices;
  matrices.hamiltonian_matrix = wavefunction_data.hamiltonian_matrix;
  matrices.overlap_matrix = wavefunction_data.overlap_matrix;

  if (matrices.hamiltonian_matrix.empty()) {
    matrices.hamiltonian_matrix.assign(matrix_size, 0.0);
  }
  if (matrices.overlap_matrix.empty()) {
    matrices.overlap_matrix.assign(matrix_size, 0.0);
    for (int i = 0; i < dimension; ++i) {
      matrices.overlap_matrix[static_cast<std::size_t>(i) * dimension + i] = 1.0;
    }
  }

  if (!orbital_space.orbital_parameter_matrix.empty() &&
      static_cast<int>(orbital_space.orbital_parameter_matrix.size()) ==
          wavefunction_data.dims.n_orbital_parameters) {
    matrices.hamiltonian_matrix[0] += orbital_space.orbital_parameter_matrix[0];
  }

  return matrices;
}

}  // namespace xmvb::vb
