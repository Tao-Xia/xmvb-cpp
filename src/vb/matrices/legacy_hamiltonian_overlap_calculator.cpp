#include "vb/matrices/legacy_hamiltonian_overlap_calculator.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "vb/matrices/legacy_hamiltonian_overlap_calculator_c.h"

namespace xmvb::vb {

HamiltonianOverlapCalculationResult
LegacyHamiltonianOverlapCalculator::calculate_from_input_file(
    const std::string& input_file_path,
    int n_threads) const {
  int n_structures = 0;
  double total_energy = 0.0;
  double one_electron_energy = 0.0;
  double* hamiltonian_matrix = nullptr;
  double* overlap_matrix = nullptr;
  char error_message[1024] = {0};

  const int status = legacy_calculate_hamiltonian_overlap(
      input_file_path.c_str(),
      n_threads,
      &n_structures,
      &total_energy,
      &one_electron_energy,
      &hamiltonian_matrix,
      &overlap_matrix,
      error_message,
      sizeof(error_message));
  if (status != 0) {
    throw std::runtime_error(error_message[0] != '\0' ? error_message : "unknown legacy error");
  }

  const std::size_t matrix_size =
      static_cast<std::size_t>(n_structures) * static_cast<std::size_t>(n_structures);

  HamiltonianOverlapCalculationResult result;
  result.n_structures = n_structures;
  result.total_energy = total_energy;
  result.one_electron_energy = one_electron_energy;
  result.hamiltonian_matrix.assign(hamiltonian_matrix, hamiltonian_matrix + matrix_size);
  result.overlap_matrix.assign(overlap_matrix, overlap_matrix + matrix_size);

  std::free(hamiltonian_matrix);
  std::free(overlap_matrix);
  return result;
}

}  // namespace xmvb::vb
