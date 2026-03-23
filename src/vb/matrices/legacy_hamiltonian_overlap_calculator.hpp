#pragma once

#include <string>
#include <vector>

namespace xmvb::vb {

struct HamiltonianOverlapCalculationResult {
  int n_structures = 0;
  double total_energy = 0.0;
  double one_electron_energy = 0.0;
  std::vector<double> hamiltonian_matrix;
  std::vector<double> overlap_matrix;
};

class LegacyHamiltonianOverlapCalculator {
public:
  HamiltonianOverlapCalculationResult calculate_from_input_file(
      const std::string& input_file_path,
      int n_threads) const;
};

}  // namespace xmvb::vb
