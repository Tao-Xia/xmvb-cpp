#pragma once

#include <vector>

namespace xmvb::vb {

struct VbScfResult {
  double total_energy = 0.0;
  double one_electron_energy = 0.0;
  double valence_bond_structure_energy = 0.0;
  double average_structure_overlap = 0.0;

  std::vector<double> electronic_state_energies;
  std::vector<double> eigenvector_matrix;
};

}  // namespace xmvb::vb
