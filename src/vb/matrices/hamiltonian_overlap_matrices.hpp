#pragma once

#include <vector>

namespace xmvb::vb {

struct HamiltonianOverlapMatrices {
  std::vector<double> hamiltonian_matrix;
  std::vector<double> overlap_matrix;
};

}  // namespace xmvb::vb
