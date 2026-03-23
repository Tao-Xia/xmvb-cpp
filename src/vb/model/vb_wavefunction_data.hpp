#pragma once

#include <vector>

#include "vb/model/vb_dimensions.hpp"

namespace xmvb::vb {

using Vector = std::vector<double>;
using Matrix = std::vector<double>;

struct VbWavefunctionData {
  VbDimensions dims;

  double nuclear_repulsion_energy = 0.0;
  double one_electron_energy = 0.0;

  Matrix hamiltonian_matrix;
  Matrix overlap_matrix;
  Matrix eigenvector_matrix;

  Vector electronic_state_energies;
  Vector state_average_weights;
};

}  // namespace xmvb::vb
