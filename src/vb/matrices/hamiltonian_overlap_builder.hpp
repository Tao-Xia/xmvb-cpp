#pragma once

#include "vb/matrices/hamiltonian_overlap_matrices.hpp"
#include "vb/model/vb_wavefunction_data.hpp"
#include "vb/orbital/orbital_space.hpp"

namespace xmvb::vb {

class HamiltonianOverlapBuilder {
public:
  HamiltonianOverlapMatrices build(
      const OrbitalSpace& orbital_space,
      VbWavefunctionData& wavefunction_data) const;
};

}  // namespace xmvb::vb
