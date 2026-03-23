#pragma once

#include "vb/model/vb_wavefunction_data.hpp"
#include "vb/orbital/orbital_parameter_vector.hpp"
#include "vb/orbital/orbital_space.hpp"

namespace xmvb::vb {

class OrbitalParameterCodec {
public:
  OrbitalSpace decode(
      const OrbitalParameterVector& parameters,
      const VbWavefunctionData& wavefunction_data) const;

  OrbitalParameterVector encode(
      const OrbitalSpace& orbital_space,
      const VbWavefunctionData& wavefunction_data) const;
};

}  // namespace xmvb::vb
