#include "vb/orbital/orbital_parameter_codec.hpp"

#include <stdexcept>

namespace xmvb::vb {

OrbitalSpace OrbitalParameterCodec::decode(
    const OrbitalParameterVector& parameters,
    const VbWavefunctionData& wavefunction_data) const {
  if (wavefunction_data.dims.n_orbital_parameters > 0 &&
      static_cast<int>(parameters.values.size()) !=
          wavefunction_data.dims.n_orbital_parameters) {
    throw std::invalid_argument(
        "parameter vector size does not match dims.n_orbital_parameters");
  }

  OrbitalSpace orbital_space;
  orbital_space.dims = wavefunction_data.dims;
  orbital_space.orbital_parameter_matrix = parameters.values;
  return orbital_space;
}

OrbitalParameterVector OrbitalParameterCodec::encode(
    const OrbitalSpace& orbital_space,
    const VbWavefunctionData& wavefunction_data) const {
  if (wavefunction_data.dims.n_orbital_parameters > 0 &&
      static_cast<int>(orbital_space.orbital_parameter_matrix.size()) !=
          wavefunction_data.dims.n_orbital_parameters) {
    throw std::invalid_argument(
        "orbital space size does not match dims.n_orbital_parameters");
  }

  OrbitalParameterVector parameters;
  parameters.values = orbital_space.orbital_parameter_matrix;
  return parameters;
}

}  // namespace xmvb::vb
