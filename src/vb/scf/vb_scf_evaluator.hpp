#pragma once

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/matrices/hamiltonian_overlap_builder.hpp"
#include "vb/model/vb_wavefunction_data.hpp"
#include "vb/orbital/orbital_parameter_codec.hpp"
#include "vb/orbital/orbital_parameter_vector.hpp"
#include "vb/scf/vb_scf_result.hpp"

namespace xmvb::vb {

class VbScfEvaluator {
public:
  VbScfEvaluator(
      OrbitalParameterCodec orbital_parameter_codec,
      HamiltonianOverlapBuilder hamiltonian_overlap_builder,
      xmvb::core::GeneralizedEigensolver generalized_eigensolver);

  VbScfResult evaluate(
      const OrbitalParameterVector& parameters,
      VbWavefunctionData& wavefunction_data) const;

private:
  OrbitalParameterCodec orbital_parameter_codec_;
  HamiltonianOverlapBuilder hamiltonian_overlap_builder_;
  xmvb::core::GeneralizedEigensolver generalized_eigensolver_;
};

}  // namespace xmvb::vb
