#pragma once

#include "pfaffian_vbscf/scf/pf_active_grad_eval.hpp"
#include "pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp"

namespace xmvb::pfaffian_vbscf {

class PfSpinAdaptedActiveGradEval {
public:
  PfActiveGradResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfSpinAdaptedBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;
};

}  // namespace xmvb::pfaffian_vbscf
