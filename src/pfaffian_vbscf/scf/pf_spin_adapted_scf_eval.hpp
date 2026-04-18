#pragma once

#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/scf/pf_scf_eval.hpp"
#include "pfaffian_vbscf/types/pf_active_space_data.hpp"
#include "pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::pfaffian_vbscf {

class PfSpinAdaptedScfEval {
public:
  PfSpinAdaptedScfEval();

  PfSpinAdaptedScfEval(
      PfActBuilder active_space_builder,
      PfMatrixBuilder matrix_builder);

  PfScfResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfSpinAdaptedBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;

  PfScfResult eval_active_space(
      const PfActiveSpaceData& active_space,
      const PfSpinAdaptedBasisData& basis,
      double nuclear_repulsion_energy = 0.0,
      double reference_energy = 0.0) const;

private:
  PfActBuilder active_space_builder_;
  PfMatrixBuilder matrix_builder_;
};

}  // namespace xmvb::pfaffian_vbscf
