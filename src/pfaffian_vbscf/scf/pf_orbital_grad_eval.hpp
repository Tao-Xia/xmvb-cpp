#pragma once

#include <vector>

#include "pfaffian_vbscf/scf/pf_active_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.hpp"
#include "pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_orbital_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfOrbitalGradResult {
  PfScfResult scf_result;
  std::vector<double> sparse_orbital_energy_gradient;
  std::vector<int> differentiable_parameter_indices;
  int n_objective_evaluations = 1;
  double orbital_prepare_dt = 0.0;
  double ao_h1e_build_dt = 0.0;
  double active_h1e_build_dt = 0.0;
  double active_2e_build_dt = 0.0;
  double pf_matrix_forward_dt = 0.0;
  double pf_adjoint_dt = 0.0;
  double forward_wall_time_seconds = 0.0;
  double active_space_grad_dt = 0.0;
  double matrix_backprop_dt = 0.0;
  double two_electron_backprop_dt = 0.0;
  double ao_h1e_backprop_dt = 0.0;
  double orbital_backprop_dt = 0.0;
  double backprop_wall_time_seconds = 0.0;
  double total_dt = 0.0;
  double total_wall_time_seconds = 0.0;
};

class PfOrbitalGradEval {
public:
  PfOrbitalGradEval();

  PfOrbitalGradEval(
      PfActiveGradEval active_gradient_evaluator,
      xmvb::vb::AoEffectiveOneElectronBackpropagator
          ao_effective_one_electron_backpropagator,
      xmvb::vb::ActiveSpaceMatrixBackpropagator
          active_space_matrix_backpropagator,
      xmvb::vb::ActiveSpaceTwoElectronBackpropagator
          active_space_two_electron_backpropagator,
      xmvb::vb::ActiveSpaceOrbitalBackpropagator
          active_space_orbital_backpropagator);

  PfOrbitalGradResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;

  PfOrbitalGradResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfSpinAdaptedBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;

private:
  PfActiveGradEval active_gradient_evaluator_;
  PfSpinAdaptedActiveGradEval spin_adapted_active_gradient_evaluator_;
  xmvb::vb::AoEffectiveOneElectronBackpropagator
      ao_effective_one_electron_backpropagator_;
  xmvb::vb::ActiveSpaceMatrixBackpropagator
      active_space_matrix_backpropagator_;
  xmvb::vb::ActiveSpaceTwoElectronBackpropagator
      active_space_two_electron_backpropagator_;
  xmvb::vb::ActiveSpaceOrbitalBackpropagator
      active_space_orbital_backpropagator_;
};

}  // namespace xmvb::pfaffian_vbscf
