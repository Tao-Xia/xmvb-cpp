#pragma once

#include <vector>

#include "pfaffian_vbscf/scf/pf_scf_eval.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "pfaffian_vbscf/types/pf_pair_profile.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/orbital/active_space_one_electron_result.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"
#include "vb/orbital/ao_effective_one_electron_result.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfActiveGradResult {
  PfScfResult scf_result;
  xmvb::vb::OrbitalPreparationResult orbital_preparation_result;
  xmvb::vb::AoEffectiveOneElectronResult ao_effective_one_electron_result;
  xmvb::vb::ActiveSpaceOneElectronResult act_h1e_result;
  xmvb::vb::ActiveSpaceTwoElectronResult act_eri_result;
  std::vector<double> sso;
  std::vector<double> sso_grad;
  std::vector<double> hho_grad;
  std::vector<double> ggo_grad;
  std::vector<double> ri_active_pair_factor_grad;
  PfPairProfile pair_grad_profile;
  double orbital_prepare_dt = 0.0;
  double ao_h1e_build_dt = 0.0;
  double active_h1e_build_dt = 0.0;
  double active_2e_build_dt = 0.0;
  double matrix_dt = 0.0;
  double adj_dt = 0.0;
  double total_dt = 0.0;
};

class PfActiveGradEval {
public:
  PfActiveGradResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;
};

}  // namespace xmvb::pfaffian_vbscf
