#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_spin_adapted_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfSpinAdaptedBasisData {
  PfBasisData primitive_basis;
  int n_states = 0;
  int spin_multiplicity = 1;
  int ms_twice = 0;
  Matrix primitive_to_adapted_coefficients;
  std::vector<PfSpinAdaptedState> states;

  bool has_blocked_open_shell() const {
    return primitive_basis.has_blocked_open_shell();
  }
};

}  // namespace xmvb::pfaffian_vbscf
