#pragma once

#include <vector>

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfSpinStringAssignment {
  std::vector<int> blocked_beta_positions;
};

struct PfSpinCouplingBlock {
  int n_open_shell_electrons = 0;
  int n_blocked_alpha = 0;
  int n_blocked_beta = 0;
  int spin_multiplicity = 1;
  int ms_twice = 0;
  Matrix primitive_to_adapted_coefficients;
  std::vector<PfSpinStringAssignment> primitive_spin_strings;
};

class PfSpinCouplingBuilder {
public:
  PfSpinCouplingBlock build(
      int n_open_shell_electrons,
      int spin_multiplicity,
      int ms_twice) const;
};

}  // namespace xmvb::pfaffian_vbscf
