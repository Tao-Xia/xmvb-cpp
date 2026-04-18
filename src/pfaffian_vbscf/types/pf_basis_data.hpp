#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Nonorthogonal Pfaffian basis used by the Pf VBSCF drivers.
 */
struct PfBasisData {
  int n_states = 0;
  int n_active_orbitals = 0;
  int n_alpha = 0;
  int n_beta = 0;
  int n_singlet_pairs = 0;
  int n_blocked_alpha = 0;
  int n_blocked_beta = 0;
  std::vector<PfState> states;

  bool has_blocked_open_shell() const {
    return n_blocked_alpha > 0 || n_blocked_beta > 0;
  }
};

}  // namespace xmvb::pfaffian_vbscf
