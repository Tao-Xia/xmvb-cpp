#pragma once

namespace xmvb::vb {

struct VbDimensions {
  int n_basis_functions = 0;
  int n_active_orbitals = 0;
  int n_virtual_orbitals = 0;
  int n_structures = 0;
  int n_state_averages = 0;
  int n_orbital_parameters = 0;
  int n_alpha_electrons = 0;
  int n_electrons = 0;
};

}  // namespace xmvb::vb
