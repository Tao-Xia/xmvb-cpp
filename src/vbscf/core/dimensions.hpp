#pragma once

#include <cstddef>

namespace xmvb::vb {

struct VbDimensions {
  std::size_t n_basis_functions = 0;
  std::size_t n_active_orbitals = 0;
  std::size_t n_virtual_orbitals = 0;
  std::size_t n_structures = 0;
  std::size_t n_state_averages = 0;
  std::size_t n_orbital_parameters = 0;
  std::size_t n_alpha_electrons = 0;
  std::size_t n_electrons = 0;
};

}  // namespace xmvb::vb
