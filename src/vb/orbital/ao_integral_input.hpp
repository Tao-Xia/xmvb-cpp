#pragma once

#include "core/shared_vector.hpp"

namespace xmvb::vb {

/**
 * @brief Raw AO integral data required by the C++ VBSCF matrix builders.
 */
struct AoIntegralInput {
  /**
   * @brief Number of AO basis functions.
   */
  int n_basis_functions = 0;

  /**
   * @brief Column-major AO core Hamiltonian matrix `HHF`.
   */
  SharedVector<double> ao_core_hamiltonian_matrix;

  /**
   * @brief Sparse AO two-electron integral values `ggf`.
   */
  SharedVector<double> ao_two_electron_integral_values;

  /**
   * @brief Flattened AO two-electron index table `g2eidx` with 4 entries per integral.
   */
  SharedVector<int> ao_two_electron_integral_indices;
};

}  // namespace xmvb::vb
