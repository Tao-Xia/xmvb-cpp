#pragma once

#include "vb/matrices/structure_types.hpp"
#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

/**
 * @brief End-to-end input bundle for the C++ VB matrix path.
 */
struct CppVbInput {
  /**
   * @brief Expanded determinant/structure data derived from the raw VB structures.
   */
  FullDeterminantStructureData structure_data;

  /**
   * @brief Sparse orbital coefficients and AO overlap used to rebuild auxiliary orbitals.
   */
  OrbitalPreparationInput orbital_preparation_input;

  /**
   * @brief AO Hamiltonian and AO two-electron integrals.
   */
  AoIntegralInput ao_integral_input;
};

}  // namespace xmvb::vb
