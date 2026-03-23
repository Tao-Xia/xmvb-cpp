#pragma once

#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Rebuilds auxiliary occupied and virtual orbitals in pure C++.
 *
 * This class is the C++ replacement for the orbital-preparation part of
 * `Orbprep`. It constructs the auxiliary orbital matrix and the active-space
 * overlap matrix directly from the sparse VB orbital parameterization and the
 * AO overlap matrix.
 */
class ActiveSpaceOrbitalPreparer {
public:
  /**
   * @brief Builds auxiliary orbitals and the active-space overlap matrix.
   *
   * @param input Orbital parameterization and AO overlap data.
   * @return OrbitalPreparationResult Prepared auxiliary orbitals and overlap matrices.
   */
  OrbitalPreparationResult prepare(const OrbitalPreparationInput& input) const;
};

}  // namespace xmvb::vb
