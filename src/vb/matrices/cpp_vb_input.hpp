#pragma once

#include <memory>

#include "vb/matrices/structure_types.hpp"
#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/libcint_input.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

struct LibcintRiIntegralProviderResult;

enum class StandardTwoElectronMode {
  Auto,
  Exact,
  ResolutionOfIdentity,
};

enum class PfTwoElectronMode {
  Auto,
  Exact,
  ResolutionOfIdentity,
};

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

  /**
   * @brief Raw libcint arrays retained for future C++-native integral generation.
   */
  LibcintInput libcint_input;

  /**
   * @brief Optional explicit RI auxiliary basis parsed from the input/runtime.
   *
   * When present, the standard RI path should prefer this basis over any
   * generated fallback so `INT=RI` follows the same auxiliary-basis semantics
   * as the legacy runtime.
   */
  LibcintInput auxiliary_libcint_input;

  /**
   * @brief Requested standard-VB two-electron representation.
   *
   * This mode selects how the active-space two-electron intermediates are
   * built. The AO-integral source may still choose an exact or Hcore-only path
   * independently so RI runs can skip exact AO four-center materialization.
   */
  StandardTwoElectronMode standard_two_electron_mode =
      StandardTwoElectronMode::Auto;

  /**
   * @brief Requested Pf two-electron representation.
   *
   * `Auto` preserves the current environment-variable fallback behavior.
   */
  PfTwoElectronMode pf_two_electron_mode = PfTwoElectronMode::Auto;

  /**
   * @brief Optional molecule-static AO-side RI cache shared across copies.
   *
   * This is populated lazily by the Pf-RI path and reused across repeated SCF
   * objective/gradient evaluations.
   */
  mutable std::shared_ptr<const LibcintRiIntegralProviderResult>
      ri_integral_provider_result;
};

}  // namespace xmvb::vb
