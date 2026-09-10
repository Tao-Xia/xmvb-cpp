#pragma once

#include <memory>

#include "vbscf/structures/structure_types.hpp"
#include "vbscf/integrals/ao/ao_integral_input.hpp"
#include "vbscf/integrals/ao/libcint_input.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"

namespace xmvb::vb {

class RiAoFactorizationProvider;
struct RiAoFactorization;

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
 * @brief End-to-end input bundle for the VBSCF matrix path.
 */
struct VbScfInput {
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
   * @brief Primary-basis libcint input used by AO integral providers.
   */
  LibcintInput libcint_input;

  /**
   * @brief Optional explicit RI auxiliary basis parsed from the input/runtime.
   *
   * When present, the RI path uses this basis. Otherwise the RI provider
   * generates the auxiliary basis defined by its numerical contract.
   */
  LibcintInput auxiliary_libcint_input;

  /**
   * @brief Requested standard-VB two-electron representation.
   *
   * This mode selects both the AO integral preparation and the active-space
   * two-electron representation. Exact mode materializes four-center AO
   * integrals; RI mode builds the factorized representation directly.
   */
  StandardTwoElectronMode standard_two_electron_mode =
      StandardTwoElectronMode::Auto;

  /**
   * @brief Requested Pf two-electron representation.
   *
   * `Auto` lets the consuming Pf evaluator select its default representation.
   */
  PfTwoElectronMode pf_two_electron_mode = PfTwoElectronMode::Auto;

  /**
   * @brief Runtime-injected backend for lazily building AO-side RI factors.
   */
  std::shared_ptr<const RiAoFactorizationProvider> ri_factorization_provider;

  /**
   * @brief Optional molecule-static AO-side RI factorization shared across copies.
   *
   * This numerical cache is populated through `ri_factorization_provider` and
   * reused across repeated SCF objective/gradient evaluations.
   */
  mutable std::shared_ptr<const RiAoFactorization> ri_factorization;
};

}  // namespace xmvb::vb
