#pragma once

#include <memory>

#include "vbscf/structures/expansion/types.hpp"
#include "integrals/ao/input.hpp"
#include "integrals/ao/libcint_input.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::integrals {
class RiAoFactorizationProvider;
struct RiAoFactorization;
} // namespace xmvb::integrals

namespace xmvb::vb {

enum class StandardTwoElectronMode {
    Auto,
    Exact,
    ResolutionOfIdentity,
};

/**
 * @brief End-to-end input bundle for the VBSCF matrix path.
 */
struct VbScfInput {
    /**
   * @brief Whether the selected structures span the complete active space.
   *
   * A complete active space is invariant under nonsingular transformations
   * among the active orbitals. Full-AO OEO optimization uses this invariant to
   * remove active--active gauge directions. Structure subsets must leave this
   * flag false because those transformations generally change their
   * variational space.
   */
    bool complete_active_space = false;

    /**
   * @brief Expanded determinant/structure data derived from the raw VB structures.
   */
    StructureExpansion structure_data;

    /**
   * @brief Sparse orbital coefficients and AO overlap used to rebuild auxiliary orbitals.
   */
    OrbitalPreparationInput orbital_preparation_input;

    /**
   * @brief AO Hamiltonian and AO two-electron integrals.
   */
    xmvb::integrals::AoIntegralInput ao_integral_input;

    /**
   * @brief Primary-basis libcint input used by AO integral providers.
   */
    xmvb::integrals::LibcintInput libcint_input;

    /**
   * @brief Optional explicit RI auxiliary basis parsed from the input deck.
   *
   * RI calculations require this basis. The input loader resolves an explicit
   * `AUXBASIS` keyword or the `${BASIS}-jkfit` default before factorization.
   */
    xmvb::integrals::LibcintInput auxiliary_libcint_input;

    /**
   * @brief Requested standard-VB two-electron representation.
   *
   * This mode selects both the AO integral preparation and the active-space
   * two-electron representation. Exact mode materializes four-center AO
   * integrals; RI mode builds the factorized representation directly.
   */
    StandardTwoElectronMode standard_two_electron_mode = StandardTwoElectronMode::Auto;

    /**
   * @brief Provider for lazily building AO-side RI factors.
   */
    std::shared_ptr<const xmvb::integrals::RiAoFactorizationProvider> ri_factorization_provider;

    /**
   * @brief Optional molecule-static AO-side RI factorization shared across copies.
   *
   * This numerical cache is populated through `ri_factorization_provider` and
   * reused across repeated SCF objective/gradient evaluations.
   */
    mutable std::shared_ptr<const xmvb::integrals::RiAoFactorization> ri_factorization;
};

} // namespace xmvb::vb
