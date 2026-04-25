#pragma once

#include <vector>

#include "runtime/input_deck_model.hpp"

namespace xmvb::vb {

/**
 * @brief Molecule-static data needed to rebuild sparse orbital supports.
 *
 * The input deck only names fragments, atoms, and Cartesian AO labels.  The
 * builder therefore needs a lightweight AO-to-atom map and the Cartesian
 * exponents for each AO so it can translate `$FRAG/$ORB` declarations into the
 * padded one-based AO index table used by the optimizer. `$ACTORB` additionally
 * uses `n_active_orbitals` so the builder can splice the explicit active rows
 * onto the historical default inactive-orbital support.
 */
struct InputDeckOrbitalSupportBuildInput {
  int n_atoms = 0;
  int n_basis_functions = 0;
  int n_orbitals = 0;
  int n_active_orbitals = 0;
  int orbital_type = 0;
  int fragment_type = 0;
  std::vector<int> ao_to_atom;
  std::vector<int> ao_cartesian_exponents;
};

/**
 * @brief Sparse orbital support chart reconstructed from pure C++ deck data.
 *
 * `orbital_basis_index_table` keeps the historical padded `(n_orbitals,
 * n_basis_functions)` layout with one-based AO indices, while the count arrays
 * record how many explicit coefficients are meaningful in each orbital row.
 */
struct InputDeckOrbitalSupportChart {
  std::vector<int> orbital_basis_index_table;
  std::vector<int> orbital_basis_counts;
  std::vector<int> original_orbital_basis_counts;
};

bool can_build_input_deck_orbital_support_chart(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input) noexcept;

InputDeckOrbitalSupportChart build_input_deck_orbital_support_chart(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input);

}  // namespace xmvb::vb
