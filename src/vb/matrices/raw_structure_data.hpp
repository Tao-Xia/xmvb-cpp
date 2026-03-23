#pragma once

#include <cstddef>
#include <vector>

namespace xmvb::vb {

/**
 * @brief Raw VB structure definitions read directly from the input deck.
 *
 * This object stores the unspecialized structure representation before the
 * legacy `rdm_vbscf` code expands it into unique determinants.
 */
struct RawStructureData {
  /**
   * @brief Number of structures.
   */
  int n_structures = 0;

  /**
   * @brief Total number of electrons.
   */
  int n_total_electrons = 0;

  /**
   * @brief Number of active electrons.
   */
  int n_active_electrons = 0;

  /**
   * @brief Spin multiplicity.
   */
  int spin_multiplicity = 1;

  /**
   * @brief Legacy wavefunction type flag.
   */
  int wavefunction_type = 0;

  /**
   * @brief Legacy VB function type flag.
   */
  int vb_function_type = 0;

  /**
   * @brief Flat structure orbital storage in legacy one-based ordering.
   *
   * The data is packed structure-by-structure. Each structure contributes
   * `n_total_electrons` orbital labels using the legacy one-based convention.
   */
  std::vector<int> raw_structure_orbitals;

  std::size_t flat_orbital_count() const {
    return static_cast<std::size_t>(n_structures) *
           static_cast<std::size_t>(n_total_electrons);
  }

  const int* structure_orbitals_data(int structure_index) const {
    return raw_structure_orbitals.data() +
           static_cast<std::size_t>(structure_index) *
               static_cast<std::size_t>(n_total_electrons);
  }
};

}  // namespace xmvb::vb
