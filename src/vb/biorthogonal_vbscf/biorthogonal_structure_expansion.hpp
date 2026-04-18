#pragma once

#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Determinant list plus selected-structure expansion matrix `T`.
 *
 * `structure_to_determinant` has dimensions `(n_determinants, n_structures)`
 * and stores the selected-structure expansion coefficients
 *
 * `| \Phi_K > = \sum_I T_{I K} | D_I >`.
 *
 * This is the exact determinant expansion already produced by the existing
 * full-structure C++ builder; the biorthogonal prototype only repackages that
 * data into the linear-algebra layout expected by the fixed-metric solver.
 */
struct BiorthogonalStructureExpansion {
  std::vector<BiorthogonalDeterminant> determinants;
  Eigen::MatrixXd structure_to_determinant;
};

/**
 * @brief Repackages one `FullDeterminantStructureData` object into `(D, T)`.
 *
 * The determinant ordering is kept identical to the incoming full-determinant
 * data so later projected Hamiltonians can be compared entry-by-entry with the
 * existing nonorthogonal structure pipeline if needed.
 */
BiorthogonalStructureExpansion build_biorthogonal_structure_expansion(
    const xmvb::vb::FullDeterminantStructureData& structure_data);

/**
 * @brief Validates determinant dimensions and the expansion matrix layout.
 */
void validate_biorthogonal_structure_expansion(
    const BiorthogonalStructureExpansion& structure_expansion,
    int expected_structure_count,
    int expected_active_orbital_count);

}  // namespace xmvb::vb::biorthogonal_vbscf
