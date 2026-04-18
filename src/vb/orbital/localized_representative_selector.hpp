#pragma once

#include <Eigen/Core>

#include "vb/pdft/physical_orbital_frame.hpp"

namespace xmvb::vb {

/**
 * @brief Builds the accepted-point localized representative selector.
 *
 * The accepted-point internal inactive-orthogonal frame `(Q_i, T_a)` and the
 * localized physical occupied representative `(C_i, C_a)` are related by the
 * small matrices
 *
 * `C_i = Q_i U_i`
 * and
 * `C_a = T_a + Q_i K_a`.
 *
 * This routine extracts `U_i`, `U_i^{-T}`, and `K_a` directly from the cached
 * accepted-point orbital blocks so later chart transport code can reuse them
 * without repeating the same block-level linear algebra.
 *
 * Dimensions:
 * - `inactive_physical_orbitals`: `(n_basis_functions, n_inactive)`
 * - `inactive_orthonormal_orbitals`: `(n_basis_functions, n_inactive)`
 * - `active_physical_orbitals`: `(n_basis_functions, n_active)`
 * - `active_auxiliary_orbitals`: `(n_basis_functions, n_active)`
 * - `basis_overlap_matrix`: `(n_basis_functions, n_basis_functions)`
 */
LocalizedRepresentativeSelector build_localized_representative_selector(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orthonormal_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

}  // namespace xmvb::vb
