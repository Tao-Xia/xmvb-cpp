#pragma once

#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Builds the fixed-kernel biorthogonal same-spin contraction contribution.
 *
 * This routine is the biorthogonal analogue of
 * `build_same_spin_matrix_backward_contribution(...)`, but it compresses the
 * determinant-pair adjoints with the exact biorthogonal coefficient families
 *
 * `L^(n)`, `Q^(n)`, `R^(n)`
 *
 * instead of the single nonorthogonal coefficient family `C^(n)`.
 *
 * The resulting contribution contains the one-electron and same-spin channels
 * obtained by contracting the exact biorthogonal determinant adjoints against
 * the current ordered unique-spin pair kernels.
 *
 * 1. `active_one_electron_gradient`,
 * 2. `packed_active_two_electron_gradient` from same-spin exchange wedges,
 * 3. `active_orbital_overlap_gradient` from the exact same-spin overlap adjoint.
 *
 * The caller must provide an enabled same-spin cache whose ordered unique-spin
 * pair entries already carry cached `same_spin_phi` payloads. This is the
 * standard accepted-point cache built by the current active-space gradient
 * forward path.
 *
 * Important: this is not yet the full exact-selected active-space gradient.
 * It validates the `L/Q/R` matrix contraction layer at fixed pair kernels, but
 * it does not include the additional chain response hidden inside the
 * biorthogonal Hamiltonian `h_bi`.
 */
xmvb::vb::SameSpinMatrixBackwardContribution
build_biorthogonal_same_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int n_active_orbitals);

/**
 * @brief Exact same-spin backward using the biorthogonal forward tile kernels.
 *
 * This variant keeps the explicit selected-space overlap adjoint on the
 * original nonorthogonal same-spin overlap cache, but pulls the partner
 * Hamiltonian channel from the exact biorthogonal forward tiles instead of the
 * legacy nonorthogonal same-spin cache.
 */
xmvb::vb::SameSpinMatrixBackwardContribution
build_biorthogonal_same_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalForwardSpinPairTileProvider& alpha_biorthogonal_provider,
    const BiorthogonalForwardSpinPairTileProvider& beta_biorthogonal_provider,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    int n_active_orbitals);

}  // namespace xmvb::vb::biorthogonal_vbscf
