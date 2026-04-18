#pragma once

#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/opposite_spin_matrix_backward.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Builds the fixed-kernel biorthogonal opposite-spin contraction contribution.
 *
 * This is the exact biorthogonal counterpart of
 * `build_opposite_spin_matrix_backward_contribution(...)`. The routine keeps
 * the current packed-pair channel factorization and replaces the nonorthogonal
 * coefficient contractions
 *
 * `C B C^T`, `C^T A C`
 *
 * with the ordered biorthogonal contractions
 *
 * `L B R^T`, `L^T A R`.
 *
 * The returned contribution contains only the opposite-spin channels obtained
 * at fixed accepted-point packed-pair kernels:
 *
 * 1. `packed_active_two_electron_gradient`,
 * 2. `active_orbital_overlap_gradient` driven by opposite-spin inverse-overlap projections.
 *
 * Important: this routine validates only the exact biorthogonal `L/R`
 * contraction on top of the existing packed-pair channel machinery. The full
 * exact-selected active-space gradient still needs the missing chain response
 * of `h_bi` with respect to the original active-space tensors.
 */
xmvb::vb::OppositeSpinMatrixBackwardContribution
build_biorthogonal_opposite_spin_matrix_backward_contribution(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const BiorthogonalSelectedStateMatrices& selected_state_matrices,
    const xmvb::vb::ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_active_orbitals);

}  // namespace xmvb::vb::biorthogonal_vbscf
