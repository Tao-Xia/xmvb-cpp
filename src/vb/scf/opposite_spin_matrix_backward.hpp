#pragma once

#include <vector>

#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb {

/**
 * @brief Matrix-form opposite-spin backward contribution on unique spin pairs.
 *
 * The returned buffers contain only the opposite-spin part of the active-space
 * adjoint. Callers are expected to add these buffers to the same-spin and
 * one-electron contributions accumulated elsewhere.
 */
struct OppositeSpinMatrixBackwardContribution {
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> packed_active_two_electron_gradient;
};

/**
 * @brief Builds the matrix-form opposite-spin backward contribution.
 *
 * This routine consumes the selected-state coefficient matrices `C^(n)` and
 * the packed-pair projections stored inside the unique same-spin cache. The
 * per-packed-pair channel matrices are rebuilt on demand in small cached tiles
 * instead of being materialized as one global family over all packed pairs.
 * The algebra is unchanged: the routine reproduces the opposite-spin
 * contribution to:
 *
 * - `packed_active_two_electron_gradient`,
 * - `active_orbital_overlap_gradient`.
 */
OppositeSpinMatrixBackwardContribution build_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals);

/**
 * @brief Builds the directional opposite-spin backward contribution.
 *
 * This is the exact first-order response of the current matrix-form
 * opposite-spin backward when the selected-state coefficient matrices vary as
 *
 * `C^(n) -> C^(n) + \delta C^(n)`
 *
 * while the accepted-point same-spin cache payloads stay fixed.
 */
OppositeSpinMatrixBackwardContribution
build_directional_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    int n_active_orbitals);

/**
 * @brief Builds the exact opposite-spin local-response contribution.
 *
 * This is the `\delta J_{\mathrm{opp}}^T \lambda` term at fixed selected-state
 * coefficients. The accepted-point opposite-spin packed-pair channels are
 * reused, while the active-space direction contributes through
 *
 * - `\delta U` from the first-order cofactor channel,
 * - `\delta x` from the inverse-overlap channel,
 * - `\delta (G x)` from the opposite-spin kernel and inverse-response.
 *
 * The implementation keeps the current block-contracted packed-pair sweep and
 * only materializes the direction-dependent per-pair payloads needed for one
 * HVP application.
 */
OppositeSpinMatrixBackwardContribution
build_local_opposite_spin_matrix_backward_contribution(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals);

}  // namespace xmvb::vb
