#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb {

/**
 * @brief Applies the accepted two-electron pair-space adjoint to RI factors.
 *
 * The returned matrix is the derivative with respect to `active_pair_factors`.
 * No packed pair-pair adjoint is constructed. All ordered same-spin pairs must
 * use the certified regular representation; inverse-free polynomial pairs are
 * intentionally handled by the separate packed path.
 */
Eigen::MatrixXd apply_regular_ri_pair_space_adjoint(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& selected_state_energies,
    int n_active_orbitals,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& active_pair_factors);

}  // namespace xmvb::vb
