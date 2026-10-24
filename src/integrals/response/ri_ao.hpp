#pragma once

#include <span>
#include <vector>
#include <Eigen/Core>

#include "integrals/response/ri.hpp"

namespace xmvb::integrals {

/** @brief Narrow AO responses needed by the orbital Hessian pullback. */
struct RiAoProjectedResponse {
    Eigen::MatrixXd inactive;             ///< 2 G(2 D + K/2) A0, AO by all inactive.
    Eigen::MatrixXd active;               ///< G(D) C, AO by active.
    Eigen::MatrixXd factor_direction{};   ///< Optional delta B, auxiliary by active pair.
    Eigen::MatrixXd fixed_two_electron{}; ///< Optional fixed-adjoint image, AO by active.
};

/** @brief Borrowed fixed-2e request in the accepted active coefficient frame.
 * `weight` maps the active direction to the AO-response density weight:
 * Wbar = direction * weight. `adjoint` is the accepted packed 2e cotangent.
 */
struct RiActiveHvpInput {
    const Eigen::MatrixXd& weight;
    const std::vector<double>& adjoint;
};

/** @brief One directional request sharing the accepted RI state and inactive map.
 * With a fixed HVP, active_direction is required and weighted_active is its
 * product with the shared fixed weight, in the accepted active frame.
 */
struct RiAoResponseDirection {
    Eigen::Ref<const Eigen::MatrixXd> horizontal_direction;
    Eigen::Ref<const Eigen::MatrixXd> weighted_active;
    const Eigen::MatrixXd* active_direction = nullptr;
};

/** @brief Shared auxiliary slabs and narrow tile projections.
 * AO projections and fixed jets have disjoint output writers. Accepted and
 * directional factor slabs are shared, with no per-thread integral copies.
 */
struct RiAoResponseWorkspace {
    workspace::Reservation residency;
    Eigen::MatrixXd direction, accepted;
    Eigen::MatrixXd direction_projection, accepted_projection, charges;
    Eigen::VectorXd coulomb;
};

/** @brief Borrowed accepted inactive transform, with exact column selection. */
struct RiInactiveFactorView {
    const Eigen::MatrixXd* factors = nullptr;
    std::span<const int> orbitals;
};

/** @brief G(C W C^T) from accepted L_A C and active-pair factors.
 * W must be symmetric. No AO eigendecomposition or repeated L_A C transform
 * is needed when the accepted factors are resident; otherwise the same
 * operator consumes bounded auxiliary tiles.
 */
Eigen::MatrixXd apply_ri_active_density_response(const RiActiveTwoElectronResponseCache& active,
                                                 const Eigen::Ref<const Eigen::MatrixXd>& weights);

/** @brief Compute a block of AO-by-orbital projections consumed by the HVP.
 * `moving_inactive_duals` and every horizontal direction contain matching
 * shared moving columns; `full_inactive_duals` contains all accepted duals A0.
 * When `inactive.factors` is present it stores L_A A0 as rows
 * `orbital*n_auxiliary+auxiliary`, and `inactive.orbitals` maps each moving
 * column to its A0 column. Without cached factors, only the moving duals
 * are transformed; a full L_A A0 transform is never required. The returned
 * inactive projection includes the factor two required by the symmetric
 * inactive-density pullback. No AO-square response is materialized.
 * With `fixed_hvp`, each weighted_active must equal its active_direction
 * times fixed_hvp->weight. Optional images use the accepted active frame.
 */
std::vector<RiAoProjectedResponse>
apply_ri_projected_ao_response_batch(const RiActiveTwoElectronResponseCache& active,
                                     const Eigen::Ref<const Eigen::MatrixXd>& moving_inactive_duals,
                                     std::span<const RiAoResponseDirection> directions,
                                     const Eigen::Ref<const Eigen::MatrixXd>& full_inactive_duals,
                                     RiAoResponseWorkspace& workspace,
                                     std::size_t workspace_bytes = workspace::limit(),
                                     RiInactiveFactorView inactive = {},
                                     const RiActiveHvpInput* fixed_hvp = nullptr);

} // namespace xmvb::integrals
