#pragma once

#include <vector>

#include <Eigen/Core>

#include "integrals/response/types.hpp"
#include "integrals/ao/input.hpp"

namespace xmvb::integrals {

/**
 * @brief Computes exact packed `\delta GGO` from one accepted-point exact 2e cache.
 *
 * The accepted AO-by-active coefficients and their accepted AO-pair images are
 * already materialized inside `accepted_cache`. Reusing them here avoids
 * rebuilding the accepted pair map and re-copying the accepted dense-active
 * buffers on every directional `\delta GGO` apply. Set
 * `compute_fixed_adjoint=false` when a later combined orbital jet will
 * compute the fixed-adjoint image; the packed integral direction is unchanged.
 */
void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals,
    bool compute_fixed_adjoint = true);

/**
 * @brief Computes several exact `delta GGO` directions with bounded workspace.
 *
 * The result has one packed active-2e derivative per column. Integral-only
 * consumers reuse accepted images without computing unused directional images.
 * Resident orbital-adjoint images use memory-budgeted column blocks; streamed
 * integral-only consumers share each accepted row tile across all directions.
 * Returned images are reused by the orbital pullback, not transformed again.
 */
Eigen::MatrixXd compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions,
    const AoIntegralInput& ao_integral_input,
    std::vector<ExactCtxPairMatrix>* directional_pair_products = nullptr,
    std::vector<Eigen::MatrixXd>* fixed_adjoint_directions = nullptr);

} // namespace xmvb::integrals
