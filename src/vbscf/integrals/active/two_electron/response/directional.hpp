#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/**
 * @brief Computes exact packed `\delta GGO` from one accepted-point exact 2e cache.
 *
 * The accepted AO-by-active coefficients and their accepted AO-pair images are
 * already materialized inside `accepted_cache`. Reusing them here avoids
 * rebuilding the accepted pair map and re-copying the accepted dense-active
 * buffers on every directional `\delta GGO` apply.
 */
void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals);

/**
 * @brief Computes several cached exact `delta GGO` directions in one AO-pair sweep.
 *
 * The result has one packed active-2e derivative per column. Directional
 * AO-pair coefficient blocks are concatenated so the fixed AO-pair graph is
 * traversed once for the whole block.
 */
Eigen::MatrixXd
compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions,
    const AoIntegralInput& ao_integral_input,
    std::vector<ExactCtxPairMatrix>* directional_pair_products = nullptr);

}  // namespace xmvb::vb
