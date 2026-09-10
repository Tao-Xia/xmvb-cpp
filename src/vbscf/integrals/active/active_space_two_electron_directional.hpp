#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"
#include "vbscf/integrals/active/active_space_two_electron_response_types.hpp"
#include "vbscf/integrals/ao/ao_integral_input.hpp"

namespace xmvb::vb {

/**
 * @brief Computes the exact packed `GGO` directional derivative from AO-driven data.
 *
 * `dense_active_coefficients` and `dense_active_direction` are dense
 * AO-by-active matrices in the repository-standard column-major Eigen layout.
 */
std::vector<double>
compute_exact_packed_active_two_electron_integral_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result =
        nullptr);

/**
 * @brief Computes exact packed `\delta GGO` into reusable caller-owned buffers.
 *
 * `workspace` keeps the transient AO-pair / active-pair tables alive across
 * repeated directional derivatives, while `delta_packed_active_two_electron_integrals`
 * receives the final packed lower-triangular `\delta GGO` tensor.
 */
void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result =
        nullptr);

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
