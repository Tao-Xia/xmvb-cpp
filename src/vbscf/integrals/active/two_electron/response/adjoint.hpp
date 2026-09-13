#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/**
 * @brief Backpropagates a changing packed active-2e adjoint at a fixed point.
 *
 * When `K B(C)` is resident, an outer-response adjoint is pulled back without
 * traversing the AO-pair graph again. Otherwise the same exact contraction is
 * evaluated in bounded row tiles. Both paths remain matrix-free with respect
 * to the orbital Hessian.
 */
Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache);

/**
 * @brief Precomputes accepted-point exact 2e HVP invariants.
 *
 * `accepted_dense_active_coefficients` is the accepted AO-by-active dense
 * active-orbital coefficient matrix. The returned cache borrows it and any
 * resident forward pair products; both owners must outlive the cache.
 */
ExactPackedActiveTwoElectronAdjointCache
build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::MatrixXd& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result);

/**
 * @brief Applies the exact fixed-adjoint 2e Hessian using one accepted-point cache.
 *
 * `dense_active_direction` uses the same AO-by-active dense matrix convention
 * as `ExactPackedActiveTwoElectronAdjointCache::accepted_active_coefficients`.
 */
Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input);

/**
 * @brief Applies the exact fixed-adjoint 2e Hessian into reusable work buffers.
 *
 * This is the allocation-aware accepted-point hot path used by exact_ctx HVPs.
 * The exact-2e workspaces now keep their AO-by-active and AO-pair-by-active-pair
 * intermediates in Eigen matrices. Any unavoidable row-major flattening is
 * isolated inside the AO-kernel storage boundary.
 */
void apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction);

/**
 * @brief Applies the exact fixed-adjoint `GGO` Hessian to one dense active-orbital tangent.
 *
 * The accepted-point packed `GGO` adjoint is held fixed while the active
 * auxiliary orbitals vary along `dense_active_direction`. The returned matrix
 * stores the directional derivative of the dense-active backpropagated
 * gradient on AO rows and active-orbital columns.
 *
 * The accepted result supplies the forward pair products and must remain alive
 * for the duration of this call.
 */
Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::MatrixXd& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result);

/**
 * @brief Reuses pre-computed forward K*mixed to skip the 2e kernel.
 *
 * When both core-direct and outer-response are active, the forward pass
 * already computed `directional_pair_products = K * mixed`.  This overload
 * reuses that result via `pair_gradients = products * gradient_matrix`,
 * avoiding one full `apply_exact_ao_pair_kernel` call per HVP.
 */
void apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& directional_pair_products,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction);
}  // namespace xmvb::vb
