#pragma once

#include <vector>

#include <Eigen/Core>

#include "integrals/active/two_electron.hpp"
#include "integrals/response/types.hpp"
#include "integrals/ao/input.hpp"

namespace xmvb::integrals {

/** @brief Pull back packed ERI adjoints without constructing HVP-only images.
 * Borrows accepted coefficients and optional pair products. Missing pair
 * products are consumed through the same bounded tiles as response actions.
 */
Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_gradient,
    const AoIntegralInput& ao,
    const ActiveSpaceTwoElectronResult& accepted);

/**
 * @brief Projects L(W C^T + C W^T) onto accepted active orbitals.
 *
 * L(D)=2 J(D)-K(D), matching the physical AO builder. Uses the shared
 * accepted Coulomb/exchange images;
 * no AO integral sweep or complete AO response is generated. W is the
 * AO-by-active weighted orbital direction. Requires the admitted exchange
 * image and resident pair products. Storage is O(n_bf*n_active).
 */
Eigen::MatrixXd
project_exact_active_density_response(const ExactPackedActiveTwoElectronAdjointCache& cache,
                                      const Eigen::Ref<const Eigen::MatrixXd>& weighted_direction);

/**
 * @brief Pulls an active H1e adjoint back to the symmetric inactive density.
 *
 * Reuses accepted Coulomb and exchange integral images. The returned AO
 * covector is the symmetric representative of the unique-ERI transpose:
 * its sum with its transpose is identical to that operator's result. Only
 * symmetric density variations are admissible in orbital preparation.
 * Requires resident pair products and the bounded exchange image.
 */
std::vector<double>
backpropagate_exact_active_h1e_gradient(const std::vector<double>& active_gradient,
                                        const ExactPackedActiveTwoElectronAdjointCache& cache);

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
ExactPackedActiveTwoElectronAdjointCache build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::MatrixXd& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result);

/** @brief Combined fixed-adjoint and changing-adjoint exact orbital jet.
 *
 * The cached accepted adjoint, `packed_gradient_direction`, accepted active
 * coefficients, and `dense_active_direction` must all use the same active
 * orbital frame. An optional directional pair product is the forward
 * `K * delta T(C)` in that frame; supplying it avoids a second AO action.
 * The result is AO-by-active and does not include any frame pullback.
 */
void apply_exact_active_two_electron_jet(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const std::vector<double>& packed_gradient_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction,
    const ExactCtxPairMatrix* directional_pair_products = nullptr);

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
} // namespace xmvb::integrals
