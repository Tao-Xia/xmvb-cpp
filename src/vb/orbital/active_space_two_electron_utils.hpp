#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"

namespace xmvb::vb {

using ExactCtxDenseMatrix = Eigen::MatrixXd;
using ExactCtxPairMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Non-owning view of one active-space two-electron representation.
 *
 * Determinant kernels need read-only access to either the legacy packed
 * `GGO` tensor or the RI factor matrix `L_{A,P}`. This view lets those
 * callers share one implementation path without copying the forward buffers.
 */
struct ActiveSpaceTwoElectronView {
  ActiveSpaceTwoElectronRepresentation representation =
      ActiveSpaceTwoElectronRepresentation::PackedExact;
  const std::vector<double>* packed_active_two_electron_integrals = nullptr;
  int n_auxiliary_functions = 0;
  const Eigen::MatrixXd* ri_active_pair_factors = nullptr;
};

/**
 * @brief Accepted-point cache for exact fixed-adjoint active-space 2e HVP work.
 *
 * The exact matrix-free second-order path repeatedly applies the same accepted
 * packed 2e adjoint to different active-orbital tangents. The buffers stored
 * here depend only on the accepted point and can therefore be built once in
 * the operator constructor instead of being rebuilt for every `H v`.
 */
struct ExactPackedActiveTwoElectronAdjointCache {
  int n_basis_functions = 0;
  int n_active_orbitals = 0;
  std::vector<int> ao_pair_first_indices;
  std::vector<int> ao_pair_second_indices;
  std::vector<int> active_pair_first_indices;
  std::vector<int> active_pair_second_indices;
  ExactCtxPairMatrix accepted_pair_coefficients;
  ExactCtxPairMatrix accepted_base_pair_products;
  ExactCtxPairMatrix active_pair_gradient_matrix;
  std::vector<double> accepted_active_pair_gradient_backprop_rows_buffer;
  ExactCtxDenseMatrix accepted_dense_active_coefficients;
  ExactCtxPairMatrix accepted_base_pair_gradients;
  std::vector<double> accepted_base_pair_gradient_matrices_buffer;
};

/**
 * @brief Reusable work buffers for accepted-point exact 2e HVP applications.
 *
 * These buffers only depend on AO-pair and active-pair dimensions at one
 * accepted point. Reusing them across repeated matrix-free `H v` calls avoids
 * reallocation of the largest exact-2e intermediates on every Krylov matvec.
 */
struct ExactPackedActiveTwoElectronApplyWorkspace {
  ExactCtxDenseMatrix dense_active_direction;
  ExactCtxDenseMatrix dense_active_gradient_direction;
  ExactCtxPairMatrix mixed_pair_coefficients;
  ExactCtxPairMatrix transformed_pair_coefficients;
  ExactCtxPairMatrix pair_gradients;
};

/**
 * @brief Reusable buffers for exact packed `\delta GGO` directional builds.
 *
 * The outer-response exact_ctx path repeatedly forms `\delta GGO` at one
 * accepted point.  Reusing the AO-pair and active-pair work buffers here
 * avoids several large allocations on every `H v` application.
 */
struct ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace {
  ExactCtxDenseMatrix dense_active_direction;
  ExactCtxPairMatrix pair_coefficients;
  ExactCtxPairMatrix base_pair_products;
  ExactCtxPairMatrix directional_pair_coefficients;
  ExactCtxPairMatrix directional_pair_products;
  ExactCtxDenseMatrix delta_active_pair_matrix;
};

/**
 * @brief Wraps legacy packed `GGO` storage in a non-owning view.
 */
ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const std::vector<double>& packed_active_two_electron_integrals);

/**
 * @brief Wraps a forward active-space two-electron result in a non-owning view.
 */
ActiveSpaceTwoElectronView make_active_space_two_electron_view(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result);

/**
 * @brief Returns the number of packed active-pair indices `P = (p, q)` with `p >= q`.
 */
int packed_active_pair_count(int n_active_orbitals);

/**
 * @brief Returns the legacy packed `GGO` storage size for `n_active_orbitals`.
 */
std::size_t packed_active_two_electron_integral_count(int n_active_orbitals);

/**
 * @brief Inverts `n_active_pairs = n_orbitals (n_orbitals + 1) / 2`.
 */
int infer_active_orbital_count_from_packed_pair_count(int n_packed_active_pairs);

/**
 * @brief Inverts the legacy packed `GGO` storage size back to `n_active_orbitals`.
 */
int infer_active_orbital_count_from_packed_integral_count(std::size_t packed_integral_count);

/**
 * @brief Evaluates the active-space pair kernel entry `G_{P,Q}`.
 *
 * The packed pair indices `P` and `Q` use the same
 * `TwoElectronIndexer::packed_pair_index(...)` convention as the legacy VB
 * packed storage.
 */
double lookup_active_space_two_electron_kernel_value(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int row_packed_pair_index,
    int column_packed_pair_index,
    int n_active_orbitals);

/**
 * @brief Applies the active-space pair kernel to a sparse packed-pair vector.
 *
 * `packed_pair_indices[k]` and `packed_pair_values[k]` define a sparse vector
 * `c_P`. The returned dense vector stores `(G c)_P` over all packed active
 * pairs. Packed-exact callers reuse the legacy `GGO` storage directly, while
 * RI callers evaluate the same contraction as `L^T (L c)`.
 */
std::vector<double> apply_active_space_two_electron_kernel_to_sparse_projection(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values);

/**
 * @brief Applies the active-space pair kernel to one sparse vector on a subset.
 *
 * This is the streamed counterpart of
 * `apply_active_space_two_electron_kernel_to_sparse_projection(...)`. Instead
 * of materializing all packed active-pair rows, the returned dense vector
 * contains only the requested `target_packed_pair_indices` in the same order.
 */
Eigen::VectorXd apply_active_space_two_electron_kernel_to_sparse_projection_subset(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<int>& target_packed_pair_indices);

/**
 * @brief Materializes packed `GGO` storage from either exact or RI inputs.
 *
 * Production forward paths should prefer the direct pair-kernel helpers above.
 * This routine is intended for diagnostics and outputs that still expect the
 * legacy packed tensor layout.
 */
std::vector<double> reconstruct_packed_active_two_electron_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals);

/**
 * @brief Computes the exact packed `GGO` directional derivative from AO-driven data.
 *
 * `dense_active_coefficients` and `dense_active_direction` are dense
 * AO-by-active matrices in the repository-standard column-major Eigen layout.
 * The implementation may still build temporary compatibility buffers
 * internally, but callers stay on `Eigen::MatrixXd`.
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

/**
 * @brief Backpropagates a changing packed active-2e adjoint at a fixed point.
 *
 * The accepted cache already stores `K B(C)`.  Therefore an outer-response
 * adjoint can be pulled back as `(K B(C)) G` without traversing the AO-pair
 * integral graph again.  This is the transpose of the cached forward map and
 * remains fully matrix-free with respect to the orbital Hessian.
 */
Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache);

/**
 * @brief Precomputes accepted-point exact 2e HVP invariants.
 *
 * `accepted_dense_active_coefficients` is the accepted AO-by-active dense
 * active-orbital coefficient matrix. The returned cache owns the accepted
 * point buffers needed by the exact fixed-adjoint 2e directional kernel.
 */
ExactPackedActiveTwoElectronAdjointCache
build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result =
        nullptr);

/**
 * @brief Applies the exact fixed-adjoint 2e Hessian using one accepted-point cache.
 *
 * `dense_active_direction` uses the same AO-by-active dense matrix convention
 * as `ExactPackedActiveTwoElectronAdjointCache::accepted_dense_active_coefficients`.
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
 * intermediates in Eigen matrices. Any unavoidable legacy row-buffer flattening
 * is isolated inside the AO-kernel compatibility boundary.
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
 * When `accepted_active_space_two_electron_result` is provided, its cached
 * dense forward products are reused to avoid rebuilding the accepted-point
 * `G B(C)` table on every HVP application.
 */
Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result =
        nullptr);

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
