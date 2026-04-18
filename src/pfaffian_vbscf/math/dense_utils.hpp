#pragma once

#include <vector>

#include "pfaffian_vbscf/data/pf_state.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Dense power and trace data for one spin-orbital kernel.
 */
struct KernelTraceData {
  std::vector<Matrix> kernel_powers;
  ScalarBuffer traces;
};

/**
 * @brief Returns the packed strict-upper-triangle size of an antisymmetric
 * matrix with the given dimension.
 *
 * @param dimension Matrix dimension.
 * @return int Number of packed independent entries.
 */
int packed_antisymmetric_size(int dimension);

/**
 * @brief Decodes a packed antisymmetric matrix into a dense spin-orbital form.
 *
 * @param packed_entries Strict-upper-triangle packed entries.
 * @param dimension Dense matrix dimension.
 * @return Matrix Dense antisymmetric matrix.
 */
Matrix decode_antisymmetric_matrix(
    const ScalarBuffer& packed_entries,
    int dimension);

/**
 * @brief Decodes the singlet-pair antisymmetric matrix stored in one Pfaffian
 * state.
 *
 * @param state Packed Pfaffian state.
 * @return Matrix Dense antisymmetric state matrix.
 */
Matrix decode_pf_state(const PfState& state);

/**
 * @brief Decodes only the close-shell alpha/beta pairing block stored in one
 * Pfaffian state.
 *
 * This avoids materializing the full `2m x 2m` antisymmetric pairing matrix
 * when the close-shell forward path only needs the spatial `m x m` pairing
 * block.
 *
 * @param state Packed Pfaffian state.
 * @return Matrix Dense `alpha/beta` pairing block.
 */
Matrix decode_pf_alpha_beta_block(const PfState& state);

/**
 * @brief Builds the block-diagonal spin-orbital embedding of a spatial matrix.
 *
 * @param spatial_matrix Spatial-orbital matrix of size `m x m`.
 * @return Matrix Spin-orbital matrix of size `2m x 2m`.
 */
Matrix build_spin_block_diagonal_metric(const ConstMatrixRef& spatial_matrix);

/**
 * @brief Builds the generating kernel `K = L Sigma R Sigma^T`.
 *
 * @param left Left spin-orbital state matrix.
 * @param sigma Spin-orbital overlap metric.
 * @param right Right spin-orbital state matrix.
 * @return Matrix Dense kernel matrix.
 */
Matrix build_kernel(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right);

/**
 * @brief Computes traces `[Tr(K), ..., Tr(K^order)]` and stores
 * `[I, K, ..., K^(order-1)]`.
 *
 * @param kernel Dense kernel matrix.
 * @param order Largest trace power to compute.
 * @return KernelTraceData Power cache and trace series.
 */
KernelTraceData compute_kernel_trace_data(
    const ConstMatrixRef& kernel,
    int order);

/**
 * @brief Builds `K_bar = sum_p w_p p (K^(p-1))^T`.
 *
 * @param kernel_powers Cached kernel powers `[I, K, ..., K^(n-1)]`.
 * @param trace_weights Trace adjoints `[w_1, ..., w_n]`.
 * @return Matrix Dense kernel adjoint.
 */
Matrix build_kernel_trace_adjoint(
    const std::vector<Matrix>& kernel_powers,
    const ScalarBuffer& trace_weights);

/**
 * @brief Applies the chain rule from `K_bar` to the overlap metric `Sigma`.
 *
 * @param left Left spin-orbital state matrix.
 * @param sigma Spin-orbital overlap metric.
 * @param right Right spin-orbital state matrix.
 * @param kernel_adjoint Dense adjoint of the kernel.
 * @return Matrix Dense adjoint of `Sigma`.
 */
Matrix backpropagate_sigma(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right,
    const ConstMatrixRef& kernel_adjoint);

/**
 * @brief Collapses the alpha/alpha and beta/beta blocks into one spatial matrix.
 *
 * @param spin_orbital_matrix Even-dimensional spin-orbital matrix.
 * @return Matrix Spatial matrix `M_aa + M_bb`.
 */
Matrix collapse_spin_diagonal_blocks(const ConstMatrixRef& spin_orbital_matrix);

}  // namespace xmvb::pfaffian_vbscf
