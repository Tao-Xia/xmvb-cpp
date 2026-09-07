#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "vb/orbital/ao_integral_input.hpp"

namespace xmvb::vb {

/**
 * @brief Molecule-static CSR buffers for the exact AO-H1E linear operator.
 *
 * The exact AO effective one-electron map is linear in the full AO inactive
 * density storage.  This helper materializes that map once as a CSR operator
 * on `vec(P11)` / `vec(G11)` so repeated SCF and exact_ctx HVP evaluations can
 * use row-wise sparse contractions instead of rescattering every ERI into six
 * AO matrix entries.
 */
struct AoEffectiveOneElectronGraphBuffers {
  std::vector<int> row_offsets;
  std::vector<int> source_indices;
  std::vector<double> signed_weights;
  std::vector<int> transpose_source_offsets;
  std::vector<int> transpose_row_indices;
  std::vector<double> transpose_signed_weights;
};

/**
 * @brief Combined forward and transpose outputs of the AO-H1E graph operator.
 *
 * `forward_output` is the unsymmetrized `\delta G11`, while
 * `transpose_output` is the unsymmetrized pullback with respect to `P11`.
 */
struct AoEffectiveOneElectronGraphFusedResult {
  std::vector<double> forward_output;
  std::vector<double> transpose_output;
};

/**
 * @brief Builds the molecule-static CSR operator for the exact AO-H1E map.
 *
 * The per-edge weights already include the AO-integral symmetry multiplier and
 * the Coulomb/exchange prefactor (`+4` or `-1`), so later applications only
 * need sparse matrix-vector multiplies against full AO matrix storage.
 */
AoEffectiveOneElectronGraphBuffers build_ao_effective_one_electron_graph(
    const std::vector<int>& ao_effective_one_electron_linear_indices,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<std::uint8_t>& ao_two_electron_integral_symmetry_shifts,
    int n_basis_functions);

/**
 * @brief Returns whether `ao_integral_input` carries a valid AO-H1E graph.
 */
bool ao_effective_one_electron_graph_available(
    const AoIntegralInput& ao_integral_input) noexcept;

/**
 * @brief Applies the AO-H1E CSR graph in the forward direction.
 *
 * `source_matrix_storage` is the full column-major AO matrix storage for
 * `vec(P11)` or any other dense AO-side source vector.
 */
std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input);

/**
 * @brief Applies the AO-H1E CSR graph in the forward direction with OpenMP row parallelism.
 *
 * `n_threads` is the number of threads used for the row-partitioned sparse
 * contraction. Each row writes to a disjoint destination entry, so the forward
 * graph path does not need thread-local AO matrix buffers.
 */
std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads);

/**
 * @brief Applies the transpose of the AO-H1E CSR graph.
 *
 * `row_adjoint_storage` is the full column-major AO matrix storage for the
 * adjoint living on `vec(G11)`.
 */
std::vector<double> apply_ao_effective_one_electron_graph_transpose(
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input);

/**
 * @brief Applies the transpose of the AO-H1E CSR graph with OpenMP parallelism.
 *
 * When the source-owned transpose graph is available this partitions the
 * output sources directly across threads; otherwise it falls back to a bounded
 * striped reduction over the row-owned graph instead of allocating one full AO
 * matrix per worker.
 */
std::vector<double> apply_ao_effective_one_electron_graph_transpose(
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads);

/**
 * @brief Applies the AO-H1E CSR graph and its transpose for exact_ctx HVP.
 *
 * When the source-owned transpose companion graph is present, the
 * implementation uses two sparse passes over the molecule-static operator:
 * one row-owned forward apply and one source-owned transpose apply. This
 * avoids thread-local AO-matrix copies while keeping the hot multithreaded
 * path lock-free. If the transpose companion graph is unavailable, the code
 * falls back to a bounded-memory single-row-sweep reduction.
 */
AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input);

/**
 * @brief Applies the fused AO-H1E CSR graph with OpenMP row parallelism.
 *
 * The multithreaded exact_ctx hot path prefers the lock-free companion
 * transpose graph when available. Otherwise it falls back to a bounded striped
 * accumulator on the shared transpose destination instead of allocating one
 * full AO matrix per worker.
 */
AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads);

/**
 * @brief Applies the AO-H1E CSR graph into caller-provided output buffers.
 *
 * This variant exists for hot exact_ctx HVP paths that repeatedly apply the
 * same molecule-static graph and want to reuse large AO-sized work buffers
 * across matvec calls instead of reallocating them every time.
 */
void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output);

/**
 * @brief Applies the fused AO-H1E CSR graph with caller-provided thread count.
 */
void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output);

/**
 * @brief Applies the fixed AO-H1E graph to several directions in one sweep.
 *
 * Columns are independent AO-matrix vectorizations.  The graph topology and
 * weights are read once per row/source while the short direction dimension is
 * accumulated contiguously.
 */
void apply_fused_ao_effective_one_electron_graph_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& source_matrix_columns,
    const Eigen::Ref<const Eigen::MatrixXd>& row_adjoint_columns,
    const AoIntegralInput& ao_integral_input,
    int n_threads,
    Eigen::MatrixXd* forward_output_columns,
    Eigen::MatrixXd* transpose_output_columns);

}  // namespace xmvb::vb
