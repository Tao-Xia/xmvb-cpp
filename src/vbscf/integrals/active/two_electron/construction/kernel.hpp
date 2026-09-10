#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb {

/**
 * @brief Non-owning view of one active-space two-electron representation.
 *
 * Determinant kernels need read-only access to either the packed
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
 * @brief Wraps packed `GGO` storage in a non-owning view.
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
 * @brief Returns the packed `GGO` storage size for `n_active_orbitals`.
 */
std::size_t packed_active_two_electron_integral_count(int n_active_orbitals);

/**
 * @brief Inverts `n_active_pairs = n_orbitals (n_orbitals + 1) / 2`.
 */
int infer_active_orbital_count_from_packed_pair_count(int n_packed_active_pairs);

/**
 * @brief Inverts the packed `GGO` storage size back to `n_active_orbitals`.
 */
int infer_active_orbital_count_from_packed_integral_count(std::size_t packed_integral_count);

/**
 * @brief Evaluates the active-space pair kernel entry `G_{P,Q}`.
 *
 * The packed pair indices `P` and `Q` use the same
 * `TwoElectronIndexer::packed_pair_index(...)` storage convention.
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
 * pairs. Packed-exact callers reuse the `GGO` storage directly, while
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
 * packed tensor layout.
 */
std::vector<double> reconstruct_packed_active_two_electron_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals);

}  // namespace xmvb::vb
