#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "integrals/active/two_electron_result.hpp"

namespace xmvb::integrals {

/**
 * @brief Borrowed packed four-index kernel, independent of the AO backend.
 * Both exact and RI integral preparation populate this same physical contract.
 */
struct ActiveSpaceTwoElectronView {
    const std::vector<double>* packed_active_two_electron_integrals = nullptr;
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
 * `xmvb::core::PairIndex::packed_pair_index(...)` storage convention.
 */
double
lookup_active_space_two_electron_kernel_value(const ActiveSpaceTwoElectronView& two_electron_view,
                                              int row_packed_pair_index,
                                              int column_packed_pair_index,
                                              int n_active_orbitals);

/**
 * @brief Applies the active-space pair kernel to a sparse packed-pair vector.
 *
 * `packed_pair_indices[k]` and `packed_pair_values[k]` define a sparse vector
 * `c_P`. The returned dense vector stores `(G c)_P` over all packed active
 * pairs. Every caller reuses the prepared packed four-index kernel.
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
 * @brief Applies the active pair kernel to a dense block without materializing it.
 *
 * `vectors` has one packed active-pair channel per row. The prepared packed
 * kernel is consumed in bounded row tiles for both exact and RI AO backends.
 */
Eigen::MatrixXd apply_active_space_two_electron_kernel_block(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors);

/**
 * @brief Copy the prepared packed kernel for diagnostics and output.
 */
std::vector<double> reconstruct_packed_active_two_electron_integrals(
    const ActiveSpaceTwoElectronView& two_electron_view, int n_active_orbitals);

} // namespace xmvb::integrals
