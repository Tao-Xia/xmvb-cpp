#pragma once

#include <vector>

#include "vb/exact_separator/bundle.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Decoded antisymmetrized support-orbital pair.
 *
 * The pair is stored in canonical order `first_orbital < second_orbital`.
 */
struct DecodedAntisymSupportPair {
  int first_orbital = -1;
  int second_orbital = -1;
};

/**
 * @brief Checks whether two one-spin bundle layouts represent the same basis.
 *
 * This is the structural compatibility predicate needed before accumulating or
 * comparing dense bundle coefficients across different subtree builders.
 */
bool have_same_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right);

/**
 * @brief Decodes one flattened support-pair index back to `(row, col)`.
 *
 * The decoded pair follows the same column-major convention as
 * `flatten_support_pair_index(...)`.
 */
void decode_support_pair_index(
    int support_pair_index,
    int support_size,
    int* row_orbital,
    int* col_orbital);

/**
 * @brief Decodes one flattened antisymmetrized support-pair index.
 */
DecodedAntisymSupportPair decode_antisym_support_pair_index(
    int pair_index,
    int support_size);

/**
 * @brief Collects all nonzero flattened degree-1 entries of one bundle.
 *
 * This is the sparse support needed by later mixed alpha/beta bundle products
 * without forcing every caller to rescan the dense degree-1 array.
 */
std::vector<int> collect_nonzero_boundary_spin_degree1_indices(
    const BoundarySpinBundle& bundle);

/**
 * @brief Contracts the closed overlap channel from one-spin bundle degree-0 data.
 */
double contract_boundary_spin_bundle_overlap(
    const BoundarySpinBundle& bundle);

/**
 * @brief Contracts the one-electron channel from one-spin bundle degree-1 data.
 *
 * The support matrix uses the repository-wide column-major convention
 * `storage[column * support_size + row]`.
 */
double contract_boundary_spin_bundle_one_electron(
    const BoundarySpinBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    int support_size);

/**
 * @brief Contracts the same-spin two-electron channel from one-spin degree-2 data.
 */
double contract_boundary_spin_bundle_same_spin(
    const BoundarySpinBundle& bundle,
    const std::vector<double>& packed_active_two_electron_integrals);

/**
 * @brief Contracts the opposite-spin channel from one mixed degree-1 bundle.
 *
 * `alpha_total_sector_count` and `beta_total_sector_count` are required to
 * decode the flattened degree-1 bundle coordinates back to support-pair plus
 * boundary-sector indices.
 */
double contract_boundary_mixed_bundle_opposite_spin(
    const BoundaryMixedBundle& mixed,
    int alpha_total_sector_count,
    int beta_total_sector_count,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals);

}  // namespace xmvb::vb::exact_separator
