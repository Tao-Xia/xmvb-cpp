#pragma once

#include <vector>

namespace xmvb::vb {

struct AoTwoElectronPairGraph {
  std::vector<int> row_offsets;
  std::vector<int> column_pair_indices;
  std::vector<int> integral_indices;
};

/**
 * @brief Precomputes packed AO pair indices `(ij_pair, kl_pair)` for each AO ERI.
 *
 * The returned vector stores two packed pair indices per AO two-electron
 * integral in the same order as `ao_two_electron_integral_indices`.
 */
std::vector<int> build_ao_two_electron_pair_indices(
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions);

/**
 * @brief Builds a symmetric AO-pair row graph for row-wise ERI contractions.
 *
 * Each AO integral `(left_pair, right_pair, value)` contributes one row entry
 * `left_pair -> right_pair` and, when `left_pair != right_pair`, one mirrored
 * row entry `right_pair -> left_pair`. The graph reuses the original integral
 * value buffer through `integral_indices` instead of duplicating values.
 */
AoTwoElectronPairGraph build_ao_two_electron_pair_graph(
    const std::vector<int>& ao_two_electron_pair_indices,
    int n_basis_functions);

}  // namespace xmvb::vb
