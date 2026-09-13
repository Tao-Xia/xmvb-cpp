#pragma once

#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/**
 * @brief Builds a symmetric AO-pair row graph for row-wise ERI contractions.
 *
 * Each AO integral `(left_pair, right_pair, value)` contributes one row entry
 * `left_pair -> right_pair` and, when `left_pair != right_pair`, one mirrored
 * row entry `right_pair -> left_pair`. The graph reuses the original integral
 * value buffer through `eri_indices` instead of duplicating values.
 *
 * @param eri_indices Flattened AO ERI indices, four entries per integral.
 * @param n_bf Number of AO basis functions.
 * @return Deterministic symmetric AO-pair graph.
 */
AoPairGraph build_ao_pair_graph(
    const std::vector<int>& eri_indices,
    int n_bf);

}  // namespace xmvb::vb
