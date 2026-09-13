#pragma once

#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/**
 * @brief Builds a symmetric AO-pair row graph for row-wise ERI contractions.
 *
 * Each AO integral `(left_pair, right_pair, value)` contributes one row entry
 * `left_pair -> right_pair` and, when `left_pair != right_pair`, one mirrored
 * row entry `right_pair -> left_pair`. Values are stored in CSR edge order to
 * avoid an indirect lookup in every matrix-free HVP.
 *
 * @param eri_indices Flattened AO ERI indices, four entries per integral.
 * @param eri_values AO ERI values in the same integral order.
 * @param n_bf Number of AO basis functions.
 * @return Deterministic symmetric AO-pair graph.
 */
AoPairGraph build_ao_pair_graph(
    const std::vector<int>& eri_indices,
    const std::vector<double>& eri_values,
    int n_bf);

}  // namespace xmvb::vb
