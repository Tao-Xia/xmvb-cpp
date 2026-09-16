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
 * Input vectors are consumed so construction never retains a second AO ERI
 * representation.
 *
 * @param left_pairs Packed left AO-pair indices in unique-integral order.
 * @param right_pairs Packed right AO-pair indices in the same order.
 * @param values AO ERI values in the same order.
 * @param n_bf Number of AO basis functions.
 * @return Deterministic symmetric AO-pair graph.
 */
AoPairGraph build_ao_pair_graph(
    std::vector<int> left_pairs,
    std::vector<int> right_pairs,
    std::vector<double> values,
    int n_bf);

}  // namespace xmvb::vb
