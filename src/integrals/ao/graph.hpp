#pragma once

#include "integrals/ao/input.hpp"

namespace xmvb::integrals {

/** Input layout contract; Unique leaves symmetry expansion to the consumer. */
enum class AoPairLayout { Automatic, Unique, Chunked };

/**
 * @brief Builds an exact AO-pair operator in the requested input layout.
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
 * @param layout Unique moves the validated input stream without CSR expansion.
 * @return Exact AO-pair operator with one retained integral representation.
 */
AoPairGraph build_ao_pair_graph(std::vector<PairAddress> left_pairs,
                                std::vector<PairAddress> right_pairs,
                                std::vector<double> values,
                                int n_bf,
                                AoPairLayout layout = AoPairLayout::Automatic);

/** Consume producer chunks without a complete aggregate or symmetry expansion. */
AoPairGraph build_ao_pair_graph(std::vector<AoIntegralChunk> chunks, int n_bf);

} // namespace xmvb::integrals
