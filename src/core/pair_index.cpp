#include <cstdint>
#include "core/pair_index.hpp"
#include "core/index.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::core {

std::size_t PairIndex::packed_pair_index(std::int64_t orbital_index_a,
                                         std::int64_t orbital_index_b) {
    if (orbital_index_a < 0 || orbital_index_b < 0) {
        throw std::invalid_argument("orbital indices must be non-negative");
    }

    const auto high_index = std::max(orbital_index_a, orbital_index_b);
    const auto low_index = std::min(orbital_index_a, orbital_index_b);
    return checked_add(triangular_count(checked_index<std::size_t>(high_index)),
                       checked_index<std::size_t>(low_index));
}

std::size_t PairIndex::packed_pair_of_pairs_index(std::int64_t packed_pair_index_a,
                                                  std::int64_t packed_pair_index_b) {
    if (packed_pair_index_a < 0 || packed_pair_index_b < 0) {
        throw std::invalid_argument("packed pair indices must be non-negative");
    }

    const auto high_index = std::max(packed_pair_index_a, packed_pair_index_b);
    const auto low_index = std::min(packed_pair_index_a, packed_pair_index_b);
    return checked_add(triangular_count(checked_index<std::size_t>(high_index)),
                       checked_index<std::size_t>(low_index));
}

std::size_t PairIndex::two_electron_storage_index(std::int64_t orbital_index_p,
                                                  std::int64_t orbital_index_q,
                                                  std::int64_t orbital_index_r,
                                                  std::int64_t orbital_index_s) {
    const auto first_packed_pair_index = packed_pair_index(orbital_index_p, orbital_index_q);
    const auto second_packed_pair_index = packed_pair_index(orbital_index_r, orbital_index_s);
    return packed_pair_of_pairs_index(checked_index<std::int64_t>(first_packed_pair_index),
                                      checked_index<std::int64_t>(second_packed_pair_index));
}

} // namespace xmvb::core
