#pragma once

#include <cstddef>
#include <cstdint>

namespace xmvb::core {

/**
 * @brief Utility for the packed index convention used by two-electron tensors.
 *
 * Both arguments and results are zero based. Pair exchange is an indexing
 * symmetry, not antisymmetrization. Two nested unordered-pair maps address
 * the corresponding symmetric two-electron tensor.
 */
class PairIndex {
  public:
    /**
   * @brief Returns the zero-based packed index of an unordered orbital pair.
   *
   * @param orbital_index_a Zero-based orbital index.
   * @param orbital_index_b Zero-based orbital index.
   * @return std::size_t Zero-based packed pair index.
   */
    static std::size_t packed_pair_index(std::int64_t orbital_index_a,
                                         std::int64_t orbital_index_b);

    /**
   * @brief Returns the zero-based packed index of an unordered pair-of-pairs.
   *
   * @param packed_pair_index_a Zero-based packed pair index.
   * @param packed_pair_index_b Zero-based packed pair index.
   * @return std::size_t Zero-based packed pair-of-pairs index.
   */
    static std::size_t packed_pair_of_pairs_index(std::int64_t packed_pair_index_a,
                                                  std::int64_t packed_pair_index_b);

    /**
   * @brief Returns the final zero-based two-electron storage index.
   *
   * The method first packs each orbital pair into a single index, then packs
   * the two pair indices again:
   * - first packed pair: `(orbital_index_p, orbital_index_q)`
   * - second packed pair: `(orbital_index_r, orbital_index_s)`
   *
   * @param orbital_index_p Zero-based index of the first orbital in pair 1.
   * @param orbital_index_q Zero-based index of the second orbital in pair 1.
   * @param orbital_index_r Zero-based index of the first orbital in pair 2.
   * @param orbital_index_s Zero-based index of the second orbital in pair 2.
   * @return std::size_t Zero-based index into the packed two-electron array.
   */
    static std::size_t two_electron_storage_index(std::int64_t orbital_index_p,
                                                  std::int64_t orbital_index_q,
                                                  std::int64_t orbital_index_r,
                                                  std::int64_t orbital_index_s);
};

} // namespace xmvb::core
