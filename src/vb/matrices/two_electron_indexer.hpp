#pragma once

namespace xmvb::vb {

/**
 * @brief Utility for the packed index convention used by legacy two-electron tensors.
 *
 * The legacy code stores antisymmetrized two-electron quantities through two
 * nested packed symmetric index maps:
 * 1. Convert an orbital pair `(p, q)` to a packed one-based index.
 * 2. Convert two packed pair indices to a final packed one-based index.
 *
 * This helper exposes the same convention with zero-based C++ arguments while
 * returning a zero-based linear offset for use with `std::vector<double>`.
 */
class TwoElectronIndexer {
public:
  /**
   * @brief Returns the zero-based packed index of an unordered orbital pair.
   *
   * @param orbital_index_a Zero-based orbital index.
   * @param orbital_index_b Zero-based orbital index.
   * @return int Zero-based packed pair index.
   */
  static int packed_pair_index(int orbital_index_a, int orbital_index_b);

  /**
   * @brief Returns the zero-based packed index of an unordered pair-of-pairs.
   *
   * @param packed_pair_index_a Zero-based packed pair index.
   * @param packed_pair_index_b Zero-based packed pair index.
   * @return int Zero-based packed pair-of-pairs index.
   */
  static int packed_pair_of_pairs_index(int packed_pair_index_a, int packed_pair_index_b);

  /**
   * @brief Returns the final zero-based legacy two-electron storage index.
   *
   * The legacy `Hamhd0` code first packs the orbital pair `(left, right)` into
   * a single index with `LAB`, then packs two such pair indices again. This
   * method mirrors that exact convention:
   * - first packed pair: `(orbital_index_p, orbital_index_q)`
   * - second packed pair: `(orbital_index_r, orbital_index_s)`
   *
   * @param orbital_index_p Zero-based index of the first orbital in pair 1.
   * @param orbital_index_q Zero-based index of the second orbital in pair 1.
   * @param orbital_index_r Zero-based index of the first orbital in pair 2.
   * @param orbital_index_s Zero-based index of the second orbital in pair 2.
   * @return int Zero-based index into the packed two-electron array.
   */
  static int two_electron_storage_index(
      int orbital_index_p,
      int orbital_index_q,
      int orbital_index_r,
      int orbital_index_s);
};

}  // namespace xmvb::vb
