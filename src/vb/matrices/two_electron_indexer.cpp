#include "vb/matrices/two_electron_indexer.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::vb {

int TwoElectronIndexer::packed_pair_index(int orbital_index_a, int orbital_index_b) {
  if (orbital_index_a < 0 || orbital_index_b < 0) {
    throw std::invalid_argument("orbital indices must be non-negative");
  }

  const int high_index = std::max(orbital_index_a, orbital_index_b);
  const int low_index = std::min(orbital_index_a, orbital_index_b);
  return high_index * (high_index + 1) / 2 + low_index;
}

int TwoElectronIndexer::packed_pair_of_pairs_index(
    int packed_pair_index_a,
    int packed_pair_index_b) {
  if (packed_pair_index_a < 0 || packed_pair_index_b < 0) {
    throw std::invalid_argument("packed pair indices must be non-negative");
  }

  const int high_index = std::max(packed_pair_index_a, packed_pair_index_b);
  const int low_index = std::min(packed_pair_index_a, packed_pair_index_b);
  return high_index * (high_index + 1) / 2 + low_index;
}

int TwoElectronIndexer::two_electron_storage_index(
    int orbital_index_p,
    int orbital_index_q,
    int orbital_index_r,
    int orbital_index_s) {
  const int first_packed_pair_index = packed_pair_index(orbital_index_p, orbital_index_q);
  const int second_packed_pair_index = packed_pair_index(orbital_index_r, orbital_index_s);
  return packed_pair_of_pairs_index(first_packed_pair_index, second_packed_pair_index);
}

}  // namespace xmvb::vb
