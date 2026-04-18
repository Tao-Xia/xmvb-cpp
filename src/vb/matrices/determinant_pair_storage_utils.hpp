#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace xmvb::vb {

struct UnorderedDeterminantPairIndices {
  int left = 0;
  int right = 0;
};

inline std::size_t unordered_determinant_pair_count(int n_determinants) {
  return xmvb::to_size(n_determinants) *
      xmvb::to_size(n_determinants + 1) / 2;
}

inline std::size_t canonical_determinant_pair_storage_index(
    int determinant_index_left,
    int determinant_index_right) {
  if (determinant_index_left < determinant_index_right) {
    std::swap(determinant_index_left, determinant_index_right);
  }
  return xmvb::to_size(determinant_index_left) *
             xmvb::to_size(determinant_index_left + 1) / 2 +
      xmvb::to_size(determinant_index_right);
}

inline UnorderedDeterminantPairIndices determinant_pair_from_storage_index(
    std::size_t storage_index) {
  const long double storage_index_ld = static_cast<long double>(storage_index);
  int determinant_index_left = static_cast<int>(
      (std::sqrt(8.0L * storage_index_ld + 1.0L) - 1.0L) * 0.5L);
  std::size_t row_start =
      xmvb::to_size(determinant_index_left) *
      xmvb::to_size(determinant_index_left + 1) / 2;
  while (row_start > storage_index) {
    --determinant_index_left;
    row_start =
        xmvb::to_size(determinant_index_left) *
        xmvb::to_size(determinant_index_left + 1) / 2;
  }
  while (xmvb::to_size(determinant_index_left + 1) *
             xmvb::to_size(determinant_index_left + 2) / 2 <=
         storage_index) {
    ++determinant_index_left;
    row_start =
        xmvb::to_size(determinant_index_left) *
        xmvb::to_size(determinant_index_left + 1) / 2;
  }
  return {
      determinant_index_left,
      static_cast<int>(storage_index - row_start)};
}

inline void advance_unordered_determinant_pair(
    UnorderedDeterminantPairIndices* determinant_pair) {
  if (determinant_pair == nullptr) {
    return;
  }
  if (determinant_pair->right < determinant_pair->left) {
    ++determinant_pair->right;
  } else {
    ++determinant_pair->left;
    determinant_pair->right = 0;
  }
}

}  // namespace xmvb::vb
