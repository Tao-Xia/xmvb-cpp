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
  return n_determinants *
      (n_determinants + 1) / 2;
}

/**
 * @brief Chooses a deterministic chunk size for striped determinant-pair work.
 *
 * The unordered pair storage is triangular: contiguous ranges can align with
 * determinant rows that have different skip rates or cache behavior.  Splitting
 * the linear storage into a few chunks per worker and assigning those chunks in
 * thread-index order gives better load balance while preserving a stable
 * mapping from pair index to thread for a fixed thread count.
 */
inline std::size_t unordered_determinant_pair_parallel_chunk_size(
    std::size_t n_determinant_pairs,
    int n_threads) {
  if (n_determinant_pairs == 0) {
    return 1;
  }
  const std::size_t thread_count =
      n_threads > 0 ? static_cast<std::size_t>(n_threads) : 1;
  const std::size_t target_chunks =
      std::max<std::size_t>(1, thread_count * 8);
  const std::size_t chunk_size =
      (n_determinant_pairs + target_chunks - 1) / target_chunks;
  return std::max<std::size_t>(
      1,
      std::min<std::size_t>(chunk_size, 4096));
}

inline std::size_t canonical_determinant_pair_storage_index(
    int determinant_index_left,
    int determinant_index_right) {
  if (determinant_index_left < determinant_index_right) {
    std::swap(determinant_index_left, determinant_index_right);
  }
  return determinant_index_left *
             (determinant_index_left + 1) / 2 +
      determinant_index_right;
}

inline UnorderedDeterminantPairIndices determinant_pair_from_storage_index(
    std::size_t storage_index) {
  const long double storage_index_ld = static_cast<long double>(storage_index);
  int determinant_index_left = static_cast<int>(
      (std::sqrt(8.0L * storage_index_ld + 1.0L) - 1.0L) * 0.5L);
  std::size_t row_start =
      determinant_index_left *
      (determinant_index_left + 1) / 2;
  while (row_start > storage_index) {
    --determinant_index_left;
    row_start =
        determinant_index_left *
        (determinant_index_left + 1) / 2;
  }
  while ((determinant_index_left + 1) *
             (determinant_index_left + 2) / 2 <=
         storage_index) {
    ++determinant_index_left;
    row_start =
        determinant_index_left *
        (determinant_index_left + 1) / 2;
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
