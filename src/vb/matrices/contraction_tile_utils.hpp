#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>


namespace xmvb::vb {

/**
 * @brief Half-open index range shared by support-local and tile-local kernels.
 *
 * The bounded-memory contraction code repeatedly slices sorted support lists
 * and square unique-pair grids into small half-open windows `[begin, end)`.
 * Keeping one shared range type makes the forward and backward tile schedulers
 * agree on the same indexing convention.
 */
struct SupportWindow {
  int begin = 0;
  int end = 0;

  int size() const {
    return end - begin;
  }

  bool empty() const {
    return begin >= end;
  }
};

/**
 * @brief Finds the support-local window overlapping one global half-open range.
 *
 * Support vectors are stored in ascending unique-spin id order. This helper
 * returns the local slice that intersects `[global_begin, global_end)` so
 * callers can gather only the touched rows or columns before doing the dense
 * local contraction.
 */
inline SupportWindow find_support_window(
    const std::vector<int>& support,
    int global_begin,
    int global_end) {
  const auto begin_iterator =
      std::lower_bound(support.begin(), support.end(), global_begin);
  const auto end_iterator =
      std::lower_bound(begin_iterator, support.end(), global_end);
  return SupportWindow{
      static_cast<int>(std::distance(support.begin(), begin_iterator)),
      static_cast<int>(std::distance(support.begin(), end_iterator)),
  };
}

/**
 * @brief Returns the square-grid tile index containing one global element.
 */
inline int tile_index_for_element(int element_index, int tile_size) {
  if (element_index < 0) {
    throw std::invalid_argument("element_index must be non-negative");
  }
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  return element_index / tile_size;
}

/**
 * @brief Returns the number of tiles needed to cover one extent.
 */
inline int tile_count_for_extent(int extent, int tile_size) {
  if (extent < 0) {
    throw std::invalid_argument("extent must be non-negative");
  }
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  return extent == 0 ? 0 : (extent + tile_size - 1) / tile_size;
}

/**
 * @brief Returns the half-open global window covered by one tile id.
 */
inline SupportWindow tile_window_for_index(
    int tile_index,
    int tile_size,
    int extent) {
  if (tile_index < 0) {
    throw std::invalid_argument("tile_index must be non-negative");
  }
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (extent < 0) {
    throw std::invalid_argument("extent must be non-negative");
  }
  const int begin = tile_index * tile_size;
  return SupportWindow{begin, std::min(extent, begin + tile_size)};
}

/**
 * @brief Returns the flattened task index for one square tile pair.
 */
inline int square_tile_task_index(int row_tile, int column_tile, int n_tiles) {
  if (row_tile < 0 || column_tile < 0 || n_tiles <= 0) {
    throw std::invalid_argument(
        "square tile task index requires non-negative tiles and positive n_tiles");
  }
  return row_tile * n_tiles + column_tile;
}

/**
 * @brief Chooses a square-ish tile size for bounded-memory parallel work.
 *
 * The configured tile size defines the maximum working-set size. When there
 * are multiple worker threads, a slightly smaller square tile can expose more
 * independent work without violating the same bounded-memory contract.
 */
inline int choose_square_parallel_tile_size(
    int extent,
    int configured_tile_size,
    int n_threads) {
  if (extent <= 0) {
    throw std::invalid_argument("extent must be positive");
  }
  if (configured_tile_size <= 0) {
    throw std::invalid_argument("configured_tile_size must be positive");
  }
  if (n_threads <= 0) {
    throw std::invalid_argument("n_threads must be positive");
  }

  const int clamped_tile_size = std::min(extent, configured_tile_size);
  if (clamped_tile_size <= 1) {
    return clamped_tile_size;
  }
  if (extent <= clamped_tile_size ||
      (extent) * (extent) <= 4096u) {
    return clamped_tile_size;
  }
  if (n_threads <= 1) {
    return clamped_tile_size;
  }

  const int target_tiles_per_dimension = std::min(
      extent,
      std::max(
          1,
          static_cast<int>(
              std::ceil(std::sqrt(2.0 * static_cast<double>(n_threads))))));
  const int parallel_tile_size = std::max(
      1,
      (extent + target_tiles_per_dimension - 1) / target_tiles_per_dimension);
  return std::min(clamped_tile_size, parallel_tile_size);
}

/**
 * @brief Lifts active ordered pair indices to their touched square tile tasks.
 *
 * Query-driven backward paths already know which ordered unique-spin pairs are
 * active. This helper turns that sparse pair set into the sparse tile-task set
 * needed by the streamed tile kernels, without visiting empty square tiles.
 */
inline std::vector<int> build_active_square_tile_task_indices(
    const std::vector<std::size_t>& active_pair_indices,
    int leading_dimension,
    int tile_size) {
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (leading_dimension <= 0 || active_pair_indices.empty()) {
    return {};
  }

  const int n_tiles = tile_count_for_extent(leading_dimension, tile_size);
  const std::size_t leading_dimension_size = leading_dimension;
  std::vector<int> active_tile_indices;
  active_tile_indices.reserve(active_pair_indices.size());
  for (const std::size_t pair_storage_index : active_pair_indices) {
    const int row_index = static_cast<int>(pair_storage_index / leading_dimension_size);
    const int column_index =
        static_cast<int>(pair_storage_index % leading_dimension_size);
    const int row_tile = tile_index_for_element(row_index, tile_size);
    const int column_tile = tile_index_for_element(column_index, tile_size);
    active_tile_indices.push_back(
        square_tile_task_index(row_tile, column_tile, n_tiles));
  }
  std::sort(active_tile_indices.begin(), active_tile_indices.end());
  active_tile_indices.erase(
      std::unique(active_tile_indices.begin(), active_tile_indices.end()),
      active_tile_indices.end());
  return active_tile_indices;
}

}  // namespace xmvb::vb
