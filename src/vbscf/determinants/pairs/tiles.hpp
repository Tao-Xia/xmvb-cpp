#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "core/index.hpp"
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
    std::int64_t begin = 0;
    std::int64_t end = 0;

    std::int64_t size() const {
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
inline SupportWindow find_support_window(const std::vector<std::int64_t>& support,
                                         std::int64_t global_begin,
                                         std::int64_t global_end) {
    const auto begin_iterator = std::lower_bound(support.begin(), support.end(), global_begin);
    const auto end_iterator = std::lower_bound(begin_iterator, support.end(), global_end);
    return SupportWindow{
        static_cast<std::int64_t>(std::distance(support.begin(), begin_iterator)),
        static_cast<std::int64_t>(std::distance(support.begin(), end_iterator)),
    };
}

/**
 * @brief Returns the square-grid tile index containing one global element.
 */
inline std::int64_t tile_index_for_element(std::int64_t element_index, std::int64_t tile_size) {
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
inline std::int64_t tile_count_for_extent(std::int64_t extent, std::int64_t tile_size) {
    if (extent < 0) {
        throw std::invalid_argument("extent must be non-negative");
    }
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    return extent == 0 ? 0 : extent / tile_size + (extent % tile_size != 0);
}

/**
 * @brief Returns the half-open global window covered by one tile id.
 */
inline SupportWindow
tile_window_for_index(std::int64_t tile_index, std::int64_t tile_size, std::int64_t extent) {
    if (tile_index < 0) {
        throw std::invalid_argument("tile_index must be non-negative");
    }
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    if (extent < 0) {
        throw std::invalid_argument("extent must be non-negative");
    }
    const auto begin = core::checked_index<std::int64_t>(core::checked_multiply(
        core::checked_index<std::size_t>(tile_index), core::checked_index<std::size_t>(tile_size)));
    return SupportWindow{begin,
                         begin >= extent ? extent : begin + std::min(tile_size, extent - begin)};
}

/**
 * @brief Returns the flattened task index for one square tile pair.
 */
inline std::int64_t
square_tile_task_index(std::int64_t row_tile, std::int64_t column_tile, std::int64_t n_tiles) {
    if (row_tile < 0 || column_tile < 0 || n_tiles <= 0) {
        throw std::invalid_argument(
            "square tile task index requires non-negative tiles and positive n_tiles");
    }
    return core::checked_index<std::int64_t>(
        core::checked_add(core::checked_multiply(core::checked_index<std::size_t>(row_tile),
                                                 core::checked_index<std::size_t>(n_tiles)),
                          core::checked_index<std::size_t>(column_tile)));
}

/**
 * @brief Lifts active ordered pair indices to their touched square tile tasks.
 *
 * Query-driven backward paths already know which ordered unique-spin pairs are
 * active. This helper turns that sparse pair set into the sparse tile-task set
 * needed by the streamed tile kernels, without visiting empty square tiles.
 */
inline std::vector<std::int64_t>
build_active_square_tile_task_indices(const std::vector<std::size_t>& active_pair_indices,
                                      std::int64_t leading_dimension,
                                      std::int64_t tile_size) {
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    if (leading_dimension <= 0 || active_pair_indices.empty()) {
        return {};
    }

    const std::int64_t n_tiles = tile_count_for_extent(leading_dimension, tile_size);
    const std::size_t leading_dimension_size = leading_dimension;
    std::vector<std::int64_t> active_tile_indices;
    active_tile_indices.reserve(active_pair_indices.size());
    for (const std::size_t pair_storage_index : active_pair_indices) {
        const std::int64_t row_index =
            static_cast<std::int64_t>(pair_storage_index / leading_dimension_size);
        const std::int64_t column_index =
            static_cast<std::int64_t>(pair_storage_index % leading_dimension_size);
        const std::int64_t row_tile = tile_index_for_element(row_index, tile_size);
        const std::int64_t column_tile = tile_index_for_element(column_index, tile_size);
        active_tile_indices.push_back(square_tile_task_index(row_tile, column_tile, n_tiles));
    }
    std::sort(active_tile_indices.begin(), active_tile_indices.end());
    active_tile_indices.erase(std::unique(active_tile_indices.begin(), active_tile_indices.end()),
                              active_tile_indices.end());
    return active_tile_indices;
}

} // namespace xmvb::vb
