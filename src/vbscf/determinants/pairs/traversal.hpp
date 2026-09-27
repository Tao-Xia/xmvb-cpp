#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Orders one right-string tile to minimize successive update rank.
 *
 * The returned indices are global string indices in `[right_begin,
 * right_end)`.  When the left string belongs to the tile, its diagonal pair
 * is used as the exact traversal anchor.
 */
std::vector<int> build_pair_update_traversal(
    const std::vector<std::vector<int>>& unique_spin_strings,
    int left_index,
    int right_begin,
    int right_end);

}  // namespace xmvb::vb
