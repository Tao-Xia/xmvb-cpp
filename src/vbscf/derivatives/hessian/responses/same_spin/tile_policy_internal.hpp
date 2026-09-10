#pragma once

namespace xmvb::vb::detail {

// Upper bound for temporary pair tiles. This controls only workspace blocking;
// it never selects a mathematical contraction path or changes the result.
inline constexpr int kSameSpinTileExtent = 64;

}  // namespace xmvb::vb::detail
