#pragma once

#include <cstdint>

namespace xmvb::mcscf {

/** @brief Logical spin-product address, independent of resident tile dimensions.
 * Individual spin-string and tile indices remain int. Their product need not
 * fit in int, even when only a few selected structures are stored.
 */
using ProductIndex = std::int64_t;

constexpr ProductIndex product_index(int alpha, int beta, int n_beta) noexcept {
    return static_cast<ProductIndex>(alpha) * n_beta + beta;
}

} // namespace xmvb::mcscf
