#pragma once

#include <cstdint>

#include "core/index.hpp"

namespace xmvb::mcscf {

/** @brief Signed spin-string address; negative values can mark absent links. */
using StringIndex = std::int64_t;

/** @brief Logical spin-product address, independent of resident tile dimensions. */
using ProductIndex = std::int64_t;

// The address arithmetic is checked on the host before any tile is materialized.
inline constexpr ProductIndex
product_index(StringIndex alpha, StringIndex beta, StringIndex n_beta) {
    if (alpha < 0 || beta < 0 || n_beta <= 0 || beta >= n_beta)
        throw std::invalid_argument("invalid spin-product coordinates");
    return core::checked_index<ProductIndex>(
        core::checked_add(core::checked_multiply(core::checked_index<std::size_t>(alpha),
                                                 core::checked_index<std::size_t>(n_beta)),
                          core::checked_index<std::size_t>(beta)));
}

} // namespace xmvb::mcscf
