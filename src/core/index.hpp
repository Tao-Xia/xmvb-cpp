#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace xmvb::core {

static_assert(std::numeric_limits<std::size_t>::digits >= 64,
              "64-bit native allocation addresses are required");

/** @brief Checked count arithmetic, before allocation or address construction.
 * Counts/byte sizes use size_t; signed matrix and string addresses use the
 * native 64-bit index. Orbital labels, fermion signs and vendor-required local
 * integer arguments stay narrow only at their bounded, checked interfaces.
 */
constexpr std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("index sum exceeds size_t range");
    return a + b;
}

constexpr std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("index product exceeds size_t range");
    return a * b;
}

/** @brief Explicit checked integer bridge, notably for external library ABIs. */
template <class To, class From> constexpr To checked_index(From value) {
    static_assert(std::is_integral_v<To> && std::is_integral_v<From>);
    if (!std::in_range<To>(value))
        throw std::overflow_error("integer index conversion exceeds destination range");
    return static_cast<To>(value);
}

constexpr std::size_t triangular_count(std::size_t n) {
    const auto next = checked_add(n, 1);
    return n % 2 ? checked_multiply(n, next / 2) : checked_multiply(n / 2, next);
}

/** @brief Two independent native addresses; never pack them into 32-bit halves. */
struct IndexPair {
    std::size_t first = 0, second = 0;
    bool operator==(const IndexPair&) const = default;
};

struct IndexPairHash {
    std::size_t operator()(const IndexPair& pair) const noexcept {
        const auto a = std::hash<std::size_t>{}(pair.first);
        const auto b = std::hash<std::size_t>{}(pair.second);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
};

} // namespace xmvb::core
