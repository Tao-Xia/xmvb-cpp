#include <climits>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "core/index.hpp"
#include "core/pair_index.hpp"
#include "integrals/ao/input.hpp"
#include "mcscf/ci/column_cache.hpp"
#include "mcscf/ci/product_index.hpp"
#include "mcscf/ci/strings.hpp"
#include "mcscf/ci/links.hpp"
#include "linalg/views.hpp"

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("wide index boundary failed");
}
template <class F> void overflow(F&& f) {
    try {
        f();
    } catch (const std::overflow_error&) {
        return;
    }
    throw std::runtime_error("overflow was not rejected");
}
struct Tile {
    std::int64_t address;
    std::size_t bytes() const {
        return sizeof(address);
    }
};
} // namespace
int main() {
    using namespace xmvb;
    static_assert(sizeof(mcscf::StringIndex) >= 8);
    static_assert(sizeof(integrals::PairAddress) >= 8);
    static_assert(sizeof(decltype(mcscf::SpinLink::source)) >= 8);
    static_assert(sizeof(decltype(integrals::AoPairGraph::row_offsets)::value_type) >= 8);
    static_assert(core::checked_multiply(48620, 48620) == 2363904400ULL);
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    require(core::checked_add(maximum - 1, 1) == maximum);
    require(core::checked_multiply(maximum, 1) == maximum);
    overflow([&] { core::checked_add(maximum, 1); });
    overflow([&] { core::checked_multiply(maximum, 2); });
    overflow([&] { core::checked_index<std::size_t>(-1); });
    overflow([&] { core::checked_index<int>(std::int64_t{INT_MAX} + 1); });
    overflow([&] { core::triangular_count(maximum); });
    constexpr std::size_t n = 4294967295ULL;
    require(core::triangular_count(n) == 9223372034707292160ULL);
    require(core::PairIndex::packed_pair_index(65536, 0) == 2147516416ULL);
    require(mcscf::product_index(48619, 48619, 48620) == 2363904399LL);
    const linalg::ConstMatrixView logical{nullptr, 48620, 48620, 48620};
    require(linalg::element_count(logical) == 2363904400LL);
    const linalg::ConstMatrixView impossible{nullptr, INT64_MAX, 2, INT64_MAX};
    overflow([&] { linalg::element_count(impossible); });
    overflow([&] { mcscf::product_index(INT64_MAX, 0, 2); });
    const auto high = std::size_t{1} << 32;
    std::unordered_map<core::IndexPair, int, core::IndexPairHash> pairs;
    pairs[{1, 2}] = 1;
    pairs[{high + 1, 2}] = 2;
    pairs[{1, high + 2}] = 3;
    require(pairs.size() == 3);
    mcscf::ColumnCache<Tile> cache(
        128,
        [](std::int64_t key) { return Tile{key}; },
        {},
        [](std::int64_t) { return sizeof(std::int64_t); });
    require(cache.get(1).address == 1);
    require(cache.get(high + 1).address == static_cast<std::int64_t>(high + 1));
    require(cache.get(1).address == 1);
    // Two resident strings, but combinatorial ranks span far beyond INT_MAX.
    const std::uint64_t low = (std::uint64_t{1} << 30) - 1;
    const std::uint64_t high_mask = low << 30;
    const std::uint64_t masks[] = {high_mask, low};
    mcscf::SpinStrings strings(60, masks);
    require(mcscf::string_count(60, 30) == 118264581564861424ULL);
    require(strings.index_from_rank(mcscf::string_count(60, 30) - 1) == 0);
    require(strings.index_from_rank(0) == 1);
    require(strings.index(high_mask) == 0 && strings.index(low) == 1);
    std::cout << "native address, CSR, cache and combinatorial boundaries passed\n";
}
