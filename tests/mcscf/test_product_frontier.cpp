// Standalone exact-support regression; compile with -Isrc and C++23.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mcscf/ci/product_paths.hpp"
#include "mcscf/ci/product_support.hpp"

namespace {
using Paths = xmvb::mcscf::ProductPaths;
using Mask = Paths::Mask;
using Seed = Paths::Seed;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<Mask> strings(int n, int electrons) {
    std::vector<Mask> result;
    for (Mask mask = 0; mask < (Mask{1} << n); ++mask)
        if (__builtin_popcountll(mask) == electrons)
            result.push_back(mask);
    return result;
}

// Independent prefix-deficit oracle, applied to the original seed family.
int deficit(Mask target, Mask seed, int n) {
    int difference = 0, maximum = 0;
    for (int k = 0; k < n; ++k) {
        difference += static_cast<int>((seed >> k) & 1) - static_cast<int>((target >> k) & 1);
        maximum = std::max(maximum, difference);
    }
    return maximum;
}

bool reachable(Mask alpha, Mask beta, const std::vector<Seed>& original, int n, int depth) {
    for (const auto& [sa, sb] : original)
        if (deficit(alpha, sa, n) + deficit(beta, sb, n) <= depth)
            return true;
    return false;
}

void check_case(int n,
                int na,
                int nb,
                const std::vector<Seed>& input,
                const std::vector<Seed>& original,
                bool covalent = false) {
    const Paths paths(n, na, nb, input, covalent);
    const auto alphas = strings(n, na), betas = strings(n, nb);
    for (int depth : {0, 1}) {
        std::map<Seed, bool> expected, actual, by_column;
        for (Mask b : betas)
            for (Mask a : alphas)
                if (reachable(a, b, original, n, depth))
                    expected.emplace(Seed{a, b}, reachable(a, b, original, n, 0));

        const auto counts = paths.visit(depth, [&](Mask a, Mask b, bool support) {
            require(actual.emplace(Seed{a, b}, support).second,
                    "duplicate product in free traversal");
        });
        require(actual == expected, "free traversal differs from original-seed oracle");
        require(counts.products == expected.size(), "free product count is wrong");

        for (Mask b : betas)
            paths.visit_column(b, depth, [&](Mask a, Mask emitted_b, bool support) {
                require(emitted_b == b, "column traversal changed beta");
                require(by_column.emplace(Seed{a, b}, support).second,
                        "duplicate product in column traversal");
            });
        require(by_column == expected, "column traversal differs from oracle");

        auto columns = paths.cursor(depth);
        std::vector<Mask> emitted_betas;
        Mask b;
        while (columns.next_column(b))
            emitted_betas.push_back(b);
        std::vector<Mask> expected_betas;
        for (Mask candidate : betas)
            if (std::any_of(expected.begin(), expected.end(), [candidate](const auto& entry) {
                    return entry.first.second == candidate;
                }))
                expected_betas.push_back(candidate);
        std::sort(emitted_betas.begin(), emitted_betas.end());
        require(emitted_betas == expected_betas, "beta-column set differs from oracle");
    }
}

std::uint64_t random_word(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void check_random() {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (const auto& sector :
         {std::pair{5, 2}, std::pair{6, 1}, std::pair{6, 0}, std::pair{6, 4}}) {
        const int n = sector.first, na = sector.second;
        const int nb = n == 6 && na == 1 ? 3 : 1;
        const auto alphas = strings(n, na), betas = strings(n, nb);
        for (int trial = 0; trial < 50; ++trial) {
            std::vector<Seed> selected;
            const int length = trial == 0 ? 0 : 1 + static_cast<int>(random_word(state) % 24);
            for (int j = 0; j < length; ++j) {
                const Mask a = alphas[random_word(state) % alphas.size()];
                const Mask b = betas[random_word(state) % betas.size()];
                selected.emplace_back(a, b);
                if (j % 5 == 0)
                    selected.emplace_back(a, b); // duplicate inputs
            }
            std::vector<Seed> distinct = selected;
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            std::size_t maximal_count = 0;
            for (const Seed& candidate : distinct) {
                const bool dominated =
                    std::any_of(distinct.begin(), distinct.end(), [&](const Seed& other) {
                        return other != candidate &&
                               deficit(candidate.first, other.first, n) == 0 &&
                               deficit(candidate.second, other.second, n) == 0;
                    });
                maximal_count += !dominated;
            }
            const Seed maximum{alphas.back(), betas.back()};
            const bool complete =
                std::find(distinct.begin(), distinct.end(), maximum) != distinct.end();
            require(Paths(n, na, nb, selected).generator_count() == (complete ? 0 : maximal_count),
                    "random maximal-generator count differs from dominance oracle");
            check_case(n, na, nb, selected, selected);
        }
    }
}

void check_special_cases() {
    const int n = 8, electrons = 4;
    const auto masks = strings(n, electrons);
    std::vector<Seed> ionic;
    for (Mask a : masks)
        for (Mask b : masks)
            if (a != b)
                ionic.emplace_back(a, b);
    const Paths ionic_paths(n, electrons, electrons, ionic);
    require(ionic.size() == 4830 && ionic_paths.generator_count() == 2,
            "ION(0:3) must have exactly two maximal generators");
    const auto zero = ionic_paths.visit(0, [](Mask, Mask, bool) {});
    const auto one = ionic_paths.visit(1, [](Mask, Mask, bool) {});
    require(zero.products == 4899 && zero.support == 4899 && one.products == 4900 &&
                one.support == 4899,
            "ION(0:3) exact closure is wrong");
    const Mask top = masks.back(), coatom = Mask{232};
    check_case(n, electrons, electrons, ionic, {{top, coatom}, {coatom, top}});

    const int small_n = 5;
    const auto a = strings(small_n, 2), b = strings(small_n, 1);
    const Seed maximum{a.back(), b.back()};
    const std::vector<Seed> full_seeds{{a.front(), b.front()}, maximum};
    const Paths full(small_n, 2, 1, full_seeds);
    require(full.generator_count() == 0, "full-sector shortcut lost");
    check_case(small_n, 2, 1, full_seeds, full_seeds);

    const Mask universe = (Mask{1} << small_n) - 1;
    std::vector<Seed> covalent;
    for (Mask mask : a)
        covalent.emplace_back(mask, universe ^ mask);
    const Paths certified(small_n, 2, 3, {}, true);
    require(certified.generator_count() == 0, "covalent shortcut lost");
    check_case(small_n, 2, 3, {}, covalent, true);
}

// Reusable column plans must match the independent original-seed oracle,
// including unequal spins and caller permutations. No product table is kept.
void check_column_plans() {
    using Support = xmvb::mcscf::ProductSupport;
    const auto check_bounds = [](const Support& support,
                                 int beta,
                                 int depth,
                                 const std::vector<std::int64_t>& expected,
                                 int count) {
        for (int upper = 0; upper <= count; ++upper) {
            std::vector<std::int64_t> actual{-1}, bounded{-1};
            support.append_column_rows(beta, actual, depth, upper);
            for (const int a : expected)
                if (a >= 0 && a < upper)
                    bounded.push_back(a);
            require(actual == bounded, "bounded column emission changed caller indices or support");
        }
    };
    for (bool permuted : {false, true})
        for (int n = 1; n <= 7; ++n)
            for (int na = 0; na <= n; ++na) {
                auto a = strings(n, na), b = strings(n, n - na);
                if (permuted) {
                    std::reverse(a.begin(), a.end());
                    std::rotate(a.begin(), a.begin() + a.size() / 3, a.end());
                }
                std::rotate(b.begin(), b.begin() + b.size() / 2, b.end());
                const Mask full = (Mask{1} << n) - 1;
                std::vector<Seed> seeds;
                std::vector<xmvb::mcscf::ProductIndex> products;
                for (int i = 0; i < static_cast<int>(a.size()); ++i) {
                    const auto j = std::find(b.begin(), b.end(), full ^ a[i]) - b.begin();
                    seeds.emplace_back(a[i], b[j]);
                    products.push_back(xmvb::mcscf::product_index(i, j, b.size()));
                }
                const Support support(n, a, b, products);
                for (int depth = 0; depth <= 1; ++depth)
                    for (int j = 0; j < static_cast<int>(b.size()); ++j) {
                        std::vector<std::int64_t> actual{-1}, expected{-1};
                        support.append_column_rows(j, actual, depth);
                        for (int i = 0; i < static_cast<int>(a.size()); ++i)
                            if (reachable(a[i], b[j], seeds, n, depth))
                                expected.push_back(i);
                        require(actual == expected,
                                "shared covalent column plan differs from seed oracle");
                        check_bounds(support, j, depth, expected, a.size());
                    }
                require(support.columns().size() == b.size(), "covalent beta columns were lost");
                // A genuinely arbitrary selection must keep its generator constraints.
                products.resize(std::min<std::size_t>(products.size(), 2));
                seeds.resize(products.size());
                const Support selected(n, a, b, products);
                std::vector<std::int64_t> columns;
                for (int depth = 0; depth <= 1; ++depth)
                    for (int j = 0; j < static_cast<int>(b.size()); ++j) {
                        std::vector<std::int64_t> actual, expected;
                        selected.append_column_rows(j, actual, depth);
                        for (int i = 0; i < static_cast<int>(a.size()); ++i)
                            if (reachable(a[i], b[j], seeds, n, depth))
                                expected.push_back(i);
                        require(actual == expected,
                                "arbitrary column plan enlarged the selected support");
                        check_bounds(selected, j, depth, expected, a.size());
                        if (!depth && !actual.empty())
                            columns.push_back(j);
                    }
                require(selected.columns() == columns, "shared beta column list is wrong");
            }
    const auto a = strings(14, 7);
    const xmvb::mcscf::CovalentSupport plan(14, a, a);
    require(plan.dynamic_bytes() < 8 * (a.size() + a.size()) * 15 * 9,
            "column plan retained product-sized metadata");
}
} // namespace

int main() {
    // Logical CAS(18,18) addresses must not overflow, without allocating U^2.
    const auto large = strings(18, 9);
    const auto last = xmvb::mcscf::product_index(48619, 48619, 48620);
    require(last == 2363904399LL, "wide product index overflow");
    const xmvb::mcscf::ProductSupport support(18, large, large, {last});
    require(support.full(), "wide product seed decoded incorrectly");
    check_random();
    check_special_cases();
    check_column_plans();
    std::cout << "product_frontier_exact_support=PASS\n";
}
