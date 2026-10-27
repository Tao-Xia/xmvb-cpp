/** @file Exact channel-support census; no production operator is modified. */
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"

namespace {
using Bits = std::uint64_t;
using Channels = std::array<Bits, 2>;
using Support = xmvb::mcscf::ProductSupport;

/** @brief Fixed-spin strings in occupied-orbital lexicographic order. */
void append_strings(int n, int left, int first, Bits mask, std::vector<Bits>& result) {
    if (!left) {
        result.push_back(mask);
        return;
    }
    for (int p = first; p <= n - left; ++p)
        append_strings(n, left - 1, p + 1, mask | (Bits{1} << p), result);
}

struct Sector {
    std::vector<Bits> strings;
    std::unordered_map<Bits, int> index;
    Sector(int n, int electrons) {
        append_strings(n, electrons, 0, 0, strings);
        for (int i = 0; i < static_cast<int>(strings.size()); ++i)
            index.emplace(strings[i], i);
    }
    std::vector<std::vector<int>> occupations(int n) const {
        std::vector<std::vector<int>> result(strings.size());
        for (std::size_t i = 0; i < strings.size(); ++i)
            for (int p = 0; p < n; ++p)
                if (strings[i] & (Bits{1} << p))
                    result[i].push_back(p);
        return result;
    }
};

/** @brief Channels with a preimage in S, independent of numerical amplitudes. */
Channels
channels(const Support& support, const Sector& alpha, const Sector& beta, int n, int a, int b) {
    Channels result{};
    const Bits am = alpha.strings[a], bm = beta.strings[b];
    const bool own = support.contains(a, b);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q <= p; ++q) {
            const Bits flip = (Bits{1} << p) | (Bits{1} << q);
            const auto changes = [&](Bits value) {
                return ((value >> p) & 1) != ((value >> q) & 1);
            };
            const bool live =
                p == q ? own && ((am | bm) & (Bits{1} << p))
                       : (changes(am) && support.contains(alpha.index.at(am ^ flip), b)) ||
                             (changes(bm) && support.contains(a, beta.index.at(bm ^ flip)));
            if (live) {
                const int pair = p * (p + 1) / 2 + q;
                result[pair / 64] |= Bits{1} << (pair % 64);
            }
        }
    return result;
}

/** @brief Covalent boundary certificate: repair every deficient prefix at once. */
Channels covalent_boundary(Bits alpha, Bits beta, int n) {
    int electrons = 0, first = n, last = -1;
    for (int p = 0; p < n; ++p) {
        electrons += ((alpha >> p) & 1) + ((beta >> p) & 1);
        if (electrons < p + 1) {
            first = std::min(first, p);
            last = p;
        }
    }
    if (last < 0)
        throw std::logic_error("boundary certificate received an interior node");
    Channels result{};
    const Bits early = (Bits{1} << (first + 1)) - 1;
    const Bits late = ((Bits{1} << n) - 1) ^ ((Bits{1} << (last + 1)) - 1);
    for (const Bits spin : {alpha, beta}) {
        const Bits empty = (~spin) & early;
        Bits occupied = spin & late;
        while (occupied) {
            const int q = std::countr_zero(occupied);
            occupied &= occupied - 1;
            for (Bits sites = empty; sites; sites &= sites - 1) {
                const int p = std::countr_zero(sites), pair = q * (q + 1) / 2 + p;
                result[pair / 64] |= Bits{1} << (pair % 64);
            }
        }
    }
    return result;
}

/** @brief Small complete-carrier oracle using the production generator action. */
void check_channels() {
    constexpr int n = 4, pairs = 10;
    for (const auto [na, nb] : {std::pair{2, 2}, std::pair{1, 3}, std::pair{0, 4}}) {
        const Sector alpha(n, na), beta(n, nb);
        const Support support(n, alpha.strings, beta.strings);
        const xmvb::mcscf::ActiveIntegrals integrals{Eigen::MatrixXd::Zero(n, n),
                                                     Eigen::MatrixXd::Zero(pairs, pairs)};
        const xmvb::mcscf::DirectCiSigmaAction sigma(
            alpha.occupations(n), beta.occupations(n), integrals);
        const int rows = alpha.strings.size(), cols = beta.strings.size();
        // Only this n=4 oracle has complete product arrays (at most 36 entries).
        std::vector<Channels> observed(rows * cols);
        for (int sa = 0; sa < rows; ++sa)
            for (int sb = 0; sb < cols; ++sb) {
                if (!support.contains(sa, sb))
                    continue;
                Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(rows, cols);
                seed(sa, sb) = 1;
                for (int p = 0; p < n; ++p)
                    for (int q = 0; q <= p; ++q) {
                        Eigen::MatrixXd generator = Eigen::MatrixXd::Zero(n, n);
                        generator(p, q) = generator(q, p) = 1;
                        const auto image = sigma.apply_one_body_generator(seed, generator);
                        const int pair = p * (p + 1) / 2 + q;
                        for (int a = 0; a < rows; ++a)
                            for (int b = 0; b < cols; ++b)
                                if (image(a, b) != 0)
                                    observed[a * cols + b][pair / 64] |= Bits{1} << (pair % 64);
                    }
            }
        for (int a = 0; a < rows; ++a)
            for (int b = 0; b < cols; ++b) {
                if (observed[a * cols + b] != channels(support, alpha, beta, n, a, b))
                    throw std::runtime_error(
                        "channel support differs from generator basis actions");
                if (!support.contains(a, b) && support.contains(a, b, 1) &&
                    observed[a * cols + b] !=
                        covalent_boundary(alpha.strings[a], beta.strings[b], n))
                    throw std::runtime_error(
                        "unequal-spin prefix certificate changed generator support");
            }
    }
    std::cout << "small_generator_oracle=PASS\n";
}

/** @brief Tile-bounded grouping census: never retain a product-space mask. */
void census(const std::string& space, int tile_rows) {
    constexpr int n = 12, pairs = n * (n + 1) / 2;
    const Sector alpha(n, 6), beta(n, 6);
    const auto make_support = [&]() {
        if (space == "cov")
            return Support(n, alpha.strings, beta.strings);
        std::vector<xmvb::mcscf::ProductIndex> seeds;
        if (space == "full") {
            const int top = alpha.index.at(((Bits{1} << 6) - 1) << 6);
            seeds.push_back(top * static_cast<int>(beta.strings.size()) + top);
        } else if (space == "single") {
            // One singlet product of six disjoint pairs: original support has 2^6 terms.
            for (Bits choices = 0; choices < (Bits{1} << 6); ++choices) {
                Bits a = 0;
                for (int p = 0; p < 6; ++p)
                    a |= Bits{1} << (2 * p + ((choices >> p) & 1));
                const Bits b = ((Bits{1} << n) - 1) ^ a;
                seeds.push_back(alpha.index.at(a) * static_cast<int>(beta.strings.size()) +
                                beta.index.at(b));
            }
        } else
            throw std::invalid_argument("unknown support family");
        return Support(n, alpha.strings, beta.strings, seeds);
    };
    const Support support = make_support();
    std::array<std::uint64_t, pairs + 1> widths{};
    std::array<std::uint64_t, 64> group_sizes{};
    std::map<Channels, std::uint64_t> groups;
    std::vector<int> boundary_widths;
    std::uint64_t contiguous_capacity = 0, cyclic_capacity = 0;
    std::uint64_t targets = 0, source_targets = 0, sum_width = 0, sum_square = 0, batches = 0,
                  group_count = 0, largest_group = 0, peak_groups = 0, source_width = 0,
                  source_square = 0, prefix_checks = 0;
    int live = 0;
    const auto flush = [&]() {
        if (!live)
            return;
        ++batches;
        group_count += groups.size();
        peak_groups = std::max<std::uint64_t>(peak_groups, groups.size());
        for (const auto& [mask, count] : groups) {
            ++group_sizes[std::bit_width(count) - 1];
            largest_group = std::max(largest_group, count);
        }
        // An arithmetic ownership model, not a wall-time or cache prediction.
        std::array<std::uint64_t, 32> contiguous{}, cyclic{};
        for (int owner = 0; owner < 32; ++owner)
            for (std::size_t i = boundary_widths.size() * owner / 32;
                 i < boundary_widths.size() * (owner + 1) / 32;
                 ++i)
                contiguous[owner] += boundary_widths[i] * boundary_widths[i];
        for (std::size_t i = 0; i < boundary_widths.size(); ++i)
            cyclic[i % 32] += boundary_widths[i] * boundary_widths[i];
        contiguous_capacity += 32 * (*std::max_element(contiguous.begin(), contiguous.end()));
        cyclic_capacity += 32 * (*std::max_element(cyclic.begin(), cyclic.end()));
        boundary_widths.clear();
        groups.clear();
        live = 0;
    };
    auto cursor = support.cursor(1);
    xmvb::mcscf::StringIndex a, b;
    bool in_support;
    while (cursor.next(a, b, in_support)) {
        if (a > b)
            continue; // The production even-exchange representative.
        const auto mask = channels(support, alpha, beta, n, a, b);
        if (space == "cov" && !in_support) {
            if (mask != covalent_boundary(alpha.strings[a], beta.strings[b], n))
                throw std::runtime_error("prefix boundary certificate changed channel support");
            ++prefix_checks;
        }
        const auto width = std::popcount(mask[0]) + std::popcount(mask[1]);
        ++targets;
        source_targets += in_support;
        ++widths[width];
        sum_width += width;
        sum_square += width * width;
        if (in_support) {
            source_width += width;
            source_square += width * width;
        } else
            boundary_widths.push_back(width);
        ++groups[mask];
        if (++live == tile_rows)
            flush();
    }
    flush();
    std::cout << std::setprecision(12) << "space=" << space << " tile_rows=" << tile_rows
              << " targets=" << targets << " source_targets=" << source_targets
              << " mean_channels=" << static_cast<double>(sum_width) / targets
              << " universal_gemm_work_ratio=" << static_cast<double>(sum_width) / (targets * pairs)
              << " gram_work_ratio=" << static_cast<double>(sum_square) / (targets * pairs * pairs)
              << " groups=" << group_count
              << " mean_group_rows=" << static_cast<double>(targets) / group_count
              << " largest_group=" << largest_group << " peak_tile_groups=" << peak_groups
              << " batches=" << batches << '\n';
    if (space == "cov")
        std::cout << "prefix_boundary_certificate=PASS nodes=" << prefix_checks << '\n';
    std::cout << "work_partition," << space << ',' << tile_rows
              << ",workers=32,total_work=" << sum_square - source_square
              << ",contiguous_capacity=" << contiguous_capacity
              << ",cyclic_capacity=" << cyclic_capacity << '\n';
    for (const bool interior : {true, false}) {
        const auto count = interior ? source_targets : targets - source_targets;
        const auto width = interior ? source_width : sum_width - source_width;
        const auto square = interior ? source_square : sum_square - source_square;
        std::cout << "region," << space << ',' << tile_rows
                  << ",kind=" << (interior ? "source" : "boundary") << ",nodes=" << count
                  << ",mean_channels=" << (count ? static_cast<double>(width) / count : 0.0)
                  << ",channel_sum=" << width << ",channel_square_sum=" << square << '\n';
    }
    for (int k = 0; k <= pairs; ++k)
        if (widths[k])
            std::cout << "width," << space << ',' << tile_rows << ',' << k << ',' << widths[k]
                      << '\n';
    for (int k = 0; k < 64; ++k)
        if (group_sizes[k])
            std::cout << "group_rows_bin," << space << ',' << tile_rows << ',' << (Bits{1} << k)
                      << ',' << group_sizes[k] << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 2 || std::string(argv[1]) != "--oracle"))
        throw std::invalid_argument("usage: audit_channel_support [--oracle]");
    check_channels();
    if (argc == 2)
        return 0;
    for (const int tile_rows : {4096, 32768})
        for (const auto* space : {"single", "cov", "full"})
            census(space, tile_rows);
}
