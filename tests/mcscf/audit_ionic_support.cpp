// Standalone combinatorial diagnostic: no integrals or optimizer changes.
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mcscf/ci/product_support.hpp"

int main() {
    using Support = xmvb::mcscf::ProductSupport;
    std::vector<Support::Mask> strings;
    for (Support::Mask bits = 0; bits < 256; ++bits)
        if (__builtin_popcountll(bits) == 4)
            strings.push_back(bits);
    const int size = static_cast<int>(strings.size());
    std::vector<int> seeds;
    for (int a = 0; a < size; ++a)
        for (int b = 0; b < size; ++b)
            if (__builtin_popcountll(strings[a] & strings[b]) <= 3)
                seeds.push_back(a * size + b);
    // Highest and unique coatom in the componentwise occupied-orbital order.
    const int top = size - 1;
    const auto found = std::find(strings.begin(), strings.end(), Support::Mask{232});
    if (found == strings.end())
        throw std::runtime_error("coatom is missing");
    const int second = static_cast<int>(found - strings.begin());
    const std::vector<int> frontier{top * size + second, second * size + top};
    const Support all(8, strings, strings, seeds);
    const Support reduced(8, strings, strings, frontier);
    std::cout << "raw_seeds=" << seeds.size() << " frontier_seeds=" << frontier.size() << '\n';
    for (int depth : {0, 1}) {
        auto generate = [&](const Support& support, const char* label) {
            std::vector<int> result;
            auto cursor = support.cursor(depth);
            int a, b;
            bool in_support;
            while (cursor.next(a, b, in_support))
                result.push_back(a * size + b);
            const auto counts = cursor.counts();
            std::cout << label << " depth=" << depth << " products=" << result.size()
                      << " prefixes=" << counts.prefixes << " words=" << counts.words << '\n';
            std::sort(result.begin(), result.end());
            return result;
        };
        const auto original = generate(all, "original");
        const auto compact = generate(reduced, "frontier");
        if (original != compact || compact.size() != (depth == 0 ? 4899u : 4900u))
            throw std::runtime_error("frontier changed the exact closure");
        const bool contains_top =
            std::binary_search(compact.begin(), compact.end(), top * size + top);
        if (contains_top != (depth == 1))
            throw std::runtime_error("wrong maximal-pair membership");
    }
    std::cout << "exact_closure_parity=PASS\n";
}
