/** @file Exact channel-support census, not a molecular timing benchmark.
 * Streams the production support cursor, retaining only sampled nodes. The
 * stride controls diagnostic sampling, never electronic-structure screening.
 */
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "mcscf/ci/product_support.hpp"

using Mask = std::uint64_t;
struct Node {
    int a, b;
    bool supported;
};
struct Links {
    int source, pair;
};

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::stoi(argv[1]) : 16;
    const int stride = argc > 2 ? std::stoi(argv[2]) : 257;
    if (n < 2 || n > 16 || n % 2 || stride < 1)
        throw std::invalid_argument("even n <=16; positive stride");
    const auto start = std::chrono::steady_clock::now();
    const int pairs = n * (n + 1) / 2;
    const auto pair = [](int p, int q) {
        if (p < q)
            std::swap(p, q);
        return p * (p + 1) / 2 + q;
    };
    std::vector<Mask> strings;
    std::vector<int> address(Mask{1} << n, -1);
    for (Mask mask = 0; mask < (Mask{1} << n); ++mask)
        if (std::popcount(mask) == n / 2) {
            address[mask] = strings.size();
            strings.push_back(mask);
        }
    std::vector<std::vector<Links>> links(strings.size());
    for (int a = 0; a < static_cast<int>(strings.size()); ++a)
        for (int p = 0; p < n; ++p)
            if ((strings[a] >> p) & 1)
                for (int q = 0; q < n; ++q)
                    if (!((strings[a] >> q) & 1))
                        links[a].push_back(
                            {address[strings[a] ^ (Mask{1} << p) ^ (Mask{1} << q)], pair(p, q)});
    const xmvb::mcscf::ProductSupport support(n, strings, strings);
    auto cursor = support.cursor(1);
    std::vector<Node> nodes;
    std::uint64_t total = 0, supported = 0;
    int a, b;
    bool inside;
    while (cursor.next(a, b, inside))
        if (a <= b) {
            if (total % stride == 0)
                nodes.push_back({a, b, inside});
            ++total;
            supported += inside;
        }
    struct Count {
        int input, output, envelope, group;
        bool valid;
    };
    std::vector<Count> counts(nodes.size());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto node = nodes[i];
        std::vector<unsigned char> input(pairs, 0), output(pairs, 0), envelope(pairs, 0);
        for (int p = 0; p < n; ++p)
            if (((strings[node.a] | strings[node.b]) >> p) & 1) {
                output[pair(p, p)] = 1;
                input[pair(p, p)] = node.supported;
            }
        for (const auto edge : links[node.a]) {
            output[edge.pair] = 1;
            if (support.contains(edge.source, node.b))
                input[edge.pair] = 1;
        }
        for (const auto edge : links[node.b]) {
            output[edge.pair] = 1;
            if (support.contains(node.a, edge.source))
                input[edge.pair] = 1;
        }
        // Outside a complete covalent upper-triangular support, every negative
        // prefix is -1. A single raising excitation must cross all such prefixes.
        int first = n, last = -1, prefix = 0;
        for (int p = 0; p < n; ++p) {
            prefix += static_cast<int>((strings[node.a] >> p) & 1) +
                      static_cast<int>((strings[node.b] >> p) & 1) - 1;
            if (prefix < 0) {
                first = std::min(first, p);
                last = p;
            }
        }
        if (node.supported)
            std::fill(envelope.begin(), envelope.end(), 1);
        else
            for (int p = 0; p <= first; ++p)
                for (int q = last + 1; q < n; ++q)
                    envelope[pair(p, q)] = 1;
        auto& count = counts[i];
        count = {0, 0, 0, node.supported ? -1 : first * n + last, true};
        for (int k = 0; k < pairs; ++k) {
            count.input += input[k];
            count.output += output[k];
            count.envelope += envelope[k];
            count.valid &= !input[k] || envelope[k];
        }
    }
    std::uint64_t input = 0, output = 0, envelope = 0, product = 0, gram = 0, envelope_gram = 0;
    std::map<int, std::uint64_t> histogram, groups;
    for (const auto count : counts) {
        if (!count.valid)
            throw std::runtime_error("prefix envelope omitted a reachable channel");
        input += count.input;
        output += count.output;
        envelope += count.envelope;
        product += count.input * count.output;
        gram += count.input * count.input;
        envelope_gram += count.envelope * count.envelope;
        ++histogram[count.input];
        ++groups[count.group];
    }
    const double samples = nodes.size(), dense = samples * pairs * pairs;
    std::cout.precision(12);
    std::cout
        << "n,strings,targets,supported_targets,sample_stride,samples,pairs,mean_input,mean_output,"
           "mean_envelope,sigma_input_ratio,sigma_both_ratio,gram_ratio,envelope_sigma_ratio,"
           "envelope_gram_ratio,groups,seconds\n"
        << n << ',' << strings.size() << ',' << total << ',' << supported << ',' << stride << ','
        << nodes.size() << ',' << pairs << ',' << input / samples << ',' << output / samples << ','
        << envelope / samples << ',' << input * pairs / dense << ',' << product / dense << ','
        << gram / dense << ',' << envelope * pairs / dense << ',' << envelope_gram / dense << ','
        << groups.size() << ','
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
    std::cout << "input_channels,sampled_nodes\n";
    for (const auto [channels, count] : histogram)
        std::cout << channels << ',' << count << '\n';
}
