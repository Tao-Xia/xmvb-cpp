#include "metric_checks.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#include "input/loading/generated_structures.hpp"
#include "mcscf/ci/combinations.hpp"
#include "vbscf/structures/expansion/expander.hpp"

namespace {
using namespace metric_audit;
using xmvb::mcscf::detail::spin_choose;
std::uint64_t structures(int n, int r) {
    const int k = n / 2 - r;
    return spin_choose(n, r) * spin_choose(n - r, r) * spin_choose(2 * k, k) / (k + 1);
}
struct Input {
    Id p = 0, q = 0;
    std::uint64_t structures = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seeds;
};
Input input(Bdd& b, int n, const std::string& name) {
    Input out;
    const int k = n / 2;
    if (name == "cov" || name == "ion02" || name == "full") {
        const int r = name == "cov" ? 0 : name == "ion02" ? std::min(2, k) : k;
        out.p = family(b, n, k, k, r);
        out.q = family(b, n, k, k, std::min(k, r + 2));
        for (int d = 0; d <= r; ++d)
            out.structures += structures(n, d);
        const auto expected = [&](int maxr) {
            std::uint64_t count = 0;
            for (int d = 0; d <= maxr; ++d)
                count += spin_choose(n, d) * spin_choose(n - d, d) * spin_choose(n - 2 * d, k - d);
            return count;
        };
        if (b.count(out.p) != expected(r) || b.count(out.q) != expected(std::min(k, r + 2)))
            throw std::runtime_error("family support formula mismatch");
        // Independent generic excitation closure checks the family shortcut.
        if (n <= 12 && closure(b, out.p, n) != out.q)
            throw std::runtime_error("family closure identity mismatch");
        return out;
    }
    if (name != "cov3" && name != "mixed20")
        throw std::invalid_argument("unknown selected space");
    if (n > 14)
        throw std::invalid_argument("expanded subset fixtures are limited to N<=14");
    xmvb::vb::InputDeckMetadata metadata;
    metadata.declared_active_orbitals = n;
    metadata.declared_active_electrons = n;
    metadata.declared_spin_multiplicity = 1;
    metadata.structure_class_keyword = name == "cov3" ? "COV" : "ION(0:2)";
    const auto raw = xmvb::vb::build_generated_raw_structures_from_structure_class(metadata, n);
    const int count = name == "cov3" ? 3 : 20;
    std::vector<std::size_t> ids;
    for (int i = 0; i < count; ++i)
        ids.push_back(static_cast<std::uint64_t>(i) * (raw.n_structures - 1) / (count - 1));
    const auto expanded = xmvb::vb::StructureExpander().expand_subset(raw, ids);
    out.structures = expanded.n_structures;
    for (std::size_t pair = 0; pair < expanded.pair_alpha.size(); ++pair) {
        const auto a = expanded.alpha_strings.mask(expanded.pair_alpha[pair]);
        const auto beta = expanded.beta_strings.mask(expanded.pair_beta[pair]);
        out.seeds.emplace_back(a, beta);
        out.p = b.either(out.p, b.point(interleave(a, beta, n)));
    }
    std::sort(out.seeds.begin(), out.seeds.end());
    out.seeds.erase(std::unique(out.seeds.begin(), out.seeds.end()), out.seeds.end());
    if (b.count(out.p) != out.seeds.size())
        throw std::runtime_error("selected incidence union mismatch");
    out.q = closure(b, out.p, n);
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            std::cout
                << "Exact graph, unequal-spin, vacuum and value/gradient/HVP jet checks pass\n";
            return 0;
        }
        if (argc != 7)
            throw std::invalid_argument(
                "usage: test_metric_order N SPACE ORDER SPLIT_BITS PLAN_GiB SECONDS");
        const int n = std::stoi(argv[1]), splits = std::stoi(argv[4]);
        if (n < 4 || n > 24 || n % 2 || splits < 0 || splits > 2)
            throw std::invalid_argument("census extent outside diagnostic scope");
        const auto begin = std::chrono::steady_clock::now();
        Bdd b(2 * n, std::stod(argv[5]) * (1ULL << 30), std::stod(argv[6]));
        auto source = input(b, n, argv[2]);
        const auto np = b.count(source.p), nq = b.count(source.q);
        Id beta_support = source.q;
        for (int i = 0; i < n; ++i)
            beta_support =
                b.either(b.fix(beta_support, 2 * i, false), b.fix(beta_support, 2 * i, true));
        const auto beta_columns = b.count(beta_support) >> n;
        Audit audit(b, source.p, source.q, schedule(n, argv[3]));
        const auto full = audit.census(audit.q);
        audit.partition(splits);
        Counts replay;
        for (std::size_t i = 0; i < audit.chunks.size(); ++i)
            replay += audit.census(audit.chunks[i]);
        if (replay.source != full.source || replay.updates < full.updates ||
            replay.scales < full.scales)
            throw std::runtime_error("disjoint source partition accounting mismatch");
        const auto u = spin_choose(n, n / 2), links = u * (n / 2) * (n - n / 2) / 2;
        const auto column_updates = 4 * links * beta_columns, column_scales = 4 * u * beta_columns;
        const long double column_dots = static_cast<long double>(np) * beta_columns;
        const auto print = [&](const char* tag, const Counts& c) {
            const double peak_gib = 8.0 * c.peak_union / (1ULL << 30);
            std::cout
                << tag << ',' << n << ',' << argv[2] << ',' << argv[3] << ',' << source.structures
                << ',' << np << ',' << nq << ',' << c.updates << ',' << c.scales << ','
                << c.peak_nodes << ',' << c.peak_union << ',' << peak_gib << ','
                << (tag == std::string("whole") ? 1 : audit.chunks.size()) << ','
                << static_cast<double>(c.updates) / full.updates << ','
                << static_cast<long double>(2 * c.updates + c.scales) /
                       (2 * column_updates + column_scales + 2 * column_dots)
                << ',' << b.memory.peak / static_cast<double>(1ULL << 30) << ',' << b.peak_nodes
                << ',' << b.collections << ','
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                << '\n';
        };
        std::cout << std::setprecision(17)
                  << "kind,n,space,order,structures,P,Q,updates,scales,peak_nodes,peak_union,"
                     "frontier_GiB,source_batches,edge_multiplicity,work_vs_column,planner_GiB,BDD_"
                     "peak_nodes,GC,census_seconds\n";
        print("whole", full);
        print("partition", replay);
        std::cout << "column_model," << n << ',' << column_updates << ',' << column_scales << ','
                  << column_dots << ',' << beta_columns << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "metric-order census: " << e.what() << '\n';
        return 1;
    }
}
