#pragma once

#include "metric_bdd.hpp"

#include <array>
#include <functional>
#include <map>
#include <string>
#include <tuple>

namespace metric_audit {
struct Gate {
    int spin, removed = -1, inserted = -1;
    bool diagonal() const {
        return removed < 0;
    }
    int from() const {
        return 2 * removed + spin;
    }
    int to() const {
        return 2 * inserted + spin;
    }
    Gate adjoint() const {
        return {spin, inserted, removed};
    }
};

inline std::vector<Gate> upper(int n, int spin) {
    std::vector<Gate> result{{spin}};
    for (int q = 1; q < n; ++q)
        for (int p = q - 1; p >= 0; --p)
            result.push_back({spin, q, p});
    return result;
}
inline std::vector<Gate> stage(int n, int spin, bool adjoint) {
    auto v = upper(n, spin);
    if (adjoint) {
        std::reverse(v.begin(), v.end());
        for (auto& g : v)
            g = g.adjoint();
    }
    return v;
}
inline std::vector<Gate> schedule(int n, const std::string& name) {
    static const std::map<std::string, std::array<int, 4>> orders{{"aabb", {0, 1, 2, 3}},
                                                                  {"abab", {0, 2, 1, 3}},
                                                                  {"abba", {0, 2, 3, 1}},
                                                                  {"bbaa", {2, 3, 0, 1}},
                                                                  {"baba", {2, 0, 3, 1}},
                                                                  {"baab", {2, 0, 1, 3}}};
    std::vector<Gate> out;
    if (auto i = orders.find(name); i != orders.end()) {
        for (int s : i->second) {
            auto v = stage(n, s / 2, s % 2);
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    }
    if (name != "zip" && name != "qwave")
        throw std::invalid_argument("unknown schedule");
    out = {{0}, {1}};
    for (int q = 1; q < n; ++q) {
        if (name == "qwave")
            for (int spin = 0; spin < 2; ++spin)
                for (int p = q - 1; p >= 0; --p)
                    out.push_back({spin, q, p});
        else
            for (int p = q - 1; p >= 0; --p)
                for (int spin = 0; spin < 2; ++spin)
                    out.push_back({spin, q, p});
    }
    auto primal = out;
    for (auto i = primal.rbegin(); i != primal.rend(); ++i)
        out.push_back(i->adjoint());
    return out;
}

/** Complete fixed-particle family with at most max_double doubly occupied sites. */
inline Id family(Bdd& b, int n, int na, int nb, int max_double) {
    std::map<std::tuple<int, int, int, int>, Id> memo;
    const auto build = [&](auto&& self, int i, int a, int beta, int d) -> Id {
        if (a < 0 || beta < 0 || d < 0 || a > n - i || beta > n - i)
            return 0;
        if (i == n)
            return a == 0 && beta == 0;
        const auto key = std::tuple{i, a, beta, d};
        if (auto p = memo.find(key); p != memo.end())
            return p->second;
        Id child[2][2];
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                child[x][y] = self(self, i + 1, a - x, beta - y, d - x * y);
        const auto lo = b.make(2 * i + 1, child[0][0], child[0][1]);
        const auto hi = b.make(2 * i + 1, child[1][0], child[1][1]);
        const auto result = b.make(2 * i, lo, hi);
        memo.emplace(key, result);
        return result;
    };
    return build(build, 0, na, nb, max_double);
}

inline Id closure(Bdd& b, Id source, int n) {
    auto result = source;
    for (int depth = 0; depth < 2; ++depth) {
        const auto base = result;
        for (int spin = 0; spin < 2; ++spin)
            for (int q = 0; q < n; ++q)
                for (int p = 0; p < n; ++p)
                    if (q != p)
                        result = b.either(result, b.image(base, 2 * q + spin, 2 * p + spin));
    }
    return result;
}

struct Counts {
    std::uint64_t updates = 0, scales = 0, peak_nodes = 0, peak_union = 0, layer_nodes = 0;
    std::uint64_t source = 0, target = 0;
    Counts& operator+=(const Counts& b) {
        updates += b.updates;
        scales += b.scales;
        layer_nodes += b.layer_nodes;
        peak_nodes = std::max(peak_nodes, b.peak_nodes);
        peak_union = std::max(peak_union, b.peak_union);
        source += b.source;
        target += b.target;
        return *this;
    }
};

/** A full census or an exact disjoint input-partition replay of the same action. */
class Audit {
  public:
    Bdd& b;
    Id p, q;
    std::vector<Gate> gates;
    std::vector<Id> needed, chunks;
    Audit(Bdd& diagram, Id selected, Id input, std::vector<Gate> order)
        : b(diagram), p(selected), q(input), gates(std::move(order)), needed(gates.size() + 1) {
        needed.back() = p;
        for (std::size_t i = gates.size(); i-- > 0;) {
            const auto g = gates[i].adjoint();
            needed[i] = g.diagonal() ? needed[i + 1] : b.step(needed[i + 1], g.from(), g.to());
            maybe_collect();
        }
    }
    void maybe_collect(Id* s = nullptr, Id* j = nullptr) {
        if (b.needs_collection())
            collect(s, j);
    }
    void collect(Id* s = nullptr, Id* j = nullptr) {
        std::vector<Id*> roots{&p, &q};
        for (auto& r : needed)
            roots.push_back(&r);
        for (auto& r : chunks)
            roots.push_back(&r);
        if (s)
            roots.push_back(s);
        if (j)
            roots.push_back(j);
        b.collect(roots);
    }
    void partition(int beta_bits) {
        chunks = {q};
        for (int i = 0; i < beta_bits; ++i) {
            std::vector<Id> next;
            for (auto c : chunks)
                for (bool bit : {false, true}) {
                    const auto part = b.both(c, b.literal(2 * i + 1, bit));
                    if (part)
                        next.push_back(part);
                }
            chunks = std::move(next);
            maybe_collect();
        }
    }
    Counts census(Id input) {
        Id s = input, j = b.both(s, needed.front());
        Counts result;
        result.source = b.count(j);
        result.target = b.count(b.both(p, needed.back()));
        result.peak_nodes = b.count(j);
        result.peak_union = result.peak_nodes;
        for (std::size_t i = 0; i < gates.size(); ++i) {
            maybe_collect(&s, &j);
            const auto g = gates[i];
            if (g.diagonal())
                result.scales += b.count(b.both(s, needed[i + 1]));
            else
                result.updates += b.edges(s, needed[i + 1], g.from(), g.to());
            const Id next = g.diagonal() ? s : b.step(s, g.from(), g.to());
            const Id nextj = b.both(next, needed[i + 1]);
            const auto nodes = b.count(nextj);
            result.layer_nodes += nodes;
            result.peak_nodes = std::max(result.peak_nodes, nodes);
            result.peak_union = std::max(result.peak_union, b.count(b.either(j, nextj)));
            s = next;
            j = nextj;
        }
        result.target = b.count(j);
        return result;
    }
};
} // namespace metric_audit
