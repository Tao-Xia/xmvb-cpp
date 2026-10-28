#pragma once

#include "metric_schedule.hpp"

#include <bit>
#include <cmath>
#include <set>
#include <unordered_map>

namespace metric_audit {
inline std::uint64_t interleave(std::uint64_t a, std::uint64_t beta, int n) {
    std::uint64_t out = 0;
    for (int i = 0; i < n; ++i)
        out |= ((a >> i) & 1) << (2 * i), out |= ((beta >> i) & 1) << (2 * i + 1);
    return out;
}
inline std::uint64_t spin_mask(std::uint64_t state, int spin, int n) {
    std::uint64_t out = 0;
    for (int i = 0; i < n; ++i)
        out |= ((state >> (2 * i + spin)) & 1) << i;
    return out;
}
inline bool source_of(std::uint64_t state, Gate g) {
    return ((state >> g.from()) & 1) && !((state >> g.to()) & 1);
}
inline std::uint64_t moved(std::uint64_t state, Gate g) {
    return state ^ (std::uint64_t{1} << g.from()) ^ (std::uint64_t{1} << g.to());
}
struct Jet {
    double v = 0, d = 0, dd = 0;
};
inline Jet add(Jet a, Jet b) {
    return {a.v + b.v, a.d + b.d, a.dd + b.dd};
}
inline Jet multiply(Jet a, Jet b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d, a.dd * b.v + 2 * a.d * b.d + a.v * b.dd};
}
inline Jet coefficient(std::uint64_t state, Gate g, int n) {
    if (g.diagonal()) {
        auto mask = spin_mask(state, g.spin, n);
        Jet result{1, 0, 0};
        for (int i = 0; i < n; ++i)
            if (mask & (std::uint64_t{1} << i))
                result = multiply(result, {1 + 0.01 * (i + 1), 0.002, 0});
        return result;
    }
    const int p = std::min(g.removed, g.inserted), q = std::max(g.removed, g.inserted);
    const auto between = spin_mask(state, g.spin, n) &
                         (((std::uint64_t{1} << q) - 1) ^ ((std::uint64_t{1} << (p + 1)) - 1));
    const double sign = std::popcount(between) % 2 ? -1 : 1;
    return {sign * ((p + q) % 3 == 0 ? 0 : 0.01 * (p + q + 1)), sign * 0.003 * (q - p), 0};
}
using Values = std::unordered_map<std::uint64_t, Jet>;
inline Values numeric(const Values& input,
                      const std::vector<Gate>& gates,
                      int n,
                      const Bdd* b = nullptr,
                      const std::vector<Id>* needed = nullptr) {
    auto value = input;
    for (std::size_t i = 0; i < gates.size(); ++i) {
        const auto g = gates[i];
        Values next;
        for (auto [state, x] : value) {
            if (g.diagonal())
                next[state] = multiply(coefficient(state, g, n), x);
            else {
                next[state] = add(next[state], x);
                if (source_of(state, g)) {
                    const auto target = moved(state, g);
                    next[target] = add(next[target], multiply(coefficient(state, g, n), x));
                }
            }
        }
        if (needed)
            for (auto it = next.begin(); it != next.end();) {
                if (!b->contains((*needed)[i + 1], it->first))
                    it = next.erase(it);
                else
                    ++it;
            }
        value = std::move(next);
    }
    return value;
}
inline std::set<std::uint64_t> set_step(const std::set<std::uint64_t>& input, Gate g) {
    auto out = input;
    if (!g.diagonal())
        for (auto v : input)
            if (source_of(v, g))
                out.insert(moved(v, g));
    return out;
}
inline std::set<std::uint64_t> intersection(const std::set<std::uint64_t>& a,
                                            const std::set<std::uint64_t>& b) {
    std::set<std::uint64_t> c;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(c, c.end()));
    return c;
}
inline Counts direct_count(const std::set<std::uint64_t>& p,
                           const std::set<std::uint64_t>& q,
                           const std::vector<Gate>& gates) {
    std::vector<std::set<std::uint64_t>> back(gates.size() + 1);
    back.back() = p;
    for (std::size_t i = gates.size(); i-- > 0;)
        back[i] = set_step(back[i + 1], gates[i].adjoint());
    auto s = q, j = intersection(q, back.front());
    Counts c;
    c.source = j.size();
    c.peak_nodes = j.size();
    c.peak_union = j.size();
    for (std::size_t i = 0; i < gates.size(); ++i) {
        const auto g = gates[i];
        if (g.diagonal())
            c.scales += intersection(s, back[i + 1]).size();
        else
            for (auto state : s)
                if (source_of(state, g) && back[i + 1].contains(moved(state, g)))
                    ++c.updates;
        s = set_step(s, g);
        const auto next = intersection(s, back[i + 1]);
        auto united = j;
        united.insert(next.begin(), next.end());
        c.peak_union = std::max<std::uint64_t>(c.peak_union, united.size());
        c.peak_nodes = std::max<std::uint64_t>(c.peak_nodes, next.size());
        c.layer_nodes += next.size();
        j = next;
    }
    c.target = j.size();
    return c;
}
inline void self_test() {
    for (const auto space :
         {std::array{4, 2, 2}, std::array{5, 3, 2}, std::array{6, 3, 1}, std::array{4, 0, 2}}) {
        const auto [n, na, nb] = space;
        std::vector<std::uint64_t> universe;
        for (std::uint64_t a = 0; a < (1ULL << n); ++a)
            if (std::popcount(a) == na)
                for (std::uint64_t b = 0; b < (1ULL << n); ++b)
                    if (std::popcount(b) == nb)
                        universe.push_back(interleave(a, b, n));
        const std::set<std::uint64_t> p{
            universe.front(), universe[universe.size() / 2], universe.back()};
        std::set<std::uint64_t> q;
        for (auto target : universe)
            for (auto source : p)
                if (std::popcount(target ^ source) <= 4) {
                    q.insert(target);
                    break;
                }
        Values inputs;
        for (auto state : q)
            inputs[state] = {std::sin(0.17 * (state + 1)),
                             0.1 * std::cos(0.03 * state),
                             0.07 * std::sin(0.09 * state)};
        const auto reference = numeric(inputs, schedule(n, "aabb"), n);
        for (const auto& name : {"aabb", "abab", "abba", "bbaa", "baba", "baab", "zip", "qwave"}) {
            Bdd b(2 * n, 256ULL << 20, 30);
            Id pp = 0;
            for (auto v : p)
                pp = b.either(pp, b.point(v));
            auto qq = closure(b, pp, n);
            if (b.count(qq) != q.size())
                throw std::runtime_error("two-electron closure mismatch");
            for (auto v : q)
                if (!b.contains(qq, v))
                    throw std::runtime_error("closure membership mismatch");
            Audit audit(b, pp, qq, schedule(n, name));
            const auto actual = audit.census(audit.q), expected = direct_count(p, q, audit.gates);
            if (actual.updates != expected.updates || actual.scales != expected.scales ||
                actual.peak_nodes != expected.peak_nodes ||
                actual.peak_union != expected.peak_union ||
                actual.layer_nodes != expected.layer_nodes)
                throw std::runtime_error("BDD versus enumerated graph mismatch");
            // Force compaction independently of the planner budget and verify all
            // live roots, including disjoint source batches, survive remapping.
            audit.partition(2);
            audit.collect();
            const auto collected = audit.census(audit.q);
            if (collected.updates != expected.updates ||
                collected.peak_union != expected.peak_union)
                throw std::runtime_error("graph compaction changed census");
            Values summed;
            Counts replay;
            for (std::size_t batch = 0; batch < audit.chunks.size(); ++batch) {
                Values part;
                for (auto [state, value] : inputs)
                    if (b.contains(audit.chunks[batch], state))
                        part.emplace(state, value);
                const auto image = numeric(part, audit.gates, n, &b, &audit.needed);
                for (auto [state, value] : image)
                    summed[state] = add(summed[state], value);
                replay += audit.census(audit.chunks[batch]);
            }
            if (replay.source != expected.source || replay.updates < expected.updates)
                throw std::runtime_error("source partition census mismatch");
            const auto pruned = numeric(inputs, audit.gates, n, &b, &audit.needed);
            for (auto state : p) {
                const auto x = pruned.contains(state) ? pruned.at(state) : Jet{},
                           y = reference.at(state), z = summed.at(state);
                for (auto pair : {std::pair{x.v, y.v},
                                  std::pair{x.d, y.d},
                                  std::pair{x.dd, y.dd},
                                  std::pair{z.v, y.v},
                                  std::pair{z.d, y.d},
                                  std::pair{z.dd, y.dd}})
                    if (std::abs(pair.first - pair.second) >
                        3e-12 * std::max(1.0, std::abs(pair.second)))
                        throw std::runtime_error("projected value/gradient/HVP jet mismatch");
            }
        }
    }
}
} // namespace metric_audit
