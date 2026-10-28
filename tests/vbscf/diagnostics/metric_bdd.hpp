#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace metric_audit {
using Id = std::uint32_t;

/** Budget tracks allocator requests for planning, not numerical workspace. */
class Budget : public std::pmr::memory_resource {
  public:
    std::size_t current = 0, peak = 0;
    explicit Budget(std::size_t bytes) : limit(bytes) {}
    std::size_t limit;

  private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes > limit - current)
            throw std::runtime_error("planning_memory_limit");
        auto* p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        current += bytes;
        peak = std::max(peak, current);
        return p;
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        current -= bytes;
        std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};

/** Exact reduced decision diagram over interleaved alpha/beta occupation bits.
 * Its numerical payload is integer topology only; no product coefficient array.
 */
class Bdd {
  public:
    Budget memory;

  private:
    struct Node {
        int var;
        Id lo, hi;
    };
    struct Key {
        int var;
        Id lo, hi;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        std::size_t operator()(const Key& k) const noexcept {
            return (std::uint64_t(k.lo) * 0x9e3779b97f4a7c15ULL) ^
                   (std::uint64_t(k.hi) * 0xbf58476d1ce4e5b9ULL) ^ k.var;
        }
    };
    std::pmr::vector<Node> nodes;
    std::pmr::unordered_map<Key, Id, Hash> unique;
    std::pmr::unordered_map<std::uint64_t, Id> ors, ands, restrictions;
    std::pmr::unordered_map<Id, std::uint64_t> counts;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    double seconds;
    Id combine(Id a, Id b, bool is_or) {
        if (a > b)
            std::swap(a, b);
        if (a == b)
            return a;
        if (is_or) {
            if (a == 1 || b == 1)
                return 1;
            if (!a)
                return b;
        } else {
            if (!a)
                return 0;
            if (a == 1)
                return b;
        }
        const auto key = (std::uint64_t(a) << 32) | b;
        auto& memo = is_or ? ors : ands;
        if (auto p = memo.find(key); p != memo.end())
            return p->second;
        const int v = std::min(nodes[a].var, nodes[b].var);
        const auto na = nodes[a], nb = nodes[b];
        const auto lo = combine(na.var == v ? na.lo : a, nb.var == v ? nb.lo : b, is_or);
        const auto hi = combine(na.var == v ? na.hi : a, nb.var == v ? nb.hi : b, is_or);
        const auto result = make(v, lo, hi);
        memo.emplace(key, result);
        return result;
    }
    std::uint64_t below(Id id) {
        if (id < 2)
            return id;
        if (auto p = counts.find(id); p != counts.end())
            return p->second;
        const auto n = nodes[id];
        const auto result = (below(n.lo) << (nodes[n.lo].var - n.var - 1)) +
                            (below(n.hi) << (nodes[n.hi].var - n.var - 1));
        counts.emplace(id, result);
        return result;
    }

  public:
    int variables;
    std::size_t collections = 0, peak_nodes = 0;
    Bdd(int vars, std::size_t bytes, double max_seconds)
        : memory(bytes), nodes(&memory), unique(&memory), ors(&memory), ands(&memory),
          restrictions(&memory), counts(&memory), seconds(max_seconds), variables(vars) {
        nodes.push_back({vars, 0, 0});
        nodes.push_back({vars, 1, 1});
    }
    Id make(int v, Id lo, Id hi) {
        if (lo == hi)
            return lo;
        Key key{v, lo, hi};
        if (auto p = unique.find(key); p != unique.end())
            return p->second;
        if (nodes.size() % 16384 == 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
                seconds)
            throw std::runtime_error("census_time_limit");
        if (nodes.size() >= UINT32_MAX)
            throw std::runtime_error("planning_index_limit");
        const auto id = static_cast<Id>(nodes.size());
        nodes.push_back({v, lo, hi});
        unique.emplace(key, id);
        peak_nodes = std::max(peak_nodes, nodes.size());
        return id;
    }
    Id either(Id a, Id b) {
        return combine(a, b, true);
    }
    Id both(Id a, Id b) {
        return combine(a, b, false);
    }
    Id literal(int v, bool x) {
        return make(v, x ? 0 : 1, x ? 1 : 0);
    }
    Id fix(Id root, int var, bool value) {
        if (root < 2 || nodes[root].var > var)
            return root;
        const auto n = nodes[root];
        if (n.var == var)
            return value ? n.hi : n.lo;
        const auto key = (std::uint64_t(root) << 8) | (var << 1) | value;
        if (auto p = restrictions.find(key); p != restrictions.end())
            return p->second;
        const auto lo = fix(n.lo, var, value), hi = fix(n.hi, var, value);
        const auto result = make(n.var, lo, hi);
        restrictions.emplace(key, result);
        return result;
    }
    Id fix2(Id root, int a, bool x, int b, bool y) {
        return fix(fix(root, a, x), b, y);
    }
    Id point(std::uint64_t bits) {
        Id root = 1;
        for (int v = variables - 1; v >= 0; --v)
            root = make(v, (bits >> v) & 1 ? 0 : root, (bits >> v) & 1 ? root : 0);
        return root;
    }
    bool contains(Id root, std::uint64_t bits) const {
        while (root >= 2) {
            const auto n = nodes[root];
            root = (bits >> n.var) & 1 ? n.hi : n.lo;
        }
        return root;
    }
    std::uint64_t count(Id root) {
        return below(root) << nodes[root].var;
    }
    Id image(Id root, int removed, int inserted) {
        auto part = fix2(root, removed, true, inserted, false);
        part = both(part, literal(removed, false));
        return both(part, literal(inserted, true));
    }
    Id step(Id root, int removed, int inserted) {
        return either(root, image(root, removed, inserted));
    }
    std::uint64_t edges(Id source, Id needed, int removed, int inserted) {
        const auto a = fix2(source, removed, true, inserted, false);
        const auto b = fix2(needed, removed, false, inserted, true);
        return count(both(a, b)) / 4; // Two cofactored bits are free in the count.
    }
    /** Collect only between operations; callers enumerate all live root slots. */
    void collect(std::span<Id*> roots) {
        ors.clear();
        ands.clear();
        restrictions.clear();
        counts.clear();
        std::pmr::vector<std::uint8_t> live(nodes.size(), 0, &memory);
        const auto mark = [&](auto&& self, Id id) -> void {
            if (id < 2 || live[id])
                return;
            live[id] = 1;
            self(self, nodes[id].lo);
            self(self, nodes[id].hi);
        };
        for (auto* r : roots)
            mark(mark, *r);
        std::pmr::vector<Id> remap(nodes.size(), 0, &memory);
        remap[1] = 1;
        unique.clear();
        Id next = 2;
        for (Id old = 2; old < nodes.size(); ++old)
            if (live[old]) {
                const auto n = nodes[old];
                const Node v{n.var, remap[n.lo], remap[n.hi]};
                nodes[next] = v;
                remap[old] = next;
                unique.emplace(Key{v.var, v.lo, v.hi}, next++);
            }
        for (auto* r : roots)
            *r = remap[*r];
        nodes.resize(next);
        ++collections;
    }
    bool needs_collection() const {
        return memory.current > memory.limit / 3;
    }
};
} // namespace metric_audit
