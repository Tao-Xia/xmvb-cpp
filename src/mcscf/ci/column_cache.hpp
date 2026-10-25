#pragma once

#include <algorithm>
#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace xmvb::mcscf {

/** Exact-demand column lifetime shared by host and device executors. Payloads
 * report bytes(); eviction consumes a partial image before releasing storage.
 * The coordinator is single-owner. Borrowed hits must be consumed before any
 * insertion that might evict them.
 */
template <class Tile> class ColumnCache {
  public:
    using Build = std::function<Tile(int)>;
    using Consume = std::function<void(int, Tile&)>;
    using Price = std::function<std::size_t(int)>;
    ColumnCache(std::size_t bytes, Build build, Consume consume = {}, Price price = {})
        : budget_(std::max<std::size_t>(1, bytes)), build_(std::move(build)),
          consume_(std::move(consume)), price_(std::move(price)) {}
    bool contains(int key) const {
        return index_.contains(key);
    }
    Tile& get(int key) {
        const auto found = index_.find(key);
        if (found != index_.end()) {
            panels_.splice(panels_.begin(), panels_, found->second);
            return found->second->tile;
        }
        if (!price_)
            throw std::logic_error("cache miss requires a prebuild tile price");
        reserve(price_(key));
        return insert(key, build_(key));
    }
    /** Release old payload before a miss allocates. Oversize tiles evict all. */
    void reserve(std::size_t bytes) {
        while (!panels_.empty() && (bytes > budget_ || resident_ > budget_ - bytes))
            evict_one();
    }
    bool evict_one() {
        if (panels_.empty())
            return false;
        auto& last = panels_.back();
        resident_ -= last.tile.bytes();
        if (consume_)
            consume_(last.key, last.tile);
        index_.erase(last.key);
        panels_.pop_back();
        ++evictions;
        return true;
    }
    Tile& insert(int key, Tile tile) {
        const auto bytes = tile.bytes();
        reserve(bytes);
        panels_.push_front({key, std::move(tile)});
        resident_ += bytes;
        peak_bytes = std::max(peak_bytes, resident_);
        index_[key] = panels_.begin();
        return panels_.front().tile;
    }
    void release(int key) {
        const auto found = index_.find(key);
        if (found == index_.end())
            return;
        auto panel = found->second;
        resident_ -= panel->tile.bytes();
        if (consume_)
            consume_(panel->key, panel->tile);
        panels_.erase(panel);
        index_.erase(found);
        ++releases;
    }
    void finish() {
        if (consume_)
            for (auto& panel : panels_)
                consume_(panel.key, panel.tile);
        panels_.clear();
        index_.clear();
        resident_ = 0;
    }
    std::size_t evictions = 0, peak_bytes = 0, releases = 0;

  private:
    struct Entry {
        int key;
        Tile tile;
    };
    std::size_t budget_, resident_ = 0;
    Build build_;
    Consume consume_;
    Price price_;
    std::list<Entry> panels_;
    std::unordered_map<int, typename std::list<Entry>::iterator> index_;
};

} // namespace xmvb::mcscf
