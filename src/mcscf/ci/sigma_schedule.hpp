#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "mcscf/ci/sigma_plan.hpp"

namespace xmvb::mcscf::detail {

/** @brief Scalar execution boundaries shared by one immutable spin-link family.
 * No target rows, halo, addresses, signs or numerical panels are retained here.
 * Each algebraic layout owns one stream, capped by the single-spin dimensions.
 */
class SigmaSchedule {
  public:
    enum class Kind { Image, Transition, Directional, Count };
    struct Key {
        std::shared_ptr<const char> support;
        SigmaTileBudget budget;
        int parity;
        bool release;
    };
    struct Stream {
        Key key;
        std::vector<SigmaTileBoundary> tiles;
        std::size_t peak_payload = 0;
        bool byte_limited = false, truncated = false;

        explicit Stream(Key value) : key(std::move(value)) {}
        void record(SigmaTileBoundary boundary, bool limited, std::size_t limit) {
            if (tiles.size() == limit) {
                truncated = true;
                return;
            }
            if (tiles.size() == tiles.capacity())
                tiles.reserve(std::min(limit, tiles.capacity() ? 2 * tiles.capacity() : 1));
            tiles.push_back(boundary);
            peak_payload = std::max(peak_payload, boundary.payload);
            byte_limited |= limited;
        }
        bool usable(const Key& next) const {
            auto geometry = key.budget;
            geometry.bytes = next.budget.bytes;
            return key.support == next.support && key.parity == next.parity &&
                   key.release == next.release && geometry == next.budget &&
                   peak_payload <= next.budget.bytes &&
                   !(byte_limited && next.budget.bytes > key.budget.bytes);
        }
        std::size_t bytes() const noexcept {
            return sizeof(Stream) + tiles.capacity() * sizeof(SigmaTileBoundary);
        }
    };

    std::shared_ptr<const Stream> acquire(Kind kind, const Key& key) const {
        std::lock_guard lock(mutex_);
        const auto& stream = streams_[static_cast<std::size_t>(kind)];
        return stream && stream->usable(key) ? stream : nullptr;
    }
    void publish(Kind kind, std::shared_ptr<const Stream> stream) {
        std::lock_guard lock(mutex_);
        streams_[static_cast<std::size_t>(kind)] = std::move(stream);
        std::size_t bytes = 0;
        for (const auto& value : streams_)
            if (value)
                bytes += value->bytes();
        bytes_.store(bytes, std::memory_order_relaxed);
    }
    std::size_t bytes() const noexcept {
        return bytes_.load(std::memory_order_relaxed);
    }

  private:
    mutable std::mutex mutex_;
    std::array<std::shared_ptr<const Stream>, static_cast<std::size_t>(Kind::Count)> streams_;
    std::atomic<std::size_t> bytes_{0};
};

} // namespace xmvb::mcscf::detail
