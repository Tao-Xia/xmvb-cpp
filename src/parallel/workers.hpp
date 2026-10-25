#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include "core/timing.hpp"

namespace xmvb::parallel {

/**
 * @brief Opt-in per-worker loop wall time, excluding the following barrier.
 *
 * Storage is one cache line per requested worker, not a copy of operator data.
 * Finish is called by the owner after all workers have completed. Ratios
 * describe loop imbalance/start skew, not cache misses or useful CPU time.
 */
class WorkerLoopTimes {
  public:
    explicit WorkerLoopTimes(int workers)
        : samples_(xmvb::profiling::phase_totals().enabled ? workers : 0) {}

    void start(int worker) {
        if (!samples_.empty())
            samples_[worker].start = std::chrono::steady_clock::now();
    }
    void stop(int worker) {
        if (!samples_.empty())
            samples_[worker].elapsed =
                std::max<long long>(1,
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - samples_[worker].start)
                                        .count());
    }
    void finish(xmvb::profiling::Phase phase) const {
        unsigned long long minimum = std::numeric_limits<unsigned long long>::max();
        unsigned long long maximum = 0, sum = 0, count = 0;
        auto first = std::chrono::steady_clock::time_point::max();
        auto last_start = std::chrono::steady_clock::time_point::min();
        auto last_end = last_start;
        for (const auto& sample : samples_) {
            if (sample.elapsed == 0)
                continue;
            minimum = std::min(minimum, sample.elapsed);
            maximum = std::max(maximum, sample.elapsed);
            sum += sample.elapsed;
            ++count;
            first = std::min(first, sample.start);
            last_start = std::max(last_start, sample.start);
            last_end = std::max(last_end, sample.start + std::chrono::nanoseconds(sample.elapsed));
        }
        if (count == 0)
            return;
        auto& totals = xmvb::profiling::phase_totals();
        const auto index = static_cast<std::size_t>(phase);
        const auto critical =
            std::chrono::duration_cast<std::chrono::nanoseconds>(last_end - first).count();
        const auto skew =
            std::chrono::duration_cast<std::chrono::nanoseconds>(last_start - first).count();
        totals.worker_calls[index].fetch_add(1, std::memory_order_relaxed);
        totals.worker_min[index].fetch_add(minimum, std::memory_order_relaxed);
        totals.worker_max[index].fetch_add(maximum, std::memory_order_relaxed);
        totals.worker_sum[index].fetch_add(sum, std::memory_order_relaxed);
        totals.worker_capacity[index].fetch_add(count * critical, std::memory_order_relaxed);
        totals.worker_skew[index].fetch_add(skew, std::memory_order_relaxed);
        totals.worker_critical[index].fetch_add(critical, std::memory_order_relaxed);
    }

  private:
    struct alignas(64) Sample {
        std::chrono::steady_clock::time_point start;
        unsigned long long elapsed = 0;
    };
    std::vector<Sample> samples_;
};

} // namespace xmvb::parallel
