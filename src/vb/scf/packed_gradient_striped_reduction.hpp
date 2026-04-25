#pragma once

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace xmvb::vb {

inline int packed_gradient_positive_env_override(
    const char* env_name,
    int default_value) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return default_value;
  }
  const int parsed_value = std::stoi(env_value);
  if (parsed_value <= 0) {
    throw std::invalid_argument(
        std::string(env_name) + " must be positive");
  }
  return parsed_value;
}

inline int packed_gradient_stripe_size() {
  return packed_gradient_positive_env_override(
      "XMVB_CPP_PACKED_GRADIENT_STRIPE_SIZE",
      8192);
}

inline int packed_gradient_stripe_cache_slots() {
  return packed_gradient_positive_env_override(
      "XMVB_CPP_PACKED_GRADIENT_STRIPE_CACHE_SLOTS",
      4);
}

/**
 * @brief Dense direct writer for serial packed-gradient accumulation.
 *
 * The matrix-form backward kernels write packed two-electron contributions as
 * `(packed_index, value)` updates. Serial paths can forward those updates
 * straight into the final dense vector through this lightweight adapter.
 */
class DensePackedGradientAccumulator {
public:
  explicit DensePackedGradientAccumulator(std::vector<double>* target)
      : target_(target) {
    if (target_ == nullptr) {
      throw std::invalid_argument("packed gradient target must not be null");
    }
  }

  void operator()(int packed_index, double value) const {
    if (value == 0.0) {
      return;
    }
    if (packed_index < 0 ||
        packed_index >= static_cast<int>(target_->size())) {
      throw std::out_of_range("packed gradient index is out of range");
    }
    (*target_)[packed_index] += value;
  }

private:
  std::vector<double>* target_ = nullptr;
};

/**
 * @brief Shared striped reducer for bounded-memory packed-gradient writes.
 *
 * Round 7 removes the `O(n_threads * N_2e)` pattern where every worker owns a
 * full dense packed-gradient vector. Each worker now keeps only a small cache
 * of dense stripes; once a stripe cache slot is evicted, the buffered updates
 * are reduced into the shared target under one stripe-local lock.
 */
class PackedGradientStripedReducer {
public:
  class ThreadLocalAccumulator {
  public:
    explicit ThreadLocalAccumulator(PackedGradientStripedReducer* reducer)
        : reducer_(reducer),
          stripe_indices_(
              reducer->cache_slots_,
              -1),
          stripe_buffers_(
              reducer->cache_slots_,
              std::vector<double>(
                  static_cast<std::size_t>(reducer->stripe_size_),
                  0.0)),
          touched_offsets_(reducer->cache_slots_) {
      if (reducer_ == nullptr) {
        throw std::invalid_argument("striped reducer must not be null");
      }
    }

    ~ThreadLocalAccumulator() {
      flush_all();
    }

    void operator()(int packed_index, double value) {
      add(packed_index, value);
    }

    void add(int packed_index, double value) {
      if (value == 0.0) {
        return;
      }
      if (packed_index < 0 ||
          packed_index >= static_cast<int>(reducer_->target_->size())) {
        throw std::out_of_range("packed gradient index is out of range");
      }

      const int stripe_index = packed_index / reducer_->stripe_size_;
      const int stripe_offset = packed_index % reducer_->stripe_size_;
      const int slot = acquire_slot(stripe_index);
      auto& stripe_buffer = stripe_buffers_[slot];
      double& buffered_value = stripe_buffer[stripe_offset];
      if (buffered_value == 0.0) {
        touched_offsets_[slot].push_back(stripe_offset);
      }
      buffered_value += value;
    }

    void flush_all() noexcept {
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        flush_slot(slot);
      }
    }

  private:
    int acquire_slot(int stripe_index) {
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        if (stripe_indices_[slot] == stripe_index) {
          return slot;
        }
      }
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        if (stripe_indices_[slot] < 0) {
          stripe_indices_[slot] = stripe_index;
          return slot;
        }
      }

      const int victim_slot = next_victim_slot_;
      next_victim_slot_ = (next_victim_slot_ + 1) % reducer_->cache_slots_;
      flush_slot(victim_slot);
      stripe_indices_[victim_slot] = stripe_index;
      return victim_slot;
    }

    void flush_slot(int slot) noexcept {
      const int stripe_index = stripe_indices_[slot];
      auto& touched_offsets = touched_offsets_[slot];
      if (stripe_index < 0 || touched_offsets.empty()) {
        stripe_indices_[slot] = -1;
        touched_offsets.clear();
        return;
      }

      const std::size_t stripe_begin =
          static_cast<std::size_t>(stripe_index) *
          static_cast<std::size_t>(reducer_->stripe_size_);
      const std::size_t stripe_length = std::min(
          static_cast<std::size_t>(reducer_->stripe_size_),
          reducer_->target_->size() - stripe_begin);
      std::lock_guard<std::mutex> guard(
          reducer_->stripe_mutexes_[stripe_index]);
      auto& stripe_buffer = stripe_buffers_[slot];
      for (const int stripe_offset : touched_offsets) {
        if (stripe_offset < 0 ||
            static_cast<std::size_t>(stripe_offset) >= stripe_length) {
          continue;
        }
        (*reducer_->target_)[stripe_begin + stripe_offset] +=
            stripe_buffer[stripe_offset];
        stripe_buffer[stripe_offset] = 0.0;
      }
      touched_offsets.clear();
      stripe_indices_[slot] = -1;
    }

    PackedGradientStripedReducer* reducer_ = nullptr;
    std::vector<int> stripe_indices_;
    std::vector<std::vector<double>> stripe_buffers_;
    std::vector<std::vector<int>> touched_offsets_;
    int next_victim_slot_ = 0;
  };

  explicit PackedGradientStripedReducer(
      std::vector<double>* target,
      int stripe_size = packed_gradient_stripe_size(),
      int cache_slots = packed_gradient_stripe_cache_slots())
      : target_(target),
        stripe_size_(stripe_size),
        cache_slots_(cache_slots),
        stripe_mutexes_(compute_stripe_count(target, stripe_size)) {
    if (target_ == nullptr) {
      throw std::invalid_argument("packed gradient target must not be null");
    }
    if (stripe_size_ <= 0 || cache_slots_ <= 0) {
      throw std::invalid_argument(
          "striped reducer parameters must be positive");
    }
  }

private:
  static std::size_t compute_stripe_count(
      const std::vector<double>* target,
      int stripe_size) {
    if (target == nullptr) {
      throw std::invalid_argument("packed gradient target must not be null");
    }
    if (stripe_size <= 0) {
      throw std::invalid_argument("packed gradient stripe_size must be positive");
    }
    return (target->size() + static_cast<std::size_t>(stripe_size) - 1) /
           static_cast<std::size_t>(stripe_size);
  }

  std::vector<double>* target_ = nullptr;
  int stripe_size_ = 0;
  int cache_slots_ = 0;
  std::vector<std::mutex> stripe_mutexes_;

  friend class ThreadLocalAccumulator;
};

}  // namespace xmvb::vb
