#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>

#include "core/slab.hpp"
#include "core/workspace.hpp"

namespace xmvb::mcscf::detail {

/** @brief Exclusive scratch lease; cache one idle slab, never one per thread.
 * Concurrent or recursive actions still own disjoint buffers. Earlier returns
 * release their buffer while another lease is active, so the idle cache cannot
 * add a slab to the concurrent working set. Only pointer ownership is moved.
 */
class ChannelWorkspace {
  struct Storage {
    workspace::Reservation residency;
    workspace::Temporary temporary;
    workspace::Slab data;
    std::size_t capacity=0,preferred=0;
    void swap(Storage& other) noexcept {
      data.swap(other.data);
      std::swap(capacity,other.capacity);
      std::swap(preferred,other.preferred);
      std::swap(temporary,other.temporary);
      std::swap(residency,other.residency);
    }
    void clear() {
      data.reset();capacity=0;
      temporary.resize(0);residency.try_resize(0);
    }
  };
  struct Pool {
    std::mutex mutex;
    Storage idle;
    std::size_t active=0;
    ~Pool() { workspace::set_reusable(0,nullptr); }
  };
  static Pool& pool() { static Pool shared; return shared; }
  static void release_idle() {
    auto& shared=pool();std::lock_guard lock(shared.mutex);
    workspace::set_reusable(0,release_idle);
    shared.idle.clear();
  }
  inline static thread_local ChannelWorkspace* action_=nullptr;
  Storage owned_;
  Storage* storage_=&owned_;
  ChannelWorkspace* lender_=nullptr;
  bool in_use_=false,reused_=false;
  std::size_t budget_,allocations_=0;

public:
  explicit ChannelWorkspace(std::size_t budget,bool action_owner=false):budget_(budget) {
    if (!action_owner && action_ && !action_->in_use_) {
      lender_=action_;lender_->in_use_=true;storage_=lender_->storage_;
    } else {
      auto& shared=pool();
      std::lock_guard lock(shared.mutex);
      workspace::set_reusable(0,release_idle);
      ++shared.active;owned_.swap(shared.idle);
    }
    if (storage_->capacity>budget_/sizeof(double)) {
      storage_->data.resize(budget_,budget_);
      storage_->capacity=budget_/sizeof(double);
      storage_->temporary.resize(storage_->capacity*sizeof(double));
      if (storage_->residency.bytes()) storage_->residency.try_resize(storage_->capacity*sizeof(double));
    }
    reused_=storage_->data.get()!=nullptr;
  }
  ~ChannelWorkspace() {
    if (lender_) { lender_->in_use_=false;return; }
    if (owned_.residency.bytes()) owned_.residency.try_resize(bytes());
    auto& shared=pool();
    std::lock_guard lock(shared.mutex);
    if (--shared.active==0 && owned_.capacity<=budget_/sizeof(double))
      owned_.swap(shared.idle);
    workspace::set_reusable(shared.idle.residency.bytes(),release_idle);
  }
  ChannelWorkspace(const ChannelWorkspace&)=delete;
  ChannelWorkspace& operator=(const ChannelWorkspace&)=delete;
  static ChannelWorkspace* bind(ChannelWorkspace* value) {
    return std::exchange(action_,value);
  }
  workspace::Reservation& residency() { return storage_->residency; }
  static void limit_action(std::size_t bytes) {
    if (action_ && !action_->in_use_ && action_->bytes()>bytes) {
      auto& slab=*action_->storage_;
      slab.data.resize(bytes,bytes);
      slab.capacity=bytes/sizeof(double);
      slab.temporary.resize(slab.capacity*sizeof(double));
      if (slab.residency.bytes()) slab.residency.try_resize(slab.capacity*sizeof(double));
    }
  }
  void retain_idle() {
    // The action still owns this capacity. Transfer only the surviving slab;
    // unused capacity is returned when the outer action scope is destroyed.
    if (!storage_->residency.try_resize(bytes())) storage_->clear();
  }
  void reserve(std::size_t required,std::size_t payload) {
    if (required>budget_/sizeof(double))
      throw std::runtime_error("CI workspace resource_limit: sigma slab exceeds its partition");
    const auto data_bytes=required*sizeof(double);
    const auto overhead=payload>data_bytes ? payload-data_bytes:0;
    if (overhead>budget_ || data_bytes>budget_-overhead)
      throw std::runtime_error("CI workspace resource_limit: sigma tile exceeds its partition");
    const auto allowed=(budget_-overhead)/sizeof(double);
    storage_->preferred=std::max(storage_->preferred,required);
    if (required<=storage_->capacity && storage_->capacity<=allowed) return;
    // Reuse the largest observed requirement that fits this tile's actual
    // data+index allowance. Keep this size history even when data is evicted.
    const auto capacity=std::min(storage_->preferred,allowed);
    if (!lender_ && !storage_->residency.try_resize(capacity*sizeof(double)))
      throw std::runtime_error("CI workspace resource_limit: sigma reservation failed");
    const bool allocated=storage_->data.resize(capacity*sizeof(double),budget_);
    storage_->capacity=capacity;
    storage_->temporary.resize(capacity*sizeof(double));
    allocations_+=allocated;
  }
  double* get() const noexcept { return storage_->data.get(); }
  double& operator[](std::size_t index) const noexcept { return storage_->data.get()[index]; }
  bool reused() const noexcept { return reused_; }
  std::size_t allocations() const noexcept { return allocations_; }
  std::size_t bytes() const noexcept { return storage_->capacity*sizeof(double); }
};

} // namespace xmvb::mcscf::detail
