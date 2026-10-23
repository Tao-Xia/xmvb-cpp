#pragma once

#include "core/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xmvb::workspace {

class Reservation;
namespace detail { inline thread_local Reservation* current = nullptr; }

/** @brief Shared numerical payload allowance for CI/RI/DFT; not an RSS limit.
 * Actions reserve working capacity; retained panels keep their own charge.
 * Consumers share bounded partitions; idle scratch is reclaimable. Static AO
 * integrals, immutable excitation links and eigensolver vectors are separate.
 */
namespace detail {
inline std::atomic<std::size_t> limit{0};
inline std::atomic<std::size_t> retained{0};
inline std::atomic<std::size_t> temporary{0};
inline std::atomic<std::size_t> temporary_peak{0};
inline std::atomic<std::size_t> reusable{0};
inline std::atomic<void(*)()> reclaim{nullptr};
}

inline std::size_t limit() {
  auto bytes=detail::limit.load();
  if (!bytes) {
    const auto detected=memory::default_workspace();
    detail::limit.compare_exchange_strong(bytes,detected);
    bytes=detail::limit.load();
  }
  return bytes;
}
inline std::size_t retained() noexcept { return detail::retained.load(); }
inline bool has_scope() noexcept { return detail::current!=nullptr; }
/** One shared idle scratch pool. Its bytes remain charged until released;
 * admission may reclaim it when another owner needs that capacity. */
inline void set_reusable(std::size_t bytes,void(*release)()) noexcept {
  detail::reclaim.store(release);
  detail::reusable.store(bytes);
}
/** Discard idle pooled scratch at a quiescent planning boundary. */
inline void reclaim_idle() {
  if (has_scope()) throw std::logic_error("cannot reclaim workspace during an action");
  if (const auto release=detail::reclaim.load()) release();
}
inline std::size_t temporary_bytes() noexcept { return detail::temporary.load(); }
inline std::size_t peak_temporary_bytes() noexcept { return detail::temporary_peak.load(); }

/** Reset at a quiescent measurement boundary; live pooled storage remains. */
inline void reset_temporary_peak() noexcept {
  detail::temporary_peak.store(temporary_bytes());
}
inline std::size_t available();

/** @brief Configure before work; existing reservations remain charged. */
inline void set_limit(std::size_t bytes) {
  if (!bytes) throw std::invalid_argument("workspace must be positive");
  detail::limit.store(bytes);
}

/** Observational ownership of temporary numerical payload, including idle
 * reusable slabs. This is separate from retained-capacity reservations:
 * callers already partition kernel allowances before allocating scratch.
 * Subtracting these bytes again would change those admission decisions.
 * Moving the ticket transfers its charge without a release/reacquire gap.
 * Metadata, uninstrumented buffers and allocator overhead are not RSS-covered.
 */
class Temporary {
 public:
  Temporary() = default;
  explicit Temporary(std::size_t bytes) noexcept { resize(bytes); }
  Temporary(const Temporary&) = delete;
  Temporary& operator=(const Temporary&) = delete;
  Temporary(Temporary&& other) noexcept
      : bytes_(std::exchange(other.bytes_,0)) {}
  Temporary& operator=(Temporary&& other) noexcept {
    if (this!=&other) {
      resize(0);
      bytes_=std::exchange(other.bytes_,0);
    }
    return *this;
  }
  ~Temporary() { resize(0); }

  std::size_t bytes() const noexcept { return bytes_; }
  void resize(std::size_t bytes) noexcept {
    if (bytes>bytes_) {
      const auto live=detail::temporary.fetch_add(bytes-bytes_)+bytes-bytes_;
      auto peak=detail::temporary_peak.load();
      while (peak<live && !detail::temporary_peak.compare_exchange_weak(peak,live)) {}
    } else if (bytes<bytes_) detail::temporary.fetch_sub(bytes_-bytes);
    bytes_=bytes;
  }

 private:
  std::size_t bytes_=0;
};

/** @brief Move-only ownership of working capacity or retained payload. */
class Reservation {
 public:
  Reservation() = default;
  Reservation(const Reservation&) = delete;
  Reservation& operator=(const Reservation&) = delete;
  Reservation(Reservation&& other) noexcept : bytes_(std::exchange(other.bytes_, 0)) {}
  Reservation& operator=(Reservation&& other) noexcept {
    if (this != &other) {
      try_resize(0);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }
  ~Reservation() { try_resize(0); }
  std::size_t bytes() const noexcept { return bytes_; }
  bool try_resize(std::size_t bytes) noexcept {
    // A nested owner receives already reserved capacity, rather than spending
    // the same process allowance again. Shrinking returns it to the action.
    if (detail::current && detail::current!=this) {
      auto& parent=*detail::current;
      if (bytes>bytes_) {
        const auto increment=bytes-bytes_;
        if (increment>parent.bytes_) return false;
        parent.bytes_-=increment;
      } else parent.bytes_+=bytes_-bytes;
      bytes_=bytes;
      return true;
    }
    if (bytes <= bytes_) {
      detail::retained.fetch_sub(bytes_ - bytes);
      bytes_ = bytes;
      return true;
    }
    const auto increment = bytes - bytes_;
    // Allocation admission remains noexcept, including first-use discovery.
    std::size_t cap;
    try { cap=limit(); } catch (...) { return false; }
    auto used = retained();
    bool reclaimed=false;
    for (;;) {
      cap=detail::limit.load();
      if (used > cap || increment > cap - used) {
        const auto release=detail::reclaim.load();
        if (!reclaimed && release && detail::reusable.load()) {
          reclaimed=true;release();used=retained();continue;
        }
        return false;
      }
      if (detail::retained.compare_exchange_weak(used, used + increment)) {
        bytes_ = bytes;
        return true;
      }
    }
  }
 private:
  std::size_t bytes_ = 0;
};

inline std::size_t available() {
  if (detail::current) return detail::current->bytes();
  const auto cap=limit(),held=retained();
  const auto used=held-std::min(held,detail::reusable.load());
  return used<cap ? cap-used:0;
}

/** Reserve one action's working capacity before creating its tiles. Retained
 * child owners take capacity from this reservation and can outlive the action.
 * Ordinary temporary buffers use the action's existing bounded partitions.
 * Nested scopes borrow the parent; they do not reserve a second allowance.
 */
class Scope {
 public:
  Scope(std::size_t requested,Reservation& reusable):previous_(detail::current) {
    if (previous_) { bytes_=std::min(requested,available()); return; }
    reservation_=std::move(reusable);
    bytes_=std::min(requested,available()+reservation_.bytes());
    if (!bytes_ || !reservation_.try_resize(bytes_)) {
      reusable=std::move(reservation_);
      throw std::runtime_error("CI workspace resource_limit: action reservation failed");
    }
    detail::current=&reservation_;
  }
  ~Scope() { detail::current=previous_; }
  Scope(const Scope&)=delete;
  Scope& operator=(const Scope&)=delete;
  std::size_t bytes() const noexcept { return bytes_; }
 private:
  Reservation* previous_;
  Reservation reservation_;
  std::size_t bytes_=0;
};

/** @brief Allocate a shared retained matrix, leaving the caller's minimum
 * scratch available. Aliasing owners/copies are charged only once.
 * An empty pointer requests mathematically identical tiled regeneration.
 */
template<class Matrix>
std::shared_ptr<Matrix> make_matrix(std::size_t rows, std::size_t cols,
                                  std::size_t scratch = 0) {
  constexpr auto scalar_bytes=sizeof(typename Matrix::Scalar);
  if (rows && cols>std::numeric_limits<std::size_t>::max()/scalar_bytes/rows)
    throw std::length_error("workspace matrix size overflow");
  const auto bytes = rows * cols * scalar_bytes;
  const auto remaining=available();
  if (scratch > remaining || bytes > remaining - scratch) return {};
  struct Storage {
    Reservation reservation;
    Matrix matrix;
  };
  auto storage = std::make_shared<Storage>();
  if (!storage->reservation.try_resize(bytes)) return {};
  storage->matrix.resize(rows, cols);
  auto* matrix = &storage->matrix;
  return std::shared_ptr<Matrix>(std::move(storage), matrix);
}

} // namespace xmvb::workspace
