#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

namespace xmvb::workspace {

/** CPU double scratch with a stable virtual address inside its reservation.
 * Growing the usable prefix does not copy data; shrinking discards only tail
 * pages. A larger address reservation discards the old contents. Logical byte
 * capacity is accounted by the caller; address space is not resident memory.
 */
class Slab {
 public:
  Slab()=default;
  ~Slab() { if (data_) munmap(data_,mapped_); }
  Slab(const Slab&)=delete;
  Slab& operator=(const Slab&)=delete;
  void swap(Slab& other) noexcept {
    std::swap(data_,other.data_);std::swap(mapped_,other.mapped_);
    std::swap(pages_,other.pages_);
  }
  double* get() const noexcept { return data_; }
  /** Returns true only if a new virtual mapping was needed. */
  bool resize(std::size_t bytes,std::size_t reservation) {
    const bool allocate=bytes>mapped_;
    if (allocate) {
      const auto size=round_up(std::max(bytes,reservation));
      void* memory=mmap(nullptr,size,PROT_READ|PROT_WRITE,
          MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
      if (memory==MAP_FAILED) throw std::bad_alloc();
      // A failed address reservation leaves the current workspace usable.
      // The new mapping has no resident payload before the old one is freed.
      if (data_) munmap(data_,mapped_);
      // Trivial default initialization starts the double array's lifetime
      // without writing or faulting the reserved pages.
      data_=::new(memory) double[size/sizeof(double)];
      mapped_=size;pages_=0;
    }
    const auto pages=round_up(bytes);
    if (pages<pages_)
      madvise(reinterpret_cast<std::byte*>(data_)+pages,pages_-pages,MADV_DONTNEED);
    pages_=pages;
    return allocate;
  }
  void reset() noexcept {
    if (pages_) madvise(data_,pages_,MADV_DONTNEED);
    pages_=0;
  }
 private:
  static std::size_t round_up(std::size_t bytes) {
    static const auto page=static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    if (!page || page>static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes>std::numeric_limits<std::size_t>::max()-(page-1)) throw std::bad_alloc();
    return ((bytes+page-1)/page)*page;
  }
  double* data_=nullptr;
  std::size_t mapped_=0,pages_=0;
};

} // namespace xmvb::workspace
