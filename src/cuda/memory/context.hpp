#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

namespace xmvb::cuda {

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed (cuBLAS status " +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

inline void check_cusolver(cusolverStatus_t status,const char* operation) {
  if (status!=CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation)+" failed (cuSOLVER status "+
        std::to_string(static_cast<int>(status))+")");
}

struct TransferCounters {
  std::size_t current_device_bytes = 0;
  std::size_t peak_device_bytes = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_bytes = 0;
};

struct AllocationSite {
  std::string file;
  std::uint_least32_t line = 0;
  std::string phase;

  [[nodiscard]] bool operator<(const AllocationSite& other) const noexcept {
    return std::tie(file,line,phase)<std::tie(other.file,other.line,other.phase);
  }
};

struct AllocationGroup {
  std::size_t bytes = 0;
  std::size_t allocations = 0;
};

/** Owns a non-default stream and its serialized cuBLAS/cuSOLVER handles.
 * Numerical objects borrow these handles; none owns another vendor context.
 */
class Context {
 public:
  /** Use 90% of device memory free after CUDA and vendor handles are ready.
   * The remaining 10% is reserved for CUDA libraries, runtime work, and
   * allocations outside this Context. Explicit budgets remain available for
   * tests and callers with a stricter resource policy.
   */
  Context() : Context(std::nullopt) {}
  explicit Context(std::size_t device_byte_budget)
      : Context(std::optional<std::size_t>(device_byte_budget)) {}

 private:
  explicit Context(std::optional<std::size_t> requested_budget) {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    try {
      int device=0;
      check_cuda(cudaGetDevice(&device),"cudaGetDevice");
      cudaMemPoolProps properties{};
      properties.allocType=cudaMemAllocationTypePinned;
      properties.location.type=cudaMemLocationTypeDevice;
      properties.location.id=device;
      check_cuda(cudaMemPoolCreate(&pool_,&properties),"cudaMemPoolCreate");
      check_cublas(cublasCreate(&blas_), "cublasCreate");
      check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream");
      check_cublas(cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode");
      check_cusolver(cusolverDnCreate(&solver_),"cusolverDnCreate Context");
      check_cusolver(cusolverDnSetStream(solver_,stream_),"cusolverDnSetStream Context");
      std::size_t total=0;
      check_cuda(cudaMemGetInfo(&initial_device_free_bytes_,&total),
          "query initial CUDA free memory");
      // Subtract a rounded-up reserve without multiplying a large byte count.
      const auto reserve=initial_device_free_bytes_/10+
          (initial_device_free_bytes_%10!=0);
      device_byte_budget_=requested_budget.value_or(initial_device_free_bytes_-reserve);
      std::uint64_t release_threshold=device_byte_budget_;
      check_cuda(cudaMemPoolSetAttribute(pool_,cudaMemPoolAttrReleaseThreshold,
          &release_threshold),"cudaMemPoolSetAttribute");
    } catch (...) {
      if (solver_ != nullptr) cusolverDnDestroy(solver_);
      if (blas_ != nullptr) cublasDestroy(blas_);
      if (pool_ != nullptr) cudaMemPoolDestroy(pool_);
      cudaStreamDestroy(stream_);
      throw;
    }
  }

 public:
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context() {
    // All owned allocations and frees are ordered on this stream. Complete
    // them once at shutdown, rather than synchronizing every temporary free.
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (pool_ != nullptr) cudaMemPoolDestroy(pool_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] cublasHandle_t blas() const noexcept { return blas_; }
  [[nodiscard]] cusolverDnHandle_t solver() const noexcept { return solver_; }
  [[nodiscard]] const TransferCounters& counters() const noexcept {
    return counters_;
  }
  void enable_memory_audit() {
    if (counters_.peak_device_bytes != 0)
      throw std::logic_error("CUDA memory audit must start before device allocations");
    memory_audit_enabled_ = true;
  }
  [[nodiscard]] bool memory_audit_enabled() const noexcept { return memory_audit_enabled_; }
  [[nodiscard]] const std::map<AllocationSite,AllocationGroup>&
  peak_allocation_groups() const noexcept { return peak_groups_; }
  [[nodiscard]] std::size_t device_byte_budget() const noexcept {
    return device_byte_budget_;
  }
  [[nodiscard]] std::size_t initial_device_free_bytes() const noexcept {
    return initial_device_free_bytes_;
  }
  /** Budget headroom limited by currently available device storage. The pool's
   * reserved but unused bytes are reusable even though cudaMemGetInfo reports
   * them as occupied. All Context work uses one stream, so queued frees can be
   * reused in stream order without a host synchronization.
   */
  [[nodiscard]] std::size_t available_device_bytes() const {
    const auto resident=counters_.current_device_bytes;
    if (resident>device_byte_budget_)
      throw std::logic_error("CUDA Context accounting exceeds byte budget");
    const auto headroom=device_byte_budget_-resident;
    std::size_t free=0,total=0;
    check_cuda(cudaMemGetInfo(&free,&total),"query CUDA available memory");
    std::uint64_t reserved=0,used=0;
    check_cuda(cudaMemPoolGetAttribute(pool_,cudaMemPoolAttrReservedMemCurrent,&reserved),
        "query CUDA pool reserved memory");
    check_cuda(cudaMemPoolGetAttribute(pool_,cudaMemPoolAttrUsedMemCurrent,&used),
        "query CUDA pool used memory");
    const auto outside_pool=std::min(headroom,free);
    const auto reusable=reserved>used ? reserved-used:0;
    return outside_pool+std::min<std::uint64_t>(headroom-outside_pool,reusable);
  }
  [[nodiscard]] const char* allocation_phase() const noexcept { return allocation_phase_; }
  void set_allocation_phase(const char* phase) noexcept { allocation_phase_=phase; }
  void synchronize() const { check_cuda(cudaStreamSynchronize(stream_), "CUDA stream synchronization"); }

  void* allocate(std::size_t bytes,
      std::source_location location=std::source_location::current()) {
    if (bytes == 0) return nullptr;
    if (bytes > device_byte_budget_ - counters_.current_device_bytes) {
      throw std::runtime_error("CUDA device allocation exceeds byte budget: requested="+
          std::to_string(bytes)+", resident="+std::to_string(counters_.current_device_bytes)+
          ", budget="+std::to_string(device_byte_budget_)+", phase="+allocation_phase_);
    }
    void* pointer = nullptr;
    check_cuda(cudaMallocFromPoolAsync(&pointer,bytes,pool_,stream_),
               "cudaMallocFromPoolAsync");
    if (memory_audit_enabled_) {
      try {
        const AllocationSite site{location.file_name(),location.line(),allocation_phase_};
        live_allocations_.emplace(pointer,LiveAllocation{site,bytes});
        try {
          auto group=current_groups_.try_emplace(site).first;
          group->second.bytes+=bytes;
          ++group->second.allocations;
          try {
            if (counters_.current_device_bytes+bytes>counters_.peak_device_bytes) {
              auto snapshot=current_groups_;
              peak_groups_.swap(snapshot);
            }
          } catch (...) {
            group->second.bytes-=bytes;
            --group->second.allocations;
            if (!group->second.allocations) current_groups_.erase(group);
            throw;
          }
        } catch (...) {
          live_allocations_.erase(pointer);
          throw;
        }
      } catch (...) {
        cudaFreeAsync(pointer,stream_);
        throw;
      }
    }
    counters_.current_device_bytes += bytes;
    counters_.peak_device_bytes =
        std::max(counters_.peak_device_bytes, counters_.current_device_bytes);
    return pointer;
  }

  void deallocate(void* pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr) return;
    if (memory_audit_enabled_) {
      const auto live=live_allocations_.find(pointer);
      if (live!=live_allocations_.end()) {
        const auto group=current_groups_.find(live->second.site);
        if (group!=current_groups_.end()) {
          group->second.bytes-=live->second.bytes;
          --group->second.allocations;
          if (!group->second.allocations) current_groups_.erase(group);
        }
        live_allocations_.erase(live);
      }
    }
    // The stream dependency preserves every preceding consumer and lets CUDA
    // reuse the storage for subsequent work without a host/device barrier.
    cudaFreeAsync(pointer,stream_);
    counters_.current_device_bytes -= bytes;
  }

  /** Queue H2D on this stream without waiting. The caller borrows src to CUDA:
   * its contents and lifetime must remain valid until this copy completes.
   */
  void enqueue_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_),
               "host-to-device copy");
    counters_.host_to_device_bytes += bytes;
  }

  void copy_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    enqueue_to_device(dst, src, bytes);
    // The source may be caller-owned temporary host memory.
    synchronize();
  }

  void copy_to_host(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_),
               "device-to-host copy");
    synchronize();
    counters_.device_to_host_bytes += bytes;
  }

  /** Pitched matrix transfers; host ownership need only last until return. */
  void copy_to_device_2d(void* dst,std::size_t dst_pitch,const void* src,
      std::size_t src_pitch,std::size_t width,std::size_t height) {
    if (!width || !height) return;
    check_cuda(cudaMemcpy2DAsync(dst,dst_pitch,src,src_pitch,width,height,
        cudaMemcpyHostToDevice,stream_),"host-to-device matrix copy");
    synchronize();
    counters_.host_to_device_bytes+=width*height;
  }

  void copy_to_host_2d(void* dst,std::size_t dst_pitch,const void* src,
      std::size_t src_pitch,std::size_t width,std::size_t height) {
    if (!width || !height) return;
    check_cuda(cudaMemcpy2DAsync(dst,dst_pitch,src,src_pitch,width,height,
        cudaMemcpyDeviceToHost,stream_),"device-to-host matrix copy");
    synchronize();
    counters_.device_to_host_bytes+=width*height;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaMemPool_t pool_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  std::size_t device_byte_budget_=0;
  std::size_t initial_device_free_bytes_=0;
  TransferCounters counters_;
  const char* allocation_phase_="unspecified";
  struct LiveAllocation {
    AllocationSite site;
    std::size_t bytes;
  };
  bool memory_audit_enabled_=false;
  std::unordered_map<void*,LiveAllocation> live_allocations_;
  std::map<AllocationSite,AllocationGroup> current_groups_;
  std::map<AllocationSite,AllocationGroup> peak_groups_;
};

/** Failure context only; no allocation, synchronization, or numerical check. */
class AllocationPhase {
 public:
  AllocationPhase(Context& context,const char* phase)
      : context_(context),previous_(context.allocation_phase()) { context_.set_allocation_phase(phase); }
  ~AllocationPhase() { context_.set_allocation_phase(previous_); }
  AllocationPhase(const AllocationPhase&)=delete;
  AllocationPhase& operator=(const AllocationPhase&)=delete;
 private:
  Context& context_;
  const char* previous_;
};

}  // namespace xmvb::cuda
