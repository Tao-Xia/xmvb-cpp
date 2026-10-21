#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

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

struct TransferCounters {
  std::size_t current_device_bytes = 0;
  std::size_t peak_device_bytes = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_bytes = 0;
};

/** Owns a non-default stream and its cuBLAS handle; allocations belong to it. */
class Context {
 public:
  explicit Context(
      std::size_t device_byte_budget = std::numeric_limits<std::size_t>::max())
      : device_byte_budget_(device_byte_budget) {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    try {
      check_cublas(cublasCreate(&blas_), "cublasCreate");
      check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream");
      check_cublas(cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode");
    } catch (...) {
      if (blas_ != nullptr) cublasDestroy(blas_);
      cudaStreamDestroy(stream_);
      throw;
    }
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context() {
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] cublasHandle_t blas() const noexcept { return blas_; }
  [[nodiscard]] const TransferCounters& counters() const noexcept {
    return counters_;
  }
  [[nodiscard]] std::size_t device_byte_budget() const noexcept {
    return device_byte_budget_;
  }
  [[nodiscard]] const char* allocation_phase() const noexcept { return allocation_phase_; }
  void set_allocation_phase(const char* phase) noexcept { allocation_phase_=phase; }
  void synchronize() const { check_cuda(cudaStreamSynchronize(stream_), "CUDA stream synchronization"); }

  void* allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > device_byte_budget_ - counters_.current_device_bytes) {
      throw std::runtime_error("CUDA device allocation exceeds byte budget: requested="+
          std::to_string(bytes)+", resident="+std::to_string(counters_.current_device_bytes)+
          ", budget="+std::to_string(device_byte_budget_)+", phase="+allocation_phase_);
    }
    void* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    counters_.current_device_bytes += bytes;
    counters_.peak_device_bytes =
        std::max(counters_.peak_device_bytes, counters_.current_device_bytes);
    return pointer;
  }

  void deallocate(void* pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr) return;
    // cudaFree synchronizes pending uses of this pointer before releasing it.
    cudaFree(pointer);
    counters_.current_device_bytes -= bytes;
  }

  void copy_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_),
               "host-to-device copy");
    // The source may be caller-owned temporary host memory.
    synchronize();
    counters_.host_to_device_bytes += bytes;
  }

  void copy_to_host(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_),
               "device-to-host copy");
    synchronize();
    counters_.device_to_host_bytes += bytes;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  std::size_t device_byte_budget_;
  TransferCounters counters_;
  const char* allocation_phase_="unspecified";
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
