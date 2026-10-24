#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

/** Reusable pinned source slot. Destination ownership stays with the caller.
 * Reuse waits only for the previous H2D event, never its later consumers.
 */
class PinnedTransfer {
 public:
  explicit PinnedTransfer(Context& context) : context_(context) {
    check_cuda(cudaEventCreateWithFlags(&uploaded_, cudaEventDisableTiming),
               "cudaEventCreateWithFlags pinned upload");
  }
  PinnedTransfer(const PinnedTransfer&) = delete;
  PinnedTransfer& operator=(const PinnedTransfer&) = delete;
  PinnedTransfer(PinnedTransfer&&) = delete;
  PinnedTransfer& operator=(PinnedTransfer&&) = delete;

  ~PinnedTransfer() noexcept {
    // The event is recorded immediately after H2D, so this does not wait for
    // later consumers. Keep the host slot alive through any pending copy.
    if (upload_pending_) {
      if (!event_recorded_ || cudaEventSynchronize(uploaded_) != cudaSuccess)
        cudaStreamSynchronize(context_.stream());
    }
    if (host_ != nullptr) cudaFreeHost(host_);
    cudaEventDestroy(uploaded_);
  }

  template <class Fill>
  void assign(void* destination,std::size_t bytes, Fill&& fill) {
    wait_for_upload();
    if (bytes > host_capacity_) {
      if (host_ != nullptr) {
        check_cuda(cudaFreeHost(host_), "cudaFreeHost pinned upload");
        host_ = nullptr;
        host_capacity_ = 0;
      }
      check_cuda(cudaMallocHost(&host_, bytes), "cudaMallocHost pinned upload");
      host_capacity_ = bytes;
    }
    std::forward<Fill>(fill)(std::span<std::byte>{
        static_cast<std::byte*>(host_), bytes});
    if (bytes == 0) return;
    context_.enqueue_to_device(destination, host_, bytes);
    // If event recording fails, the destructor still protects the source.
    upload_pending_ = true;
    event_recorded_ = false;
    check_cuda(cudaEventRecord(uploaded_, context_.stream()),
               "cudaEventRecord pinned upload");
    event_recorded_ = true;
  }

  void assign(void* destination,std::span<const std::byte> source) {
    assign(destination,source.size(), [source](std::span<std::byte> target) {
      if (!source.empty()) std::memcpy(target.data(), source.data(), source.size());
    });
  }

 private:
  void wait_for_upload() {
    if (!upload_pending_) return;
    // Only the preceding H2D is needed before the pinned slot can be reused.
    // On a record failure, synchronize the stream instead of an unrecorded event.
    if (!event_recorded_) {
      context_.synchronize();
    } else {
      check_cuda(cudaEventSynchronize(uploaded_),
                 "cudaEventSynchronize pinned upload");
    }
    upload_pending_ = false;
    event_recorded_ = false;
  }

  Context& context_;
  void* host_ = nullptr;
  std::size_t host_capacity_ = 0;
  cudaEvent_t uploaded_ = nullptr;
  bool upload_pending_ = false;
  bool event_recorded_ = false;
};

/** One pinned staging slot and one exact-sized, stream-owned device tile. */
class PinnedUpload {
 public:
  explicit PinnedUpload(Context& context):context_(context),transfer_(context) {}

  template<class Fill> void assign(std::size_t bytes,Fill&& fill) {
    if (device_.size()!=bytes) {
      device_=Buffer<std::byte>{};
      device_=Buffer<std::byte>(context_,bytes);
    }
    transfer_.assign(device_.data(),bytes,std::forward<Fill>(fill));
    active_size_=bytes;
  }
  void assign(std::span<const std::byte> source) {
    assign(source.size(),[source](std::span<std::byte> target) {
      if (!source.empty()) std::memcpy(target.data(),source.data(),source.size());
    });
  }
  std::byte* data() noexcept { return active_size_ ? device_.data():nullptr; }
  const std::byte* data() const noexcept { return active_size_ ? device_.data():nullptr; }
  std::size_t size() const noexcept { return active_size_; }

 private:
  Context& context_;
  PinnedTransfer transfer_;
  Buffer<std::byte> device_;
  std::size_t active_size_=0;
};

}  // namespace xmvb::cuda
