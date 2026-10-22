#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

/** One reusable pinned host slot and one exact-sized device tile.
 * Context must outlive this object. Consumers must use the Context stream;
 * replacing the device tile queues its release after those consumers.
 */
class PinnedUpload {
 public:
  explicit PinnedUpload(Context& context) : context_(context) {
    check_cuda(cudaEventCreateWithFlags(&uploaded_, cudaEventDisableTiming),
               "cudaEventCreateWithFlags pinned upload");
  }
  PinnedUpload(const PinnedUpload&) = delete;
  PinnedUpload& operator=(const PinnedUpload&) = delete;
  PinnedUpload(PinnedUpload&&) = delete;
  PinnedUpload& operator=(PinnedUpload&&) = delete;

  ~PinnedUpload() noexcept {
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
  void assign(std::size_t bytes, Fill&& fill) {
    wait_for_upload();
    active_size_ = 0;
    if (bytes > host_capacity_) {
      if (host_ != nullptr) {
        check_cuda(cudaFreeHost(host_), "cudaFreeHost pinned upload");
        host_ = nullptr;
        host_capacity_ = 0;
      }
      check_cuda(cudaMallocHost(&host_, bytes), "cudaMallocHost pinned upload");
      host_capacity_ = bytes;
    }
    if (device_.size() != bytes) {
      device_ = Buffer<std::byte>{};  // stream-ordered free of the old tile
      device_ = Buffer<std::byte>(context_, bytes);
    }
    std::forward<Fill>(fill)(std::span<std::byte>{
        static_cast<std::byte*>(host_), bytes});
    if (bytes == 0) return;
    context_.enqueue_to_device(device_.data(), host_, bytes);
    // If event recording fails, the destructor still protects the source.
    upload_pending_ = true;
    event_recorded_ = false;
    check_cuda(cudaEventRecord(uploaded_, context_.stream()),
               "cudaEventRecord pinned upload");
    event_recorded_ = true;
    active_size_ = bytes;
  }

  void assign(std::span<const std::byte> source) {
    assign(source.size(), [source](std::span<std::byte> target) {
      if (!source.empty()) std::memcpy(target.data(), source.data(), source.size());
    });
  }

  [[nodiscard]] std::byte* data() noexcept {
    return active_size_ == 0 ? nullptr : device_.data();
  }
  [[nodiscard]] const std::byte* data() const noexcept {
    return active_size_ == 0 ? nullptr : device_.data();
  }
  [[nodiscard]] std::size_t size() const noexcept { return active_size_; }

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
  Buffer<std::byte> device_;
  cudaEvent_t uploaded_ = nullptr;
  bool upload_pending_ = false;
  bool event_recorded_ = false;
  std::size_t active_size_ = 0;
};

}  // namespace xmvb::cuda
