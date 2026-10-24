#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

template <class T> class Buffer {
  public:
    Buffer() = default;
    Buffer(Context& context,
           std::size_t count,
           std::source_location location = std::source_location::current())
        : context_(&context), count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("CUDA buffer size overflow");
        }
        data_ = static_cast<T*>(context.allocate(count * sizeof(T), location));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept {
        *this = std::move(other);
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            reset();
            context_ = std::exchange(other.context_, nullptr);
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    ~Buffer() {
        reset();
    }

    [[nodiscard]] T* data() noexcept {
        return data_;
    }
    [[nodiscard]] const T* data() const noexcept {
        return data_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }

  private:
    void reset() noexcept {
        if (context_ != nullptr)
            context_->deallocate(data_, count_ * sizeof(T));
        context_ = nullptr;
        data_ = nullptr;
        count_ = 0;
    }
    Context* context_ = nullptr;
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <class T> class Matrix {
  public:
    Matrix() = default;
    Matrix(Context& context,
           std::int64_t rows,
           std::int64_t cols,
           std::source_location location = std::source_location::current())
        : rows_(rows), cols_(cols), storage_(context, checked_count(rows, cols), location) {}
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    [[nodiscard]] std::int64_t rows() const noexcept {
        return rows_;
    }
    [[nodiscard]] std::int64_t cols() const noexcept {
        return cols_;
    }
    [[nodiscard]] T* data() noexcept {
        return storage_.data();
    }
    [[nodiscard]] const T* data() const noexcept {
        return storage_.data();
    }
    [[nodiscard]] linalg::BasicMatrixView<T> view() noexcept {
        return {storage_.data(), rows_, cols_, rows_};
    }
    [[nodiscard]] linalg::BasicMatrixView<const T> view() const noexcept {
        return {storage_.data(), rows_, cols_, rows_};
    }

  private:
    static std::size_t checked_count(std::int64_t rows, std::int64_t cols) {
        if (rows < 0 || cols < 0 ||
            (cols > 0 &&
             static_cast<std::uint64_t>(rows) >
                 std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(cols))) {
            throw std::overflow_error("CUDA matrix dimensions overflow");
        }
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }
    std::int64_t rows_ = 0;
    std::int64_t cols_ = 0;
    Buffer<T> storage_;
};

inline void copy_to_device(Context& context,
                           linalg::MatrixView destination,
                           const Eigen::Ref<const Eigen::MatrixXd>& source) {
    if (destination.rows != source.rows() || destination.cols != source.cols() ||
        (destination.cols > 0 && destination.ld < destination.rows)) {
        throw std::invalid_argument("CUDA matrix upload dimensions");
    }
    context.copy_to_device_2d(destination.data,
                              destination.ld * sizeof(double),
                              source.data(),
                              source.outerStride() * sizeof(double),
                              destination.rows * sizeof(double),
                              destination.cols);
}

inline void copy_to_host(Context& context,
                         Eigen::Ref<Eigen::MatrixXd> destination,
                         linalg::ConstMatrixView source) {
    if (destination.rows() != source.rows || destination.cols() != source.cols ||
        (source.cols > 0 && source.ld < source.rows)) {
        throw std::invalid_argument("CUDA matrix download dimensions");
    }
    context.copy_to_host_2d(destination.data(),
                            destination.outerStride() * sizeof(double),
                            source.data,
                            source.ld * sizeof(double),
                            source.rows * sizeof(double),
                            source.cols);
}

} // namespace xmvb::cuda
