#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "core/index.hpp"

namespace xmvb::linalg {

/** A borrowed, column-major matrix. Its pointer is never owned by the view. */
template <class T> struct BasicMatrixView {
    T* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t ld = 0;

    [[nodiscard]] bool empty() const noexcept {
        return rows == 0 || cols == 0;
    }

    [[nodiscard]] BasicMatrixView column(std::int64_t index) const {
        return columns(index, 1);
    }

    [[nodiscard]] BasicMatrixView columns(std::int64_t first, std::int64_t count) const {
        if (first < 0 || count < 0 || first > cols || count > cols - first) {
            throw std::out_of_range("matrix view column range");
        }
        const auto offset = core::checked_index<std::ptrdiff_t>(core::checked_multiply(
            core::checked_index<std::size_t>(first), core::checked_index<std::size_t>(ld)));
        return {data == nullptr ? nullptr : data + offset, rows, count, ld};
    }

    template <class U = T>
        requires(!std::is_const_v<U>)
    [[nodiscard]] operator BasicMatrixView<const U>() const noexcept {
        return {data, rows, cols, ld};
    }
};

using MatrixView = BasicMatrixView<double>;
using ConstMatrixView = BasicMatrixView<const double>;

/** @brief Native logical element count, checked before flattening a vector. */
template <class T> std::int64_t element_count(BasicMatrixView<T> view) {
    return core::checked_index<std::int64_t>(core::checked_multiply(
        core::checked_index<std::size_t>(view.rows), core::checked_index<std::size_t>(view.cols)));
}

} // namespace xmvb::linalg
