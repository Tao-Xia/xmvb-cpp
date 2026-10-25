#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

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
        return {data == nullptr ? nullptr : data + first * ld, rows, count, ld};
    }

    template <class U = T>
        requires(!std::is_const_v<U>)
    [[nodiscard]] operator BasicMatrixView<const U>() const noexcept {
        return {data, rows, cols, ld};
    }
};

using MatrixView = BasicMatrixView<double>;
using ConstMatrixView = BasicMatrixView<const double>;

} // namespace xmvb::linalg
