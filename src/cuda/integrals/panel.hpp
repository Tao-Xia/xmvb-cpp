#pragma once

#include <cstdint>

namespace xmvb::cuda {

/** Borrowed row-major AO-pair/channel data; the stride is in elements. */
template <class T> struct BasicPairPanel {
    T* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t ld = 0;
};
using PairPanel = BasicPairPanel<double>;
using ConstPairPanel = BasicPairPanel<const double>;

} // namespace xmvb::cuda
