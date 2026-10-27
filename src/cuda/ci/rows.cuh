#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace xmvb::cuda {

/** One representative in beta-column order, with alpha contiguous. */
struct ProductRow {
    std::int64_t alpha, beta;
    std::int64_t rhs, column;
    double weight;
};

/** Decode without a stored product-sized index map. Nonzero parity requires
 * identical spin spaces and a topology-certified transpose sector.
 */
__host__ __device__ inline ProductRow
product_row(std::int64_t index, std::int64_t na, std::int64_t nb, int parity) {
    if (!parity) {
        const auto column = index / na;
        return {index % na, column % nb, column / nb, column, 1.0};
    }
    // Host admission certifies that the complete triangle fits int64_t.
    // Divide first: n*(n+1) can overflow even when its half is representable.
    const auto triangular = [](std::int64_t n) {
        const auto a = static_cast<std::uint64_t>(n);
        return static_cast<std::int64_t>((a & 1) ? a * ((a + 1) / 2) : (a / 2) * (a + 1));
    };
    const auto triangle = triangular(na);
    const auto rhs = index / triangle, local = index % triangle;
    const auto reverse = triangle - 1 - local;
    auto k = static_cast<std::int64_t>((sqrt(8.0 * reverse + 1.0) - 1.0) * 0.5);
    while (triangular(k) > reverse)
        --k;
    while (k + 1 < na && triangular(k + 1) <= reverse)
        ++k;
    const auto b = na - 1 - k;
    const auto prefix = triangle - triangular(na - b);
    const auto a = b + (local - prefix);
    return {a, b, rhs, rhs * nb + b, a == b ? 1.0 : 2.0};
}

} // namespace xmvb::cuda
