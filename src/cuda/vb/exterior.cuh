#include <cstdint>
#pragma once

#include "cuda/vb/transform.hpp"

namespace xmvb::cuda::detail {

/** Source views for a column factor. Updated and read-only q sectors are disjoint. */
template <bool Right, bool Indirect, bool Tangent> struct ExteriorValues {
    linalg::MatrixView primal, tangent;
    double* const* columns;
    double* const* tangent_columns;
    std::int64_t row, column;

    __device__ double value(std::int64_t spin) const {
        if constexpr (Indirect)
            return columns[spin][row];
        else if constexpr (Right)
            return primal.data[row + primal.ld * (column + spin)];
        else
            return primal.data[spin + primal.ld * column];
    }
    __device__ double delta(std::int64_t spin) const {
        if constexpr (!Tangent)
            return 0;
        else if constexpr (Indirect)
            return tangent_columns[spin][row];
        else if constexpr (Right)
            return tangent.data[row + tangent.ld * (column + spin)];
        else
            return tangent.data[spin + tangent.ld * column];
    }
    __device__ void store(std::int64_t spin, double value, double delta) const {
        if constexpr (Indirect) {
            columns[spin][row] = value;
            if constexpr (Tangent)
                tangent_columns[spin][row] = delta;
        } else if constexpr (Right) {
            primal.data[row + primal.ld * (column + spin)] = value;
            if constexpr (Tangent)
                tangent.data[row + tangent.ld * (column + spin)] = delta;
        } else {
            primal.data[spin + primal.ld * column] = value;
            if constexpr (Tangent)
                tangent.data[spin + tangent.ld * column] = delta;
        }
    }
};

__device__ inline std::int64_t choose(ExteriorTransform::Index index, int p, int k) {
    return index.choose[p * (index.orbitals + 1) + k];
}

/** Apply all I+c_pq a_p^dagger a_q factors for fixed q, in the original order.
 * Products of two different updates vanish because they annihilate q twice.
 * A rolling colex-rank difference addresses replacements in O(q) work per row;
 * arbitrary caller ordering uses only one immutable O(U) permutation.
 */
template <bool Tangent, class Values>
__device__ double2 column_factor(ExteriorTransform::Index index,
                                 std::int64_t spin,
                                 int q,
                                 bool adjoint,
                                 bool inverse,
                                 const double* coefficients,
                                 const double* delta_coefficients,
                                 Values values,
                                 const std::uint64_t* allowed) {
    const auto mask = index.masks[spin];
    const auto below = mask & ((std::uint64_t{1} << q) - 1);
    const int total = __popcll(below);
    const bool reverse = adjoint != inverse;
    std::int64_t rank = spin;
    if (index.order) {
        rank = 0;
        int k = 1;
        for (auto bits = mask; bits; bits &= bits - 1)
            rank += choose(index, __ffsll(bits) - 1, k++);
    }
    std::int64_t correction = 0;
    if (reverse) {
        int k = 1;
        for (auto bits = below; bits; bits &= bits - 1) {
            const int p = __ffsll(bits) - 1;
            correction += choose(index, p, k + (adjoint ? 1 : -1)) - choose(index, p, k);
            ++k;
        }
    }
    int k = reverse ? 0 : total;
    double value = values.value(spin), delta = values.delta(spin);
    for (int step = 0; step < q; ++step) {
        const int p = reverse ? step : q - 1 - step;
        const bool occupied = mask & (std::uint64_t{1} << p);
        if (reverse && occupied) {
            ++k;
            correction -= choose(index, p, k + (adjoint ? 1 : -1)) - choose(index, p, k);
        }
        if (occupied != adjoint) {
            const auto source_rank =
                adjoint ? rank - choose(index, q, total + 1) + choose(index, p, k + 1) + correction
                        : rank + choose(index, q, total) - choose(index, p, k) + correction;
            const std::int64_t source = index.order ? index.order[source_rank] : source_rank;
            if (!allowed || (allowed[source / 64] & (std::uint64_t{1} << (source % 64)))) {
                const double sign = (total - k) & 1 ? -1.0 : 1.0;
                const int factor = q * (q - 1) / 2 + q - 1 - p;
                const double c = coefficients[factor];
                const double x = values.value(source);
                if constexpr (Tangent)
                    delta += sign * (delta_coefficients[factor] * x + c * values.delta(source));
                value += (inverse ? -c : c) * sign * x;
            }
        }
        if (!reverse && occupied) {
            correction += choose(index, p, k + (adjoint ? 1 : -1)) - choose(index, p, k);
            --k;
        }
    }
    return make_double2(value, delta);
}

/** Exact fixed-q update with no shared-vector requirement or source atomics. */
template <bool Right, bool Indirect = false, bool Tangent = false>
__global__ void exterior_column(linalg::MatrixView primal,
                                linalg::MatrixView tangent,
                                ExteriorTransform::Index index,
                                std::int64_t dimension,
                                int q,
                                const double* coefficients,
                                const double* delta_coefficients,
                                bool adjoint,
                                bool inverse,
                                const std::uint64_t* allowed = nullptr,
                                int support_columns = 0,
                                int words = 0,
                                double* const* columns = nullptr,
                                double* const* tangent_columns = nullptr) {
    const auto work = primal.rows * primal.cols;
    for (auto item = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; item < work;
         item += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const auto row = Right ? item % primal.rows : 0;
        const auto column = Right ? (item / primal.rows / dimension) * dimension : item / dimension;
        const std::int64_t spin = Right ? (item / primal.rows) % dimension : item % dimension;
        if (bool(index.masks[spin] & (std::uint64_t{1} << q)) != adjoint)
            continue;
        const auto* support = allowed ? allowed + (column % support_columns) * words : nullptr;
        if (support && !(support[spin / 64] & (std::uint64_t{1} << (spin % 64))))
            continue;
        const ExteriorValues<Right, Indirect, Tangent> values{
            primal, tangent, columns, tangent_columns, row, column};
        const auto result = column_factor<Tangent>(
            index, spin, q, adjoint, inverse, coefficients, delta_coefficients, values, support);
        values.store(spin, result.x, result.y);
    }
}

/** Scaling for noncontiguous numerical column tiles, without transposed stripes. */
template <bool Tangent = false>
__global__ void scale_exterior_tiles(double* const* columns,
                                     double* const* tangent,
                                     std::int64_t rows,
                                     std::int64_t dimension,
                                     const double* scales,
                                     const double* delta_scales = nullptr) {
    for (auto item = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < rows * dimension;
         item += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t spin = item / rows;
        const auto row = item % rows;
        const double value = columns[spin][row];
        if constexpr (Tangent)
            tangent[spin][row] = scales[spin] * tangent[spin][row] + delta_scales[spin] * value;
        columns[spin][row] = scales[spin] * value;
    }
}
} // namespace xmvb::cuda::detail
