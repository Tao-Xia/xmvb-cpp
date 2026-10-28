#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define XMVB_ROW_HD __host__ __device__
#else
#define XMVB_ROW_HD
#endif

namespace xmvb::mcscf::detail {

/** Borrowed integer topology for the CPU's descending occupied-row recursion. */
struct ExteriorRowTopology {
    int orbitals, electrons;
    std::int64_t dimension;
    const std::uint64_t* binomial;
    const std::int64_t* column_ranks;
    XMVB_ROW_HD std::uint64_t choose(int n, int k) const noexcept {
        return n < 0 || k < 0 || k > n ? 0 : binomial[n * (orbitals + 1) + k];
    }
};

/** The identical row recursion on host or device. Scratch is two spin vectors.
 * Occupied orbitals and index scratch contain electrons entries. The output
 * is a requested compound row in the caller's spin-column permutation.
 */
XMVB_ROW_HD inline void exterior_row(const ExteriorRowTopology& plan,
                                     const double* upper,
                                     int ld,
                                     const int* occupied,
                                     int* indices,
                                     double* previous,
                                     double* current,
                                     double* output) {
    previous[0] = 1.0;
    for (int k = 1; k <= plan.electrons; ++k) {
        const int base = plan.electrons - k;
        const auto count = plan.choose(plan.orbitals - base, k);
        for (std::uint64_t i = 0; i < count; ++i)
            current[i] = 0.0;
        for (int p = 0; p < k; ++p)
            indices[p] = occupied[base + p];
        for (;;) {
            std::uint64_t rank = 0, suffix = 0, prefix = 0;
            for (int p = 0; p < k; ++p) {
                rank += plan.choose(indices[p] - base, p + 1);
                if (p)
                    suffix += plan.choose(indices[p] - base - 1, p);
            }
            double value = 0.0;
            for (int p = 0; p < k; ++p) {
                if (indices[0] != base || p == 0) {
                    const double coefficient = upper[occupied[base] + ld * indices[p]];
                    if (coefficient != 0.0)
                        value += (p % 2 ? -coefficient : coefficient) * previous[prefix + suffix];
                }
                prefix += plan.choose(indices[p] - base - 1, p + 1);
                if (p + 1 < k)
                    suffix -= plan.choose(indices[p + 1] - base - 1, p + 1);
            }
            current[rank] = value;
            int p = 0;
            while (p < k && indices[p] + 1 == (p + 1 < k ? indices[p + 1] : plan.orbitals))
                ++p;
            if (p == k)
                break;
            ++indices[p];
            for (int j = 0; j < p; ++j)
                indices[j] = occupied[base + j];
        }
        auto* temporary = previous;
        previous = current;
        current = temporary;
    }
    for (std::int64_t column = 0; column < plan.dimension; ++column)
        output[column] = previous[plan.column_ranks[column]];
}
} // namespace xmvb::mcscf::detail

#undef XMVB_ROW_HD
