#include <climits>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/index.hpp"
#include "cuda/ci/rows.cuh"
#include "cuda/memory/buffer.hpp"

namespace {
struct Sample {
    std::int64_t n, index;
};
__global__ void decode(const Sample* samples, xmvb::cuda::ProductRow* result, std::size_t count) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        result[i] = xmvb::cuda::product_row(samples[i].index, samples[i].n, samples[i].n, 1);
}
std::uint64_t triangle(std::uint64_t n) {
    return static_cast<std::uint64_t>(static_cast<__uint128_t>(n) * (n + 1) / 2);
}
} // namespace
int main() {
    xmvb::cuda::Context context;
    std::vector<Sample> samples;
    for (const std::int64_t n : {48620LL, 2147483648LL, 4294967295LL}) {
        const auto count = static_cast<std::int64_t>(triangle(n));
        for (const auto local : {std::int64_t{0}, n - 1, n, count / 2, count - 1})
            samples.push_back({n, local});
    }
    const auto small_triangle = static_cast<std::int64_t>(triangle(48620));
    samples.push_back({48620, 3 * small_triangle - 1});
    xmvb::cuda::Buffer<Sample> input(context, samples.size());
    xmvb::cuda::Buffer<xmvb::cuda::ProductRow> output(context, samples.size());
    context.copy_to_device(input.data(), samples.data(), samples.size() * sizeof(Sample));
    decode<<<1, 64, 0, context.stream()>>>(input.data(), output.data(), samples.size());
    std::vector<xmvb::cuda::ProductRow> actual(samples.size());
    context.copy_to_host(actual.data(), output.data(), actual.size() * sizeof(actual[0]));
    context.synchronize();
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto [n, full_index] = samples[i];
        const auto count = static_cast<std::int64_t>(triangle(n));
        const auto rhs = full_index / count, index = full_index % count;
        std::int64_t low = 0, high = n;
        const auto prefix = [&](std::int64_t b) { return triangle(n) - triangle(n - b); };
        while (high - low > 1) {
            const auto mid = low + (high - low) / 2;
            if (prefix(mid) <= static_cast<std::uint64_t>(index))
                low = mid;
            else
                high = mid;
        }
        const auto a = low + (index - static_cast<std::int64_t>(prefix(low)));
        const auto& value = actual[i];
        if (value.alpha != a || value.beta != low || value.rhs != rhs ||
            value.column != rhs * n + low || value.weight != (a == low ? 1.0 : 2.0))
            throw std::runtime_error("GPU triangular global-address decode mismatch");
    }
    std::cout << "GPU native index boundaries passed without large allocations\n";
}
