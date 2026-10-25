#include "cuda/ci/spin_self.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {
using View = linalg::ConstMatrixView;
using Out = linalg::MatrixView;
constexpr int threads = 256;
int grid(std::int64_t n) {
    return blas_int((n + threads - 1) / threads);
}

__global__ void
gather_occupied(View frame, const int* occupied, double* factors, int n, int k, int count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::int64_t>(n) * k * count) {
        const int row = i % n, col = (i / n) % k, string = i / (n * k);
        factors[i] = frame.data[row + frame.ld * occupied[string * k + col]];
    }
}

__global__ void
matrix_pointers(double* lu, double* inverse, double** left, double** right, int k, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        left[i] = lu + i * k * k;
        right[i] = inverse + i * k * k;
    }
}

__global__ void
self_overlap(const double* lu, const int* pivots, double* overlap, int k, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double value = 1.0;
        for (int j = 0; j < k; ++j) {
            value *= lu[i * k * k + j + j * k];
            if (pivots[i * k + j] != j + 1)
                value = -value;
        }
        overlap[i] = value;
    }
}

__device__ double density(const double* d, int n, int p, int q) {
    return 0.5 * (d[p + n * q] + d[q + n * p]);
}

__global__ void channel_means(const double* densities,
                              const int* first,
                              const int* second,
                              Out means,
                              int n,
                              int begin,
                              int count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::int64_t>(count) * means.cols) {
        const int u = i % count, pair = i / count;
        const int p = first[pair], q = second[pair];
        means.data[begin + u + means.ld * pair] =
            (p == q ? 1.0 : 2.0) * density(densities + u * n * n, n, p, q);
    }
}

__global__ void self_energy(const double* densities,
                            const int* first,
                            const int* second,
                            View kernel,
                            double* energy,
                            int n,
                            int count) {
    const int u = blockIdx.x;
    if (u >= count)
        return;
    using Reduce = cub::BlockReduce<double, threads>;
    __shared__ typename Reduce::TempStorage scratch;
    const double* d = densities + u * n * n;
    double sum = 0.0;
    for (int pqrs = threadIdx.x; pqrs < kernel.rows * kernel.cols; pqrs += blockDim.x) {
        const int left = pqrs % kernel.rows, right = pqrs / kernel.rows;
        const int p = first[left], q = second[left], r = first[right], s = second[right];
        double moment = 0.0;
        for (int a = 0; a < (p == q ? 1 : 2); ++a)
            for (int b = 0; b < (r == s ? 1 : 2); ++b) {
                const int i = a ? q : p, j = a ? p : q, k = b ? s : r, l = b ? r : s;
                // <a_i^+ a_j a_k^+ a_l> for a normalized Slater state.
                moment += density(d, n, i, j) * density(d, n, k, l) +
                          (j == k ? density(d, n, i, l) : 0.0) -
                          density(d, n, i, l) * density(d, n, j, k);
            }
        sum += kernel.data[left + kernel.ld * right] * moment;
    }
    sum = Reduce(scratch).Sum(sum);
    if (threadIdx.x == 0)
        energy[u] = sum;
}

__global__ void vacuum(double* overlap, double* energy, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        overlap[i] = 1.0;
        energy[i] = 0.0;
    }
}

void check_info(Context& c, const Buffer<int>& info, int count) {
    std::vector<int> host(count);
    c.copy_to_host(host.data(), info.data(), count * sizeof(int));
    if (std::any_of(host.begin(), host.end(), [](int value) { return value != 0; }))
        throw std::runtime_error("nonorthogonal spin-self Gram factorization failed");
}
} // namespace

SpinSelf::SpinSelf(Context& context, int orbitals, std::span<const std::uint64_t> masks)
    : context_(&context), orbitals_(orbitals),
      occupied_(masks.empty() ? 0 : std::popcount(masks.front())), strings_(blas_int(masks.size())),
      channels_(orbitals * (orbitals + 1) / 2),
      occupations_(context, static_cast<std::size_t>(strings_) * occupied_),
      first_(context, channels_), second_(context, channels_),
      moments_(context, strings_, channels_), energy_(context, strings_, 1),
      overlap_(context, strings_, 1) {
    if (orbitals <= 0 || orbitals > 64 || strings_ == 0)
        throw std::invalid_argument("invalid spin-self topology");
    std::vector<int> occupations, first, second;
    for (auto mask : masks) {
        if (std::popcount(mask) != occupied_)
            throw std::invalid_argument("spin-self strings have different particle counts");
        while (mask) {
            occupations.push_back(std::countr_zero(mask));
            mask &= mask - 1;
        }
    }
    for (int p = 0; p < orbitals; ++p)
        for (int q = 0; q <= p; ++q) {
            first.push_back(p);
            second.push_back(q);
        }
    context.copy_to_device(
        occupations_.data(), occupations.data(), occupations.size() * sizeof(int));
    context.copy_to_device(first_.data(), first.data(), first.size() * sizeof(int));
    context.copy_to_device(second_.data(), second.data(), second.size() * sizeof(int));
}

void SpinSelf::update(View frame, View kernel) {
    const nvtx3::scoped_range range{"ci.nonorthogonal_spin_self"};
    if (frame.rows != orbitals_ || frame.cols != orbitals_ || frame.ld < frame.rows ||
        kernel.rows != channels_ || kernel.cols != channels_ || kernel.ld < kernel.rows)
        throw std::invalid_argument("spin-self frame/kernel shape mismatch");
    auto& c = *context_;
    if (!occupied_) {
        vacuum<<<grid(strings_), threads, 0, c.stream()>>>(
            overlap_.data(), energy_.data(), strings_);
        check_cuda(
            cudaMemsetAsync(moments_.data(), 0, strings_ * channels_ * sizeof(double), c.stream()),
            "zero vacuum channel means");
        check_cuda(cudaGetLastError(), "vacuum self diagonal");
        return;
    }
    const int n = orbitals_, k = occupied_;
    const std::size_t per_string = (2ULL * n * k + 2ULL * k * k + 1ULL * n * n) * sizeof(double) +
                                   2 * sizeof(double*) + (k + 1) * sizeof(int);
    const auto available = c.available_device_bytes();
    const int tile = static_cast<int>(std::min<std::size_t>(strings_, available / per_string));
    if (!tile)
        throw std::runtime_error("spin-self workspace cannot fit one string");
    Matrix<double> factors(c, n, k * tile), lu(c, k, k * tile), inverse(c, k, k * tile),
        contracted(c, n, k * tile), densities(c, n, n * tile);
    Buffer<double*> left(c, tile), right(c, tile);
    Buffer<int> pivots(c, k * tile), info(c, tile);
    const double one = 1.0, zero = 0.0;
    for (int begin = 0; begin < strings_; begin += tile) {
        const int count = std::min(tile, strings_ - begin);
        gather_occupied<<<grid(1LL * n * k * count), threads, 0, c.stream()>>>(
            frame, occupations_.data() + begin * k, factors.data(), n, k, count);
        matrix_pointers<<<grid(count), threads, 0, c.stream()>>>(
            lu.data(), inverse.data(), left.data(), right.data(), k, count);
        check_cuda(cudaGetLastError(), "spin-self occupied gather");
        check_cublas(cublasDgemmStridedBatched(c.blas(),
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               k,
                                               k,
                                               n,
                                               &one,
                                               factors.data(),
                                               n,
                                               n * k,
                                               factors.data(),
                                               n,
                                               n * k,
                                               &zero,
                                               lu.data(),
                                               k,
                                               k * k,
                                               count),
                     "spin-self occupied Gram GEMM");
        check_cublas(
            cublasDgetrfBatched(c.blas(), k, left.data(), k, pivots.data(), info.data(), count),
            "spin-self batched LU");
        check_info(c, info, count);
        self_overlap<<<grid(count), threads, 0, c.stream()>>>(
            lu.data(), pivots.data(), overlap_.data() + begin, k, count);
        check_cublas(cublasDgetriBatched(c.blas(),
                                         k,
                                         reinterpret_cast<const double* const*>(left.data()),
                                         k,
                                         pivots.data(),
                                         right.data(),
                                         k,
                                         info.data(),
                                         count),
                     "spin-self batched Gram inverse");
        check_info(c, info, count);
        check_cublas(cublasDgemmStridedBatched(c.blas(),
                                               CUBLAS_OP_N,
                                               CUBLAS_OP_N,
                                               n,
                                               k,
                                               k,
                                               &one,
                                               factors.data(),
                                               n,
                                               n * k,
                                               inverse.data(),
                                               k,
                                               k * k,
                                               &zero,
                                               contracted.data(),
                                               n,
                                               n * k,
                                               count),
                     "spin-self inverse projection");
        check_cublas(cublasDgemmStridedBatched(c.blas(),
                                               CUBLAS_OP_N,
                                               CUBLAS_OP_T,
                                               n,
                                               n,
                                               k,
                                               &one,
                                               contracted.data(),
                                               n,
                                               n * k,
                                               factors.data(),
                                               n,
                                               n * k,
                                               &zero,
                                               densities.data(),
                                               n,
                                               n * n,
                                               count),
                     "spin-self normalized density");
        channel_means<<<grid(1LL * count * channels_), threads, 0, c.stream()>>>(
            densities.data(), first_.data(), second_.data(), moments_.view(), n, begin, count);
        self_energy<<<count, threads, 0, c.stream()>>>(densities.data(),
                                                       first_.data(),
                                                       second_.data(),
                                                       kernel,
                                                       energy_.data() + begin,
                                                       n,
                                                       count);
        check_cuda(cudaGetLastError(), "spin-self channel contraction");
    }
}

} // namespace xmvb::cuda
