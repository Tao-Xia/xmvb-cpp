#include <cstdint>
#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>
#include <cub/warp/warp_reduce.cuh>

#include "cuda/integrals/packed.hpp"
#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

unsigned int grid(std::int64_t work) {
    return static_cast<unsigned int>(
        std::min<std::int64_t>(65535, (work / 256 + (work % 256 != 0))));
}

struct ExchangeTerms {
    double a, b, c, d;
    __device__ ExchangeTerms operator+(ExchangeTerms other) const {
        return {a + other.a, b + other.b, c + other.c, d + other.d};
    }
};
using ExchangeReduce = cub::WarpReduce<ExchangeTerms>;

/** One canonical integral column owns four output panels, as in CPU unique
 * contraction. Packed columns are contiguous, so each retained value is read
 * once, not rediscovered through four-index exchange queries. The nonzero
 * bitmap skips exact zeros without storing another integral/index stream.
 * Both density orientations remain independent. Global atomics merge only
 * completed AO panels; per-integral additions stay in block shared memory.
 */
__global__ void exchange_packed_columns(PackedAoPairMatrix::DeviceView eri,
                                        const int* first,
                                        const int* second,
                                        linalg::ConstMatrixView density,
                                        linalg::MatrixView output,
                                        double weight,
                                        int column_width) {
    extern __shared__ double sums[];
    __shared__ ExchangeReduce::TempStorage reduction[8];
    const int q = blockIdx.x, i = first[q], j = second[q];
    const int begin = blockIdx.y * column_width,
              count = min(column_width, static_cast<int>(output.rows) - begin);
    for (int k = threadIdx.x; k < 4 * count; k += blockDim.x)
        sums[k] = 0.0;
    __syncthreads();
    const std::int64_t qb = q / eri.edge;
    const int local = q % eri.edge;
    for (auto pb = qb; pb < eri.blocks; ++pb) {
        const auto d = pb - qb, tile = d * eri.blocks - d * (d - 1) / 2 + qb;
        const auto base = tile * eri.edge * eri.edge + static_cast<std::int64_t>(local) * eri.edge;
        for (int start = 0; start < eri.edge; start += blockDim.x) {
            const int row = start + threadIdx.x;
            const auto p = pb * eri.edge + row, index = base + row;
            const bool physical = row < eri.edge && p < eri.pairs;
            const bool nonzero =
                physical && p >= q && (eri.nonzero[index / 32] & (1u << (index % 32)));
            const double value = nonzero ? weight * eri.values[index] : 0.0;
            const int k = physical ? first[p] : -1, l = physical ? second[p] : -1;
            ExchangeTerms terms{0, 0, 0, 0};
            if (value != 0.0) {
                terms.a = value * density.data[j + density.ld * l];
                if (i != j)
                    terms.b = value * density.data[i + density.ld * l];
                if (p != q) {
                    terms.c = value * density.data[l + density.ld * j];
                    if (i != j)
                        terms.d = value * density.data[l + density.ld * i];
                }
            }
            const int lane = threadIdx.x & 31;
            const int previous = __shfl_up_sync(0xffffffff, k, 1);
            const bool head = lane == 0 || previous != k;
            const auto reduced =
                ExchangeReduce(reduction[threadIdx.x / 32]).HeadSegmentedSum(terms, head);
            if (head && k >= begin && k < begin + count) {
                const int col = k - begin;
                if (reduced.a != 0.0)
                    atomicAdd(sums + col, reduced.a);
                if (reduced.b != 0.0)
                    atomicAdd(sums + count + col, reduced.b);
                if (reduced.c != 0.0)
                    atomicAdd(sums + 2 * count + col, reduced.c);
                if (reduced.d != 0.0)
                    atomicAdd(sums + 3 * count + col, reduced.d);
            }
            const auto add = [&](int target, int partner) {
                if (target < begin || target >= begin + count)
                    return;
                const int col = target - begin;
                atomicAdd(sums + col, value * density.data[j + density.ld * partner]);
                if (i != j)
                    atomicAdd(sums + count + col, value * density.data[i + density.ld * partner]);
                if (p != q) {
                    atomicAdd(sums + 2 * count + col,
                              value * density.data[partner + density.ld * j]);
                    if (i != j)
                        atomicAdd(sums + 3 * count + col,
                                  value * density.data[partner + density.ld * i]);
                }
            };
            if (value != 0.0 && k != l)
                add(l, k);
            __syncwarp();
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < 4 * count; k += blockDim.x) {
        const double value = sums[k];
        if (value == 0.0)
            continue;
        const int segment = k / count, column = begin + k % count, ao = (segment & 1) ? j : i;
        atomicAdd(segment < 2 ? output.data + ao + output.ld * column
                              : output.data + column + output.ld * ao,
                  value);
    }
}

// D(l,c,nu)=P(nu,l;c), not P(l,nu;c). Retain the existing K convention
// even for numerically nonsymmetric input; no symmetrization is introduced.
__global__ void pack_density(linalg::ConstMatrixView input,
                             linalg::MatrixView d,
                             int first_nu,
                             int first_channel,
                             int channels) {
    const auto n = input.rows, work = d.rows * d.cols;
    for (auto k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < work;
         k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
        const auto l = k % n, c = k / n % channels, nu = first_nu + k / (n * channels);
        d.data[k] = input.data[nu + input.ld * (l + n * (first_channel + c))];
    }
}

__global__ void reduce_exchange(linalg::ConstMatrixView partial,
                                linalg::MatrixView output,
                                int mu,
                                int first_channel,
                                int channels,
                                int nu_count,
                                double weight) {
    const auto n = output.rows, work = n * channels;
    for (auto k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < work;
         k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
        double value = 0.0;
        for (int nu = 0; nu < nu_count; ++nu)
            value += partial.data[k + n * channels * nu];
        output.data[mu + output.ld * (k % n + n * (first_channel + k / n))] += weight * value;
    }
}

} // namespace

void ExactAoPairOperator::packed_exchange(linalg::ConstMatrixView density,
                                          linalg::MatrixView output,
                                          double weight,
                                          int channels) const {
    const nvtx3::scoped_range range{"ao.packed_exchange"};
    if (channels == 1) {
        const int width = std::min<std::size_t>(
            n_bf_,
            (exchange_shared_values_ * sizeof(double) - 8 * sizeof(ExchangeReduce::TempStorage)) /
                (4 * sizeof(double)));
        const dim3 blocks(blas_int(n_pairs_), (n_bf_ + width - 1) / width);
        exchange_packed_columns<<<blocks, 256, 4 * width * sizeof(double), context_->stream()>>>(
            packed_->view(),
            pair_first_.data(),
            pair_second_.data(),
            density,
            output,
            weight,
            width);
        check_cuda(cudaGetLastError(), "CUDA packed integral-column exchange");
        return;
    }
    const auto available = response_capacity() / sizeof(double);
    const auto n = static_cast<std::int64_t>(n_bf_), square = n * n;
    if (available < square + n)
        throw std::runtime_error("CUDA packed exchange budget cannot hold one AO plane slice");
    // One AO slice plus a partial sum is the minimum, also the storage needed
    // by the ordinary packed J contraction. Larger batches share the same plane.
    int batch = static_cast<int>(std::min<std::size_t>(channels, (available - square) / (2 * n)));
    batch = std::max(1, batch);
    const auto per_nu = square + (batch == 1 ? n : 2 * n * batch);
    const int width = static_cast<int>(std::min<std::size_t>(n, available / per_nu));
    Matrix<double> plane(*context_, n, n * width);
    Matrix<double> partial(*context_, n, static_cast<std::int64_t>(batch) * width);
    Matrix<double> d;
    if (batch > 1)
        d = Matrix<double>(*context_, n, static_cast<std::int64_t>(batch) * width);
    const double one = 1.0, zero = 0.0;
    for (int first_nu = 0; first_nu < n_bf_; first_nu += width) {
        const int count = std::min(width, n_bf_ - first_nu);
        if (batch == channels && batch > 1) {
            const auto tile = d.view().columns(0, static_cast<std::int64_t>(channels) * count);
            pack_density<<<grid(tile.rows * tile.cols), 256, 0, context_->stream()>>>(
                density, tile, first_nu, 0, channels);
            check_cuda(cudaGetLastError(), "CUDA packed exchange density channels");
        }
        for (int mu = 0; mu < n_bf_; ++mu) {
            packed_->ao_plane(mu, plane.view().columns(0, n * count), first_nu);
            for (int first_c = 0; first_c < channels; first_c += batch) {
                const int nc = std::min(batch, channels - first_c);
                if (batch > 1) {
                    const auto tile = d.view().columns(0, static_cast<std::int64_t>(nc) * count);
                    if (batch < channels) {
                        pack_density<<<grid(tile.rows * tile.cols), 256, 0, context_->stream()>>>(
                            density, tile, first_nu, first_c, nc);
                        check_cuda(cudaGetLastError(), "CUDA tiled exchange density channels");
                    }
                    check_cublas(cublasDgemmStridedBatched_64(context_->blas(),
                                                              CUBLAS_OP_T,
                                                              CUBLAS_OP_N,
                                                              n_bf_,
                                                              nc,
                                                              n_bf_,
                                                              &one,
                                                              plane.data(),
                                                              n_bf_,
                                                              square,
                                                              d.data(),
                                                              n_bf_,
                                                              n * nc,
                                                              &zero,
                                                              partial.data(),
                                                              n_bf_,
                                                              n * nc,
                                                              count),
                                 "CUDA packed batched exchange GEMM");
                } else {
                    // B is a 1-by-nbf row of the supplied density, hence OP_T and
                    // stride 1 between different nu. OP_N would silently read columns.
                    check_cublas(cublasDgemmStridedBatched_64(context_->blas(),
                                                              CUBLAS_OP_T,
                                                              CUBLAS_OP_T,
                                                              n_bf_,
                                                              1,
                                                              n_bf_,
                                                              &one,
                                                              plane.data(),
                                                              n_bf_,
                                                              square,
                                                              density.data + first_nu +
                                                                  density.ld * n * first_c,
                                                              density.ld,
                                                              1,
                                                              &zero,
                                                              partial.data(),
                                                              n_bf_,
                                                              n,
                                                              count),
                                 "CUDA packed scalar exchange GEMM");
                }
                reduce_exchange<<<grid(n * nc), 256, 0, context_->stream()>>>(
                    partial.view(), output, mu, first_c, nc, count, weight);
                check_cuda(cudaGetLastError(), "CUDA packed exchange row reduction");
            }
        }
    }
}

} // namespace xmvb::cuda
