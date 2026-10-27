#include <cstdint>
#include "cuda/ci/raw.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/linalg/blas.hpp"
#include <array>
#include "mcscf/ci/canonical.hpp"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace xmvb::cuda {
namespace {
using View = linalg::ConstMatrixView;
using Out = linalg::MatrixView;
using Mask = std::uint64_t;
unsigned grid(std::int64_t size) {
    return static_cast<unsigned>(std::min<std::int64_t>(65535, (size / 256 + (size % 256 != 0))));
}
void zero(Context& c, Out out) {
    check_cuda(
        cudaMemset2DAsync(
            out.data, out.ld * sizeof(double), 0, out.rows * sizeof(double), out.cols, c.stream()),
        "clear canonical image panel");
}
/** Sum equal integral channels within the active warp before global accumulation.
 * No per-thread or per-block four-index numerical replicas are allocated.
 */
__device__ void integral_add(double* address, double value) {
    auto active = cooperative_groups::coalesced_threads();
    const auto key = reinterpret_cast<unsigned long long>(address);
    auto equal = cooperative_groups::labeled_partition(active, key);
    if (equal.size() == 1) {
        atomicAdd(address, value);
        return;
    }
    const double sum = cooperative_groups::reduce(equal, value, cooperative_groups::plus<double>());
    if (equal.thread_rank() == 0)
        atomicAdd(address, sum);
}
__device__ void select_partition(Out& output, int slot) {
    if (output.data)
        output.data += static_cast<std::int64_t>(slot) * output.rows * output.cols;
}
__device__ void select_partition(Out& ah, Out& ag, RawCiBatch& batch, int partitions) {
    const int slot = blockIdx.x % partitions;
    select_partition(ah, slot);
    select_partition(ag, slot);
    for (int l = 0; l < batch.count; ++l) {
        select_partition(batch.lanes[l].one_adjoint, slot);
        select_partition(batch.lanes[l].two_adjoint, slot);
    }
}
struct Element {
    View one, two;
    Out one_adjoint, two_adjoint;
    const RawCiBatch& batch;
    double source, seed, delta_source[2], delta_seed[2], value = 0, delta[2]{};
    __device__ __forceinline__ void term(int a, int b, double c, bool pair) {
        const auto op = pair ? two : one;
        value += c * op.data[a + op.ld * b];
        const auto adj = pair ? two_adjoint : one_adjoint;
        if (adj.data && seed * source != 0)
            integral_add(adj.data + a + adj.ld * b, c * seed * source);
        for (int l = 0; l < batch.count; ++l) {
            const auto& lane = batch.lanes[l];
            const auto dop = pair ? lane.two : lane.one;
            if (dop.data)
                delta[l] += c * dop.data[a + dop.ld * b];
            const auto da = pair ? lane.two_adjoint : lane.one_adjoint;
            const double weight = delta_seed[l] * source + seed * delta_source[l];
            if (da.data && weight != 0)
                integral_add(da.data + a + da.ld * b, c * weight);
        }
    }
};
struct OneTerm {
    Element& e;
    __device__ void operator()(int p, int q, double c) {
        e.term(p, q, c, false);
    }
};
struct TwoTerm {
    Element& e;
    int n;
    __device__ void operator()(int p, int q, int r, int s, double c) {
        e.term(p * n + q, r * n + s, c, true);
    }
};
__device__ std::int64_t
lookup(Mask mask, const Mask* sorted, const std::int64_t* ids, std::int64_t rows) {
    std::int64_t low = 0, high = rows;
    while (low < high) {
        const auto mid = low + (high - low) / 2;
        if (sorted[mid] < mask)
            low = mid + 1;
        else
            high = mid;
    }
    return low < rows && sorted[low] == mask ? ids[low] : -1;
}
__device__ __forceinline__ void edge(Mask ta,
                                     Mask tb,
                                     Mask sa,
                                     Mask sb,
                                     std::int64_t p,
                                     int column,
                                     const Mask* sorted,
                                     const std::int64_t* ids,
                                     std::int64_t rows,
                                     int n,
                                     View x,
                                     View seed,
                                     View one,
                                     View two,
                                     Out z,
                                     Out ah,
                                     Out ag,
                                     const RawCiBatch& batch,
                                     double* sums = nullptr,
                                     std::int64_t target_row = -1) {
    bool active = false;
    for (int v = 0; v < x.cols; ++v)
        active |= x.data[p + x.ld * v] != 0;
    for (int l = 0; l < batch.count; ++l)
        active |= batch.lanes[l].source.data && batch.lanes[l].source.data[p] != 0;
    if (!active)
        return;
    const std::int64_t a = target_row >= 0 ? target_row : lookup(ta, sorted, ids, rows);
    if (a < 0)
        return;
    Element e{one, two, ah, ag, batch, x.data[p], seed.data ? seed.data[a + seed.ld * column] : 0};
    for (int l = 0; l < batch.count; ++l) {
        e.delta_source[l] = batch.lanes[l].source.data ? batch.lanes[l].source.data[p] : 0;
        e.delta_seed[l] = batch.lanes[l].seed.data[a + batch.lanes[l].seed.ld * column];
    }
    mcscf::canonical::terms(ta, tb, sa, sb, OneTerm{e}, TwoTerm{e, n});
    if (x.cols > 1) {
        for (int v = 0; v < x.cols; ++v) {
            const double value = e.value * x.data[p + x.ld * v];
            auto* out = z.data + a + z.ld * (column * x.cols + v);
            if (target_row >= 0)
                *out += value;
            else if (value != 0)
                atomicAdd(out, value);
        }
        return;
    }
    const double value = e.value * e.source;
    if (sums)
        sums[0] += value;
    else if (e.source != 0)
        atomicAdd(z.data + a + z.ld * column, value);
    for (int l = 0; l < batch.count; ++l) {
        const double dv = e.delta[l] * e.source + e.value * e.delta_source[l];
        if (sums)
            sums[l + 1] += dv;
        else
            atomicAdd(batch.lanes[l].image.data + a + batch.lanes[l].image.ld * column, dv);
    }
}
/** Source order enumerates only exact zero/single/double excitation neighborhoods. */
__global__ void raw_image(const std::int64_t* alpha,
                          const std::int64_t* beta,
                          std::int64_t groups,
                          const Mask* am,
                          const Mask* bm,
                          const Mask* sorted,
                          const std::int64_t* ids,
                          std::int64_t rows,
                          int n,
                          const std::int64_t* columns,
                          int count,
                          View x,
                          View seed,
                          View one,
                          View two,
                          Out z,
                          Out ah,
                          Out ag,
                          RawCiBatch batch,
                          int partitions) {
    select_partition(ah, ag, batch, partitions);
    const Mask full = (Mask{1} << n) - 1;
    for (auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < groups * count;
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const auto p = index % groups;
        const int j = index / groups;
        bool active = false;
        for (int v = 0; v < x.cols; ++v)
            active |= x.data[p + x.ld * v] != 0;
        for (int l = 0; l < batch.count; ++l)
            active |= batch.lanes[l].source.data && batch.lanes[l].source.data[p] != 0;
        if (!active)
            continue;
        const Mask sa = am[alpha[p]], sb = bm[beta[p]], tb = bm[columns[j]];
        const int distance = __popcll(sb & ~tb);
        if (distance > 2)
            continue;
        edge(sa, tb, sa, sb, p, j, sorted, ids, rows, n, x, seed, one, two, z, ah, ag, batch);
        if (distance < 2)
            for (Mask rm = sa; rm; rm &= rm - 1) {
                const Mask r = rm & -rm;
                for (Mask cm = full ^ sa; cm; cm &= cm - 1) {
                    const Mask c = cm & -cm;
                    edge(sa ^ r ^ c,
                         tb,
                         sa,
                         sb,
                         p,
                         j,
                         sorted,
                         ids,
                         rows,
                         n,
                         x,
                         seed,
                         one,
                         two,
                         z,
                         ah,
                         ag,
                         batch);
                }
            }
        if (distance == 0)
            for (Mask r0 = sa; r0; r0 &= r0 - 1)
                for (Mask r1 = r0 & (r0 - 1); r1; r1 &= r1 - 1)
                    for (Mask c0 = full ^ sa; c0; c0 &= c0 - 1)
                        for (Mask c1 = c0 & (c0 - 1); c1; c1 &= c1 - 1)
                            edge(sa ^ (r0 & -r0) ^ (r1 & -r1) ^ (c0 & -c0) ^ (c1 & -c1),
                                 tb,
                                 sa,
                                 sb,
                                 p,
                                 j,
                                 sorted,
                                 ids,
                                 rows,
                                 n,
                                 x,
                                 seed,
                                 one,
                                 two,
                                 z,
                                 ah,
                                 ag,
                                 batch);
    }
}
struct CovalentConsumer {
    Mask a, b, full;
    std::int64_t column, row, rows;
    int n;
    const Mask* sorted;
    const std::int64_t* ids;
    View x, seed, one, two;
    Out z, ah, ag;
    const RawCiBatch& batch;
    double* sums;
    __device__ void operator()(Mask source) const {
        const std::int64_t p = lookup(source, sorted, ids, rows);
        edge(a,
             b,
             source,
             full ^ source,
             p,
             column,
             sorted,
             ids,
             rows,
             n,
             x,
             seed,
             one,
             two,
             z,
             ah,
             ag,
             batch,
             sums,
             row);
    }
};
__device__ Mask site(Mask bits, int position) {
    while (position--)
        bits &= bits - 1;
    return bits & -bits;
}
__device__ Mask pair_sites(Mask bits, int rank) {
    int high = 1;
    while (rank >= high) {
        rank -= high;
        ++high;
    }
    return site(bits, high) | site(bits, rank);
}
/** One owner per allowed COV target; no full-alpha scan or image atomics. */
__global__ void covalent_image(RawCiTopology t,
                               const std::int64_t* columns,
                               int count,
                               View x,
                               View seed,
                               View one,
                               View two,
                               Out z,
                               Out ah,
                               Out ag,
                               RawCiBatch batch,
                               int partitions) {
    select_partition(ah, ag, batch, partitions);
    const Mask full = (Mask{1} << t.orbitals) - 1;
    const int singles = t.alpha_electrons * t.beta_electrons;
    const int alpha_pairs = t.alpha_electrons * (t.alpha_electrons - 1) / 2;
    const int beta_pairs = t.beta_electrons * (t.beta_electrons - 1) / 2;
    const int targets = 1 + singles + alpha_pairs * beta_pairs;
    for (auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < static_cast<std::int64_t>(targets) * count;
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int j = index / targets, local = index % targets;
        const Mask tb = t.beta_masks[columns[j]], holes = full ^ tb;
        Mask ta = holes;
        if (local > singles) {
            const int pair = local - 1 - singles;
            ta ^= pair_sites(tb, pair / alpha_pairs) ^ pair_sites(holes, pair % alpha_pairs);
        } else if (local) {
            const int single = local - 1;
            ta ^= site(tb, single / t.alpha_electrons) ^ site(holes, single % t.alpha_electrons);
        }
        const std::int64_t a = lookup(ta, t.sorted_masks, t.sorted_ids, t.rows);
        double sums[3]{};
        mcscf::canonical::covalent_sources(ta,
                                           tb,
                                           full,
                                           CovalentConsumer{ta,
                                                            tb,
                                                            full,
                                                            j,
                                                            a,
                                                            t.rows,
                                                            t.orbitals,
                                                            t.sorted_masks,
                                                            t.sorted_ids,
                                                            x,
                                                            seed,
                                                            one,
                                                            two,
                                                            z,
                                                            ah,
                                                            ag,
                                                            batch,
                                                            sums});
        if (x.cols == 1) {
            z.data[a + z.ld * j] = sums[0];
            for (int l = 0; l < batch.count; ++l)
                batch.lanes[l].image.data[a + batch.lanes[l].image.ld * j] = sums[l + 1];
        }
    }
}
__global__ void fill_ones(Out out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.rows)
        out.data[i] = 1;
}
/** Partitioned orbital cotangents, then one cuBLAS reduction per consumer.
 * The extra numerical payload is bounded by the current image panel (or one
 * orbital cotangent for tiny panels), never a copy per GPU thread or all blocks.
 */
struct IntegralReduction {
    std::array<Out, 3> one_outputs{}, two_outputs{};
    std::array<Matrix<double>, 3> one, two;
    Matrix<double> ones;
    int partitions = 1;
    IntegralReduction(
        Context& c, const RawCiTopology& t, Out z, Out& ah, Out& ag, RawCiBatch& batch) {
        one_outputs[0] = ah;
        two_outputs[0] = ag;
        for (int l = 0; l < batch.count; ++l) {
            one_outputs[l + 1] = batch.lanes[l].one_adjoint;
            two_outputs[l + 1] = batch.lanes[l].two_adjoint;
        }
        std::size_t channels = 0;
        for (int i = 0; i <= batch.count; ++i) {
            channels += one_outputs[i].rows * one_outputs[i].cols;
            channels += two_outputs[i].rows * two_outputs[i].cols;
        }
        if (!channels)
            return;
        const auto panel = static_cast<std::size_t>(z.rows * z.cols) * (1 + batch.count);
        partitions = std::min<std::size_t>(t.multiprocessors,
                                           std::max<std::size_t>(1, panel / (channels + 1)));
        // One partition has no reduction: accumulate directly into its output.
        if (partitions == 1)
            return;
        ones = Matrix<double>(c, partitions, 1);
        fill_ones<<<grid(partitions), 256, 0, c.stream()>>>(ones.view());
        for (int i = 0; i <= batch.count; ++i) {
            const auto prepare = [&](Out output, Matrix<double>& storage) {
                if (!output.data)
                    return Out{};
                storage = Matrix<double>(c, output.rows * output.cols, partitions);
                zero(c, storage.view());
                return Out{storage.data(), output.rows, output.cols, output.rows};
            };
            const auto h = prepare(one_outputs[i], one[i]);
            const auto g = prepare(two_outputs[i], two[i]);
            if (!i) {
                ah = h;
                ag = g;
            } else {
                batch.lanes[i - 1].one_adjoint = h;
                batch.lanes[i - 1].two_adjoint = g;
            }
        }
    }
    void finish(Context& c) const {
        if (!ones.data())
            return;
        for (int i = 0; i < 3; ++i) {
            const auto reduce = [&](Out target, const Matrix<double>& partial) {
                if (target.data) {
                    const auto size = target.rows * target.cols;
                    gemm(c, {target.data, size, 1, size}, partial.view(), ones.view(), 1, 1);
                }
            };
            reduce(one_outputs[i], one[i]);
            reduce(two_outputs[i], two[i]);
        }
    }
};
} // namespace
void apply_raw_ci(Context& c,
                  const RawCiTopology& t,
                  View x,
                  const std::int64_t* columns,
                  int count,
                  View one,
                  View two,
                  Out z,
                  View seed,
                  Out ah,
                  Out ag,
                  RawCiBatch batch) {
    IntegralReduction reduction(c, t, z, ah, ag, batch);
    zero(c, z);
    for (int l = 0; l < batch.count; ++l)
        zero(c, batch.lanes[l].image);
    if (t.covalent) {
        const auto pairs = [](int k) { return k * (k - 1) / 2; };
        const std::int64_t targets = 1 + t.alpha_electrons * t.beta_electrons +
                                     pairs(t.alpha_electrons) * pairs(t.beta_electrons);
        covalent_image<<<grid(targets * count), 256, 0, c.stream()>>>(
            t, columns, count, x, seed, one, two, z, ah, ag, batch, reduction.partitions);
    } else {
        raw_image<<<grid(t.groups * count), 256, 0, c.stream()>>>(t.alpha,
                                                                  t.beta,
                                                                  t.groups,
                                                                  t.alpha_masks,
                                                                  t.beta_masks,
                                                                  t.sorted_masks,
                                                                  t.sorted_ids,
                                                                  t.rows,
                                                                  t.orbitals,
                                                                  columns,
                                                                  count,
                                                                  x,
                                                                  seed,
                                                                  one,
                                                                  two,
                                                                  z,
                                                                  ah,
                                                                  ag,
                                                                  batch,
                                                                  reduction.partitions);
    }
    check_cuda(cudaGetLastError(), "canonical source image and adjoints");
    reduction.finish(c);
}
} // namespace xmvb::cuda
