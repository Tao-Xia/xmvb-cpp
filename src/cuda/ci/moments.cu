#include "core/index.hpp"
#include <cstdint>
#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

namespace xmvb::cuda {
namespace {
constexpr int threads = 256;
constexpr std::int64_t moment_chunk = 4096;

struct StreamEvent {
    std::int64_t source, local, slot;
    int created, annihilated;
    double sign;
};
struct EventBatch {
    std::int64_t begin, end;
    std::vector<std::int64_t> sources;
};

// The metadata may cover the beta links in this small panel. Numerical
// spectator storage below is bounded by the panel width.
std::pair<std::vector<StreamEvent>, std::vector<EventBatch>>
beta_event_batches(const mcscf::SpinConnections& graph, int first, int count) {
    std::vector<StreamEvent> events;
    for (int local = 0; local < count; ++local)
        for (const auto& link : graph.singles(first + local))
            events.push_back({link.source, local, 0, link.created, link.annihilated, link.sign});
    std::stable_sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.source < b.source;
    });
    std::vector<EventBatch> batches;
    for (std::size_t begin = 0; begin < events.size();) {
        EventBatch batch{core::checked_index<std::int64_t>(begin), 0, {}};
        std::size_t end = begin;
        for (; end < events.size(); ++end) {
            if (end == begin || events[end].source != events[end - 1].source) {
                if (batch.sources.size() == static_cast<std::size_t>(count))
                    break;
                batch.sources.push_back(events[end].source);
            }
            events[end].slot = core::checked_index<std::int64_t>(batch.sources.size()) - 1;
        }
        batch.end = core::checked_index<std::int64_t>(end);
        batches.push_back(std::move(batch));
        begin = end;
    }
    return {std::move(events), std::move(batches)};
}

struct AddMoments {
    __device__ double2 operator()(double2 a, double2 b) const {
        return make_double2(a.x + b.x, a.y + b.y);
    }
};
/** One independent oriented bilinear; common readers share the same tile. */
struct StreamedMomentTerm {
    linalg::ConstMatrixView left, right;
    linalg::MatrixView output;
    double scale;
    bool transpose;
};
template <int Count> struct MomentBatch {
    StreamedMomentTerm terms[Count];
};
template <int Count> struct MomentSum {
    double values[2 * Count]{};
};
template <int Count> struct AddMomentSum {
    __device__ MomentSum<Count> operator()(MomentSum<Count> a, const MomentSum<Count>& b) const {
#pragma unroll
        for (int i = 0; i < 2 * Count; ++i)
            a.values[i] += b.values[i];
        return a;
    }
};

/** Link lookup and occupation decoding are shared across all derivative lanes. */
template <int Count, bool SharedPairs>
__global__ void reduce_streamed_alpha(MomentBatch<Count> batch,
                                      const std::int64_t* alpha,
                                      const std::int64_t* beta,
                                      int n_alpha,
                                      int n_beta,
                                      int first,
                                      bool alpha_only) {
    using Reduce = cub::BlockReduce<MomentSum<Count>, threads>;
    __shared__ typename Reduce::TempStorage storage;
    const int channel = blockIdx.x;
    int high = 0;
    while ((high + 1) * (high + 2) / 2 <= channel)
        ++high;
    const int low = channel - high * (high + 1) / 2;
    const int diagonal = high * (high + 1) / 2 + high;
    MomentSum<Count> sum;
    const auto size = static_cast<std::int64_t>(n_alpha) * batch.terms[0].left.cols;
    for (auto begin = static_cast<std::int64_t>(blockIdx.y) * moment_chunk; begin < size;
         begin += static_cast<std::int64_t>(gridDim.y) * moment_chunk)
        for (auto index = begin + threadIdx.x; index < size && index < begin + moment_chunk;
             index += threads) {
            const std::int64_t a = index % n_alpha, local = index / n_alpha;
            double weights[Count];
            bool nonzero = false;
#pragma unroll
            for (int t = 0; t < Count; ++t) {
                const auto left = batch.terms[t].left;
                if constexpr (SharedPairs && Count == 4) {
                    if (t >= 2)
                        weights[t] = weights[t - 2];
                    else
                        weights[t] = left.data[a + left.ld * local];
                } else
                    weights[t] = left.data[a + left.ld * local];
                nonzero |= weights[t] != 0.0;
            }
            // Exact support pruning, shared by all independent outputs.
            if (!nonzero)
                continue;
            std::int64_t source = a, slot = 0;
            double sign = 1;
            if (high == low) {
                sign = (alpha[static_cast<std::int64_t>(diagonal) * n_alpha + a] != 0) +
                       (alpha_only
                            ? 0
                            : (beta[static_cast<std::int64_t>(diagonal) * n_beta + first + local] !=
                               0));
            } else {
                const int link = alpha[static_cast<std::int64_t>(channel) * n_alpha + a];
                if (!link)
                    continue;
                source = link > 0 ? link - 1 : -link - 1;
                sign = link > 0 ? 1.0 : -1.0;
                slot = alpha[static_cast<std::int64_t>(diagonal) * n_alpha + a] != 0 ? 0 : 1;
            }
            if constexpr (SharedPairs) {
                const auto r0 = batch.terms[0].right;
                const double v0 = r0.data[source + r0.ld * local];
                sum.values[slot] += sign * weights[0] * v0;
                sum.values[2 + slot] += sign * weights[1] * v0;
                if constexpr (Count == 4) {
                    const auto r1 = batch.terms[2].right;
                    const double v1 = r1.data[source + r1.ld * local];
                    sum.values[4 + slot] += sign * weights[0] * v1;
                    sum.values[6 + slot] += sign * weights[1] * v1;
                }
            } else {
#pragma unroll
                for (int t = 0; t < Count; ++t) {
                    const auto term = batch.terms[t];
                    if (weights[t] != 0.0)
                        sum.values[2 * t + slot] +=
                            sign * weights[t] * term.right.data[source + term.right.ld * local];
                }
            }
        }
    sum = Reduce(storage).Reduce(sum, AddMomentSum<Count>{});
    if (threadIdx.x == 0) {
#pragma unroll
        for (int t = 0; t < Count; ++t) {
            const auto term = batch.terms[t];
            const std::int64_t row = term.transpose ? low : high, col = term.transpose ? high : low;
            atomicAdd(term.output.data + row + term.output.ld * col,
                      term.scale * sum.values[2 * t]);
            if (high != low)
                atomicAdd(term.output.data + col + term.output.ld * row,
                          term.scale * sum.values[2 * t + 1]);
        }
    }
}

template <int Count, bool SharedPairs>
__global__ void
reduce_streamed_beta(MomentBatch<Count> batch, const StreamEvent* events, int begin) {
    using Reduce = cub::BlockReduce<MomentSum<Count>, threads>;
    __shared__ typename Reduce::TempStorage storage;
    const auto event = events[begin + blockIdx.x];
    MomentSum<Count> sum;
    for (std::int64_t a = threadIdx.x; a < batch.terms[0].left.rows; a += threads) {
        if constexpr (SharedPairs) {
            const auto l0 = batch.terms[0].left, l1 = batch.terms[1].left,
                       r0 = batch.terms[0].right;
            const double w0 = l0.data[a + l0.ld * event.local],
                         w1 = l1.data[a + l1.ld * event.local];
            const double v0 = r0.data[a + r0.ld * event.slot];
            sum.values[0] += w0 * v0;
            sum.values[2] += w1 * v0;
            if constexpr (Count == 4) {
                const auto r1 = batch.terms[2].right;
                const double v1 = r1.data[a + r1.ld * event.slot];
                sum.values[4] += w0 * v1;
                sum.values[6] += w1 * v1;
            }
        } else {
#pragma unroll
            for (int t = 0; t < Count; ++t) {
                const auto term = batch.terms[t];
                sum.values[2 * t] += term.left.data[a + term.left.ld * event.local] *
                                     term.right.data[a + term.right.ld * event.slot];
            }
        }
    }
    sum = Reduce(storage).Reduce(sum, AddMomentSum<Count>{});
    if (threadIdx.x == 0) {
#pragma unroll
        for (int t = 0; t < Count; ++t) {
            const auto term = batch.terms[t];
            const std::int64_t row = term.transpose ? event.annihilated : event.created;
            const int col = term.transpose ? event.created : event.annihilated;
            atomicAdd(term.output.data + row + term.output.ld * col,
                      term.scale * event.sign * sum.values[2 * t]);
        }
    }
}

/** CUB supplies reductions; only irregular fermionic link gathering is custom. */
template <int Count, bool SharedPairs = false>
void launch_moments(Context& c,
                    std::span<const StreamedMomentTerm> terms,
                    dim3 grid,
                    const std::int64_t* alpha,
                    const std::int64_t* beta,
                    int na,
                    int nb,
                    int first,
                    bool alpha_only,
                    const StreamEvent* events,
                    int begin,
                    int end) {
    MomentBatch<Count> batch;
    std::copy_n(terms.begin(), Count, batch.terms);
    if (events)
        reduce_streamed_beta<Count, SharedPairs>
            <<<end - begin, threads, 0, c.stream()>>>(batch, events, begin);
    else
        reduce_streamed_alpha<Count, SharedPairs>
            <<<grid, threads, 0, c.stream()>>>(batch, alpha, beta, na, nb, first, alpha_only);
    check_cuda(cudaGetLastError(), "CUDA fused streamed generator reduction");
}
void launch_moments(Context& c,
                    std::span<const StreamedMomentTerm> terms,
                    dim3 grid,
                    const std::int64_t* alpha,
                    const std::int64_t* beta,
                    int na,
                    int nb,
                    int first,
                    bool alpha_only,
                    const StreamEvent* events = nullptr,
                    std::int64_t begin = 0,
                    std::int64_t end = 0) {
    const auto same = [](auto a, auto b) {
        return a.data == b.data && a.ld == b.ld && a.rows == b.rows && a.cols == b.cols;
    };
    // The cached coupled jet is a 2x2 bilinear: two left panels, two right
    // readers, four oriented outputs. Read each numerical input once.
    if (terms.size() == 4 && same(terms[0].left, terms[2].left) &&
        same(terms[1].left, terms[3].left) && same(terms[0].right, terms[1].right) &&
        same(terms[2].right, terms[3].right)) {
        launch_moments<4, true>(
            c, terms, grid, alpha, beta, na, nb, first, alpha_only, events, begin, end);
        return;
    }
    if (terms.size() == 2 && same(terms[0].right, terms[1].right)) {
        launch_moments<2, true>(
            c, terms, grid, alpha, beta, na, nb, first, alpha_only, events, begin, end);
        return;
    }
    switch (terms.size()) {
#define XMVB_MOMENT_CASE(N)                                                                        \
    case N:                                                                                        \
        launch_moments<N>(                                                                         \
            c, terms, grid, alpha, beta, na, nb, first, alpha_only, events, begin, end);           \
        break
        XMVB_MOMENT_CASE(1);
        XMVB_MOMENT_CASE(2);
        XMVB_MOMENT_CASE(3);
        XMVB_MOMENT_CASE(4);
#undef XMVB_MOMENT_CASE
        default:
            throw std::logic_error("invalid streamed generator batch width");
    }
}

__global__ void finish_streamed(linalg::ConstMatrixView raw,
                                linalg::MatrixView output,
                                bool symmetric_sum,
                                bool add) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output.rows * output.cols)
        return;
    const std::int64_t row = index % output.rows, col = index / output.rows;
    if (row > col) {
        if (!add)
            output.data[row + output.ld * col] = 0.0;
        return;
    }
    double value = raw.data[row + raw.ld * col];
    if (symmetric_sum && row != col)
        value += raw.data[col + raw.ld * row];
    if (symmetric_sum && row == col)
        value *= 2.0;
    if (add)
        output.data[row + output.ld * col] += value;
    else
        output.data[row + output.ld * col] = value;
}

__global__ void apply_streamed_alpha(linalg::ConstMatrixView input,
                                     linalg::ConstMatrixView generator,
                                     const std::int64_t* alpha,
                                     const std::int64_t* beta,
                                     int n_alpha,
                                     int n_beta,
                                     int first,
                                     int orbitals,
                                     bool alpha_only,
                                     linalg::MatrixView output) {
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= static_cast<std::int64_t>(n_alpha) * input.cols)
        return;
    const std::int64_t a = static_cast<std::int64_t>(index % n_alpha),
                       local = static_cast<int>(index / n_alpha);
    const std::int64_t b = first + local;
    const double self = input.data[a + input.ld * local];
    double value = 0.0;
    for (int p = 0; p < orbitals; ++p) {
        const int diagonal = p * (p + 1) / 2 + p;
        const int occupied =
            (alpha[static_cast<std::int64_t>(diagonal) * n_alpha + a] != 0) +
            (alpha_only ? 0 : (beta[static_cast<std::int64_t>(diagonal) * n_beta + b] != 0));
        value += occupied * generator.data[p + generator.ld * p] * self;
    }
    for (int high = 1; high < orbitals; ++high) {
        const int diagonal = high * (high + 1) / 2 + high;
        const bool alpha_high = alpha[static_cast<std::int64_t>(diagonal) * n_alpha + a] != 0;
        for (int low = 0; low < high; ++low) {
            const int channel = high * (high + 1) / 2 + low;
            const int signed_source = alpha[static_cast<std::int64_t>(channel) * n_alpha + a];
            if (!signed_source)
                continue;
            const std::int64_t source = signed_source > 0 ? signed_source - 1 : -signed_source - 1;
            const int created = alpha_high ? high : low, annihilated = alpha_high ? low : high;
            value += (signed_source > 0 ? 1.0 : -1.0) *
                     generator.data[created + generator.ld * annihilated] *
                     input.data[source + input.ld * local];
        }
    }
    output.data[a + output.ld * local] = value;
}

__global__ void apply_streamed_beta(linalg::ConstMatrixView input,
                                    linalg::ConstMatrixView generator,
                                    const StreamEvent* events,
                                    int begin,
                                    linalg::MatrixView output) {
    const std::int64_t a = blockIdx.y * blockDim.x + threadIdx.x;
    if (a >= output.rows)
        return;
    const auto e = events[begin + blockIdx.x];
    const double factor = e.sign * generator.data[e.created + generator.ld * e.annihilated];
    if (factor != 0.0)
        atomicAdd(output.data + a + output.ld * e.local,
                  factor * input.data[a + input.ld * e.slot]);
}

// One packed channel, both E_pq orientations. A partition owns beta
// columns and visits their contiguous alpha rows. The beta link and the
// representative triangle boundary are decoded once per column, not once
// per product and orbital channel. No product-sized index map is needed.
__global__ void reduce_moments(linalg::ConstMatrixView left,
                               linalg::ConstMatrixView right,
                               const std::int64_t* alpha,
                               const std::int64_t* beta,
                               int n_alpha,
                               int n_beta,
                               int parity,
                               double2* partials) {
    using Reduce = cub::BlockReduce<double2, threads>;
    __shared__ typename Reduce::TempStorage scratch;
    const int channel = blockIdx.x;
    int high = 0;
    while ((high + 1) * (high + 2) / 2 <= channel)
        ++high;
    const int low = channel - high * (high + 1) / 2;
    const int diagonal = high * (high + 1) / 2 + high;
    double2 sum = make_double2(0.0, 0.0);
    for (std::int64_t column = blockIdx.y; column < right.cols; column += gridDim.y) {
        const std::int64_t b = static_cast<std::int64_t>(column % n_beta);
        const int sb = beta[static_cast<std::int64_t>(channel) * n_beta + b];
        const bool beta_upper = beta[static_cast<std::int64_t>(diagonal) * n_beta + b] != 0;
        for (std::int64_t a = (parity ? b : 0) + threadIdx.x; a < n_alpha; a += threads) {
            const int sa = alpha[static_cast<std::int64_t>(channel) * n_alpha + a];
            if (!sa && !sb)
                continue;
            const double weight = (parity && a != b ? 2.0 : 1.0) * left.data[a + left.ld * column];
            if (high == low) {
                sum.x += weight * ((sa != 0) + (sb != 0)) * right.data[a + right.ld * column];
                continue;
            }
            double upper = 0.0, lower = 0.0;
            if (sa) {
                const double value =
                    (sa > 0 ? 1.0 : -1.0) * right.data[abs(sa) - 1 + right.ld * column];
                if (alpha[static_cast<std::int64_t>(diagonal) * n_alpha + a])
                    upper += value;
                else
                    lower += value;
            }
            if (sb) {
                const double value =
                    (sb > 0 ? 1.0 : -1.0) * right.data[a + right.ld * (column - b + abs(sb) - 1)];
                if (beta_upper)
                    upper += value;
                else
                    lower += value;
            }
            sum.x += weight * upper;
            sum.y += weight * lower;
        }
    }
    sum = Reduce(scratch).Reduce(sum, AddMoments{});
    if (threadIdx.x == 0)
        partials[static_cast<std::int64_t>(channel) * gridDim.y + blockIdx.y] = sum;
}

__global__ void store_moments(const double2* partials, int partitions, linalg::MatrixView output) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output.rows * output.cols)
        return;
    const std::int64_t row = index % output.rows, column = index / output.rows;
    const int high = max(row, column), low = min(row, column);
    const int channel = high * (high + 1) / 2 + low;
    double sum = 0.0;
    for (int i = 0; i < partitions; ++i) {
        const auto value = partials[static_cast<std::int64_t>(channel) * partitions + i];
        sum += row >= column ? value.x : value.y;
    }
    output.data[row + output.ld * column] = sum;
}
} // namespace

void DirectCiSigma::generator_moments(std::span<const linalg::ConstMatrixView> left,
                                      linalg::ConstMatrixView right,
                                      std::span<const linalg::MatrixView> outputs) const {
    const nvtx3::scoped_range range{"ci.generator_moments"};
    if (left.empty() || left.size() != outputs.size() || right.rows != n_alpha_ ||
        right.cols <= 0 || right.cols % n_beta_ || right.ld < right.rows || !right.data)
        throw std::invalid_argument("CUDA generator moment dimensions");
    for (std::size_t i = 0; i < left.size(); ++i) {
        const auto l = left[i];
        const auto out = outputs[i];
        if (l.rows != right.rows || l.cols != right.cols || l.ld < l.rows || !l.data ||
            out.rows != n_orbitals_ || out.cols != n_orbitals_ || out.ld < out.rows || !out.data ||
            out.data == l.data || out.data == right.data)
            throw std::invalid_argument("invalid CUDA generator moment views");
    }
    if (!workspace_->moment_partitions) {
        int device = 0, multiprocessors = 0, blocks = 0;
        check_cuda(cudaGetDevice(&device), "generator moment device");
        check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                   "generator moment multiprocessors");
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reduce_moments, threads, 0),
            "generator moment occupancy");
        workspace_->moment_partitions =
            std::max(1, (multiprocessors * blocks + n_channels_ - 1) / n_channels_);
    }
    const int partitions = static_cast<std::int64_t>(
        std::min<std::int64_t>(workspace_->moment_partitions, right.cols));
    const auto count = static_cast<std::size_t>(n_channels_) * partitions;
    Buffer<double2> partials(*context_, count);
    for (std::size_t i = 0; i < left.size(); ++i) {
        reduce_moments<<<dim3(n_channels_, partitions), threads, 0, context_->stream()>>>(
            left[i],
            right,
            alpha_->sources.data(),
            beta_->sources.data(),
            n_alpha_,
            n_beta_,
            exchange_parity_,
            partials.data());
        check_cuda(cudaGetLastError(), "CUDA direct link generator moments");
        const auto cells = static_cast<std::int64_t>(n_orbitals_) * n_orbitals_;
        store_moments<<<static_cast<unsigned int>((cells + threads - 1) / threads),
                        threads,
                        0,
                        context_->stream()>>>(partials.data(), partitions, outputs[i]);
        check_cuda(cudaGetLastError(), "CUDA generator moment partial reduction");
    }
}

void DirectCiSigma::generator_adjoint_streamed(int first,
                                               std::span<const CiStreamedGeneratorBatch> requests,
                                               std::span<const linalg::MatrixView> outputs,
                                               bool symmetric_sum,
                                               bool spin_exchange_reduced,
                                               bool add) const {
    const nvtx3::scoped_range range{"ci.generator_adjoint_streamed"};
    if (outputs.empty() || requests.empty() || requests.front().terms.empty() || first < 0 ||
        requests.front().terms.front().left.cols <= 0 ||
        requests.front().terms.front().left.cols > n_beta_ ||
        first > n_beta_ - requests.front().terms.front().left.cols ||
        (spin_exchange_reduced && (!exchange_parity_ || !shared_spin_graph_)))
        throw std::invalid_argument("invalid CUDA streamed generator panel");
    const std::int64_t count = static_cast<std::int64_t>(requests.front().terms.front().left.cols);
    for (const auto output : outputs)
        if (!output.data || output.rows != n_orbitals_ || output.cols != n_orbitals_ ||
            output.ld < output.rows)
            throw std::invalid_argument("invalid CUDA streamed generator output");
    for (std::size_t i = 0; i < outputs.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (outputs[i].data == outputs[j].data)
                throw std::invalid_argument("duplicate CUDA streamed generator output");
    for (const auto& request : requests) {
        if (!request.read || request.terms.empty() || !std::isfinite(request.scale))
            throw std::invalid_argument("invalid CUDA streamed generator reader");
        for (const auto& term : request.terms) {
            const auto left = term.left;
            if (term.output >= outputs.size() || !left.data || left.rows != n_alpha_ ||
                left.cols != count || left.ld < left.rows || outputs[term.output].data == left.data)
                throw std::invalid_argument("invalid CUDA streamed generator left or output");
        }
    }

    auto& context = *context_;
    std::vector<std::int64_t> rows(n_alpha_), betas(count);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(betas.begin(), betas.end(), first);
    Matrix<double> tile(context, n_alpha_, count);
    std::vector<Matrix<double>> reductions;
    reductions.reserve(outputs.size());
    const auto orbital_cells = static_cast<std::size_t>(n_orbitals_) * n_orbitals_;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        reductions.emplace_back(context, n_orbitals_, n_orbitals_);
        check_cuda(
            cudaMemsetAsync(
                reductions.back().data(), 0, orbital_cells * sizeof(double), context.stream()),
            "clear CUDA streamed generator reduction");
    }
    std::vector<StreamEvent> events;
    std::vector<EventBatch> batches;
    Buffer<StreamEvent> device_events;
    if (!spin_exchange_reduced) {
        auto plan = beta_event_batches(beta_->topology, first, count);
        events = std::move(plan.first);
        batches = std::move(plan.second);
        device_events = Buffer<StreamEvent>(context, events.size());
        context.copy_to_device(
            device_events.data(), events.data(), events.size() * sizeof(StreamEvent));
    }
    // Two readers can share one link sweep. The extra storage is one bounded
    // single-spin tile, admitted from tracked headroom without a device query.
    const bool pair_readers =
        std::adjacent_find(requests.begin(), requests.end(), [](const auto& a, const auto& b) {
            return a.terms.size() + b.terms.size() <= 4;
        }) != requests.end();
    Matrix<double> other_tile;
    const auto tile_bytes = static_cast<std::size_t>(n_alpha_) * count * sizeof(double);
    if (pair_readers &&
        tile_bytes <= context.device_byte_budget() - context.counters().current_device_bytes)
        other_tile = Matrix<double>(context, n_alpha_, count);
    const auto cells = static_cast<std::int64_t>(n_alpha_) * count;
    const dim3 grid(n_channels_,
                    static_cast<unsigned>(
                        std::min<std::int64_t>(65535, (cells + moment_chunk - 1) / moment_chunk)));
    for (std::size_t r = 0; r < requests.size();) {
        const bool paired = other_tile.data() && r + 1 < requests.size() &&
                            requests[r].terms.size() + requests[r + 1].terms.size() <= 4;
        const int readers = paired ? 2 : 1;
        const std::array<linalg::MatrixView, 2> tiles{tile.view(), other_tile.view()};
        std::vector<StreamedMomentTerm> terms;
        for (int j = 0; j < readers; ++j) {
            const auto& request = requests[r + j];
            request.read(betas, rows, tiles[j]);
            for (const auto& term : request.terms)
                terms.push_back({term.left,
                                 tiles[j],
                                 reductions[term.output].view(),
                                 request.scale * (spin_exchange_reduced ? 2.0 : 1.0),
                                 request.transpose});
        }
        const auto reduce =
            [&](const StreamEvent* event = nullptr, std::int64_t begin = 0, std::int64_t end = 0) {
                for (std::size_t t = 0; t < terms.size(); t += 4)
                    launch_moments(context,
                                   std::span<const StreamedMomentTerm>{terms}.subspan(
                                       t, std::min<std::size_t>(4, terms.size() - t)),
                                   grid,
                                   alpha_->sources.data(),
                                   beta_->sources.data(),
                                   n_alpha_,
                                   n_beta_,
                                   first,
                                   spin_exchange_reduced,
                                   event,
                                   begin,
                                   end);
            };
        reduce();
        if (!spin_exchange_reduced)
            for (const auto& batch : batches) {
                for (int j = 0; j < readers; ++j)
                    requests[r + j].read(
                        batch.sources, rows, tiles[j].columns(0, batch.sources.size()));
                // Exchange reduction is disabled here, so request scales are
                // identical for alpha and beta contractions.
                reduce(device_events.data(), batch.begin, batch.end);
            }
        r += readers;
    }
    for (std::size_t l = 0; l < outputs.size(); ++l) {
        finish_streamed<<<static_cast<unsigned int>((orbital_cells + threads - 1) / threads),
                          threads,
                          0,
                          context.stream()>>>(reductions[l].view(), outputs[l], symmetric_sum, add);
        check_cuda(cudaGetLastError(), "CUDA streamed generator finish");
    }
}

void DirectCiSigma::spin_generator_adjoints(bool beta,
                                            std::span<const CiSpinGeneratorTerm> terms,
                                            std::span<const linalg::MatrixView> outputs,
                                            bool add) const {
    const int rows = beta ? n_beta_ : n_alpha_;
    auto& c = *context_;
    for (auto out : outputs)
        if (!add)
            check_cuda(cudaMemset2DAsync(out.data,
                                         out.ld * sizeof(double),
                                         0,
                                         out.rows * sizeof(double),
                                         out.cols,
                                         c.stream()),
                       "clear single-spin moment");
    const auto& graph = beta ? beta_ : alpha_;
    for (std::size_t first = 0; first < terms.size(); first += 4) {
        std::vector<StreamedMomentTerm> batch;
        for (std::size_t t = first; t < std::min(first + 4, terms.size()); ++t) {
            const auto& term = terms[t];
            if (term.output >= outputs.size() || term.left.rows != rows ||
                term.right.rows != rows || term.left.cols != term.right.cols)
                throw std::invalid_argument("invalid single-spin moment panels");
            batch.push_back({term.left, term.right, outputs[term.output], term.scale, false});
        }
        const auto cells = static_cast<std::int64_t>(rows) * batch.front().left.cols;
        launch_moments(c,
                       batch,
                       dim3(n_channels_,
                            static_cast<unsigned>(std::min<std::int64_t>(
                                65535, (cells + moment_chunk - 1) / moment_chunk))),
                       graph->sources.data(),
                       nullptr,
                       rows,
                       0,
                       0,
                       true);
    }
}

void DirectCiSigma::apply_one_body_generator_columns(int first,
                                                     int count,
                                                     const CiTileReader& columns,
                                                     linalg::ConstMatrixView generator,
                                                     linalg::MatrixView image,
                                                     bool alpha_only) const {
    const nvtx3::scoped_range range{"ci.apply_generator_columns"};
    if (!columns || first < 0 || count <= 0 || count > n_beta_ || first > n_beta_ - count ||
        !generator.data || generator.rows != n_orbitals_ || generator.cols != n_orbitals_ ||
        generator.ld < generator.rows || !image.data || image.rows != n_alpha_ ||
        image.cols != count || image.ld < image.rows || image.data == generator.data)
        throw std::invalid_argument("invalid CUDA one-body generator columns");
    auto& context = *context_;
    std::vector<std::int64_t> rows(n_alpha_), betas(count);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(betas.begin(), betas.end(), first);
    Matrix<double> tile(context, n_alpha_, count);
    columns(betas, rows, tile.view());
    const auto cells = static_cast<std::size_t>(n_alpha_) * count;
    apply_streamed_alpha<<<static_cast<unsigned int>((cells + threads - 1) / threads),
                           threads,
                           0,
                           context.stream()>>>(tile.view(),
                                               generator,
                                               alpha_->sources.data(),
                                               beta_->sources.data(),
                                               n_alpha_,
                                               n_beta_,
                                               first,
                                               n_orbitals_,
                                               alpha_only,
                                               image);
    check_cuda(cudaGetLastError(), "CUDA one-body alpha generator columns");
    if (alpha_only)
        return;
    auto [events, batches] = beta_event_batches(beta_->topology, first, count);
    Buffer<StreamEvent> device_events(context, events.size());
    context.copy_to_device(
        device_events.data(), events.data(), events.size() * sizeof(StreamEvent));
    for (const auto& batch : batches) {
        columns(batch.sources, rows, tile.view().columns(0, batch.sources.size()));
        apply_streamed_beta<<<dim3(batch.end - batch.begin, (n_alpha_ + threads - 1) / threads),
                              threads,
                              0,
                              context.stream()>>>(
            tile.view(), generator, device_events.data(), batch.begin, image);
        check_cuda(cudaGetLastError(), "CUDA one-body beta generator columns");
    }
}
} // namespace xmvb::cuda
