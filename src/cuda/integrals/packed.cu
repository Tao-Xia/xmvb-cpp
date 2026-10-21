#include "cuda/integrals/packed.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

constexpr int kMaxEdge = 512;

unsigned int grid(std::int64_t work) {
  return static_cast<unsigned int>(
      std::min<std::int64_t>(65535, (work - 1) / 256 + 1));
}

std::size_t checked_bytes(std::int64_t count) {
  if (count < 0 || static_cast<std::uint64_t>(count) >
                       std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error("packed AO-pair workspace size overflow");
  return static_cast<std::size_t>(count) * sizeof(double);
}

std::size_t staging_bytes(std::int64_t rows, std::int64_t edges) {
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (rows < 0 || edges < 0 ||
      static_cast<std::uint64_t>(rows + 1) > limit / sizeof(int) ||
      static_cast<std::uint64_t>(edges) >
          (limit - static_cast<std::size_t>(rows + 1) * sizeof(int)) /
              (sizeof(int) + sizeof(double)))
    throw std::overflow_error("packed AO-pair staging size overflow");
  return static_cast<std::size_t>(rows + 1) * sizeof(int) +
         static_cast<std::size_t>(edges) * (sizeof(int) + sizeof(double));
}

__device__ void add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
  atomicAdd(address, value);
#else
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#endif
}

__global__ void pack_csr_rows(const int* offsets, const int* columns,
                              const double* source_values,
                              std::int64_t first_row, int row_count,
                              int first_edge, int edge_count,
                              int block_edge, std::int64_t blocks,
                              double* packed_values) {
  for (std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) +
                            threadIdx.x;
       index < edge_count;
       index += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const int global_edge = first_edge + static_cast<int>(index);
    int lo = 0, hi = row_count;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (offsets[mid + 1] <= global_edge) lo = mid + 1;
      else hi = mid;
    }
    const std::int64_t row = first_row + lo;
    const std::int64_t col = columns[index];
    const std::int64_t p = row / block_edge;
    const std::int64_t q = col / block_edge;
    if (p < q) continue;  // Off-diagonal tiles use their lower CSR direction.
    const std::int64_t d = p - q;
    const std::int64_t tile = d * blocks - d * (d - 1) / 2 + q;
    const std::int64_t element =
        tile * block_edge * block_edge + (col % block_edge) * block_edge +
        row % block_edge;
    add_double(packed_values + element, source_values[index]);
  }
}

// A canonical integral supplies one symmetric AO-pair matrix entry. Store
// only its lower block tile; diagonal tiles also need the transpose lane.
// Atomic addition preserves the existing additive duplicate-entry contract.
__global__ void pack_unique(const int* rows,const int* columns,const double* values,
    std::int64_t count,int edge,std::int64_t blocks,double* packed) {
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<count;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto row=static_cast<std::int64_t>(max(rows[k],columns[k]));
    const auto col=static_cast<std::int64_t>(min(rows[k],columns[k]));
    const auto p=row/edge,q=col/edge,d=p-q;
    const auto tile=d*blocks-d*(d-1)/2+q;
    const auto i=row%edge,j=col%edge,base=tile*edge*edge;
    add_double(packed+base+j*edge+i,values[k]);
    if (p==q && i!=j) add_double(packed+base+i*edge+j,values[k]);
  }
}

__global__ void clear_panel(PairPanel target, std::int64_t work) {
  for (std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) +
                            threadIdx.x;
       index < work;
       index += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    target.data[(index / target.cols) * target.ld + index % target.cols] = 0.0;
  }
}

__global__ void scatter_scratch(const double* scratch, PairPanel target,
                                std::int64_t first_row,
                                std::int64_t block_first,
                                std::int64_t copy_begin,
                                std::int64_t copy_count) {
  const std::int64_t work = copy_count * target.cols;
  for (std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) +
                            threadIdx.x;
       index < work;
       index += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const auto row = copy_begin + index / target.cols;
    const auto channel = index % target.cols;
    target.data[(row - first_row) * target.ld + channel] =
        scratch[(row - block_first) * target.cols + channel];
  }
}

__global__ void fill_ao_plane(PackedAoPairMatrix::DeviceView matrix,
                              int mu, int first_nu, linalg::MatrixView plane,
                              std::int64_t work) {
  const auto n = plane.rows;
  for (std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) +
                            threadIdx.x;
       index < work;
       index += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const auto l = index % n;
    const auto k = index / n % n;
    const auto local_nu = index / (n * n);
    const auto nu = first_nu + local_nu;
    const auto a = static_cast<std::int64_t>(mu > nu ? mu : nu);
    const auto b = static_cast<std::int64_t>(mu > nu ? nu : mu);
    const auto c = k > l ? k : l;
    const auto d = k > l ? l : k;
    const auto row = a * (a + 1) / 2 + b;
    const auto col = c * (c + 1) / 2 + d;
    plane.data[l + plane.ld * (local_nu * n + k)] = matrix.value(row, col);
  }
}

bool spans_overlap(ConstPairPanel source, PairPanel target) {
  // The span includes stride padding, deliberately rejecting uncertain aliases.
  const auto a = reinterpret_cast<std::uintptr_t>(source.data);
  const auto b = reinterpret_cast<std::uintptr_t>(target.data);
  const auto a_count = static_cast<std::uint64_t>(source.rows - 1) * source.ld +
                       source.cols;
  const auto b_count = static_cast<std::uint64_t>(target.rows - 1) * target.ld +
                       target.cols;
  if (a_count > (std::numeric_limits<std::uintptr_t>::max() - a) /
                    sizeof(double) ||
      b_count > (std::numeric_limits<std::uintptr_t>::max() - b) /
                    sizeof(double))
    return true;
  return a < b + b_count * sizeof(double) &&
         b < a + a_count * sizeof(double);
}

}  // namespace

std::size_t PackedAoPairMatrix::storage_bytes(std::int64_t pairs) noexcept {
  if (pairs <= 0) return std::numeric_limits<std::size_t>::max();
  const auto edge = static_cast<std::uint64_t>(std::min<std::int64_t>(kMaxEdge, pairs));
  const auto blocks = static_cast<std::uint64_t>((pairs - 1) / edge + 1);
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (blocks > limit / (blocks + 1)) return limit;
  const auto tiles = blocks * (blocks + 1) / 2;
  if (tiles > limit / edge / edge / sizeof(double)) return limit;
  return static_cast<std::size_t>(tiles * edge * edge * sizeof(double));
}

PackedAoPairMatrix::PackedAoPairMatrix(
    Context& context, const integrals::AoPairGraph& graph, int n_bf)
    : context_(&context), n_bf_(n_bf),
      n_pairs_(static_cast<std::int64_t>(n_bf) * (static_cast<std::int64_t>(n_bf) + 1) / 2) {
  static const nvtx3::registered_string range_name{"ao.pack_input"};
  const nvtx3::scoped_range range{range_name};
  const bool unique=graph.has_compact_integral_stream();
  if (n_bf <= 0 || n_pairs_ > std::numeric_limits<int>::max() ||
      graph.pair_first.size() != static_cast<std::size_t>(n_pairs_) ||
      graph.pair_second.size() != static_cast<std::size_t>(n_pairs_))
    throw std::invalid_argument("invalid packed AO-pair topology");
  if (unique) {
    if (graph.integral_rows.size()!=graph.integral_columns.size() ||
        graph.integral_rows.size()!=graph.integral_values.size())
      throw std::invalid_argument("invalid packed AO-pair unique stream");
  } else if (!graph.has_row_graph() ||
      graph.row_offsets.size() != static_cast<std::size_t>(n_pairs_ + 1) ||
      graph.columns.size() != graph.values.size() ||
      graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
      graph.columns.size() != static_cast<std::size_t>(graph.row_offsets.back()))
    throw std::invalid_argument("invalid packed AO-pair CSR graph");
  edge_ = static_cast<int>(std::min<std::int64_t>(kMaxEdge, n_pairs_));
  n_blocks_ = (n_pairs_ - 1) / edge_ + 1;
  const auto bytes = storage_bytes(n_pairs_);
  if (bytes == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("packed AO-pair storage size overflow");
  AllocationPhase phase(context, "packed AO-pair matrix construction");
  values_ = Buffer<double>(context, bytes / sizeof(double));
  check_cuda(cudaMemsetAsync(values_.data(), 0, bytes, context.stream()),
             "zero packed AO-pair tiles");

  if (unique) {
    if (graph.integral_values.empty()) return;
    std::size_t free=0,total=0;
    check_cuda(cudaMemGetInfo(&free,&total),"CUDA unique ERI staging capacity");
    const auto available=std::min(free,context.device_byte_budget()-
        context.counters().current_device_bytes);
    // One block-row worth of input is sufficient to stream the conversion;
    // tighter budgets split that row further. There is no full input copy.
    const auto count=std::min({graph.integral_values.size(),
        static_cast<std::size_t>(edge_)*static_cast<std::size_t>(n_pairs_),
        available/(2*sizeof(int)+sizeof(double))});
    if (!count) throw std::runtime_error("unique ERI staging cannot hold one entry");
    Buffer<int> rows(context,count),columns(context,count);
    Buffer<double> staging(context,count);
    for (std::size_t begin=0;begin<graph.integral_values.size();begin+=count) {
      const auto size=std::min(count,graph.integral_values.size()-begin);
      context.copy_to_device(rows.data(),graph.integral_rows.data()+begin,size*sizeof(int));
      context.copy_to_device(columns.data(),graph.integral_columns.data()+begin,size*sizeof(int));
      context.copy_to_device(staging.data(),graph.integral_values.data()+begin,size*sizeof(double));
      pack_unique<<<grid(size),256,0,context.stream()>>>(rows.data(),columns.data(),
          staging.data(),size,edge_,n_blocks_,values_.data());
      check_cuda(cudaGetLastError(),"CUDA unique ERI tile packing");
    }
    return;
  }

  // Keep only one bounded CSR row chunk on device; its staging memory is freed
  // before the next chunk. A row is never split, preserving a simple CSR map.
  std::int64_t begin = 0;
  while (begin < n_pairs_) {
    const auto available = context.device_byte_budget() -
                           context.counters().current_device_bytes;
    std::int64_t end = begin;
    while (end < n_pairs_ && end - begin < edge_) {
      const auto edges = static_cast<std::int64_t>(graph.row_offsets[end + 1]) -
                         graph.row_offsets[begin];
      if (staging_bytes(end - begin + 1, edges) > available) break;
      ++end;
    }
    if (end == begin)
      throw std::runtime_error("packed AO-pair CSR row staging exceeds device budget");
    const int first_edge = graph.row_offsets[begin];
    const int edge_count = graph.row_offsets[end] - first_edge;
    if (edge_count > 0) {
      Buffer<int> offsets(context, end - begin + 1);
      Buffer<int> columns(context, edge_count);
      Buffer<double> staging_values(context, edge_count);
      context.copy_to_device(offsets.data(), graph.row_offsets.data() + begin,
                             static_cast<std::size_t>(end - begin + 1) * sizeof(int));
      context.copy_to_device(columns.data(), graph.columns.data() + first_edge,
                             static_cast<std::size_t>(edge_count) * sizeof(int));
      context.copy_to_device(staging_values.data(), graph.values.data() + first_edge,
                             static_cast<std::size_t>(edge_count) * sizeof(double));
      pack_csr_rows<<<grid(edge_count), 256, 0, context.stream()>>>(
          offsets.data(), columns.data(), staging_values.data(), begin,
          static_cast<int>(end - begin), first_edge, edge_count,
          edge_, n_blocks_, values_.data());
      check_cuda(cudaGetLastError(), "pack AO-pair CSR rows");
    }
    begin = end;
  }
}

std::size_t PackedAoPairMatrix::workspace_bytes(
    std::int64_t channels, std::int64_t first_row,
    std::int64_t row_count) const {
  if (channels <= 0 || channels > std::numeric_limits<int>::max() ||
      first_row < 0 || first_row > n_pairs_ ||
      row_count < 0 || row_count > n_pairs_ - first_row)
    throw std::invalid_argument("invalid packed AO-pair apply dimensions");
  if (row_count == 0 ||
      (first_row % edge_ == 0 && (first_row + row_count) % edge_ == 0))
    return 0;
  if (channels > std::numeric_limits<std::int64_t>::max() / edge_)
    throw std::overflow_error("packed AO-pair workspace size overflow");
  return checked_bytes(channels * edge_);
}

void PackedAoPairMatrix::apply(ConstPairPanel source, PairPanel target,
                               std::int64_t first_row) const {
  const nvtx3::scoped_range range{"ao.packed_pair_apply"};
  if (!source.data || !target.data || source.rows != n_pairs_ ||
      source.cols <= 0 || source.cols != target.cols ||
      source.ld < source.cols || target.ld < target.cols ||
      source.ld > std::numeric_limits<int>::max() ||
      target.ld > std::numeric_limits<int>::max())
    throw std::invalid_argument("invalid packed AO-pair panels");
  const auto scratch_bytes = workspace_bytes(source.cols, first_row, target.rows);
  if (target.rows == 0) return;
  if (spans_overlap(source, target))
    throw std::invalid_argument("overlapping packed AO-pair panels");

  AllocationPhase phase(*context_, "packed AO-pair action scratch");
  Buffer<double> scratch(*context_, scratch_bytes / sizeof(double));
  const auto work = target.rows * target.cols;
  clear_panel<<<grid(work), 256, 0, context_->stream()>>>(target, work);
  check_cuda(cudaGetLastError(), "clear packed AO-pair target");

  const auto output_end = first_row + target.rows;
  const auto full_begin = (first_row + edge_ - 1) / edge_;
  const auto full_end = output_end / edge_;
  const double alpha = 1.0, beta = 1.0;
  const auto source_stride = static_cast<long long>(edge_) * source.ld;
  const auto target_stride = static_cast<long long>(edge_) * target.ld;
  const auto tile_stride = static_cast<long long>(edge_) * edge_;
  const int channels = static_cast<int>(source.cols);
  const int tail = static_cast<int>(n_pairs_ - (n_blocks_ - 1) * edge_);

  auto full_batch = [&](std::int64_t d, bool lower,
                        std::int64_t out_begin, std::int64_t out_end) {
    if (out_begin >= out_end) return;
    const auto tile_q = lower ? out_begin - d : out_begin;
    const auto source_block = lower ? tile_q : out_begin + d;
    const auto tile_index = d * n_blocks_ - d * (d - 1) / 2 + tile_q;
    const auto* tile = values_.data() + tile_index * tile_stride;
    const auto* input = source.data + source_block * source_stride;
    auto* output = target.data + (out_begin * edge_ - first_row) * target.ld;
    auto operation = lower ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Only the final source block can have fewer than B physical rows.
    const auto tail_output = lower ? n_blocks_ - 1 + d : n_blocks_ - 1 - d;
    if (tail != edge_ && tail_output >= out_begin && tail_output < out_end) {
      const auto prefix = static_cast<int>(tail_output - out_begin);
      if (prefix > 0)
        check_cublas(cublasDgemmStridedBatched(
            context_->blas(), CUBLAS_OP_N, operation, channels, edge_, edge_,
            &alpha, input, static_cast<int>(source.ld), source_stride,
            tile, edge_, tile_stride, &beta, output,
            static_cast<int>(target.ld), target_stride, prefix),
            "packed AO-pair batched GEMM");
      const auto tail_tile = tile + static_cast<std::int64_t>(prefix) * tile_stride;
      const auto tail_input = source.data + (n_blocks_ - 1) * source_stride;
      auto* tail_output_ptr = target.data +
          (tail_output * edge_ - first_row) * target.ld;
      check_cublas(cublasDgemm(
          context_->blas(), CUBLAS_OP_N, operation, channels, edge_, tail,
          &alpha, tail_input, static_cast<int>(source.ld), tail_tile, edge_,
          &beta, tail_output_ptr, static_cast<int>(target.ld)),
          "packed AO-pair tail GEMM");
      return;
    }
    check_cublas(cublasDgemmStridedBatched(
        context_->blas(), CUBLAS_OP_N, operation, channels, edge_, edge_,
        &alpha, input, static_cast<int>(source.ld), source_stride,
        tile, edge_, tile_stride, &beta, output,
        static_cast<int>(target.ld), target_stride,
        static_cast<int>(out_end - out_begin)),
        "packed AO-pair batched GEMM");
  };

  for (std::int64_t d = 0; d < n_blocks_; ++d) {
    full_batch(d, true, std::max(full_begin, d), full_end);
    if (d > 0)
      full_batch(d, false, full_begin,
                 std::min(full_end, n_blocks_ - d));
  }

  auto partial_block = [&](std::int64_t block) {
    const auto block_first = block * edge_;
    const auto copy_begin = std::max(first_row, block_first);
    const auto copy_end = std::min(output_end, block_first + edge_);
    if (copy_end <= copy_begin ||
        (copy_begin == block_first && copy_end == block_first + edge_))
      return;
    check_cuda(cudaMemsetAsync(scratch.data(), 0, scratch_bytes,
                               context_->stream()),
               "clear packed AO-pair scratch tile");
    for (std::int64_t source_block = 0; source_block < n_blocks_; ++source_block) {
      const auto d = block > source_block ? block - source_block :
                     source_block - block;
      const auto q = std::min(block, source_block);
      const auto tile_index = d * n_blocks_ - d * (d - 1) / 2 + q;
      const auto* tile = values_.data() + tile_index * tile_stride;
      const auto* input = source.data + source_block * source_stride;
      const int k = static_cast<int>(std::min<std::int64_t>(
          edge_, n_pairs_ - source_block * edge_));
      const auto operation = block >= source_block ? CUBLAS_OP_T : CUBLAS_OP_N;
      check_cublas(cublasDgemm(
          context_->blas(), CUBLAS_OP_N, operation, channels, edge_, k,
          &alpha, input, static_cast<int>(source.ld), tile, edge_, &beta,
          scratch.data(), channels), "packed AO-pair boundary GEMM");
    }
    const auto copied = copy_end - copy_begin;
    scatter_scratch<<<grid(copied * target.cols), 256, 0, context_->stream()>>>(
        scratch.data(), target, first_row, block_first, copy_begin, copied);
    check_cuda(cudaGetLastError(), "scatter packed AO-pair boundary");
  };
  const auto first_block = first_row / edge_;
  const auto last_block = (output_end - 1) / edge_;
  partial_block(first_block);
  if (last_block != first_block) partial_block(last_block);
}

void PackedAoPairMatrix::ao_plane(int mu, linalg::MatrixView plane,
                                  int first_nu) const {
  const nvtx3::scoped_range range{"ao.packed_ao_plane"};
  if (mu < 0 || mu >= n_bf_ || !plane.data || plane.rows != n_bf_ ||
      first_nu < 0 || first_nu >= n_bf_ ||
      plane.cols <= 0 || plane.cols % n_bf_ != 0 ||
      plane.cols / n_bf_ > n_bf_ - first_nu ||
      plane.ld < plane.rows)
    throw std::invalid_argument("invalid packed AO plane dimensions");
  const auto work = plane.rows * plane.cols;
  fill_ao_plane<<<grid(work), 256, 0, context_->stream()>>>(
      view(), mu, first_nu, plane, work);
  check_cuda(cudaGetLastError(), "packed AO-pair plane");
}

}  // namespace xmvb::cuda
