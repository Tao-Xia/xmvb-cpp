#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

void check_sparse(cusparseStatus_t status, const char* operation) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed (cuSPARSE status " +
                             std::to_string(static_cast<int>(status)) + ")");
}

unsigned int blocks(std::int64_t count) {
  if (count <= 0) return 0;
  return static_cast<unsigned int>(std::min<std::int64_t>(65535, (count - 1) / 256 + 1));
}

void check_launch(const char* name) { check_cuda(cudaGetLastError(), name); }

__device__ void atomic_add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
  atomicAdd(address, value);
#else
  auto* integer = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *integer, assumed;
  do {
    assumed = old;
    old = atomicCAS(integer, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#endif
}

std::int64_t active_pairs(std::int64_t active) {
  if (active <= 0 || active > std::numeric_limits<int>::max() ||
      active > std::numeric_limits<std::int64_t>::max() / (active + 1))
    throw std::invalid_argument("invalid CUDA active orbital count");
  const std::int64_t pairs = active * (active + 1) / 2;
  if (pairs > std::numeric_limits<int>::max())
    throw std::overflow_error("CUDA active pair count exceeds cuBLAS range");
  return pairs;
}

std::int64_t checked_product(std::int64_t left, std::int64_t right) {
  if (left < 0 || right < 0 ||
      (right && left > std::numeric_limits<std::int64_t>::max() / right))
    throw std::overflow_error("CUDA integral work size overflow");
  return left * right;
}

__device__ void decode_pair(std::int64_t p, int& first, int& second) {
  std::int64_t i = static_cast<std::int64_t>(
      (sqrt(8.0 * static_cast<double>(p) + 1.0) - 1.0) * 0.5);
  while (i * (i + 1) / 2 > p) --i;
  while ((i + 1) * (i + 2) / 2 <= p) ++i;
  first = static_cast<int>(i);
  second = static_cast<int>(p - i * (i + 1) / 2);
}

__device__ double feature(linalg::ConstMatrixView c,
                          const int* pair_first, const int* pair_second,
                          std::int64_t row, int a, int b) {
  const int i = pair_first[row], j = pair_second[row];
  double value = c.data[i + c.ld * a] * c.data[j + c.ld * b];
  if (i != j) value += c.data[j + c.ld * a] * c.data[i + c.ld * b];
  return value;
}

__device__ double mixed_feature(linalg::ConstMatrixView c,
                                linalg::ConstMatrixView d,
                                const int* pair_first, const int* pair_second,
                                std::int64_t row, int a, int b) {
  const int i = pair_first[row], j = pair_second[row];
  double value = d.data[i + d.ld * a] * c.data[j + c.ld * b] +
                 c.data[i + c.ld * a] * d.data[j + d.ld * b];
  if (i != j)
    value += d.data[j + d.ld * a] * c.data[i + c.ld * b] +
             c.data[j + c.ld * a] * d.data[i + d.ld * b];
  return value;
}

__global__ void feature_rows(linalg::ConstMatrixView c,
                             linalg::ConstMatrixView d,
                             const int* first, const int* second,
                             std::int64_t begin, std::int64_t channel_begin,
                             PairPanel output,
                             PairPanel directional, std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k / output.cols;
    const std::int64_t p = k % output.cols;
    int a, b;
    decode_pair(channel_begin + p, a, b);
    output.data[row * output.ld + p] = feature(c, first, second, begin + row, a, b);
    if (d.data)
      directional.data[row * directional.ld + p] =
          mixed_feature(c, d, first, second, begin + row, a, b);
  }
}

__global__ void mixed_feature_rows(linalg::ConstMatrixView c,
                                   linalg::ConstMatrixView d,
                                   const int* first, const int* second,
                                   std::int64_t begin, std::int64_t channel_begin,
                                   PairPanel output,
                                   std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k / output.cols;
    const std::int64_t p = k % output.cols;
    int a, b;
    decode_pair(channel_begin + p, a, b);
    output.data[row * output.ld + p] =
        mixed_feature(c, d, first, second, begin + row, a, b);
  }
}

__global__ void generated_csr(linalg::ConstMatrixView c,
                              linalg::ConstMatrixView d,
                              const int* first, const int* second,
                              const int* offsets, const int* columns,
                              const double* values, std::int64_t begin,
                              PairPanel output, PairPanel directional,
                              std::int64_t work, bool direction_only) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k / output.cols;
    const std::int64_t p = k % output.cols;
    int a, b;
    decode_pair(p, a, b);
    double sum = 0.0, delta = 0.0;
    for (int edge = offsets[begin + row]; edge < offsets[begin + row + 1]; ++edge) {
      const int source = columns[edge];
      const double integral = values[edge];
      if (!direction_only) sum += integral * feature(c, first, second, source, a, b);
      if (d.data)
        delta += integral * mixed_feature(c, d, first, second, source, a, b);
    }
    output.data[row * output.ld + p] = direction_only ? delta:sum;
    if (directional.data) directional.data[row * directional.ld + p] = delta;
  }
}

__global__ void generated_compact(linalg::ConstMatrixView c,
                                  linalg::ConstMatrixView d,
                                  const int* first, const int* second,
                                  const int* integral_rows,
                                  const int* integral_columns,
                                  const double* values,
                                  std::int64_t integral_count,
                                  std::int64_t begin, PairPanel output,
                                  PairPanel directional, std::int64_t work,
                                  bool direction_only) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t edge = k / output.cols;
    const std::int64_t p = k % output.cols;
    if (edge >= integral_count) continue;
    int a, b;
    decode_pair(p, a, b);
    const int left = integral_rows[edge], right = integral_columns[edge];
    const double value = values[edge];
    if (left >= begin && left < begin + output.rows) {
      atomic_add_double(output.data + (left - begin) * output.ld + p,
                        value * (direction_only ? mixed_feature(c,d,first,second,right,a,b):
                                                   feature(c, first, second, right, a, b)));
      if (directional.data)
        atomic_add_double(directional.data + (left - begin) * directional.ld + p,
                          value * mixed_feature(c, d, first, second, right, a, b));
    }
    if (left != right && right >= begin && right < begin + output.rows) {
      atomic_add_double(output.data + (right - begin) * output.ld + p,
                        value * (direction_only ? mixed_feature(c,d,first,second,left,a,b):
                                                   feature(c, first, second, left, a, b)));
      if (directional.data)
        atomic_add_double(directional.data + (right - begin) * directional.ld + p,
                          value * mixed_feature(c, d, first, second, left, a, b));
    }
  }
}

__global__ void compact_apply(const int* rows, const int* columns,
                              const double* values, std::int64_t edges,
                              ConstPairPanel source, PairPanel target,
                              std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t edge = k / source.cols;
    const std::int64_t column = k % source.cols;
    if (edge >= edges) continue;
    const int left = rows[edge], right = columns[edge];
    const double value = values[edge];
    atomic_add_double(target.data + left * target.ld + column,
                      value * source.data[right * source.ld + column]);
    if (left != right)
      atomic_add_double(target.data + right * target.ld + column,
                        value * source.data[left * source.ld + column]);
  }
}

__global__ void pack_symmetric_density(linalg::ConstMatrixView density,
                                       const int* first, const int* second,
                                       double* packed, std::int64_t count) {
  for (std::int64_t p = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       p < count; p += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const int i = first[p], j = second[p];
    packed[p] = (i == j ? 1.0 : 2.0) * density.data[i + density.ld * j];
  }
}

__global__ void pack_density_batch(linalg::ConstMatrixView densities,
    const int* first,const int* second,double* packed,std::int64_t pairs,int channels) {
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<pairs*channels;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto p=k/channels;const int channel=static_cast<int>(k%channels);
    const int i=first[p],j=second[p];
    packed[k]=(i==j ? 1.0:2.0)*densities.data[i+densities.ld*(j+channel*densities.rows)];
  }
}

__global__ void exchange_unique(const int* rows, const int* columns,
                                const int* edges, const int* csr_columns,
                                const double* values, std::int64_t count,
                                const int* first, const int* second,
                                linalg::ConstMatrixView density,
                                linalg::MatrixView output,
                                double weight, bool compact,int channels) {
  for (std::int64_t p = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       p < count; p += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t edge = compact ? p : edges[p];
    const int left = rows[p];
    const int right = compact ? columns[p] : csr_columns[edge];
    const double value = weight * values[edge];
    const int pair_a[2] = {left, right};
    const int pair_b[2] = {right, left};
    for (int swap_pairs = 0; swap_pairs < (left == right ? 1 : 2); ++swap_pairs) {
      const int u = pair_a[swap_pairs], v = pair_b[swap_pairs];
      const int ui = first[u], uj = second[u], vi = first[v], vj = second[v];
      for (int swap_u = 0; swap_u < (ui == uj ? 1 : 2); ++swap_u) {
        const int a = swap_u ? uj : ui, b = swap_u ? ui : uj;
        for (int swap_v = 0; swap_v < (vi == vj ? 1 : 2); ++swap_v) {
          const int c = swap_v ? vj : vi, d = swap_v ? vi : vj;
          // Decode each unique ERI and symmetry orbit once for all densities.
          for (int channel=0;channel<channels;++channel)
            atomic_add_double(output.data+a+output.ld*(c+channel*output.rows),
                value*density.data[b+density.ld*(d+channel*density.rows)]);
        }
      }
    }
  }
}

__global__ void finish_jk_batch(const double* image,linalg::MatrixView output,
    double weight,std::int64_t work,int channels) {
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto row=k%output.rows,column=k/output.rows;
    const auto col=column%output.rows,channel=column/output.rows;
    const auto hi=row>col ? row:col,lo=row>col ? col:row;
    const auto pair=hi*(hi+1)/2+lo;
    output.data[row+output.ld*column]+=weight*image[pair*channels+channel];
  }
}

__global__ void finish_jk(const double* coulomb_image,
                          linalg::MatrixView output, double weight,
                          std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const int row = static_cast<int>(k % output.rows);
    const int col = static_cast<int>(k / output.rows);
    const int hi = row > col ? row : col, lo = row > col ? col : row;
    const std::int64_t pair = static_cast<std::int64_t>(hi) * (hi + 1) / 2 + lo;
    output.data[row + output.ld * col] += weight * coulomb_image[pair];
  }
}

__global__ void add_dense(linalg::MatrixView output,
                          linalg::ConstMatrixView source,
                          std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k % output.rows, col = k / output.rows;
    output.data[row + output.ld * col] += source.data[row + source.ld * col];
  }
}

__global__ void trace_density(linalg::ConstMatrixView density,
                              linalg::ConstMatrixView first,
                              linalg::ConstMatrixView second,
                              double factor, double* result,
                              std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k % density.rows, col = k / density.rows;
    const double a = first.data[col + first.ld * row];
    const double b = second.data ? second.data[col + second.ld * row] : 0.0;
    atomic_add_double(result, factor * density.data[row + density.ld * col] * (a + b));
  }
}

__global__ void symmetrize(const linalg::ConstMatrixView g,
                           const linalg::ConstMatrixView dg,
                           double* h, double* dh, int pairs,
                           std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const int row = static_cast<int>(k % pairs), col = static_cast<int>(k / pairs);
    h[k] = g.data[row + g.ld * col] + g.data[col + g.ld * row];
    if (dg.data) dh[k] = dg.data[row + dg.ld * col] + dg.data[col + dg.ld * row];
  }
}

__global__ void pair_pullback(PairPanel pair_gradient,
                              linalg::ConstMatrixView c,
                              const int* first, const int* second,
                              std::int64_t begin,
                              linalg::MatrixView gradient,
                              std::int64_t work) {
  for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       k < work; k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
    const std::int64_t row = k / pair_gradient.cols;
    const std::int64_t p = k % pair_gradient.cols;
    const int i = first[begin + row], j = second[begin + row];
    int a, b;
    decode_pair(p, a, b);
    const double value = pair_gradient.data[row * pair_gradient.ld + p];
    atomic_add_double(gradient.data + i + gradient.ld * a,
                      value * c.data[j + c.ld * b]);
    atomic_add_double(gradient.data + j + gradient.ld * b,
                      value * c.data[i + c.ld * a]);
    if (i != j) {
      atomic_add_double(gradient.data + j + gradient.ld * a,
                        value * c.data[i + c.ld * b]);
      atomic_add_double(gradient.data + i + gradient.ld * b,
                        value * c.data[j + c.ld * a]);
    }
  }
}

void zero_matrix(Context& context, linalg::MatrixView matrix) {
  check_cuda(cudaMemset2DAsync(matrix.data,matrix.ld*sizeof(double),0,
      matrix.rows*sizeof(double),matrix.cols,context.stream()),"zero CUDA matrix");
}

void multiply_pair_tile(Context& context, PairPanel result, ConstPairPanel product,
                        const double* h, int pairs, double beta = 0.0) {
  const double alpha = 1.0;
  check_cublas(cublasDgemm(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N,
                          pairs, blas_int(product.rows), pairs,
                          &alpha, h, pairs, product.data, blas_int(product.ld),
                          &beta, result.data, blas_int(result.ld)),
               "pair-adjoint tile GEMM");
}

void contract_tile(Context& context, linalg::MatrixView out,
                   ConstPairPanel left, ConstPairPanel right, double beta) {
  const double alpha = 1.0;
  check_cublas(cublasDgemm(context.blas(), CUBLAS_OP_N, CUBLAS_OP_T,
                          blas_int(out.rows), blas_int(out.cols),
                          blas_int(left.rows), &alpha, left.data,
                          blas_int(left.ld), right.data, blas_int(right.ld),
                          &beta, out.data, blas_int(out.ld)),
               "active pair contraction GEMM");
}

void complete_pair_direction(Context& context, linalg::ConstMatrixView unsymmetric,
                             linalg::MatrixView symmetric) {
  constexpr double one = 1.0;
  check_cublas(cublasDgeam(context.blas(), CUBLAS_OP_N, CUBLAS_OP_T,
                          blas_int(symmetric.rows), blas_int(symmetric.cols),
                          &one, unsymmetric.data, blas_int(unsymmetric.ld),
                          &one, unsymmetric.data, blas_int(unsymmetric.ld),
                          symmetric.data, blas_int(symmetric.ld)),
               "complete active pair direction");
}

std::int64_t tile_size(Context& context, std::int64_t n_ao_pairs,
                       std::int64_t active_pairs, int images,
                       std::int64_t output_elements) {
  const auto& counters = context.counters();
  const std::size_t budget = context.device_byte_budget();
  if (counters.current_device_bytes > budget)
    throw std::runtime_error("CUDA integral device accounting exceeds budget");
  std::size_t device_free = 0, device_total = 0;
  check_cuda(cudaMemGetInfo(&device_free, &device_total),
             "query CUDA integral free memory");
  const std::size_t available = std::min(
      budget - counters.current_device_bytes, device_free);
  // The requested output is already live. Reserve its measured size once more
  // for library/reduction scratch; fixed intermediates are allocated before
  // calling this function and are reflected in the live counters/free bytes.
  const std::size_t reserve = static_cast<std::size_t>(
      checked_product(output_elements, sizeof(double)));
  const std::size_t row_bytes = static_cast<std::size_t>(
      checked_product(checked_product(active_pairs, images), sizeof(double)));
  if (available <= reserve || (available - reserve) / row_bytes < 1)
    throw std::runtime_error("CUDA integral budget cannot hold one AO-pair row tile");
  const std::size_t affordable_rows = (available - reserve) / row_bytes;
  return affordable_rows >= static_cast<std::size_t>(n_ao_pairs)
      ? n_ao_pairs : static_cast<std::int64_t>(affordable_rows);
}

std::size_t spmm_scratch_bytes(cusparseHandle_t sparse,
    cusparseSpMatDescr_t csr,std::int64_t rows,int channels,
    std::int64_t target_ld,double* source_data,double* target_data) {
  cusparseDnMatDescr_t source=nullptr,target=nullptr;
  check_sparse(cusparseCreateDnMat(&source,rows,channels,channels,
      source_data,CUDA_R_64F,CUSPARSE_ORDER_ROW),
      "create generated AO-pair source descriptor");
  try {
    check_sparse(cusparseCreateDnMat(&target,rows,channels,target_ld,
        target_data,CUDA_R_64F,CUSPARSE_ORDER_ROW),
        "create generated AO-pair target descriptor");
    const double alpha=1.0,beta=0.0;
    std::size_t bytes=0;
    check_sparse(cusparseSpMM_bufferSize(sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,csr,source,&beta,target,
        CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,&bytes),
        "query generated AO-pair SpMM scratch");
    cusparseDestroyDnMat(target);
    cusparseDestroyDnMat(source);
    return bytes;
  } catch (...) {
    if (target) cusparseDestroyDnMat(target);
    cusparseDestroyDnMat(source);
    throw;
  }
}

}  // namespace

ExactAoPairOperator::ExactAoPairOperator(
    Context& context, const integrals::AoPairGraph& graph, int n_basis_functions)
    : context_(&context), n_bf_(n_basis_functions),
      n_pairs_(static_cast<std::int64_t>(n_basis_functions) *
               (static_cast<std::int64_t>(n_basis_functions) + 1) / 2) {
  if (n_bf_ <= 0 || n_pairs_ > std::numeric_limits<int>::max() ||
      graph.pair_first.size() != static_cast<std::size_t>(n_pairs_) ||
      graph.pair_second.size() != static_cast<std::size_t>(n_pairs_))
    throw std::invalid_argument("invalid CUDA AO-pair dimensions");
  for (int pair = 0; pair < n_pairs_; ++pair) {
    if (graph.pair_first[pair] < 0 || graph.pair_first[pair] >= n_bf_ ||
        graph.pair_second[pair] < 0 || graph.pair_second[pair] > graph.pair_first[pair] ||
        static_cast<std::int64_t>(graph.pair_first[pair]) *
                (graph.pair_first[pair] + 1) / 2 + graph.pair_second[pair] != pair)
      throw std::invalid_argument("invalid CUDA AO-pair component order");
  }
  compact_ = graph.has_compact_integral_stream();
  if (compact_) {
    if (graph.integral_rows.size() != graph.integral_columns.size() ||
        graph.integral_rows.size() != graph.integral_values.size())
      throw std::invalid_argument("invalid compact AO integral stream");
    n_integrals_ = graph.integral_rows.size();
    n_unique_ = n_integrals_;
    for (std::int64_t k = 0; k < n_integrals_; ++k)
      if (graph.integral_rows[k] < 0 || graph.integral_rows[k] >= n_pairs_ ||
          graph.integral_columns[k] < 0 || graph.integral_columns[k] >= n_pairs_)
        throw std::invalid_argument("compact AO integral index out of range");
  } else {
    if (graph.row_offsets.size() != static_cast<std::size_t>(n_pairs_ + 1) ||
        graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
        graph.columns.size() != graph.values.size() ||
        graph.columns.size() != static_cast<std::size_t>(graph.row_offsets.back()))
      throw std::invalid_argument("invalid CUDA AO-pair CSR graph");
    for (int row = 0; row < n_pairs_; ++row)
      if (graph.row_offsets[row] > graph.row_offsets[row + 1])
        throw std::invalid_argument("unsorted CUDA AO-pair CSR offsets");
    for (int column : graph.columns)
      if (column < 0 || column >= n_pairs_)
        throw std::invalid_argument("CUDA AO-pair CSR column out of range");
    n_integrals_ = graph.columns.size();
    n_unique_ = graph.integral_rows.size();
    if (graph.integral_edges.size() != static_cast<std::size_t>(n_unique_))
      throw std::invalid_argument("invalid unique CUDA AO integral references");
    for (std::int64_t k = 0; k < n_unique_; ++k)
      if (graph.integral_rows[k] < 0 || graph.integral_rows[k] >= n_pairs_ ||
          graph.integral_edges[k] < 0 || graph.integral_edges[k] >= n_integrals_ ||
          graph.integral_edges[k] < graph.row_offsets[graph.integral_rows[k]] ||
          graph.integral_edges[k] >= graph.row_offsets[graph.integral_rows[k] + 1])
        throw std::invalid_argument("unique CUDA AO integral reference out of range");
  }
  pair_first_ = Buffer<int>(context, n_pairs_);
  pair_second_ = Buffer<int>(context, n_pairs_);
  context.copy_to_device(pair_first_.data(), graph.pair_first.data(), n_pairs_ * sizeof(int));
  context.copy_to_device(pair_second_.data(), graph.pair_second.data(), n_pairs_ * sizeof(int));
  if (compact_) {
    integral_rows_ = Buffer<int>(context, n_integrals_);
    integral_columns_ = Buffer<int>(context, n_integrals_);
    integral_values_ = Buffer<double>(context, n_integrals_);
    context.copy_to_device(integral_rows_.data(), graph.integral_rows.data(),
                           n_integrals_ * sizeof(int));
    context.copy_to_device(integral_columns_.data(), graph.integral_columns.data(),
                           n_integrals_ * sizeof(int));
    context.copy_to_device(integral_values_.data(), graph.integral_values.data(),
                           n_integrals_ * sizeof(double));
  } else {
    row_offsets_ = Buffer<int>(context, n_pairs_ + 1);
    columns_ = Buffer<int>(context, n_integrals_);
    values_ = Buffer<double>(context, n_integrals_);
    context.copy_to_device(row_offsets_.data(), graph.row_offsets.data(),
                           (n_pairs_ + 1) * sizeof(int));
    context.copy_to_device(columns_.data(), graph.columns.data(),
                           n_integrals_ * sizeof(int));
    context.copy_to_device(values_.data(), graph.values.data(),
                           n_integrals_ * sizeof(double));
    integral_rows_ = Buffer<int>(context, n_unique_);
    integral_edges_ = Buffer<int>(context, n_unique_);
    context.copy_to_device(integral_rows_.data(), graph.integral_rows.data(),
                           n_unique_ * sizeof(int));
    context.copy_to_device(integral_edges_.data(), graph.integral_edges.data(),
                           n_unique_ * sizeof(int));
    if (n_integrals_ != 0) {
      check_sparse(cusparseCreate(&sparse_), "cusparseCreate");
      try {
        check_sparse(cusparseSetStream(sparse_, context.stream()), "cusparseSetStream");
        check_sparse(cusparseCreateCsr(&csr_, n_pairs_, n_pairs_, n_integrals_,
                                      row_offsets_.data(), columns_.data(), values_.data(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                     "cusparseCreateCsr");
      } catch (...) {
        cusparseDestroy(sparse_);
        sparse_ = nullptr;
        throw;
      }
    }
  }
}

ExactAoPairOperator::~ExactAoPairOperator() {
  if (csr_) cusparseDestroySpMat(csr_);
  if (sparse_) cusparseDestroy(sparse_);
}

void ExactAoPairOperator::validate_coefficients(linalg::ConstMatrixView c) const {
  if (!c.data || c.rows != n_bf_ || c.cols <= 0 || c.ld < c.rows)
    throw std::invalid_argument("invalid CUDA active coefficients");
  active_pairs(c.cols);
}

void ExactAoPairOperator::validate_panel(ConstPairPanel p) const {
  if (!p.data || p.rows != n_pairs_ || p.cols <= 0 || p.ld < p.cols)
    throw std::invalid_argument("invalid CUDA source AO-pair panel");
}

void ExactAoPairOperator::validate_panel(PairPanel p) const {
  if (!p.data || p.rows != n_pairs_ || p.cols <= 0 || p.ld < p.cols)
    throw std::invalid_argument("invalid CUDA target AO-pair panel");
}

void ExactAoPairOperator::apply(ConstPairPanel source, PairPanel target) const {
  validate_panel(source);
  validate_panel(target);
  if (source.cols != target.cols || source.data == target.data)
    throw std::invalid_argument("CUDA AO-pair action panel mismatch or alias");
  if (n_integrals_ == 0) {
    for (std::int64_t row = 0; row < n_pairs_; ++row)
      check_cuda(cudaMemsetAsync(target.data + row * target.ld, 0,
                                 target.cols * sizeof(double), context_->stream()),
                 "zero empty AO-pair target");
    return;
  }
  if (compact_) {
    for (std::int64_t row = 0; row < n_pairs_; ++row)
      check_cuda(cudaMemsetAsync(target.data + row * target.ld, 0,
                                 target.cols * sizeof(double), context_->stream()),
                 "zero compact AO-pair target");
    const std::int64_t work = checked_product(n_integrals_, source.cols);
    if (work) {
      compact_apply<<<blocks(work), 256, 0, context_->stream()>>>(
          integral_rows_.data(), integral_columns_.data(), integral_values_.data(),
          n_integrals_, source, target, work);
      check_launch("compact AO-pair action");
    }
    return;
  }
  cusparseDnMatDescr_t a = nullptr, b = nullptr;
  check_sparse(cusparseCreateDnMat(&a, n_pairs_, source.cols, source.ld,
                                  const_cast<double*>(source.data), CUDA_R_64F,
                                  CUSPARSE_ORDER_ROW), "cusparseCreateDnMat source");
  try {
    check_sparse(cusparseCreateDnMat(&b, n_pairs_, target.cols, target.ld,
                                    target.data, CUDA_R_64F, CUSPARSE_ORDER_ROW),
                 "cusparseCreateDnMat target");
    const double alpha = 1.0, beta = 0.0;
    std::size_t bytes = 0;
    check_sparse(cusparseSpMM_bufferSize(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, csr_, a, &beta, b, CUDA_R_64F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
                 "cusparseSpMM_bufferSize");
    Buffer<char> scratch(*context_, bytes);
    check_sparse(cusparseSpMM(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, csr_, a, &beta, b, CUDA_R_64F,
                             CUSPARSE_SPMM_ALG_DEFAULT, scratch.data()),
                 "cusparseSpMM AO-pair action");
  } catch (...) {
    if (b) cusparseDestroyDnMat(b);
    cusparseDestroyDnMat(a);
    throw;
  }
  cusparseDestroyDnMat(b);
  cusparseDestroyDnMat(a);
}

void ExactAoPairOperator::pair_features(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    std::int64_t begin, PairPanel output, PairPanel directional) const {
  validate_coefficients(c);
  if (d.data && (d.rows != c.rows || d.cols != c.cols || d.ld < d.rows))
    throw std::invalid_argument("invalid CUDA active direction");
  const std::int64_t pairs = active_pairs(c.cols);
  if (begin < 0 || output.rows < 0 || output.rows > n_pairs_ - begin ||
      output.cols != pairs || output.ld < pairs || (output.rows && !output.data) ||
      ((d.data != nullptr) != (directional.data != nullptr)) ||
      (d.data && (directional.rows != output.rows || directional.cols != pairs ||
                  directional.ld < pairs)))
    throw std::invalid_argument("invalid CUDA pair feature tile");
  const std::int64_t work = checked_product(output.rows, pairs);
  if (!work) return;
  feature_rows<<<blocks(work), 256, 0, context_->stream()>>>(
      c, d, pair_first_.data(), pair_second_.data(), begin, 0,
      output, directional, work);
  check_launch("CUDA AO-pair features");
}

void ExactAoPairOperator::generated_rows_impl(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    std::int64_t begin, PairPanel output, PairPanel directional,
    bool direction_only) const {
  validate_coefficients(c);
  if (d.data && (d.rows != c.rows || d.cols != c.cols || d.ld < d.rows))
    throw std::invalid_argument("invalid CUDA active direction");
  const std::int64_t pairs = active_pairs(c.cols);
  if (begin < 0 || begin > n_pairs_ || output.rows < 0 ||
      output.rows > n_pairs_ - begin || output.cols != pairs ||
      output.ld < pairs || (output.rows && !output.data) ||
      (direction_only && (!d.data || directional.data)) ||
      (!direction_only && ((d.data != nullptr) != (directional.data != nullptr))) ||
      (directional.data && (directional.rows != output.rows || directional.cols != pairs ||
                  directional.ld < pairs)))
    throw std::invalid_argument("invalid CUDA generated AO-pair tile");
  const std::int64_t work = checked_product(output.rows, pairs);
  if (!work) return;
  if (!compact_ && n_integrals_ != 0 && begin == 0 && output.rows == n_pairs_) {
    // Full AO rows admit one graph SpMM per active-pair channel slab. The
    // generated CSR kernel below is retained for row subsets, where a full
    // source feature panel would repeat work and increase residency.
    std::size_t device_free=0,device_total=0;
    check_cuda(cudaMemGetInfo(&device_free,&device_total),
               "query CUDA generated AO-pair free memory");
    const auto resident=context_->counters().current_device_bytes;
    const auto budget=context_->device_byte_budget();
    if (resident>budget)
      throw std::runtime_error("CUDA generated AO-pair accounting exceeds budget");
    const auto available=std::min(budget-resident,device_free);
    const auto channel_bytes=static_cast<std::size_t>(
        checked_product(n_pairs_,sizeof(double)));
    const auto max_channels=static_cast<int>(std::min<std::size_t>(
        pairs,available/channel_bytes));
    const auto spmm_scratch=[&](int channels) {
      return spmm_scratch_bytes(sparse_,csr_,n_pairs_,channels,output.ld,
                                output.data,output.data);
    };
    int slab_channels=0;
    // The library may require different scratch for the final short slab.
    // Query both exact widths, so apply() cannot overrun the live budget.
    for (int candidate=max_channels;candidate>0;candidate/=2) {
      const int remainder=static_cast<int>(pairs%candidate);
      const auto scratch=std::max(spmm_scratch(candidate),
                                  remainder ? spmm_scratch(remainder):std::size_t{0});
      const auto feature_bytes=channel_bytes*static_cast<std::size_t>(candidate);
      if (feature_bytes<=available && scratch<=available-feature_bytes) {
        slab_channels=candidate;
        break;
      }
    }
    if (slab_channels>0) {
      Buffer<double> features(*context_,checked_product(n_pairs_,slab_channels));
      for (std::int64_t channel=0;channel<pairs;channel+=slab_channels) {
        const auto count=std::min<std::int64_t>(slab_channels,pairs-channel);
        const PairPanel source{features.data(),n_pairs_,count,count};
        const PairPanel product{output.data+channel,n_pairs_,count,output.ld};
        const auto slab_work=checked_product(n_pairs_,count);
        if (direction_only)
          mixed_feature_rows<<<blocks(slab_work),256,0,context_->stream()>>>(
              c,d,pair_first_.data(),pair_second_.data(),0,channel,source,slab_work);
        else
          feature_rows<<<blocks(slab_work),256,0,context_->stream()>>>(
              c,{},pair_first_.data(),pair_second_.data(),0,channel,
              source,{},slab_work);
        check_launch("full-row CUDA AO-pair features");
        apply({source.data,source.rows,source.cols,source.ld},product);
        if (directional.data) {
          mixed_feature_rows<<<blocks(slab_work),256,0,context_->stream()>>>(
              c,d,pair_first_.data(),pair_second_.data(),0,channel,
              source,slab_work);
          check_launch("full-row CUDA directional AO-pair features");
          apply({source.data,source.rows,source.cols,source.ld},
                {directional.data+channel,n_pairs_,count,directional.ld});
        }
      }
      return;
    }
  }
  if (compact_) {
    for (std::int64_t row = 0; row < output.rows; ++row) {
      check_cuda(cudaMemsetAsync(output.data + row * output.ld, 0,
                                 pairs * sizeof(double), context_->stream()),
                 "zero compact generated AO-pair tile");
      if (directional.data)
        check_cuda(cudaMemsetAsync(directional.data + row * directional.ld, 0,
                                   pairs * sizeof(double), context_->stream()),
                   "zero compact directional AO-pair tile");
    }
    const std::int64_t compact_work = checked_product(n_integrals_, pairs);
    if (compact_work) {
      generated_compact<<<blocks(compact_work), 256, 0, context_->stream()>>>(
          c, d, pair_first_.data(), pair_second_.data(), integral_rows_.data(),
          integral_columns_.data(), integral_values_.data(), n_integrals_,
          begin, output, directional, compact_work, direction_only);
      check_launch("compact generated AO-pair rows");
    }
  } else {
    generated_csr<<<blocks(work), 256, 0, context_->stream()>>>(
        c, d, pair_first_.data(), pair_second_.data(), row_offsets_.data(),
        columns_.data(), values_.data(), begin, output, directional, work,direction_only);
    check_launch("CSR generated AO-pair rows");
  }
}

void ExactAoPairOperator::generated_rows(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    std::int64_t begin, PairPanel output, PairPanel directional) const {
  generated_rows_impl(c, d, begin, output, directional);
}

Buffer<double> ExactAoPairOperator::reserve_products(int n_active) const {
  const auto count=checked_product(n_pairs_,active_pairs(n_active));
  const auto bytes=static_cast<std::size_t>(checked_product(count,sizeof(double)));
  std::size_t free=0,total=0;
  check_cuda(cudaMemGetInfo(&free,&total),"query accepted AO-pair products memory");
  const auto used=context_->counters().current_device_bytes;
  const auto budget=context_->device_byte_budget();
  if (used>budget) throw std::runtime_error("CUDA accepted pair accounting exceeds budget");
  // One retained image, another physical point, and two full response panels.
  // Streaming remains entirely on device when this complete working set cannot fit.
  if (bytes>std::min(free,budget-used)/4) return {};
  return Buffer<double>(*context_,count);
}

void ExactAoPairOperator::forward(linalg::ConstMatrixView c,
    linalg::MatrixView kernel,PairPanel accepted) const {
  const nvtx3::scoped_range range{"ao.pair_forward"};
  validate_coefficients(c);
  const std::int64_t pairs = active_pairs(c.cols);
  if (!kernel.data || kernel.rows != pairs || kernel.cols != pairs || kernel.ld < pairs)
    throw std::invalid_argument("invalid CUDA active pair kernel");
  if (accepted.data && (accepted.rows!=n_pairs_ || accepted.cols!=pairs || accepted.ld<pairs))
    throw std::invalid_argument("invalid CUDA accepted pair products");
  const std::int64_t rows = tile_size(*context_, n_pairs_, pairs, 2,
                                      checked_product(pairs, pairs));
  Buffer<double> b(*context_, rows * pairs), p;
  if (!accepted.data) p=Buffer<double>(*context_,rows*pairs);
  zero_matrix(*context_, kernel);
  for (std::int64_t begin = 0; begin < n_pairs_; begin += rows) {
    const std::int64_t count = std::min(rows, n_pairs_ - begin);
    PairPanel bp{b.data(), count, pairs, pairs};
    PairPanel pp{accepted.data ? accepted.data+begin*accepted.ld:p.data(),
        count,pairs,accepted.data ? accepted.ld:pairs};
    pair_features(c, {}, begin, bp);
    // A full-row forward tile already owns B. Reuse it directly as the
    // SpMM source instead of regenerating the same active-pair features.
    bool direct_spmm=false;
    if (!compact_ && n_integrals_ != 0 && begin == 0 && count == n_pairs_) {
      std::size_t device_free=0,device_total=0;
      check_cuda(cudaMemGetInfo(&device_free,&device_total),
                 "query CUDA forward SpMM free memory");
      const auto available=std::min(context_->device_byte_budget()-
          context_->counters().current_device_bytes,device_free);
      direct_spmm=spmm_scratch_bytes(sparse_,csr_,n_pairs_,static_cast<int>(pairs),
          pp.ld,bp.data,pp.data)<=available;
    }
    if (direct_spmm) apply({bp.data,bp.rows,bp.cols,bp.ld},pp);
    else
      generated_rows_impl(c, {}, begin, pp, {});
    contract_tile(*context_, kernel,
                  {bp.data, bp.rows, bp.cols, bp.ld},
                  {pp.data, pp.rows, pp.cols, pp.ld}, 1.0);
  }
}

void ExactAoPairOperator::directional(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    linalg::MatrixView kernel_direction,ConstPairPanel accepted) const {
  const nvtx3::scoped_range range{"ao.pair_direction"};
  validate_coefficients(c);
  const std::int64_t pairs = active_pairs(c.cols);
  if (!d.data || d.rows != c.rows || d.cols != c.cols || d.ld < d.rows ||
      !kernel_direction.data || kernel_direction.rows != pairs ||
      kernel_direction.cols != pairs || kernel_direction.ld < pairs)
    throw std::invalid_argument("invalid CUDA directional pair kernel");
  if (accepted.data && (accepted.rows!=n_pairs_ || accepted.cols!=pairs || accepted.ld<pairs))
    throw std::invalid_argument("invalid CUDA accepted pair products");
  // K is symmetric. The CPU exact response forms U=dB^T(KB) and completes
  // dM=U+U^T; KdB and the accepted B tile are unnecessary for this consumer.
  Matrix<double> unsymmetric(*context_, pairs, pairs);
  const std::int64_t rows = tile_size(*context_, n_pairs_, pairs, 2,
                                      checked_product(pairs, pairs));
  Buffer<double> db(*context_, rows * pairs), p;
  if (!accepted.data) p=Buffer<double>(*context_,rows*pairs);
  zero_matrix(*context_, unsymmetric.view());
  for (std::int64_t begin = 0; begin < n_pairs_; begin += rows) {
    const std::int64_t count = std::min(rows, n_pairs_ - begin);
    PairPanel dbp{db.data(), count, pairs, pairs};
    ConstPairPanel pp{accepted.data ? accepted.data+begin*accepted.ld:p.data(),
        count,pairs,accepted.data ? accepted.ld:pairs};
    const std::int64_t work = checked_product(count, pairs);
    mixed_feature_rows<<<blocks(work), 256, 0, context_->stream()>>>(
        c, d, pair_first_.data(), pair_second_.data(), begin, 0, dbp, work);
    check_launch("CUDA directional AO-pair features");
    if (!accepted.data) generated_rows_impl(c,{},begin,{p.data(),count,pairs,pairs},{});
    contract_tile(*context_, unsymmetric.view(),
                  {dbp.data, count, pairs, pairs},
                  pp, 1.0);
  }
  complete_pair_direction(*context_, unsymmetric.view(), kernel_direction);
}

void ExactAoPairOperator::adjoint(
    linalg::ConstMatrixView c, linalg::ConstMatrixView g,
    linalg::MatrixView gradient,ConstPairPanel accepted) const {
  const nvtx3::scoped_range range{"ao.pair_adjoint"};
  directional_adjoint(c, {}, g, {}, gradient,accepted);
}

void ExactAoPairOperator::directional_adjoint(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    linalg::ConstMatrixView g, linalg::ConstMatrixView dg,
    linalg::MatrixView gradient,ConstPairPanel accepted,ConstPairPanel exchange) const {
  validate_coefficients(c);
  const std::int64_t pairs = active_pairs(c.cols);
  if ((d.data && (d.rows != c.rows || d.cols != c.cols || d.ld < d.rows)) ||
      !g.data || g.rows != pairs || g.cols != pairs || g.ld < pairs ||
      (dg.data && (dg.rows != pairs || dg.cols != pairs || dg.ld < pairs)) ||
      (dg.data && !d.data) ||
      !gradient.data || gradient.rows != c.rows || gradient.cols != c.cols ||
      gradient.ld < gradient.rows)
    throw std::invalid_argument("invalid CUDA exact pair adjoint dimensions");
  if (accepted.data && (accepted.rows!=n_pairs_ || accepted.cols!=pairs || accepted.ld<pairs))
    throw std::invalid_argument("invalid CUDA accepted pair products");
  const bool projected=d.data && exchange.data;
  Buffer<double> h(*context_, pairs * pairs);
  Buffer<double> dh;
  if (dg.data) dh = Buffer<double>(*context_, pairs * pairs);
  const std::int64_t rows = tile_size(*context_, n_pairs_, pairs,
      d.data ? 4 : 2, checked_product(gradient.rows, gradient.cols));
  Buffer<double> p, t(*context_, rows * pairs);
  if (!accepted.data) p=Buffer<double>(*context_,rows*pairs);
  Buffer<double> dp, dt;
  if (d.data) {
    if (!projected) dp = Buffer<double>(*context_, rows * pairs);
    if (!projected || dg.data) dt = Buffer<double>(*context_, rows * pairs);
  }
  const std::int64_t hwork = pairs * pairs;
  symmetrize<<<blocks(hwork), 256, 0, context_->stream()>>>(
      g, dg, h.data(), dh.data(), blas_int(pairs), hwork);
  check_launch("symmetrize CUDA pair adjoint");
  zero_matrix(*context_, gradient);
  for (std::int64_t begin = 0; begin < n_pairs_; begin += rows) {
    const std::int64_t count = std::min(rows, n_pairs_ - begin);
    ConstPairPanel pp{accepted.data ? accepted.data+begin*accepted.ld:p.data(),
        count,pairs,accepted.data ? accepted.ld:pairs};
    PairPanel dpp{dp.data(), count, pairs, pairs};
    if (!accepted.data)
      generated_rows_impl(c,projected ? linalg::ConstMatrixView{}:d,begin,
          {p.data(),count,pairs,pairs},d.data && !projected ? dpp:PairPanel{});
    else if (d.data && !projected) generated_rows_impl(c,d,begin,dpp,{},true);
    PairPanel tp{t.data(), count, pairs, pairs};
    multiply_pair_tile(*context_, tp, pp,
                       h.data(), blas_int(pairs));
    const std::int64_t work = count * pairs;
    pair_pullback<<<blocks(work), 256, 0, context_->stream()>>>(
        tp, d.data ? d : c, pair_first_.data(), pair_second_.data(),
        begin, gradient, work);
    check_launch("accepted CUDA pair pullback");
    if (d.data) {
      PairPanel dtp{dt.data(), count, pairs, pairs};
      if (!projected)
        multiply_pair_tile(*context_, dtp, {dpp.data, count, pairs, pairs},
                           h.data(), blas_int(pairs));
      if (dg.data)
        multiply_pair_tile(*context_, dtp, pp,
                           dh.data(), blas_int(pairs), projected ? 0.0:1.0);
      if (!projected || dg.data) {
        pair_pullback<<<blocks(work), 256, 0, context_->stream()>>>(
            dtp, c, pair_first_.data(), pair_second_.data(), begin, gradient, work);
        check_launch("directional CUDA pair pullback");
      }
    }
  }
  if (projected) {
    // The pair sweep and exchange projection are sequential consumers. Do
    // not retain their large temporary panels simultaneously under a tight
    // device budget; the original point-owned integral images stay shared.
    p={};t={};dp={};dt={};h={};dh={};
    exchange_pair_direction(exchange,d,g,gradient);
  }
}

void ExactAoPairOperator::contract_jk(linalg::ConstMatrixView density,
                                       linalg::MatrixView output,
                                       double coulomb, double exchange,
                                       linalg::MatrixView coulomb_output) const {
  const nvtx3::scoped_range range{"ao.jk"};
  if (!density.data || density.rows != n_bf_ || density.cols != n_bf_ ||
      density.ld < n_bf_ || !output.data || output.rows != n_bf_ ||
      output.cols != n_bf_ || output.ld < n_bf_ ||
      density.data == output.data || !std::isfinite(coulomb) ||
      !std::isfinite(exchange))
    throw std::invalid_argument("invalid CUDA symmetric JK contraction");
  if (coulomb_output.data && (coulomb_output.rows!=n_bf_ ||
      coulomb_output.cols!=n_bf_ || coulomb_output.ld<n_bf_ ||
      coulomb_output.data==output.data || coulomb_output.data==density.data))
    throw std::invalid_argument("invalid CUDA separate Coulomb output");
  zero_matrix(*context_, output);
  if (exchange != 0.0 && n_unique_ != 0) {
    exchange_unique<<<blocks(n_unique_), 256, 0, context_->stream()>>>(
        integral_rows_.data(), integral_columns_.data(), integral_edges_.data(),
        columns_.data(), compact_ ? integral_values_.data() : values_.data(),
        n_unique_, pair_first_.data(), pair_second_.data(), density, output,
        exchange, compact_,1);
    check_launch("CUDA unique-ERI exchange contraction");
  }
  if (coulomb != 0.0 || coulomb_output.data) {
    Buffer<double> packed_density(*context_, n_pairs_);
    Buffer<double> j_pairs(*context_, n_pairs_);
    pack_symmetric_density<<<blocks(n_pairs_), 256, 0, context_->stream()>>>(
        density, pair_first_.data(), pair_second_.data(),
        packed_density.data(), n_pairs_);
    check_launch("pack CUDA symmetric AO density");
    apply({packed_density.data(), n_pairs_, 1, 1},
          {j_pairs.data(), n_pairs_, 1, 1});
    const std::int64_t work = checked_product(n_bf_, n_bf_);
    finish_jk<<<blocks(work), 256, 0, context_->stream()>>>(
        j_pairs.data(), output, coulomb, work);
    check_launch("finish CUDA AO Coulomb contraction");
    if (coulomb_output.data) {
      zero_matrix(*context_,coulomb_output);
      finish_jk<<<blocks(work),256,0,context_->stream()>>>(
          j_pairs.data(),coulomb_output,1.0,work);
      check_launch("reuse CUDA separate Coulomb contraction");
    }
  }
}

void ExactAoPairOperator::contract_jk_batch(linalg::ConstMatrixView density,
    linalg::MatrixView output,double coulomb,double exchange) const {
  const nvtx3::scoped_range range{"ao.jk_batch"};
  if (!density.data || density.rows!=n_bf_ || density.cols<=0 ||
      density.cols%n_bf_!=0 || density.ld<n_bf_ || !output.data ||
      output.rows!=n_bf_ || output.cols!=density.cols || output.ld<n_bf_ ||
      output.data==density.data || !std::isfinite(coulomb) || !std::isfinite(exchange))
    throw std::invalid_argument("invalid CUDA symmetric JK batch");
  const int channels=blas_int(density.cols/n_bf_);
  zero_matrix(*context_,output);
  if (exchange!=0.0 && n_unique_!=0) {
    exchange_unique<<<blocks(n_unique_),256,0,context_->stream()>>>(
        integral_rows_.data(),integral_columns_.data(),integral_edges_.data(),
        columns_.data(),compact_ ? integral_values_.data():values_.data(),
        n_unique_,pair_first_.data(),pair_second_.data(),density,output,exchange,compact_,channels);
    check_launch("CUDA batched unique-ERI exchange contraction");
  }
  if (coulomb!=0.0) {
    Buffer<double> packed(*context_,checked_product(n_pairs_,channels));
    Buffer<double> image(*context_,checked_product(n_pairs_,channels));
    const auto work=checked_product(n_pairs_,channels);
    pack_density_batch<<<blocks(work),256,0,context_->stream()>>>(
        density,pair_first_.data(),pair_second_.data(),packed.data(),n_pairs_,channels);
    check_launch("pack CUDA symmetric AO density batch");
    apply({packed.data(),n_pairs_,channels,channels},{image.data(),n_pairs_,channels,channels});
    const auto dense_work=checked_product(output.rows,output.cols);
    finish_jk_batch<<<blocks(dense_work),256,0,context_->stream()>>>(
        image.data(),output,coulomb,dense_work,channels);
    check_launch("finish CUDA batched Coulomb contraction");
  }
}

void ExactAoPairOperator::core_energy(
    linalg::ConstMatrixView density, linalg::ConstMatrixView core,
    linalg::MatrixView effective_core, double* energy) const {
  if (!energy || !core.data || core.rows != n_bf_ || core.cols != n_bf_ ||
      core.ld < n_bf_ || core.data == effective_core.data)
    throw std::invalid_argument("invalid CUDA exact core energy input");
  contract_jk(density, effective_core);
  const std::int64_t work = checked_product(n_bf_, n_bf_);
  add_dense<<<blocks(work), 256, 0, context_->stream()>>>(
      effective_core, core, work);
  check_launch("CUDA effective core addition");
  check_cuda(cudaMemsetAsync(energy, 0, sizeof(double), context_->stream()),
             "zero CUDA core energy");
  trace_density<<<blocks(work), 256, 0, context_->stream()>>>(
      density, core, effective_core, 1.0, energy, work);
  check_launch("CUDA core energy trace");
}

void ExactAoPairOperator::core_directional(
    linalg::ConstMatrixView density_direction,
    linalg::ConstMatrixView effective_core,
    linalg::MatrixView effective_core_direction,
    double* energy_direction) const {
  if (!energy_direction || !effective_core.data ||
      effective_core.rows != n_bf_ || effective_core.cols != n_bf_ ||
      effective_core.ld < n_bf_)
    throw std::invalid_argument("invalid CUDA exact core direction input");
  contract_jk(density_direction, effective_core_direction);
  check_cuda(cudaMemsetAsync(energy_direction, 0, sizeof(double), context_->stream()),
             "zero CUDA core energy direction");
  const std::int64_t work = checked_product(n_bf_, n_bf_);
  trace_density<<<blocks(work), 256, 0, context_->stream()>>>(
      density_direction, effective_core, {}, 2.0, energy_direction, work);
  check_launch("CUDA core energy directional trace");
}

void ExactAoPairOperator::active_core_density_adjoint(
    linalg::ConstMatrixView c, linalg::ConstMatrixView a,
    linalg::MatrixView density_adjoint) const {
  validate_coefficients(c);
  if (!a.data || a.rows != c.cols || a.cols != c.cols || a.ld < a.rows ||
      !density_adjoint.data || density_adjoint.rows != n_bf_ ||
      density_adjoint.cols != n_bf_ || density_adjoint.ld < n_bf_)
    throw std::invalid_argument("invalid CUDA active-core density adjoint");
  const int n = blas_int(c.cols);
  Buffer<double> symmetric(*context_, static_cast<std::size_t>(n) * n);
  symmetrize<<<blocks(static_cast<std::int64_t>(n) * n), 256, 0, context_->stream()>>>(
      a, {}, symmetric.data(), nullptr, n, static_cast<std::int64_t>(n) * n);
  check_launch("symmetrize CUDA active-core adjoint");
  Matrix<double> contracted(*context_, n_bf_, n);
  Matrix<double> active_density(*context_, n_bf_, n_bf_);
  gemm(*context_, contracted.view(), c,
       {symmetric.data(), n, n, n});
  gemm(*context_, active_density.view(), contracted.view(), c,
       0.5, 0.0, Operation::none, Operation::transpose);
  contract_jk(active_density.view(), density_adjoint);
}

void ExactAoPairOperator::active_core_density_directional_adjoint(
    linalg::ConstMatrixView c, linalg::ConstMatrixView d,
    linalg::ConstMatrixView a, linalg::ConstMatrixView da,
    linalg::MatrixView density_adjoint_direction) const {
  validate_coefficients(c);
  if (!d.data || d.rows != c.rows || d.cols != c.cols || d.ld < d.rows ||
      !a.data || a.rows != c.cols || a.cols != c.cols || a.ld < a.rows ||
      !da.data || da.rows != c.cols || da.cols != c.cols || da.ld < da.rows ||
      !density_adjoint_direction.data || density_adjoint_direction.rows != n_bf_ ||
      density_adjoint_direction.cols != n_bf_ ||
      density_adjoint_direction.ld < n_bf_)
    throw std::invalid_argument("invalid CUDA directional active-core density adjoint");
  const int n = blas_int(c.cols);
  Buffer<double> symmetric(*context_, static_cast<std::size_t>(n) * n);
  Buffer<double> directional_symmetric(*context_, static_cast<std::size_t>(n) * n);
  symmetrize<<<blocks(static_cast<std::int64_t>(n) * n), 256, 0, context_->stream()>>>(
      a, da, symmetric.data(), directional_symmetric.data(), n,
      static_cast<std::int64_t>(n) * n);
  check_launch("symmetrize CUDA directional active-core adjoint");
  Matrix<double> contracted(*context_, n_bf_, n);
  Matrix<double> directional_contracted(*context_, n_bf_, n);
  Matrix<double> active_density_direction(*context_, n_bf_, n_bf_);
  const linalg::ConstMatrixView h{symmetric.data(), n, n, n};
  const linalg::ConstMatrixView dh{directional_symmetric.data(), n, n, n};
  gemm(*context_, contracted.view(), c, h);
  gemm(*context_, directional_contracted.view(), d, h);
  gemm(*context_, directional_contracted.view(), c, dh, 1.0, 1.0);
  gemm(*context_, active_density_direction.view(),
       directional_contracted.view(), c, 0.5, 0.0,
       Operation::none, Operation::transpose);
  gemm(*context_, active_density_direction.view(), contracted.view(), d,
       0.5, 1.0, Operation::none, Operation::transpose);
  contract_jk(active_density_direction.view(), density_adjoint_direction);
}

}  // namespace xmvb::cuda
