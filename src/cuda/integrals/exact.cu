#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_search.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/integrals/packed.hpp"

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

__global__ void density_feature_rows(linalg::ConstMatrixView c,
    const int* first,const int* second,std::int64_t begin,PairPanel output) {
  const auto work=output.rows*output.cols;
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto row=k/output.cols,column=k%output.cols;
    const int i=first[begin+row],j=second[begin+row];
    output.data[row*output.ld+column]=(i==j ? 1.0:2.0)*
        c.data[i+c.ld*column]*c.data[j+c.ld*column];
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

template<class PairMatrix>
__global__ void generated_packed(PairMatrix matrix,linalg::ConstMatrixView c,
    linalg::ConstMatrixView d,const int* first,const int* second,
    std::int64_t begin,PairPanel output,PairPanel directional,std::int64_t work,
    std::int64_t sources,bool direction_only) {
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto row=k/output.cols,channel=k%output.cols;
    int a,b;decode_pair(channel,a,b);
    double value=0.0,delta=0.0;
    for (std::int64_t q=0;q<sources;++q) {
      const double integral=matrix.value(begin+row,q);
      if (integral==0.0) continue;
      if (!direction_only) value+=integral*feature(c,first,second,q,a,b);
      if (d.data) delta+=integral*mixed_feature(c,d,first,second,q,a,b);
    }
    output.data[row*output.ld+channel]=direction_only ? delta:value;
    if (directional.data) directional.data[row*directional.ld+channel]=delta;
  }
}

template<class Index>
__global__ void generated_unique(linalg::ConstMatrixView c,
                                  linalg::ConstMatrixView d,
                                  const int* first, const int* second,
                                  const Index* offsets,
                                  const Index* columns,
                                  const double* values,
                                  std::int64_t begin, PairPanel output,
                                  PairPanel directional,
                                  bool direction_only) {
  const int left=blockIdx.x;
  const auto start=offsets[left],end=offsets[left+1];
  const auto work=static_cast<std::int64_t>(end-start)*output.cols;
  for (std::int64_t k=threadIdx.x;k<work;k+=blockDim.x) {
    const auto edge=start+k/output.cols;
    const std::int64_t p = k % output.cols;
    int a, b;
    decode_pair(p, a, b);
    const int right=static_cast<int>(columns[edge]);
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

__global__ void subtract_pair_diagonal(const double* diagonal,ConstPairPanel source,
    PairPanel target,std::int64_t work) {
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto row=k/source.cols,column=k%source.cols;
    target.data[row*target.ld+column]-=diagonal[row]*source.data[row*source.ld+column];
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

template<bool RankOne=false>
__device__ void add_oriented_exchange(int u,int v,double value,
    const int* first,const int* second,linalg::ConstMatrixView density,
    linalg::MatrixView output,int channels) {
  const int ui = first[u], uj = second[u], vi = first[v], vj = second[v];
  for (int swap_u = 0; swap_u < (ui == uj ? 1 : 2); ++swap_u) {
    const int a = swap_u ? uj : ui, b = swap_u ? ui : uj;
    for (int swap_v = 0; swap_v < (vi == vj ? 1 : 2); ++swap_v) {
      const int c = swap_v ? vj : vi, d = swap_v ? vi : vj;
      for (int channel=0;channel<channels;++channel) {
        double source;
        if constexpr (RankOne)
          source=density.data[b+density.ld*channel]*density.data[d+density.ld*channel];
        else source=density.data[b+density.ld*(d+channel*density.rows)];
        atomic_add_double(output.data+a+output.ld*(c+channel*output.rows),
            value*source);
      }
    }
  }
}

template<class Index>
__device__ int unique_row(const Index* offsets,int rows,std::int64_t edge,int begin=0) {
  return begin+cub::UpperBound(offsets+begin+1,rows-begin,edge);
}

// Balance integral edges, not variably sized CSR rows. A warp finds its
// first row once; only lanes crossing that row's end need another lookup.
// Row offsets are small and shared; no full edge-to-row index is stored.
template<class Index,bool RankOne=false>
__global__ void exchange_unique(const Index* offsets,const Index* columns,
    const double* values,int rows,std::int64_t edges,const int* first,const int* second,
    linalg::ConstMatrixView density,linalg::MatrixView output,double weight,int channels) {
  const int lane=threadIdx.x&31;
  for (auto begin=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x-lane;
       begin<edges;begin+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    int left=0;
    if (!lane) left=unique_row(offsets,rows,begin);
    left=__shfl_sync(0xffffffff,left,0);
    const auto p=begin+lane;
    if (p>=edges) continue;
    if (p>=offsets[left+1]) left=unique_row(offsets,rows,p,left+1);
    const int right=static_cast<int>(columns[p]);
    const double value=weight*values[p];
    add_oriented_exchange<RankOne>(left,right,value,first,second,density,output,channels);
    if (left!=right)
      add_oriented_exchange<RankOne>(right,left,value,first,second,density,output,channels);
  }
}

// A CSR row (i,j) contributes only to output rows i and j. Merge those
// contributions in block-local panels before the global accumulation. CSR
// already contains the reverse pair orientation; do not emit it again.
__global__ void exchange_rows(const int* offsets,const int* columns,const double* values,
    const int* first,const int* second,linalg::ConstMatrixView density,
    linalg::MatrixView output,double weight,int channels,int column_width,
    int channel_width) {
  extern __shared__ double sums[];
  const int row=blockIdx.x,ui=first[row],uj=second[row];
  const int column_begin=blockIdx.y*column_width;
  const int count=min(column_width,static_cast<int>(output.rows)-column_begin);
  const int channel_begin=blockIdx.z*channel_width;
  const int channel_count=min(channel_width,channels-channel_begin);
  const int panel=count*channel_count,cells=2*panel;
  for (int k=threadIdx.x;k<cells;k+=blockDim.x) sums[k]=0.0;
  __syncthreads();
  const int begin=offsets[row];
  const int end=offsets[row+1];
  for (int edge=begin+threadIdx.x;edge<end;edge+=blockDim.x) {
    const int v=columns[edge],vi=first[v],vj=second[v];
    const double value=weight*values[edge];
    if (value==0.0) continue;
    for (int c=0;c<channel_count;++c) {
      const auto shift=static_cast<std::int64_t>(channel_begin+c)*density.rows;
      const int base=c*count;
      if (vi>=column_begin && vi<column_begin+count) {
        atomic_add_double(sums+base+vi-column_begin,
            value*density.data[uj+density.ld*(vj+shift)]);
        if (ui!=uj) atomic_add_double(sums+panel+base+vi-column_begin,
            value*density.data[ui+density.ld*(vj+shift)]);
      }
      if (vi!=vj && vj>=column_begin && vj<column_begin+count) {
        atomic_add_double(sums+base+vj-column_begin,
            value*density.data[uj+density.ld*(vi+shift)]);
        if (ui!=uj) atomic_add_double(sums+panel+base+vj-column_begin,
            value*density.data[ui+density.ld*(vi+shift)]);
      }
    }
  }
  __syncthreads();
  for (int k=threadIdx.x;k<cells;k+=blockDim.x) {
    const double value=sums[k];
    if (value==0.0) continue;
    const int local=k%panel,channel=channel_begin+local/count;
    const int column=column_begin+local%count,ao=k<panel ? ui:uj;
    atomic_add_double(output.data+ao+output.ld*(column+channel*output.rows),value);
  }
}

void launch_exchange_rows(Context& context,const int* offsets,const int* columns,
    const double* values,const int* first,const int* second,
    linalg::ConstMatrixView density,linalg::MatrixView output,double weight,
    int channels,int shared_values,std::int64_t pairs) {
  // Capacity comes from the device, not an independent numerical/cache knob.
  // Only very wide AO rows or channel blocks require another source-row visit.
  const int column_width=std::min(blas_int(output.rows),shared_values/2);
  const int channel_width=std::min(channels,shared_values/(2*column_width));
  const dim3 grid(blas_int(pairs),(blas_int(output.rows)+column_width-1)/column_width,
      (channels+channel_width-1)/channel_width);
  const auto bytes=2*static_cast<std::size_t>(column_width)*channel_width*sizeof(double);
  exchange_rows<<<grid,256,bytes,context.stream()>>>(offsets,columns,values,first,second,
        density,output,weight,channels,column_width,channel_width);
}

struct ExchangeSum {
  double forward,reverse;
};

struct AddExchange {
  __device__ ExchangeSum operator()(ExchangeSum a,ExchangeSum b) const {
    return {a.forward+b.forward,a.reverse+b.reverse};
  }
};

__device__ std::int64_t ao_pair(int a,int b) {
  const auto high=max(a,b),low=min(a,b);
  return static_cast<std::int64_t>(high)*(high+1)/2+low;
}

// One output AO pair owns both orientations. Summing unordered source pairs
// needs no scatter or atomics and preserves nonsymmetric density inputs too:
// K_mu,nu=sum_kl (mu k|nu l) D_kl, K_nu,mu uses the transposed D.
__global__ void exchange_pair_gather(PackedAoPairMatrix::DeviceView eri,
    const int* first,const int* second,linalg::ConstMatrixView density,
    linalg::MatrixView output,double weight) {
  using Reduce=cub::BlockReduce<ExchangeSum,256>;
  __shared__ typename Reduce::TempStorage temporary;
  const int mu=first[blockIdx.x],nu=second[blockIdx.x];
  ExchangeSum sum{0.0,0.0};
  for (std::int64_t q=threadIdx.x;q<eri.pairs;q+=blockDim.x) {
    const int k=first[q],l=second[q];
    const double a=eri.value(ao_pair(mu,k),ao_pair(nu,l));
    const double d=density.data[k+density.ld*l],dt=density.data[l+density.ld*k];
    sum.forward+=a*d;
    sum.reverse+=a*dt;
    if (k!=l) {
      const double b=eri.value(ao_pair(mu,l),ao_pair(nu,k));
      sum.forward+=b*dt;
      sum.reverse+=b*d;
    }
  }
  const auto value=Reduce(temporary).Reduce(sum,AddExchange{});
  if (threadIdx.x==0) {
    output.data[mu+output.ld*nu]=weight*value.forward;
    if (mu!=nu) output.data[nu+output.ld*mu]=weight*value.reverse;
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
    cusparseSpMatDescr_t csr,ConstPairPanel input,PairPanel output,bool unique=false) {
  cusparseDnMatDescr_t source=nullptr,target=nullptr;
  check_sparse(cusparseCreateDnMat(&source,input.rows,input.cols,input.ld,
      const_cast<double*>(input.data),CUDA_R_64F,CUSPARSE_ORDER_ROW),
      "create generated AO-pair source descriptor");
  try {
    check_sparse(cusparseCreateDnMat(&target,output.rows,output.cols,output.ld,
        output.data,CUDA_R_64F,CUSPARSE_ORDER_ROW),
        "create generated AO-pair target descriptor");
    const double alpha=1.0;
    std::size_t bytes=0;
    for (int pass=0;pass<1+unique;++pass) {
      const double beta=pass ? 1.0:0.0;
      std::size_t current=0;
      const auto op=pass ? CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
      check_sparse(cusparseSpMM_bufferSize(sparse,op,
          CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,csr,source,&beta,target,
          CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,&current),
          "query generated AO-pair SpMM scratch");
      bytes=std::max(bytes,current);
    }
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
  int device=0,shared_bytes=0;
  check_cuda(cudaGetDevice(&device),"query CUDA exchange device");
  check_cuda(cudaDeviceGetAttribute(&shared_bytes,cudaDevAttrMaxSharedMemoryPerBlock,device),
      "query CUDA exchange block capacity");
  exchange_shared_values_=shared_bytes/sizeof(double);
  for (int pair = 0; pair < n_pairs_; ++pair) {
    if (graph.pair_first[pair] < 0 || graph.pair_first[pair] >= n_bf_ ||
        graph.pair_second[pair] < 0 || graph.pair_second[pair] > graph.pair_first[pair] ||
        static_cast<std::int64_t>(graph.pair_first[pair]) *
                (graph.pair_first[pair] + 1) / 2 + graph.pair_second[pair] != pair)
      throw std::invalid_argument("invalid CUDA AO-pair component order");
  }
  compact_ = graph.has_compact_integral_stream();
  std::size_t directed_bytes=0,staging_bytes=0;
  if (compact_) {
    if (!graph.valid_unique_storage())
      throw std::invalid_argument("invalid compact AO integral stream");
    n_integrals_ = graph.integral_count();
    std::size_t directed=0;
    graph.for_each_chunk([&](auto rows,auto columns,auto values) {
      for (std::size_t k = 0; k < values.size(); ++k) {
        if (rows[k] < 0 || rows[k] >= n_pairs_ || columns[k] < 0 || columns[k] >= n_pairs_)
          throw std::invalid_argument("compact AO integral index out of range");
        directed+=1+(rows[k]!=columns[k]);
      }
    });
    directed_bytes=directed*(sizeof(int)+sizeof(double))+(n_pairs_+1)*sizeof(int);
    staging_bytes=2*sizeof(int)+sizeof(double);
  } else {
    if (graph.row_offsets.size() != static_cast<std::size_t>(n_pairs_ + 1) ||
        graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
        graph.columns.size() != graph.values.size() ||
        graph.columns.size() != static_cast<std::size_t>(graph.row_offsets.back()))
      throw std::invalid_argument("invalid CUDA AO-pair CSR graph");
    std::size_t largest_row=0;
    for (int row = 0; row < n_pairs_; ++row) {
      if (graph.row_offsets[row] > graph.row_offsets[row + 1])
        throw std::invalid_argument("unsorted CUDA AO-pair CSR offsets");
      largest_row=std::max(largest_row,static_cast<std::size_t>(
          graph.row_offsets[row+1]-graph.row_offsets[row]));
    }
    for (int column : graph.columns)
      if (column < 0 || column >= n_pairs_)
        throw std::invalid_argument("CUDA AO-pair CSR column out of range");
    n_integrals_ = graph.columns.size();
    directed_bytes=graph.columns.size()*(sizeof(int)+sizeof(double))+
        graph.row_offsets.size()*sizeof(int);
    staging_bytes=largest_row*(sizeof(int)+sizeof(double))+2*sizeof(int);
  }
  pair_first_ = Buffer<int>(context, n_pairs_);
  pair_second_ = Buffer<int>(context, n_pairs_);
  context.copy_to_device(pair_first_.data(), graph.pair_first.data(), n_pairs_ * sizeof(int));
  context.copy_to_device(pair_second_.data(), graph.pair_second.data(), n_pairs_ * sizeof(int));
  const auto packed_bytes=PackedAoPairMatrix::storage_bytes(n_pairs_);
  const auto available=response_capacity();
  // Input layout does not change execution planning: use exactly the implied
  // symmetric row-graph storage cost, without constructing or uploading it.
  if (packed_bytes<=directed_bytes && packed_bytes<=available &&
      staging_bytes<=available-packed_bytes) {
    packed_=std::make_unique<PackedAoPairMatrix>(context,graph,n_bf_);
    compact_=false;
    return;
  }
  if (compact_) {
    upload_unique(graph);
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
  }
  if (n_integrals_ != 0) {
    check_sparse(cusparseCreate(&sparse_), "cusparseCreate");
    try {
      check_sparse(cusparseSetStream(sparse_, context.stream()), "cusparseSetStream");
      const auto index_type=wide_ ? CUSPARSE_INDEX_64I:CUSPARSE_INDEX_32I;
      void* offsets=wide_ ? static_cast<void*>(wide_offsets_.data()):row_offsets_.data();
      void* columns=wide_ ? static_cast<void*>(wide_columns_.data()):columns_.data();
      check_sparse(cusparseCreateCsr(&csr_, n_pairs_, n_pairs_, n_integrals_,
                                    offsets, columns, values_.data(),index_type,index_type,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                   "cusparseCreateCsr");
    } catch (...) {
      cusparseDestroy(sparse_);
      sparse_ = nullptr;
      throw;
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

std::size_t ExactAoPairOperator::apply_workspace(ConstPairPanel source,PairPanel target) const {
  if (!n_integrals_) return 0;
  if (packed_) return packed_->workspace_bytes(source.cols,0,target.rows);
  return spmm_scratch_bytes(sparse_,csr_,source,target,compact_);
}

void ExactAoPairOperator::apply(ConstPairPanel source, PairPanel target) const {
  validate_panel(source);
  validate_panel(target);
  if (source.cols != target.cols || source.data == target.data)
    throw std::invalid_argument("CUDA AO-pair action panel mismatch or alias");
  if (n_integrals_ == 0) {
    zero_matrix(*context_,{target.data,target.cols,target.rows,target.ld});
    return;
  }
  if (packed_) {
    packed_->apply(source,target);
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
    const double alpha = 1.0;
    std::size_t bytes = 0;
    for (int pass=0;pass<1+compact_;++pass) {
      const auto op=pass ? CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
      const double beta=pass ? 1.0:0.0;
      std::size_t current=0;
      check_sparse(cusparseSpMM_bufferSize(sparse_, op,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, csr_, a, &beta, b, CUDA_R_64F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, &current),
                 "cusparseSpMM_bufferSize");
      bytes=std::max(bytes,current);
    }
    Buffer<char> scratch(*context_, bytes);
    for (int pass=0;pass<1+compact_;++pass) {
      const auto op=pass ? CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
      const double beta=pass ? 1.0:0.0;
      check_sparse(cusparseSpMM(sparse_, op,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, csr_, a, &beta, b, CUDA_R_64F,
                             CUSPARSE_SPMM_ALG_DEFAULT, scratch.data()),
                 "cusparseSpMM AO-pair action");
    }
    if (compact_) {
      const auto work=checked_product(n_pairs_,source.cols);
      subtract_pair_diagonal<<<blocks(work),256,0,context_->stream()>>>(
          diagonal_.data(),source,target,work);
      check_launch("remove double-counted unique CSR diagonal");
    }
  } catch (...) {
    if (b) cusparseDestroyDnMat(b);
    cusparseDestroyDnMat(a);
    throw;
  }
  cusparseDestroyDnMat(b);
  cusparseDestroyDnMat(a);
}

void ExactAoPairOperator::density_features(linalg::ConstMatrixView c,
    std::int64_t begin,PairPanel output) const {
  validate_coefficients(c);
  if (begin<0 || output.rows<0 || output.rows>n_pairs_-begin ||
      output.cols!=c.cols || output.ld<output.cols || (output.rows && !output.data))
    throw std::invalid_argument("invalid CUDA rank-one density feature tile");
  const auto work=checked_product(output.rows,output.cols);
  if (work) {
    density_feature_rows<<<blocks(work),256,0,context_->stream()>>>(
        c,pair_first_.data(),pair_second_.data(),begin,output);
    check_launch("CUDA rank-one density features");
  }
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
  if (packed_) {
    const auto available=response_capacity();
    const auto channel_bytes=static_cast<std::size_t>(n_pairs_)*sizeof(double);
    int slab=static_cast<int>(std::min<std::size_t>(pairs,available/channel_bytes));
    for (;slab>0;slab/=2) {
      const auto bytes=channel_bytes*slab;
      if (bytes<=available && packed_->workspace_bytes(slab,begin,output.rows)<=available-bytes)
        break;
    }
    if (slab) {
      Buffer<double> features(*context_,n_pairs_*slab);
      for (std::int64_t channel=0;channel<pairs;channel+=slab) {
        const auto count=std::min<std::int64_t>(slab,pairs-channel);
        const PairPanel source{features.data(),n_pairs_,count,count};
        const auto source_work=n_pairs_*count;
        if (direction_only)
          mixed_feature_rows<<<blocks(source_work),256,0,context_->stream()>>>(
              c,d,pair_first_.data(),pair_second_.data(),0,channel,source,source_work);
        else
          feature_rows<<<blocks(source_work),256,0,context_->stream()>>>(
              c,{},pair_first_.data(),pair_second_.data(),0,channel,source,{},source_work);
        check_launch("CUDA packed pair source features");
        packed_->apply({source.data,source.rows,source.cols,source.ld},
            {output.data+channel,output.rows,count,output.ld},begin);
        if (directional.data) {
          mixed_feature_rows<<<blocks(source_work),256,0,context_->stream()>>>(
              c,d,pair_first_.data(),pair_second_.data(),0,channel,source,source_work);
          check_launch("CUDA packed directional source features");
          packed_->apply({source.data,source.rows,source.cols,source.ld},
              {directional.data+channel,directional.rows,count,directional.ld},begin);
        }
      }
    } else {
      // A genuine low-budget row tile cannot hold a complete source panel.
      // Generate its exact features in-register while reading the same layout.
      generated_packed<<<blocks(work),256,0,context_->stream()>>>(packed_->view(),
          c,d,pair_first_.data(),pair_second_.data(),begin,output,directional,work,n_pairs_,
          direction_only);
      check_launch("CUDA streamed packed pair features");
    }
    return;
  }
  if (n_integrals_ != 0 && begin == 0 && output.rows == n_pairs_) {
    // Full AO rows admit a library graph action per active-pair channel slab. The
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
      return apply_workspace({output.data,n_pairs_,channels,channels},
          {output.data,n_pairs_,channels,output.ld});
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
    // The transposed view shares this row-major panel, including its pitch.
    // Clear it in one CUDA operation, not one launch per AO-pair row.
    zero_matrix(*context_,{output.data,output.cols,output.rows,output.ld});
    if (directional.data)
      zero_matrix(*context_,{directional.data,directional.cols,directional.rows,directional.ld});
    if (n_integrals_) {
      if (wide_)
        generated_unique<<<static_cast<unsigned int>(n_pairs_),256,0,context_->stream()>>>(
            c,d,pair_first_.data(),pair_second_.data(),wide_offsets_.data(),
            wide_columns_.data(),values_.data(),begin,output,directional,direction_only);
      else
        generated_unique<<<static_cast<unsigned int>(n_pairs_),256,0,context_->stream()>>>(
            c,d,pair_first_.data(),pair_second_.data(),row_offsets_.data(),
            columns_.data(),values_.data(),begin,output,directional,direction_only);
      check_launch("unique CSR generated AO-pair rows");
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
    bool direct_action=false;
    if (n_integrals_ != 0 && begin == 0 && count == n_pairs_) {
      std::size_t device_free=0,device_total=0;
      check_cuda(cudaMemGetInfo(&device_free,&device_total),
                 "query CUDA forward SpMM free memory");
      const auto available=std::min(context_->device_byte_budget()-
          context_->counters().current_device_bytes,device_free);
      direct_action=apply_workspace({bp.data,bp.rows,bp.cols,bp.ld},pp)<=available;
    }
    if (direct_action) apply({bp.data,bp.rows,bp.cols,bp.ld},pp);
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

void ExactAoPairOperator::unique_exchange(linalg::ConstMatrixView density,
    linalg::MatrixView output,double weight,int channels) const {
  if (wide_)
    exchange_unique<<<blocks(n_integrals_),256,0,context_->stream()>>>(
        wide_offsets_.data(),wide_columns_.data(),values_.data(),blas_int(n_pairs_),
        n_integrals_,pair_first_.data(),pair_second_.data(),density,output,weight,channels);
  else
    exchange_unique<<<blocks(n_integrals_),256,0,context_->stream()>>>(
        row_offsets_.data(),columns_.data(),values_.data(),blas_int(n_pairs_),
        n_integrals_,pair_first_.data(),pair_second_.data(),density,output,weight,channels);
}

void ExactAoPairOperator::rank_one_exchange(linalg::ConstMatrixView factors,
    linalg::MatrixView output) const {
  const nvtx3::scoped_range range{"ao.rank_one_exchange"};
  zero_matrix(*context_,output);
  if (!n_integrals_) return;
  const auto channels=blas_int(factors.cols);
  if (wide_)
    exchange_unique<std::int64_t,true><<<blocks(n_integrals_),256,0,context_->stream()>>>(
        wide_offsets_.data(),wide_columns_.data(),values_.data(),blas_int(n_pairs_),
        n_integrals_,pair_first_.data(),pair_second_.data(),factors,output,1.0,channels);
  else
    exchange_unique<int,true><<<blocks(n_integrals_),256,0,context_->stream()>>>(
        row_offsets_.data(),columns_.data(),values_.data(),blas_int(n_pairs_),
        n_integrals_,pair_first_.data(),pair_second_.data(),factors,output,1.0,channels);
  check_launch("CUDA rank-one exchange batch");
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
  if (exchange != 0.0 && n_integrals_ != 0) {
    if (packed_)
      exchange_pair_gather<<<blas_int(n_pairs_),256,0,context_->stream()>>>(
          packed_->view(),pair_first_.data(),pair_second_.data(),density,output,exchange);
    else if (compact_)
      unique_exchange(density,output,exchange,1);
    else
      launch_exchange_rows(*context_,
          row_offsets_.data(),columns_.data(),values_.data(),pair_first_.data(),
          pair_second_.data(),density,output,exchange,1,exchange_shared_values_,n_pairs_);
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
  if (channels==1) {
    contract_jk(density,output,coulomb,exchange);
    return;
  }
  zero_matrix(*context_,output);
  if (exchange!=0.0 && n_integrals_!=0) {
    if (packed_)
      packed_exchange(density,output,exchange,channels);
    else if (compact_)
      unique_exchange(density,output,exchange,channels);
    else
      launch_exchange_rows(*context_,
          row_offsets_.data(),columns_.data(),values_.data(),pair_first_.data(),
          pair_second_.data(),density,output,exchange,channels,exchange_shared_values_,n_pairs_);
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

namespace {
// KB(C) stores unordered active pairs. Diagonal weights are A_aa;
// off-diagonal weights are A_ab+A_ba, without another AO-pair multiplicity.
__global__ void active_density_weights(const double* symmetric,int n,double* weights,
    std::int64_t count) {
  for (std::int64_t p=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       p<count;p+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    int a,b;decode_pair(p,a,b);
    weights[p]=symmetric[a+n*b]*(a==b ? 0.5:1.0);
  }
}
} // namespace

void ExactAoPairOperator::active_core_density_adjoint(
    linalg::ConstMatrixView c, linalg::ConstMatrixView a,
    linalg::MatrixView density_adjoint,ConstPairPanel products,
    ConstPairPanel exchange) const {
  validate_coefficients(c);
  if (!a.data || a.rows != c.cols || a.cols != c.cols || a.ld < a.rows ||
      !density_adjoint.data || density_adjoint.rows != n_bf_ ||
      density_adjoint.cols != n_bf_ || density_adjoint.ld < n_bf_)
    throw std::invalid_argument("invalid CUDA active-core density adjoint");
  const int n = blas_int(c.cols);
  const auto pairs=active_pairs(n);
  if (products.data && (products.rows!=n_pairs_ || products.cols!=pairs || products.ld<pairs))
    throw std::invalid_argument("invalid CUDA density adjoint accepted products");
  Buffer<double> symmetric(*context_, static_cast<std::size_t>(n) * n);
  symmetrize<<<blocks(static_cast<std::int64_t>(n) * n), 256, 0, context_->stream()>>>(
      a, {}, symmetric.data(), nullptr, n, static_cast<std::int64_t>(n) * n);
  check_launch("symmetrize CUDA active-core adjoint");
  if (!products.data || !exchange.data) {
    Matrix<double> contracted(*context_, n_bf_, n);
    Matrix<double> active_density(*context_, n_bf_, n_bf_);
    gemm(*context_, contracted.view(), c,{symmetric.data(), n, n, n});
    gemm(*context_, active_density.view(), contracted.view(), c,
         0.5, 0.0, Operation::none, Operation::transpose);
    contract_jk(active_density.view(),density_adjoint,
        products.data ? 0.0:2.0,exchange.data ? 0.0:-1.0);
  }
  if (exchange.data) {
    if (products.data) {
      exchange_density(exchange,{symmetric.data(),n,n,n},density_adjoint,-0.5);
    } else {
      Matrix<double> k(*context_,n_bf_,n_bf_);
      exchange_density(exchange,{symmetric.data(),n,n,n},k.view(),-0.5);
      const auto work=static_cast<std::int64_t>(n_bf_)*n_bf_;
      add_dense<<<blocks(work),256,0,context_->stream()>>>(density_adjoint,k.view(),work);
      check_launch("CUDA accepted density exchange addition");
    }
  }
  if (products.data) {
    const nvtx3::scoped_range range{"ao.accepted_density_coulomb"};
    Buffer<double> weights(*context_,pairs);
    Matrix<double> image(*context_,n_pairs_,1);
    active_density_weights<<<blocks(pairs),256,0,context_->stream()>>>(
        symmetric.data(),n,weights.data(),pairs);
    check_launch("CUDA accepted density pair weights");
    gemm(*context_,image.view(),{products.data,pairs,n_pairs_,products.ld},
        {weights.data(),pairs,1,pairs},1.0,0.0,Operation::transpose);
    const auto work=static_cast<std::int64_t>(n_bf_)*n_bf_;
    finish_jk<<<blocks(work),256,0,context_->stream()>>>(image.data(),density_adjoint,2.0,work);
    check_launch("CUDA accepted density Coulomb image");
  }
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
  Matrix<double> directional_contracted(*context_, n_bf_, n);
  const linalg::ConstMatrixView h{symmetric.data(), n, n, n};
  const linalg::ConstMatrixView dh{directional_symmetric.data(), n, n, n};
  Matrix<double> contracted(*context_, n_bf_, n);
  gemm(*context_, contracted.view(), c, h);
  gemm(*context_, directional_contracted.view(), d, h);
  Matrix<double> active_density_direction(*context_, n_bf_, n_bf_);
  gemm(*context_, directional_contracted.view(), c, dh, 1.0, 1.0);
  gemm(*context_, active_density_direction.view(),
       directional_contracted.view(), c, 0.5, 0.0,
       Operation::none, Operation::transpose);
  gemm(*context_, active_density_direction.view(), contracted.view(), d,
       0.5, 1.0, Operation::none, Operation::transpose);
  contract_jk(active_density_direction.view(), density_adjoint_direction);
}

}  // namespace xmvb::cuda
