#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {
using View=linalg::ConstMatrixView;
using Out=linalg::MatrixView;

unsigned int grid(std::int64_t count) {
  return static_cast<unsigned int>(std::min<std::int64_t>(65535,(count+255)/256));
}

__device__ int pair(int i,int j) {
  const int high=max(i,j),low=min(i,j);
  return high*(high+1)/2+low;
}

// For one fixed mu, H(nu;k,b)=sum_l (mu nu|k l) C(l,b).
// Only n active channels enter the integral sweep, not n*n. The small shared
// half image is released after preparation; no AO four-index tensor exists.
__global__ void exchange_half(View c,const int* offsets,const int* columns,
    const double* values,const int* first,const int* second,int mu,double* half) {
  const int nu=blockIdx.y,n=static_cast<int>(c.cols),nbf=static_cast<int>(c.rows);
  const int p=pair(mu,nu),begin=offsets[p],end=offsets[p+1];
  const auto work=static_cast<std::int64_t>(end-begin)*n;
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int edge=begin+static_cast<int>(k/n),b=static_cast<int>(k%n);
    const int q=columns[edge],i=first[q],j=second[q];
    const double value=values[edge];
    atomicAdd(half+(static_cast<std::int64_t>(nu)*nbf+i)*n+b,
        value*c.data[j+c.ld*b]);
    if (i!=j) atomicAdd(half+(static_cast<std::int64_t>(nu)*nbf+j)*n+b,
        value*c.data[i+c.ld*b]);
  }
}

__global__ void store_exchange(View transformed,int mu,int n,PairPanel exchange) {
  const auto work=static_cast<std::int64_t>(mu+1)*n*n;
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int nu=static_cast<int>(k/(n*n)),a=static_cast<int>(k/n%n),b=static_cast<int>(k%n);
    exchange.data[static_cast<std::int64_t>(pair(mu,nu))*exchange.ld+a*n+b]=
        transformed.data[nu*n+b+transformed.ld*a];
  }
}

__global__ void density_image(ConstPairPanel exchange,View weights,Out out,double scale) {
  const int n=static_cast<int>(weights.rows),nbf=static_cast<int>(out.rows);
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<static_cast<std::int64_t>(nbf)*nbf;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int mu=static_cast<int>(k%nbf),nu=static_cast<int>(k/nbf);
    const auto* e=exchange.data+static_cast<std::int64_t>(pair(mu,nu))*exchange.ld;
    double sum=0.0;
    for (int a=0;a<n;++a) for (int b=0;b<n;++b)
      sum+=e[mu>=nu ? a*n+b:b*n+a]*weights.data[a+weights.ld*b];
    out.data[mu+out.ld*nu]=scale*sum;
  }
}

__global__ void pair_weights(View g,double* weights,int n,int pairs) {
  for (int p=blockIdx.x*blockDim.x+threadIdx.x;p<pairs;p+=gridDim.x*blockDim.x) {
    int a=0;while ((a+1)*(a+2)/2<=p) ++a;
    const int b=p-a*(a+1)/2;
    for (int q=0;q<pairs;++q) {
      int c=0;while ((c+1)*(c+2)/2<=q) ++c;
      const int d=q-c*(c+1)/2;
      const double value=g.data[q+g.ld*p]+g.data[p+g.ld*q];
      const auto width=static_cast<std::int64_t>(n)*n*n;
      atomicAdd(weights+c*width+(b*n+d)*n+a,value);
      atomicAdd(weights+d*width+(b*n+c)*n+a,value);
      atomicAdd(weights+c*width+(a*n+d)*n+b,value);
      atomicAdd(weights+d*width+(a*n+c)*n+b,value);
    }
  }
}

__global__ void exchange_rows(ConstPairPanel exchange,int begin,int rows,
    int nbf,int n,double* panel) {
  const auto width=static_cast<std::int64_t>(nbf)*n*n;
  const auto work=width*rows;
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int mu=begin+static_cast<int>(k/width),nu=static_cast<int>(k/ (n*n)%nbf);
    const int a=static_cast<int>(k/n%n),b=static_cast<int>(k%n);
    const auto p=static_cast<std::int64_t>(pair(mu,nu));
    panel[k]=exchange.data[p*exchange.ld+(mu>=nu ? a*n+b:b*n+a)];
  }
}

std::size_t available(Context& context) {
  std::size_t free=0,total=0;
  check_cuda(cudaMemGetInfo(&free,&total),"query CUDA exchange workspace memory");
  const auto budget=context.device_byte_budget(),used=context.counters().current_device_bytes;
  if (used>budget) throw std::runtime_error("CUDA exchange accounting exceeds budget");
  return std::min(free,budget-used);
}

void validate_image(ConstPairPanel exchange,int nbf,int n) {
  if (!exchange.data || exchange.rows!=static_cast<std::int64_t>(nbf)*(nbf+1)/2 ||
      exchange.cols!=static_cast<std::int64_t>(n)*n || exchange.ld<exchange.cols)
    throw std::invalid_argument("invalid CUDA accepted exchange image");
}
} // namespace

Buffer<double> ExactAoPairOperator::prepare_exchange(View c) const {
  const nvtx3::scoped_range range{"ao.exchange_prepare"};
  validate_coefficients(c);
  // Compact ERIs retain their existing exact device contractions. Preparation
  // needs the resident row graph to visit each row without rescanning the stream.
  if (compact_) return {};
  const int n=blas_int(c.cols);
  const auto n2=static_cast<std::int64_t>(n)*n,n3=n2*n;
  blas_int(n3);
  const auto free=available(*context_);
  const auto half_count=static_cast<std::int64_t>(n_bf_)*n_bf_*n;
  const auto action_count=static_cast<std::int64_t>(n_bf_)*n3+n*n3+2*n_bf_*n2;
  // Reserve the second point's image and the larger of preparation/action
  // scratch. This is admission of an exact representation, never truncation.
  if (static_cast<std::size_t>(n2)>free/(2*sizeof(double))/n_pairs_) return {};
  const auto count=n_pairs_*n2;
  const auto workspace=std::max(half_count+static_cast<std::int64_t>(n_bf_)*n2,action_count);
  if (static_cast<std::size_t>(workspace)>free/sizeof(double)-2*count) return {};
  Buffer<double> exchange(*context_,count),half(*context_,half_count);
  Matrix<double> transformed(*context_,static_cast<std::int64_t>(n_bf_)*n,n);
  const PairPanel image{exchange.data(),n_pairs_,n2,n2};
  for (int mu=0;mu<n_bf_;++mu) {
    check_cuda(cudaMemsetAsync(half.data(),0,half_count*sizeof(double),context_->stream()),
        "clear CUDA exchange half image");
    exchange_half<<<dim3(16,n_bf_),256,0,context_->stream()>>>(
        c,row_offsets_.data(),columns_.data(),values_.data(),pair_first_.data(),
        pair_second_.data(),mu,half.data());
    check_cuda(cudaGetLastError(),"CUDA one-active-index exchange transform");
    gemm(*context_,transformed.view(),{half.data(),static_cast<std::int64_t>(n_bf_)*n,
        n_bf_,static_cast<std::int64_t>(n_bf_)*n},c);
    store_exchange<<<grid(static_cast<std::int64_t>(mu+1)*n2),256,0,context_->stream()>>>(
        transformed.view(),mu,n,image);
    check_cuda(cudaGetLastError(),"CUDA triangular exchange image");
  }
  return exchange;
}

void ExactAoPairOperator::exchange_density(ConstPairPanel exchange,View weights,
    Out output,double scale) const {
  const nvtx3::scoped_range range{"ao.exchange_density"};
  const int n=blas_int(weights.rows);
  validate_image(exchange,n_bf_,n);
  if (!weights.data || weights.cols!=n || weights.ld<n || !output.data ||
      output.rows!=n_bf_ || output.cols!=n_bf_ || output.ld<n_bf_ || !std::isfinite(scale))
    throw std::invalid_argument("invalid CUDA exchange density projection");
  density_image<<<grid(static_cast<std::int64_t>(n_bf_)*n_bf_),256,0,context_->stream()>>>(
      exchange,weights,output,scale);
  check_cuda(cudaGetLastError(),"CUDA accepted exchange density projection");
}

void ExactAoPairOperator::exchange_pair_direction(ConstPairPanel exchange,View d,
    View g,Out gradient) const {
  const nvtx3::scoped_range range{"ao.exchange_pair_direction"};
  validate_coefficients(d);
  const int n=blas_int(d.cols),pairs=n*(n+1)/2;
  validate_image(exchange,n_bf_,n);
  if (!g.data || g.rows!=pairs || g.cols!=pairs || g.ld<pairs || !gradient.data ||
      gradient.rows!=n_bf_ || gradient.cols!=n || gradient.ld<n_bf_)
    throw std::invalid_argument("invalid CUDA exchange pair direction");
  const auto n2=static_cast<std::int64_t>(n)*n,n3=n2*n;
  const int channels=blas_int(n3);
  Buffer<double> weights(*context_,n*n3),mixed(*context_,static_cast<std::int64_t>(n_bf_)*n3);
  check_cuda(cudaMemsetAsync(weights.data(),0,n*n3*sizeof(double),context_->stream()),
      "clear CUDA exchange pair weights");
  pair_weights<<<grid(pairs),256,0,context_->stream()>>>(g,weights.data(),n,pairs);
  check_cuda(cudaGetLastError(),"CUDA exact pair exchange weights");
  // Row-major D*weights, stored as n^3-by-nbf column-major.
  gemm(*context_,{mixed.data(),channels,n_bf_,channels},
      {weights.data(),channels,n,channels},d,1.0,0.0,Operation::none,Operation::transpose);
  const auto row_width=static_cast<std::int64_t>(n_bf_)*n2;
  const int rows=static_cast<int>(std::min<std::size_t>(n_bf_,
      available(*context_)/(row_width*sizeof(double))));
  if (rows<1) throw std::runtime_error("CUDA exchange budget cannot hold one output row");
  Buffer<double> panel(*context_,row_width*rows);
  for (int begin=0;begin<n_bf_;begin+=rows) {
    const int count=std::min(rows,n_bf_-begin);
    exchange_rows<<<grid(row_width*count),256,0,context_->stream()>>>(
        exchange,begin,count,n_bf_,n,panel.data());
    check_cuda(cudaGetLastError(),"CUDA exchange output-row panel");
    gemm(*context_,{gradient.data+begin,count,n,gradient.ld},
        {panel.data(),row_width,count,row_width},
        {mixed.data(),n,row_width,n},1.0,1.0,Operation::transpose,Operation::transpose);
  }
}
} // namespace xmvb::cuda
