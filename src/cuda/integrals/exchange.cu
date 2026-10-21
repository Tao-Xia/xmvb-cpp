#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_reduce.cuh>

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

__global__ void store_exchange(View transformed,int mu,int n,PairPanel exchange,bool diagonal) {
  const auto width=diagonal ? n:static_cast<std::int64_t>(n)*n;
  const auto work=static_cast<std::int64_t>(mu+1)*width;
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int nu=static_cast<int>(k/width);
    const int a=diagonal ? static_cast<int>(k%n):static_cast<int>(k/n%n);
    const int b=diagonal ? a:static_cast<int>(k%n);
    const auto channel=diagonal ? a:a*n+b;
    exchange.data[static_cast<std::int64_t>(pair(mu,nu))*exchange.ld+channel]=
        transformed.data[nu*n+b+transformed.ld*a];
  }
}

// Keep the already computed one-index image, without another ERI traversal.
__global__ void store_half(const double* half,int mu,int nbf,int n,PairPanel image) {
  const auto width=static_cast<std::int64_t>(nbf)*n;
  const auto work=static_cast<std::int64_t>(mu+1)*width;
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x))
    image.data[static_cast<std::int64_t>(pair(mu,static_cast<int>(k/width)))*image.ld+
        k%width]=half[k];
}

__global__ void half_jk(ConstPairPanel half,View v,const double* j,Out out,int nbf,int n) {
  using Reduce=cub::BlockReduce<double,256>;
  __shared__ typename Reduce::TempStorage storage;
  const int mu=blockIdx.x%nbf,nu=blockIdx.x/nbf;
  double sum=0.0;
  for (int ka=threadIdx.x;ka<nbf*n;ka+=blockDim.x) {
    const int k=ka/n,a=ka%n;
    sum+=(half.data[static_cast<std::int64_t>(pair(mu,k))*half.ld+nu*n+a]+
          half.data[static_cast<std::int64_t>(pair(nu,k))*half.ld+mu*n+a])*
        v.data[k+v.ld*a];
  }
  sum=Reduce(storage).Sum(sum);
  if (threadIdx.x==0) out.data[mu+out.ld*nu]=4.0*j[pair(mu,nu)]-sum;
}

__global__ void pack_factor(View v,double* packed,int nbf,int n) {
  for (int k=blockIdx.x*blockDim.x+threadIdx.x;k<nbf*n;k+=gridDim.x*blockDim.x)
    packed[k]=v.data[k/n+v.ld*(k%n)];
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

void validate_image(ConstPairPanel exchange,int nbf,int n) {
  if (!exchange.data || exchange.rows!=static_cast<std::int64_t>(nbf)*(nbf+1)/2 ||
      exchange.cols!=static_cast<std::int64_t>(n)*n || exchange.ld<exchange.cols)
    throw std::invalid_argument("invalid CUDA accepted exchange image");
}
} // namespace

std::size_t ExactAoPairOperator::response_capacity() const {
  std::size_t free=0,total=0;
  check_cuda(cudaMemGetInfo(&free,&total),"query CUDA exact response workspace memory");
  const auto budget=context_->device_byte_budget(),used=context_->counters().current_device_bytes;
  if (used>budget) throw std::runtime_error("CUDA exact response accounting exceeds budget");
  return std::min(free,budget-used);
}

Buffer<double> ExactAoPairOperator::prepare_exchange(View c,Buffer<double>* half_products) const {
  const nvtx3::scoped_range range{"ao.exchange_prepare"};
  validate_coefficients(c);
  if (half_products) *half_products={};
  // Compact ERIs retain their existing exact device contractions. Preparation
  // needs the resident row graph to visit each row without rescanning the stream.
  if (compact_) return {};
  const int n=blas_int(c.cols);
  const auto n2=static_cast<std::int64_t>(n)*n,n3=n2*n;
  blas_int(n3);
  const auto free=response_capacity();
  const auto half_count=static_cast<std::int64_t>(n_bf_)*n_bf_*n;
  const auto action_count=static_cast<std::int64_t>(n_bf_)*n3+n*n3+2*n_bf_*n2;
  // Reserve the second point's image and the larger of preparation/action
  // scratch. This is admission of an exact representation, never truncation.
  if (static_cast<std::size_t>(n2)>free/(2*sizeof(double))/n_pairs_) return {};
  const auto count=n_pairs_*n2;
  const auto workspace=std::max(half_count+static_cast<std::int64_t>(n_bf_)*n2,action_count);
  if (static_cast<std::size_t>(workspace)>free/sizeof(double)-2*count) return {};
  Buffer<double> exchange(*context_,count);
  const PairPanel image{exchange.data(),n_pairs_,n2,n2};
  PairPanel retained;
  if (half_products) {
    const auto half_size=n_pairs_*n_bf_*n;
    // Account for both accepted/trial points and existing exchange scratch.
    // If unavailable, the exact device-streamed JK representation remains.
    if (static_cast<std::size_t>(2*half_size+workspace+count)<=response_capacity()/sizeof(double)) {
      *half_products=Buffer<double>(*context_,half_size);
      retained={half_products->data(),n_pairs_,static_cast<std::int64_t>(n_bf_)*n,
          static_cast<std::int64_t>(n_bf_)*n};
    }
  }
  build_exchange_image(c,image,false,retained);
  return exchange;
}

void ExactAoPairOperator::build_exchange_image(View c,PairPanel image,bool diagonal,
    PairPanel retained) const {
  const int n=blas_int(c.cols);
  const auto n2=static_cast<std::int64_t>(n)*n;
  const auto half_count=static_cast<std::int64_t>(n_bf_)*n_bf_*n;
  Buffer<double> half(*context_,half_count);
  Matrix<double> transformed(*context_,static_cast<std::int64_t>(n_bf_)*n,n);
  for (int mu=0;mu<n_bf_;++mu) {
    check_cuda(cudaMemsetAsync(half.data(),0,half_count*sizeof(double),context_->stream()),
        "clear CUDA exchange half image");
    exchange_half<<<dim3(16,n_bf_),256,0,context_->stream()>>>(
        c,row_offsets_.data(),columns_.data(),values_.data(),pair_first_.data(),
        pair_second_.data(),mu,half.data());
    check_cuda(cudaGetLastError(),"CUDA one-active-index exchange transform");
    if (retained.data) {
      store_half<<<grid(static_cast<std::int64_t>(mu+1)*n_bf_*n),256,0,
          context_->stream()>>>(half.data(),mu,n_bf_,n,retained);
      check_cuda(cudaGetLastError(),"retain CUDA one-index half transform");
    }
    gemm(*context_,transformed.view(),{half.data(),static_cast<std::int64_t>(n_bf_)*n,
        n_bf_,static_cast<std::int64_t>(n_bf_)*n},c);
    store_exchange<<<grid(static_cast<std::int64_t>(mu+1)*(diagonal ? n:n2)),
        256,0,context_->stream()>>>(transformed.view(),mu,n,image,diagonal);
    check_cuda(cudaGetLastError(),"CUDA triangular exchange image");
  }
}

void ExactAoPairOperator::half_density_response(ConstPairPanel half,View v,Out out) const {
  const nvtx3::scoped_range range{"ao.half_density_response"};
  validate_coefficients(v);
  const int n=blas_int(v.cols);
  const auto width=static_cast<std::int64_t>(n_bf_)*n;
  if (!half.data || half.rows!=n_pairs_ || half.cols!=width || half.ld<width ||
      !out.data || out.rows!=n_bf_ || out.cols!=n_bf_ || out.ld<n_bf_)
    throw std::invalid_argument("invalid CUDA half-density response");
  Matrix<double> packed(*context_,width,1),j(*context_,n_pairs_,1);
  pack_factor<<<grid(width),256,0,context_->stream()>>>(v,packed.data(),n_bf_,n);
  check_cuda(cudaGetLastError(),"pack CUDA half-density factor");
  gemm(*context_,j.view(),{half.data,width,n_pairs_,half.ld},packed.view(),
      1.0,0.0,Operation::transpose);
  half_jk<<<n_bf_*n_bf_,256,0,context_->stream()>>>(half,v,j.data(),out,n_bf_,n);
  check_cuda(cudaGetLastError(),"CUDA projected half-density JK");
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
      response_capacity()/(row_width*sizeof(double))));
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
