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

// R(mu,t;k,a)=4(mu T_t|k C_a)-(mu C_a|k T_t)-(mu k|T_t C_a).
// Each fixed-mu launch updates a distinct row and column. Off-diagonal
// destinations are disjoint within the launch; subsequent mu uses the stream.
__device__ void add_projection(Out response,std::int64_t row,std::int64_t col,
    double value,bool packed) {
  if (!packed) response.data[row+response.ld*col]+=value;
  else if (row>=col) response.data[row*(row+1)/2+col]+=value;
}

__global__ void add_density_projection(View u,View p,Out response,int mu,int nbf,int n,
    bool packed,int first_column) {
  const auto work=static_cast<std::int64_t>(nbf)*n*p.rows;
  for (auto index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       index<work;index+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int k=index%nbf,a=index/nbf%n,t=index/(nbf*n);
    const double value=u.data[k*n+a+u.ld*t];
    add_projection(response,mu+nbf*t,k+nbf*(a+first_column),
        4.0*value-p.data[t+p.ld*(k*n+a)],packed);
    add_projection(response,k+nbf*t,mu+nbf*(a+first_column),-value,packed);
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

Buffer<double> ExactAoPairOperator::prepare_exchange(View c,Matrix<double>* projection,
    View targets) const {
  const nvtx3::scoped_range range{"ao.exchange_prepare"};
  validate_coefficients(c);
  if (projection) *projection={};
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
  if (projection && targets.cols) *projection=reserve_density_projection(c,targets);
  build_exchange_image(c,image,false,targets,projection ? projection->view():Out{});
  return exchange;
}

void ExactAoPairOperator::build_exchange_image(View c,PairPanel image,bool diagonal,
    View targets,Out response,bool packed,int first_column) const {
  const int n=blas_int(c.cols);
  const auto n2=static_cast<std::int64_t>(n)*n;
  const auto half_count=static_cast<std::int64_t>(n_bf_)*n_bf_*n;
  Buffer<double> half(*context_,half_count);
  Matrix<double> transformed,u,p;
  // A core block is a column slice of targets. H_mu C is then already a
  // slice of U_mu=H_mu targets, needed for the density-response projection.
  // Reuse that exact slice rather than multiply the same half image twice.
  const bool shared_projection=response.data && first_column>=0 &&
      first_column+n<=targets.cols && c.ld==targets.ld &&
      c.data==targets.data+targets.ld*first_column;
  if (image.data && !shared_projection)
    transformed=Matrix<double>(*context_,static_cast<std::int64_t>(n_bf_)*n,n);
  if (response.data) {
    u=Matrix<double>(*context_,static_cast<std::int64_t>(n_bf_)*n,targets.cols);
    p=Matrix<double>(*context_,targets.cols,static_cast<std::int64_t>(n_bf_)*n);
    if (!packed)
      check_cuda(cudaMemsetAsync(response.data,0,response.rows*response.cols*sizeof(double),
          context_->stream()),"zero CUDA projected density response");
  }
  for (int mu=0;mu<n_bf_;++mu) {
    check_cuda(cudaMemsetAsync(half.data(),0,half_count*sizeof(double),context_->stream()),
        "clear CUDA exchange half image");
    exchange_half<<<dim3(16,n_bf_),256,0,context_->stream()>>>(
        c,row_offsets_.data(),columns_.data(),values_.data(),pair_first_.data(),
        pair_second_.data(),mu,half.data());
    check_cuda(cudaGetLastError(),"CUDA one-active-index exchange transform");
    if (response.data) {
      gemm(*context_,u.view(),{half.data(),static_cast<std::int64_t>(n_bf_)*n,
          n_bf_,static_cast<std::int64_t>(n_bf_)*n},targets);
      // H_k[a,nu] already occupies one contiguous n-by-nbf slice. Batched
      // targets^T H_k^T produces P[t,k,a] directly from that storage, without
      // permuting/copying the whole AO-by-AO-by-orbital half image.
      const double one=1.0,zero=0.0;
      const int nt=blas_int(targets.cols);
      check_cublas(cublasDgemmStridedBatched(context_->blas(),CUBLAS_OP_T,CUBLAS_OP_T,
          nt,n,n_bf_,&one,targets.data,blas_int(targets.ld),0,
          half.data(),n,static_cast<long long>(n_bf_)*n,&zero,
          p.data(),nt,static_cast<long long>(nt)*n,n_bf_),
          "CUDA batched density response half projection");
      add_density_projection<<<grid(static_cast<std::int64_t>(n_bf_)*n*targets.cols),
          256,0,context_->stream()>>>(u.view(),p.view(),response,mu,n_bf_,n,packed,first_column);
      check_cuda(cudaGetLastError(),"assemble CUDA projected density response");
    }
    if (!image.data) continue;
    if (!shared_projection)
      gemm(*context_,transformed.view(),{half.data(),static_cast<std::int64_t>(n_bf_)*n,
          n_bf_,static_cast<std::int64_t>(n_bf_)*n},c);
    const auto image_projection=shared_projection ? u.view().columns(first_column,n):
        transformed.view();
    store_exchange<<<grid(static_cast<std::int64_t>(mu+1)*(diagonal ? n:n2)),
        256,0,context_->stream()>>>(image_projection,mu,n,image,diagonal);
    check_cuda(cudaGetLastError(),"CUDA triangular exchange image");
  }
}

Matrix<double> ExactAoPairOperator::reserve_density_projection(View c,View targets) const {
  validate_coefficients(c);
  validate_coefficients(targets);
  if (compact_) return {};
  const auto count=static_cast<std::int64_t>(n_bf_)*n_bf_*c.cols*targets.cols;
  const auto scratch=2*static_cast<std::int64_t>(n_bf_)*n_bf_*c.cols+
      2*static_cast<std::int64_t>(n_bf_)*c.cols*targets.cols+
      static_cast<std::int64_t>(n_bf_)*c.cols*c.cols;
  if (static_cast<std::size_t>(2*count+scratch)>response_capacity()/sizeof(double)) return {};
  return Matrix<double>(*context_,static_cast<std::int64_t>(n_bf_)*targets.cols,
      static_cast<std::int64_t>(n_bf_)*c.cols);
}

Matrix<double> ExactAoPairOperator::prepare_density_projection(View c,View targets) const {
  const nvtx3::scoped_range range{"ao.density_projection_prepare"};
  auto response=reserve_density_projection(c,targets);
  if (response.data()) build_exchange_image(c,{},false,targets,response.view());
  return response;
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
