#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/integrals/packed.hpp"
#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

unsigned int grid(std::int64_t work) {
  return static_cast<unsigned int>(std::min<std::int64_t>(65535,(work+255)/256));
}

// D(l,c,nu)=P(nu,l;c), not P(l,nu;c). Retain the existing K convention
// even for numerically nonsymmetric input; no symmetrization is introduced.
__global__ void pack_density(linalg::ConstMatrixView input,linalg::MatrixView d,
    int first_nu,int first_channel,int channels) {
  const auto n=input.rows,work=d.rows*d.cols;
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto l=k%n,c=k/n%channels,nu=first_nu+k/(n*channels);
    d.data[k]=input.data[nu+input.ld*(l+n*(first_channel+c))];
  }
}

__global__ void reduce_exchange(linalg::ConstMatrixView partial,linalg::MatrixView output,
    int mu,int first_channel,int channels,int nu_count,double weight) {
  const auto n=output.rows,work=n*channels;
  for (auto k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    double value=0.0;
    for (int nu=0;nu<nu_count;++nu)
      value+=partial.data[k+n*channels*nu];
    output.data[mu+output.ld*(k%n+n*(first_channel+k/n))]+=weight*value;
  }
}

} // namespace

void ExactAoPairOperator::packed_exchange(linalg::ConstMatrixView density,
    linalg::MatrixView output,double weight,int channels) const {
  const nvtx3::scoped_range range{"ao.packed_exchange"};
  const auto available=response_capacity()/sizeof(double);
  const auto n=static_cast<std::int64_t>(n_bf_),square=n*n;
  if (available<square+n)
    throw std::runtime_error("CUDA packed exchange budget cannot hold one AO plane slice");
  // One AO slice plus a partial sum is the minimum, also the storage needed
  // by the ordinary packed J contraction. Larger batches share the same plane.
  int batch=static_cast<int>(std::min<std::size_t>(channels,
      (available-square)/(2*n)));
  batch=std::max(1,batch);
  const auto per_nu=square+(batch==1 ? n:2*n*batch);
  const int width=static_cast<int>(std::min<std::size_t>(n,available/per_nu));
  Matrix<double> plane(*context_,n,n*width);
  Matrix<double> partial(*context_,n,static_cast<std::int64_t>(batch)*width);
  Matrix<double> d;
  if (batch>1) d=Matrix<double>(*context_,n,static_cast<std::int64_t>(batch)*width);
  const double one=1.0,zero=0.0;
  for (int first_nu=0;first_nu<n_bf_;first_nu+=width) {
    const int count=std::min(width,n_bf_-first_nu);
    if (batch==channels && batch>1) {
      const auto tile=d.view().columns(0,static_cast<std::int64_t>(channels)*count);
      pack_density<<<grid(tile.rows*tile.cols),256,0,context_->stream()>>>(
          density,tile,first_nu,0,channels);
      check_cuda(cudaGetLastError(),"CUDA packed exchange density channels");
    }
    for (int mu=0;mu<n_bf_;++mu) {
      packed_->ao_plane(mu,plane.view().columns(0,n*count),first_nu);
      for (int first_c=0;first_c<channels;first_c+=batch) {
        const int nc=std::min(batch,channels-first_c);
        if (batch>1) {
          const auto tile=d.view().columns(0,static_cast<std::int64_t>(nc)*count);
          if (batch<channels) {
            pack_density<<<grid(tile.rows*tile.cols),256,0,context_->stream()>>>(
                density,tile,first_nu,first_c,nc);
            check_cuda(cudaGetLastError(),"CUDA tiled exchange density channels");
          }
          check_cublas(cublasDgemmStridedBatched(context_->blas(),CUBLAS_OP_T,CUBLAS_OP_N,
              n_bf_,nc,n_bf_,&one,plane.data(),n_bf_,square,
              d.data(),n_bf_,n*nc,&zero,partial.data(),n_bf_,n*nc,count),
              "CUDA packed batched exchange GEMM");
        } else {
          // B is a 1-by-nbf row of the supplied density, hence OP_T and
          // stride 1 between different nu. OP_N would silently read columns.
          check_cublas(cublasDgemmStridedBatched(context_->blas(),CUBLAS_OP_T,CUBLAS_OP_T,
              n_bf_,1,n_bf_,&one,plane.data(),n_bf_,square,
              density.data+first_nu+density.ld*n*first_c,blas_int(density.ld),1,
              &zero,partial.data(),n_bf_,n,count),
              "CUDA packed scalar exchange GEMM");
        }
        reduce_exchange<<<grid(n*nc),256,0,context_->stream()>>>(
            partial.view(),output,mu,first_c,nc,count,weight);
        check_cuda(cudaGetLastError(),"CUDA packed exchange row reduction");
      }
    }
  }
}

} // namespace xmvb::cuda
