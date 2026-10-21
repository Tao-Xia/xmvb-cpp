#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

// Each channel is a rank-one core density. The second output panel will
// subsequently hold exchange images, so no extra feature allocation exists.
__global__ void core_features(linalg::ConstMatrixView core,const int* first,
    const int* second,PairPanel features) {
  const auto work=features.rows*features.cols;
  for (std::int64_t k=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       k<work;k+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto p=k/features.cols,c=k%features.cols;
    const int i=first[p],j=second[p];
    features.data[p*features.ld+c]=(i==j ? 1.0:2.0)*
        core.data[i+core.ld*c]*core.data[j+core.ld*c];
  }
}

__global__ void project_core(linalg::ConstMatrixView mo,const int* first,
    const int* second,ConstPairPanel j,ConstPairPanel k,
    linalg::MatrixView coulomb,linalg::MatrixView response,int begin) {
  const auto work=mo.cols*j.cols;
  for (std::int64_t cell=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       cell<work;cell+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const auto p=cell%mo.cols,c=cell/mo.cols;
    double jp=0.0,kp=0.0;
    for (std::int64_t pair=0;pair<j.rows;++pair) {
      const int i=first[pair],l=second[pair];
      const double coefficient=(i==l ? 1.0:2.0)*mo.data[i+mo.ld*p]*mo.data[l+mo.ld*p];
      jp+=coefficient*j.data[pair*j.ld+c];
      kp+=coefficient*k.data[pair*k.ld+c];
    }
    coulomb.data[p+coulomb.ld*(begin+c)]=jp;
    response.data[p+response.ld*(begin+c)]=2.0*jp-kp;
  }
}

unsigned int grid(std::int64_t count) {
  return static_cast<unsigned int>(std::min<std::int64_t>(65535,(count+255)/256));
}

} // namespace

bool ExactAoPairOperator::core_diagonal(linalg::ConstMatrixView mo,int ncore,
    linalg::MatrixView coulomb,linalg::MatrixView response) const {
  const nvtx3::scoped_range range{"ao.core_diagonal"};
  validate_coefficients(mo);
  if (mo.cols!=n_bf_ || ncore<0 || ncore>n_bf_ ||
      coulomb.rows!=n_bf_ || coulomb.cols!=ncore || coulomb.ld<n_bf_ ||
      response.rows!=n_bf_ || response.cols!=ncore || response.ld<n_bf_ ||
      (ncore && (!coulomb.data || !response.data || coulomb.data==response.data)))
    throw std::invalid_argument("invalid CUDA projected core diagonal dimensions");
  if (ncore==0) return true;
  if (compact_) return false;
  const auto free=response_capacity()/sizeof(double);
  int width=ncore;
  // Both pair panels and one shared half-transform scratch are bounded. A
  // second working set remains free for SpMM and subsequent response work.
  for (;width>0;width/=2) {
    const auto half=static_cast<std::int64_t>(n_bf_)*n_bf_*width;
    const auto transformed=static_cast<std::int64_t>(n_bf_)*width*width;
    const auto images=2*n_pairs_*width;
    if (static_cast<std::size_t>(images+half+transformed)<=free/2) break;
  }
  if (width==0) return false;
  Buffer<double> images(*context_,2*n_pairs_*width);
  for (int begin=0;begin<ncore;begin+=width) {
    const int count=std::min(width,ncore-begin);
    const auto core=mo.columns(begin,count);
    const PairPanel j{images.data(),n_pairs_,count,width};
    const PairPanel k{images.data()+n_pairs_*width,n_pairs_,count,width};
    core_features<<<grid(n_pairs_*count),256,0,context_->stream()>>>(
        core,pair_first_.data(),pair_second_.data(),k);
    check_cuda(cudaGetLastError(),"CUDA rank-one core pair features");
    apply({k.data,k.rows,k.cols,k.ld},j);
    build_exchange_image(core,k,true);
    project_core<<<grid(static_cast<std::int64_t>(n_bf_)*count),256,0,context_->stream()>>>(
        mo,pair_first_.data(),pair_second_.data(),{j.data,j.rows,j.cols,j.ld},
        {k.data,k.rows,k.cols,k.ld},coulomb,response,begin);
    check_cuda(cudaGetLastError(),"CUDA direct core J/K diagonal projection");
  }
  return true;
}

} // namespace xmvb::cuda
