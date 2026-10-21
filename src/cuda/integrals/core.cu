#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/integrals/packed.hpp"

namespace xmvb::cuda {
namespace {

struct AddProjection {
  __device__ double2 operator()(double2 a,double2 b) const {
    return make_double2(a.x+b.x,a.y+b.y);
  }
};

struct PairExchange {
  ConstPairPanel image;
  __device__ double operator()(std::int64_t pair,int,int,int c) const {
    return image.data[pair*image.ld+c];
  }
};

struct DenseExchange {
  linalg::ConstMatrixView image;
  __device__ double operator()(std::int64_t,int i,int j,int c) const {
    const auto shift=static_cast<std::int64_t>(c)*image.rows;
    const double value=image.data[i+image.ld*(j+shift)];
    if (i==j) return value;
    return 0.5*(value+image.data[j+image.ld*(i+shift)]);
  }
};

// Share each AO-pair coefficient between J and K. CUB reduces the pair
// dimension without storing a MO-by-AO-pair feature matrix or partial sums.
template<class Exchange>
__global__ void project_core(linalg::ConstMatrixView mo,const int* first,
    const int* second,ConstPairPanel j,Exchange k,
    linalg::MatrixView coulomb,linalg::MatrixView response,int begin) {
  using Reduce=cub::BlockReduce<double2,256>;
  __shared__ typename Reduce::TempStorage scratch;
  const auto work=mo.cols*j.cols;
  for (std::int64_t cell=blockIdx.x;cell<work;cell+=gridDim.x) {
    const auto p=cell%mo.cols,c=cell/mo.cols;
    double2 sum=make_double2(0.0,0.0);
    for (std::int64_t pair=threadIdx.x;pair<j.rows;pair+=blockDim.x) {
      const int i=first[pair],l=second[pair];
      const double coefficient=(i==l ? 1.0:2.0)*mo.data[i+mo.ld*p]*mo.data[l+mo.ld*p];
      sum.x+=coefficient*j.data[pair*j.ld+c];
      sum.y+=coefficient*k(pair,i,l,c);
    }
    sum=Reduce(scratch).Reduce(sum,AddProjection{});
    if (threadIdx.x==0) {
      coulomb.data[p+coulomb.ld*(begin+c)]=sum.x;
      response.data[p+response.ld*(begin+c)]=2.0*sum.x-sum.y;
    }
    __syncthreads();
  }
}

} // namespace

bool ExactAoPairOperator::core_diagonal(linalg::ConstMatrixView mo,int ncore,
    linalg::MatrixView coulomb,linalg::MatrixView response,
    Buffer<double>* projection) const {
  return prepare_core(mo,ncore,coulomb,response,projection);
}

bool ExactAoPairOperator::unique_core_diagonal(linalg::ConstMatrixView mo,int ncore,
    linalg::MatrixView coulomb,linalg::MatrixView response) const {
  const nvtx3::scoped_range range{"ao.rank_one_core_diagonal"};
  const auto square=static_cast<std::int64_t>(n_bf_)*n_bf_;
  const auto per_core=static_cast<std::size_t>(2*n_pairs_+square)*sizeof(double);
  int width=static_cast<int>(std::min<std::size_t>(ncore,response_capacity()/per_core));
  Buffer<double> features,products;
  Matrix<double> exchange;
  for (;width>0;width/=2) {
    features=Buffer<double>(*context_,n_pairs_*width);
    products=Buffer<double>(*context_,n_pairs_*width);
    exchange=Matrix<double>(*context_,n_bf_,static_cast<std::int64_t>(n_bf_)*width);
    const auto scratch=[&](int count) {
      return apply_workspace({features.data(),n_pairs_,count,width},
          {products.data(),n_pairs_,count,width});
    };
    const int remainder=ncore%width;
    if (std::max(scratch(width),remainder ? scratch(remainder):std::size_t{0})<=
        response_capacity()) break;
    features={};products={};exchange={};
  }
  if (!width) return false;
  for (int begin=0;begin<ncore;begin+=width) {
    const int count=std::min(width,ncore-begin);
    const auto factors=mo.columns(begin,count);
    const PairPanel source{features.data(),n_pairs_,count,width};
    const PairPanel image{products.data(),n_pairs_,count,width};
    density_features(factors,0,source);
    apply({source.data,source.rows,source.cols,source.ld},image);
    const auto k=exchange.view().columns(0,static_cast<std::int64_t>(n_bf_)*count);
    rank_one_exchange(factors,k);
    const auto cells=static_cast<std::int64_t>(n_bf_)*count;
    project_core<<<static_cast<unsigned int>(std::min<std::int64_t>(65535,cells)),
        256,0,context_->stream()>>>(mo,pair_first_.data(),pair_second_.data(),
            ConstPairPanel{image.data,image.rows,image.cols,image.ld},DenseExchange{k},
            coulomb,response,begin);
    check_cuda(cudaGetLastError(),"CUDA rank-one core diagonal projection");
  }
  return true;
}

ExactAoResponses ExactAoPairOperator::prepare_responses(linalg::ConstMatrixView active,
    linalg::ConstMatrixView mo,int ncore) const {
  validate_coefficients(mo);
  if (mo.cols!=n_bf_ || ncore<0 || ncore>n_bf_)
    throw std::invalid_argument("CUDA joint AO response orbital partition");
  ExactAoResponses result;
  result.exchange=reserve_exchange(active);
  if (result.exchange.empty()) return result;
  const auto core=mo.columns(0,ncore);
  if (ncore) result.active=reserve_density_projection(active,core);
  result.core_coulomb=Matrix<double>(*context_,n_bf_,ncore);
  result.core_response=Matrix<double>(*context_,n_bf_,ncore);
  const auto n2=active.cols*active.cols;
  const ExchangeRequest request{active,core,{result.exchange.data(),n_pairs_,n2,n2},{},
      result.active.view()};
  result.core_ready=prepare_core(mo,ncore,result.core_coulomb.view(),
      result.core_response.view(),result.active.data() ? &result.inactive:nullptr,&request);
  return result;
}

bool ExactAoPairOperator::prepare_core(linalg::ConstMatrixView mo,int ncore,
    linalg::MatrixView coulomb,linalg::MatrixView response,
    Buffer<double>* projection,const ExchangeRequest* joint) const {
  const nvtx3::scoped_range range{"ao.core_diagonal"};
  validate_coefficients(mo);
  if (mo.cols!=n_bf_ || ncore<0 || ncore>n_bf_ ||
      coulomb.rows!=n_bf_ || coulomb.cols!=ncore || coulomb.ld<n_bf_ ||
      response.rows!=n_bf_ || response.cols!=ncore || response.ld<n_bf_ ||
      (ncore && (!coulomb.data || !response.data || coulomb.data==response.data)))
    throw std::invalid_argument("invalid CUDA projected core diagonal dimensions");
  if (projection) *projection={};
  if (ncore==0) {
    if (joint) build_exchange_images(std::array{*joint});
    return true;
  }
  if (compact_) return unique_core_diagonal(mo,ncore,coulomb,response);
  const auto targets=mo.columns(0,ncore);
  if (projection) {
    const auto dimension=static_cast<std::int64_t>(n_bf_)*ncore;
    const auto count=dimension*(dimension+1)/2;
    const auto scratch=std::max(static_cast<std::int64_t>(n_bf_)*n_bf_*ncore,
        joint ? static_cast<std::int64_t>(n_bf_)*n_bf_*joint->coefficients.cols:0)+
        std::max(2*static_cast<std::int64_t>(n_bf_)*ncore*ncore,
            joint ? exchange_workspace(*joint):0)+
        (packed_ ? static_cast<std::int64_t>(n_bf_)*n_bf_:0);
    if (static_cast<std::size_t>(2*count+scratch)<=response_capacity()/sizeof(double)) {
      *projection=Buffer<double>(*context_,count);
      check_cuda(cudaMemsetAsync(projection->data(),0,count*sizeof(double),context_->stream()),
          "zero CUDA symmetric core response");
    }
  }
  const auto free=response_capacity()/sizeof(double);
  int width=ncore;
  // Both J and K are extracted from the same half transform; J does not
  // require a separate pair-operator action or another integral sweep.
  for (;width>0;width/=2) {
    const bool projected=projection && projection->data();
    const auto half=std::max(static_cast<std::int64_t>(n_bf_)*n_bf_*width,
        joint ? static_cast<std::int64_t>(n_bf_)*n_bf_*joint->coefficients.cols:0);
    const auto transformed=(projected ? 0:static_cast<std::int64_t>(n_bf_)*width*width)+
        (projected ? 2:1)*static_cast<std::int64_t>(n_bf_)*width*targets.cols;
    const auto images=2*n_pairs_*width;
    const auto plane_min=packed_ ? static_cast<std::int64_t>(n_bf_)*n_bf_:0;
    if (static_cast<std::size_t>(images+half+
            std::max(transformed,joint ? exchange_workspace(*joint):0)+plane_min)<=
        (packed_ ? free:free/2)) break;
  }
  if (width==0) {
    if (projection) *projection={};
    if (joint) {
      // Sequential scheduling needs less live workspace. The same device
      // contractions and exact images are used, not a different physical model.
      build_exchange_images(std::array{*joint});
      return prepare_core(mo,ncore,coulomb,response,projection);
    }
    return false;
  }
  Buffer<double> images(*context_,2*n_pairs_*width);
  for (int begin=0;begin<ncore;begin+=width) {
    const int count=std::min(width,ncore-begin);
    const auto core=mo.columns(begin,count);
    const PairPanel j{images.data(),n_pairs_,count,width};
    const PairPanel k{images.data()+n_pairs_*width,n_pairs_,count,width};
    const auto projected=projection && projection->data()
        ? linalg::MatrixView{projection->data(),static_cast<std::int64_t>(projection->size()),1,
            static_cast<std::int64_t>(projection->size())}:linalg::MatrixView{};
    const ExchangeRequest request{core,targets,k,j,projected,true,true,begin};
    if (joint && begin==0) build_exchange_images(std::array{*joint,request});
    else build_exchange_images(std::array{request});
    {
      const nvtx3::scoped_range project_range{"ao.core_projection"};
      const auto cells=static_cast<std::int64_t>(n_bf_)*count;
      const auto blocks=static_cast<unsigned int>(std::min<std::int64_t>(65535,cells));
      project_core<<<blocks,256,0,context_->stream()>>>(mo,pair_first_.data(),pair_second_.data(),
          ConstPairPanel{j.data,j.rows,j.cols,j.ld},
          PairExchange{{k.data,k.rows,k.cols,k.ld}},coulomb,response,begin);
    }
    check_cuda(cudaGetLastError(),"CUDA direct core J/K diagonal projection");
  }
  return true;
}

} // namespace xmvb::cuda
