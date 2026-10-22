#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

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

// Each unique pair row u supplies a partial symmetric AO matrix M_u.
// Stack a bounded batch vertically: T = [M_0; ...; M_b] C is one GEMM,
// sharing C without replicated factors or an expanded integral graph.
template<class Index>
__global__ void core_exchange_rows(const Index* offsets,const Index* columns,
    const double* values,const int* first,const int* second,int begin,int count,
    int n_bf,double* rows) {
  const auto ld=static_cast<std::int64_t>(n_bf)*count;
  for (int row=blockIdx.x;row<count;row+=gridDim.x) {
    const int u=begin+row;
    for (auto edge=offsets[u]+threadIdx.x;edge<offsets[u+1];edge+=blockDim.x) {
      const int v=static_cast<int>(columns[edge]),k=first[v],l=second[v];
      atomicAdd(rows+static_cast<std::int64_t>(row)*n_bf+k+ld*l,values[edge]);
      if (k!=l)
        atomicAdd(rows+static_cast<std::int64_t>(row)*n_bf+l+ld*k,values[edge]);
    }
  }
}

__device__ void add_core_exchange(linalg::MatrixView output,int i,int j,int channel,
    double value) {
  if (value==0.0) return;
  const int high=max(i,j),low=min(i,j);
  atomicAdd(output.data+channel+
      output.ld*(static_cast<std::int64_t>(high)*(high+1)/2+low),value);
}

// K = F + F^T - D. D removes the repeated u=v quartet. Emit only the
// independent triangle; diagonal destinations receive both F orientations.
__global__ void core_exchange_scatter(linalg::ConstMatrixView factors,
    linalg::ConstMatrixView images,const int* first,const int* second,
    const double* diagonal,int begin,int count,linalg::MatrixView output) {
  const int n_bf=static_cast<int>(factors.rows);
  const auto cells=images.rows*factors.cols;
  for (auto cell=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       cell<cells;cell+=gridDim.x*static_cast<std::int64_t>(blockDim.x)) {
    const int k=cell%n_bf,row=cell/n_bf%count,channel=cell/images.rows;
    const int u=begin+row,i=first[u],j=second[u];
    const double ci=factors.data[i+factors.ld*channel];
    const double cj=factors.data[j+factors.ld*channel];
    const double t=images.data[cell],d=diagonal[u];
    double value=(i==k ? 2.0:1.0)*cj*t;
    if (k==i) value-=d*cj*cj;
    else if (i!=j && k==j) value-=d*ci*cj;
    add_core_exchange(output,i,k,channel,value);
    if (i!=j) {
      value=(j==k ? 2.0:1.0)*ci*t;
      if (k==j) value-=d*ci*ci;
      add_core_exchange(output,j,k,channel,value);
    }
  }
}

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

void ExactAoPairOperator::rank_one_exchange(linalg::ConstMatrixView factors,
    linalg::MatrixView output) const {
  const nvtx3::scoped_range range{"ao.rank_one_exchange"};
  check_cuda(cudaMemset2DAsync(output.data,output.ld*sizeof(double),0,
      output.rows*sizeof(double),output.cols,context_->stream()),"clear CUDA core exchange");
  if (!n_integrals_) return;
  const auto per_row=static_cast<std::size_t>(n_bf_)*(n_bf_+factors.cols)*sizeof(double);
  // One AO-index slice is enough to expose a wide GEMM; do not fill the
  // entire budget with a temporary dense expansion. Smaller budgets sub-tile it.
  const int width=static_cast<int>(std::min<std::size_t>(n_bf_,response_capacity()/per_row));
  if (!width) throw std::runtime_error("CUDA core exchange budget cannot hold one pair row");
  Matrix<double> rows(*context_,static_cast<std::int64_t>(n_bf_)*width,n_bf_);
  Matrix<double> images(*context_,rows.rows(),factors.cols);
  for (int begin=0;begin<n_pairs_;begin+=width) {
    const int count=static_cast<int>(std::min<std::int64_t>(width,n_pairs_-begin));
    const auto height=static_cast<std::int64_t>(n_bf_)*count;
    const linalg::MatrixView a{rows.data(),height,n_bf_,height};
    const linalg::MatrixView t{images.data(),height,factors.cols,height};
    check_cuda(cudaMemsetAsync(a.data,0,height*n_bf_*sizeof(double),context_->stream()),
        "clear CUDA core exchange row tile");
    if (wide_)
      core_exchange_rows<<<count,256,0,context_->stream()>>>(wide_offsets_.data(),
          wide_columns_.data(),values_.data(),pair_first_.data(),pair_second_.data(),
          begin,count,n_bf_,a.data);
    else
      core_exchange_rows<<<count,256,0,context_->stream()>>>(row_offsets_.data(),
          columns_.data(),values_.data(),pair_first_.data(),pair_second_.data(),
          begin,count,n_bf_,a.data);
    check_cuda(cudaGetLastError(),"CUDA core exchange row packing");
    gemm(*context_,t,a,factors);
    const auto blocks=std::min<std::int64_t>(65535,(height*factors.cols+255)/256);
    core_exchange_scatter<<<static_cast<unsigned int>(blocks),256,0,context_->stream()>>>(
        factors,t,pair_first_.data(),pair_second_.data(),diagonal_.data(),begin,count,output);
    check_cuda(cudaGetLastError(),"CUDA core exchange row projection");
  }
}

bool ExactAoPairOperator::unique_core_diagonal(linalg::ConstMatrixView mo,int ncore,
    linalg::MatrixView coulomb,linalg::MatrixView response) const {
  const nvtx3::scoped_range range{"ao.rank_one_core_diagonal"};
  const auto available=response_capacity();
  const auto row_bytes=static_cast<std::size_t>(n_bf_)*n_bf_*sizeof(double);
  const auto per_core=static_cast<std::size_t>(2*n_pairs_+n_bf_)*sizeof(double);
  // Keep space for at least one stacked exchange row and its image, in
  // addition to the two live J/K source panels. Tight budgets split channels.
  int width=available>row_bytes ? static_cast<int>(std::min<std::size_t>(ncore,
      (available-row_bytes)/per_core)):0;
  Buffer<double> features,products;
  for (;width>0;width/=2) {
    features=Buffer<double>(*context_,n_pairs_*width);
    products=Buffer<double>(*context_,n_pairs_*width);
    const auto scratch=[&](int count) {
      return apply_workspace({features.data(),n_pairs_,count,width},
          {products.data(),n_pairs_,count,width});
    };
    const int remainder=ncore%width;
    if (std::max(scratch(width),remainder ? scratch(remainder):std::size_t{0})<=
        response_capacity()) break;
    features={};products={};
  }
  if (!width) return false;
  for (int begin=0;begin<ncore;begin+=width) {
    const int count=std::min(width,ncore-begin);
    const auto factors=mo.columns(begin,count);
    const PairPanel source{features.data(),n_pairs_,count,width};
    const PairPanel image{products.data(),n_pairs_,count,width};
    density_features(factors,0,source);
    apply({source.data,source.rows,source.cols,source.ld},image);
    // J has consumed the source. Its storage now holds only the independent
    // AO triangle of symmetric K(C_i C_i^T), with channels contiguous.
    const linalg::MatrixView k{features.data(),count,n_pairs_,width};
    rank_one_exchange(factors,k);
    const auto cells=static_cast<std::int64_t>(n_bf_)*count;
    project_core<<<static_cast<unsigned int>(std::min<std::int64_t>(65535,cells)),
        256,0,context_->stream()>>>(mo,pair_first_.data(),pair_second_.data(),
            ConstPairPanel{image.data,image.rows,image.cols,image.ld},
            PairExchange{{k.data,n_pairs_,count,k.ld}},
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
  if (compact_) {
    if (joint) build_exchange_images(std::array{*joint});
    return unique_core_diagonal(mo,ncore,coulomb,response);
  }
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
