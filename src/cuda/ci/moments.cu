#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <stdexcept>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

namespace xmvb::cuda {
namespace {
constexpr int threads=256;

struct AddMoments {
  __device__ double2 operator()(double2 a,double2 b) const {
    return make_double2(a.x+b.x,a.y+b.y);
  }
};

// One packed channel, both E_pq orientations. Partitions stripe disjoint
// carrier rows; no generator image or carrier-by-channel panel is formed.
__global__ void reduce_moments(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,const int* alpha,const int* beta,
    int n_alpha,int n_beta,std::int64_t products,double2* partials) {
  using Reduce=cub::BlockReduce<double2,threads>;
  __shared__ typename Reduce::TempStorage scratch;
  const int channel=blockIdx.x;
  int high=0;
  while ((high+1)*(high+2)/2<=channel) ++high;
  const int low=channel-high*(high+1)/2;
  const int diagonal=high*(high+1)/2+high;
  double2 sum=make_double2(0.0,0.0);
  for (std::int64_t product=static_cast<std::int64_t>(blockIdx.y)*threads+threadIdx.x;
       product<products;product+=static_cast<std::int64_t>(gridDim.y)*threads) {
    const int a=product%n_alpha;
    const auto column=product/n_alpha;
    const int b=column%n_beta;
    const auto rhs=column/n_beta;
    const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
    const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
    if (!sa && !sb) continue;
    const double weight=left.data[a+left.ld*column];
    if (high==low) {
      sum.x+=weight*((sa!=0)+(sb!=0))*right.data[a+right.ld*column];
      continue;
    }
    double upper=0.0,lower=0.0;
    if (sa) {
      const double value=(sa>0 ? 1.0:-1.0)*
          right.data[abs(sa)-1+right.ld*column];
      if (alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]) upper+=value;
      else lower+=value;
    }
    if (sb) {
      const double value=(sb>0 ? 1.0:-1.0)*
          right.data[a+right.ld*(rhs*n_beta+abs(sb)-1)];
      if (beta[static_cast<std::int64_t>(diagonal)*n_beta+b]) upper+=value;
      else lower+=value;
    }
    sum.x+=weight*upper;
    sum.y+=weight*lower;
  }
  sum=Reduce(scratch).Reduce(sum,AddMoments{});
  if (threadIdx.x==0)
    partials[static_cast<std::int64_t>(channel)*gridDim.y+blockIdx.y]=sum;
}

__global__ void store_moments(const double2* partials,int partitions,
    linalg::MatrixView output) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=output.rows*output.cols) return;
  const int row=index%output.rows,column=index/output.rows;
  const int high=max(row,column),low=min(row,column);
  const int channel=high*(high+1)/2+low;
  double sum=0.0;
  for (int i=0;i<partitions;++i) {
    const auto value=partials[static_cast<std::int64_t>(channel)*partitions+i];
    sum+=row>=column ? value.x:value.y;
  }
  output.data[row+output.ld*column]=sum;
}
} // namespace

void DirectCiSigma::generator_moments(std::span<const linalg::ConstMatrixView> left,
    linalg::ConstMatrixView right,std::span<const linalg::MatrixView> outputs) const {
  const nvtx3::scoped_range range{"ci.generator_moments"};
  if (left.empty() || left.size()!=outputs.size() || right.rows!=n_alpha_ ||
      right.cols<=0 || right.cols%n_beta_ || right.ld<right.rows || !right.data)
    throw std::invalid_argument("CUDA generator moment dimensions");
  for (std::size_t i=0;i<left.size();++i) {
    const auto l=left[i];const auto out=outputs[i];
    if (l.rows!=right.rows || l.cols!=right.cols || l.ld<l.rows || !l.data ||
        out.rows!=n_orbitals_ || out.cols!=n_orbitals_ || out.ld<out.rows || !out.data ||
        out.data==l.data || out.data==right.data)
      throw std::invalid_argument("invalid CUDA generator moment views");
  }
  if (!workspace_->moment_partitions) {
    int device=0,multiprocessors=0,blocks=0;
    check_cuda(cudaGetDevice(&device),"generator moment device");
    check_cuda(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device),
        "generator moment multiprocessors");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,reduce_moments,threads,0),
        "generator moment occupancy");
    workspace_->moment_partitions=std::max(1,(multiprocessors*blocks+n_channels_-1)/n_channels_);
  }
  const auto products=right.rows*right.cols;
  const int partitions=static_cast<int>(std::min<std::int64_t>(workspace_->moment_partitions,
      (products+threads-1)/threads));
  const auto count=static_cast<std::size_t>(n_channels_)*partitions;
  Buffer<double2> partials(*context_,count);
  for (std::size_t i=0;i<left.size();++i) {
    reduce_moments<<<dim3(n_channels_,partitions),threads,0,context_->stream()>>>(
        left[i],right,alpha_->sources.data(),beta_->sources.data(),n_alpha_,n_beta_,
        products,partials.data());
    check_cuda(cudaGetLastError(),"CUDA direct link generator moments");
    const auto cells=static_cast<std::int64_t>(n_orbitals_)*n_orbitals_;
    store_moments<<<static_cast<unsigned int>((cells+threads-1)/threads),threads,0,
        context_->stream()>>>(partials.data(),partitions,outputs[i]);
    check_cuda(cudaGetLastError(),"CUDA generator moment partial reduction");
  }
}
} // namespace xmvb::cuda
