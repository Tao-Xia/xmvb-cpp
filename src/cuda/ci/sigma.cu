#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/sigma.hpp"

namespace xmvb::cuda {
namespace {

__global__ void scatter_channels(
    linalg::ConstMatrixView input, double* channels, std::int64_t begin,std::int64_t count,
    int n_alpha, int n_beta, const int* alpha, const int* beta,
    int channel_count) {
  const std::int64_t index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=count*channel_count) return;
  const auto product=begin+index%count;
  const int channel=index/count;
  const int a=product%n_alpha;
  const auto column=product/n_alpha;
  const int b=column%n_beta;
  const auto rhs=column/n_beta;
  const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
  const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
  double value=0;
  if (sa) value+=(sa>0 ? 1.0:-1.0)*input.data[abs(sa)-1+input.ld*column];
  if (sb) value+=(sb>0 ? 1.0:-1.0)*input.data[a+input.ld*(rhs*n_beta+abs(sb)-1)];
  channels[index]=value;
}

// E_P is symmetric: the same exact source/sign table implements E_P^T.
__global__ void gather_channels(const double* channels,linalg::MatrixView image,
    std::int64_t products,int n_alpha,int n_beta,const int* alpha,const int* beta,
    int first_channel,int channel_count,bool add) {
  const std::int64_t product=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (product>=products) return;
  const int a=product%n_alpha;
  const auto column=product/n_alpha;
  const int b=column%n_beta;
  const auto rhs=column/n_beta;
  double value=0;
  for (int local=0;local<channel_count;++local) {
    const int channel=first_channel+local;
    const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
    const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
    if (sa) value+=(sa>0 ? 1.0:-1.0)*channels[abs(sa)-1+n_alpha*column+products*local];
    if (sb) value+=(sb>0 ? 1.0:-1.0)*channels[a+n_alpha*(rhs*n_beta+abs(sb)-1)+products*local];
  }
  auto& destination=image.data[a+image.ld*column];
  if (add) destination+=value;
  else destination=value;
}

unsigned int grid_size(std::int64_t count) {
  constexpr std::int64_t threads = 256;
  const auto blocks = (count + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA direct-CI kernel grid is too large");
  return static_cast<unsigned int>(blocks);
}

}  // namespace

DirectCiSigma::SpinGraph::SpinGraph(
    Context& context,const mcscf::SpinConnections& source,int channels)
    : sources(context,static_cast<std::size_t>(source.n_strings())*channels) {
  std::vector<int> indices(sources.size(),0);
  const auto insert=[&](int channel,int row,int address,double sign) {
    if (channel<0 || channel>=channels || address<0 || address>=source.n_strings() ||
        (sign!=1.0 && sign!=-1.0))
      throw std::invalid_argument("invalid packed one-body channel link");
    auto& entry=indices[static_cast<std::size_t>(channel)*source.n_strings()+row];
    if (entry) throw std::logic_error("multiple sources in a one-spin packed channel");
    entry=sign>0 ? address+1:-(address+1);
  };
  for (int row=0;row<source.n_strings();++row) {
    for (const auto orbital:source.occupied(row))
      insert(orbital*(orbital+1)/2+orbital,row,row,1.0);
    for (const auto& link:source.singles(row))
      insert(link.pair,row,link.source,link.sign);
  }
  context.copy_to_device(sources.data(),indices.data(),indices.size()*sizeof(int));
}

DirectCiSigma::DirectCiSigma(
    Context& context, const mcscf::DirectCiSigmaAction& source,
    int maximum_channel_tile)
    : context_(&context), n_alpha_(source.n_alpha_strings()),
      n_beta_(source.n_beta_strings()),
      n_channels_(static_cast<int>(source.channel_kernel().rows())),
      maximum_channel_tile_(maximum_channel_tile),
      shared_spin_graph_(source.shared_spin_connections()),
      kernel_(context, static_cast<std::size_t>(n_channels_)*n_channels_),
      alpha_(context, source.alpha_connections(), n_channels_) {
  if (maximum_channel_tile <= 0 || n_channels_ <= 0 ||
      source.channel_kernel().cols() != n_channels_)
    throw std::invalid_argument("invalid CUDA direct-CI channel dimensions");
  context.copy_to_device(kernel_.data(), source.channel_kernel().data(),
                         kernel_.size()*sizeof(double));
  if (!shared_spin_graph_)
    beta_ = SpinGraph(context, source.beta_connections_view(), n_channels_);
}

std::size_t DirectCiSigma::resident_bytes() const noexcept {
  return kernel_.size()*sizeof(double) + alpha_.resident_bytes() +
         (shared_spin_graph_ ? 0 : beta_.resident_bytes()) +
      sizeof(double)*(scattered_.rows()*scattered_.cols()+transformed_.rows()*transformed_.cols());
}

void DirectCiSigma::prepare(int block_width) const {
  if (block_width<=0) throw std::invalid_argument("invalid CUDA sigma block width");
  const auto products=static_cast<std::int64_t>(n_alpha_)*n_beta_*block_width;
  if (transformed_.rows()==products) return;
  scattered_=Matrix<double>{};
  transformed_=Matrix<double>{};
  const std::size_t available=context_->device_byte_budget()-
      context_->counters().current_device_bytes;
  const std::size_t per_channel=2*static_cast<std::size_t>(products)*sizeof(double);
  const int tile=static_cast<int>(std::min<std::size_t>(
      {static_cast<std::size_t>(n_channels_),static_cast<std::size_t>(maximum_channel_tile_),
       available/per_channel}));
  if (tile<1)
    throw std::runtime_error("CUDA direct-CI budget cannot hold two single-channel slabs");
  // Keep the same two-slab capacity, but contract all channels for a row tile.
  // C is written once rather than reread/accumulated for every Q block.
  auto rows=std::min(products,std::max<std::int64_t>(1,products*tile/n_channels_));
  constexpr int warp_rows=32;
  if (rows>=warp_rows) rows=(rows/warp_rows)*warp_rows;
  if ((static_cast<std::size_t>(rows)*n_channels_+
       static_cast<std::size_t>(products)*tile)*sizeof(double)>available)
    throw std::runtime_error("CUDA direct-CI budget cannot hold a complete channel row");
  Matrix<double> scattered(*context_,rows,n_channels_);
  Matrix<double> transformed(*context_,products,tile);
  scattered_=std::move(scattered);
  transformed_=std::move(transformed);
}

void DirectCiSigma::apply(linalg::ConstMatrixView coefficients,
                          linalg::MatrixView image, bool add) const {
  const nvtx3::scoped_range range{"ci.sigma"};
  if (coefficients.rows != n_alpha_ || image.rows != n_alpha_ ||
      coefficients.cols <= 0 || coefficients.cols != image.cols ||
      coefficients.cols % n_beta_ != 0 ||
      coefficients.ld < coefficients.rows || image.ld < image.rows ||
      coefficients.data == nullptr || image.data == nullptr ||
      coefficients.data == image.data)
    throw std::invalid_argument("invalid CUDA direct-CI product panels");
  if (coefficients.cols > std::numeric_limits<int>::max() / n_alpha_)
    throw std::overflow_error("CUDA direct-CI product panel exceeds cuBLAS range");
  const std::int64_t products = coefficients.rows*coefficients.cols;
  prepare(static_cast<int>(coefficients.cols/n_beta_));
  const int tile=static_cast<int>(transformed_.cols());
  const auto* alpha=alpha_.sources.data();
  const auto* beta=shared_spin_graph_ ? alpha:beta_.sources.data();

  // A serpentine walk reuses the boundary source-row tile at the next P block.
  const auto rows=scattered_.rows();
  const auto blocks=(products+rows-1)/rows;
  std::int64_t retained_begin=-1;
  for (int p = 0; p < n_channels_; p += tile) {
    const int p_count = std::min(tile, n_channels_ - p);
    for (std::int64_t step=0;step<blocks;++step) {
      const auto begin=((p/tile)%2 ? blocks-1-step:step)*rows;
      const auto count=std::min(rows,products-begin);
      if (begin!=retained_begin) {
        scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          coefficients, scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_);
        check_cuda(cudaGetLastError(), "CUDA direct-CI channel scatter");
        retained_begin=begin;
      }
      const linalg::ConstMatrixView q_view{scattered_.data(),count,n_channels_,count};
      const linalg::ConstMatrixView kernel_view{
          kernel_.data()+p,p_count,n_channels_,n_channels_};
      const linalg::MatrixView p_view{transformed_.data()+begin,count,p_count,products};
      gemm(*context_, p_view, q_view, kernel_view,
           1.0,0.0,
           Operation::none, Operation::transpose);
    }
    gather_channels<<<grid_size(products),256,0,context_->stream()>>>(
        transformed_.data(), image, products, n_alpha_, n_beta_,
        alpha, beta, p, p_count, add || p != 0);
    check_cuda(cudaGetLastError(), "CUDA direct-CI channel gather");
  }
}

}  // namespace xmvb::cuda
