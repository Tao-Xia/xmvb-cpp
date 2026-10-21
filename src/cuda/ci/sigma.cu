#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <cmath>
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

__global__ void absorbed_one_adjoint(linalg::ConstMatrixView channel,
    double* absorbed, int n_orbitals, int n_channels) {
  const int pair=blockIdx.x*blockDim.x+threadIdx.x;
  if (pair>=n_channels) return;
  double value=0;
  for (int orbital=0;orbital<n_orbitals;++orbital) {
    const int diagonal=orbital*(orbital+1)/2+orbital;
    value+=0.5*(channel.data[diagonal+channel.ld*pair]+
                channel.data[pair+channel.ld*diagonal]);
  }
  absorbed[pair]=value;
}

__device__ __forceinline__ int packed_pair(int first,int second) {
  const int row=max(first,second);
  return row*(row+1)/2+min(first,second);
}

__device__ __forceinline__ void unpack_pair(int pair,int& row,int& column) {
  row=static_cast<int>((sqrt(8.0*pair+1.0)-1.0)*0.5);
  while ((row+1)*(row+2)/2<=pair) ++row;
  while (row*(row+1)/2>pair) --row;
  column=pair-row*(row+1)/2;
}

__global__ void build_effective_one(linalg::ConstMatrixView one_electron,
    linalg::ConstMatrixView pair_kernel,double* effective,
    int n_orbitals,int n_channels,int n_electrons) {
  const int pair=blockIdx.x*blockDim.x+threadIdx.x;
  if (pair>=n_channels) return;
  int row,column;
  unpack_pair(pair,row,column);
  double contraction=0;
  for (int orbital=0;orbital<n_orbitals;++orbital)
    contraction+=pair_kernel.data[
        packed_pair(row,orbital)+pair_kernel.ld*packed_pair(orbital,column)];
  effective[pair]=
      (one_electron.data[row+one_electron.ld*column]-0.5*contraction)/n_electrons;
}

__global__ void rebuild_channel_kernel(linalg::ConstMatrixView pair_kernel,
    const double* effective,double* kernel,int n_channels) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=n_channels*n_channels) return;
  const int p=index%n_channels,q=index/n_channels;
  int p_row,p_column,q_row,q_column;
  unpack_pair(p,p_row,p_column);
  unpack_pair(q,q_row,q_column);
  double value=0.25*(pair_kernel.data[p+pair_kernel.ld*q]+
                     pair_kernel.data[q+pair_kernel.ld*p]);
  if (p_row==p_column) value+=0.5*effective[q];
  if (q_row==q_column) value+=0.5*effective[p];
  kernel[index]=value;
}

__global__ void pullback_one_electron(const double* absorbed,
    linalg::MatrixView one_electron,int n_orbitals,int n_electrons) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=n_orbitals*n_orbitals) return;
  const int row=index%n_orbitals, column=index/n_orbitals;
  const double weight=absorbed[packed_pair(row,column)]/n_electrons;
  one_electron.data[row+one_electron.ld*column]=
      row==column ? weight:0.5*weight;
}

__global__ void pullback_pair_kernel(linalg::ConstMatrixView channel,
    const double* absorbed,linalg::MatrixView pair_kernel,
    int n_channels,int n_electrons) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=n_channels*n_channels) return;
  const int p=index%n_channels, q=index/n_channels;
  int p_row,p_column,q_row,q_column;
  unpack_pair(p,p_row,p_column);
  unpack_pair(q,q_row,q_column);
  double correction=0;
  const int common[2]={p_row,p_column};
  for (int candidate=0;candidate<2;++candidate) {
    const int orbital=common[candidate];
    if (candidate==1 && orbital==common[0]) continue;
    if (orbital!=q_row && orbital!=q_column) continue;
    const int row=orbital==p_row ? p_column:p_row;
    const int column=orbital==q_row ? q_column:q_row;
    correction+=absorbed[packed_pair(row,column)]/n_electrons;
  }
  pair_kernel.data[p+pair_kernel.ld*q]=
      0.25*(channel.data[p+channel.ld*q]+channel.data[q+channel.ld*p])-
      (p==q ? 0.5:0.25)*correction;
}

__global__ void product_diagonal(linalg::MatrixView values,
    const double* kernel,const int* alpha,const int* beta,
    int n_alpha,int n_beta,int n_orbitals,int n_channels) {
  const std::int64_t product=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (product>=static_cast<std::int64_t>(n_alpha)*n_beta) return;
  const int a=product%n_alpha, b=product/n_alpha;
  double result=0;
  for (int p=0;p<n_orbitals;++p) {
    const int pp=p*(p+1)/2+p;
    const int occupation_p=(alpha[static_cast<std::int64_t>(pp)*n_alpha+a]!=0)+
                           (beta[static_cast<std::int64_t>(pp)*n_beta+b]!=0);
    if (!occupation_p) continue;
    for (int q=0;q<n_orbitals;++q) {
      const int qq=q*(q+1)/2+q;
      const int occupation_q=(alpha[static_cast<std::int64_t>(qq)*n_alpha+a]!=0)+
                             (beta[static_cast<std::int64_t>(qq)*n_beta+b]!=0);
      result+=occupation_p*occupation_q*kernel[pp+static_cast<std::int64_t>(n_channels)*qq];
    }
  }
  for (int row=1;row<n_orbitals;++row) {
    for (int column=0;column<row;++column) {
      const int channel=row*(row+1)/2+column;
      const int links=(alpha[static_cast<std::int64_t>(channel)*n_alpha+a]!=0)+
                      (beta[static_cast<std::int64_t>(channel)*n_beta+b]!=0);
      result+=links*kernel[channel+static_cast<std::int64_t>(n_channels)*channel];
    }
  }
  values.data[a+values.ld*b]=result;
}

__global__ void apply_orbital_generator(linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView generator,linalg::MatrixView image,
    std::int64_t products,int n_alpha,int n_beta,int n_orbitals,
    const int* alpha,const int* beta) {
  const std::int64_t product=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (product>=products) return;
  const int a=product%n_alpha;
  const auto column=product/n_alpha;
  const int b=column%n_beta;
  const auto rhs=column/n_beta;
  const double self=coefficients.data[a+coefficients.ld*column];
  double value=0;
  for (int p=0;p<n_orbitals;++p) {
    const int diagonal=p*(p+1)/2+p;
    const int occupied=(alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]!=0)+
                       (beta[static_cast<std::int64_t>(diagonal)*n_beta+b]!=0);
    value+=occupied*generator.data[p+generator.ld*p]*self;
  }
  for (int high=1;high<n_orbitals;++high) {
    const int diagonal=high*(high+1)/2+high;
    const bool alpha_high=alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]!=0;
    const bool beta_high=beta[static_cast<std::int64_t>(diagonal)*n_beta+b]!=0;
    for (int low=0;low<high;++low) {
      const int channel=high*(high+1)/2+low;
      const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
      const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
      if (sa) {
        const int created=alpha_high ? high:low;
        const int annihilated=alpha_high ? low:high;
        value+=(sa>0 ? 1.0:-1.0)*
            generator.data[created+generator.ld*annihilated]*
            coefficients.data[abs(sa)-1+coefficients.ld*column];
      }
      if (sb) {
        const int created=beta_high ? high:low;
        const int annihilated=beta_high ? low:high;
        value+=(sb>0 ? 1.0:-1.0)*
            generator.data[created+generator.ld*annihilated]*
            coefficients.data[a+coefficients.ld*(rhs*n_beta+abs(sb)-1)];
      }
    }
  }
  image.data[a+image.ld*column]=value;
}

__global__ void scatter_oriented_generator(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,double* left_values,
    double* high_channels,double* low_channels,std::int64_t begin,int count,
    int n_alpha,int n_beta,int n_channels,
    const int* alpha,const int* beta) {
  const std::int64_t index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=static_cast<std::int64_t>(count)*n_channels) return;
  const int channel=index/count;
  const auto product=begin+index%count;
  const int a=product%n_alpha;
  const auto column=product/n_alpha;
  const int b=column%n_beta;
  const auto rhs=column/n_beta;
  if (channel==0) left_values[index%count]=left.data[a+left.ld*column];
  const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
  const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
  int high,low;
  unpack_pair(channel,high,low);
  double high_value=0,low_value=0;
  if (high==low) {
    const int occupied=(sa!=0)+(sb!=0);
    high_value=occupied*right.data[a+right.ld*column];
  } else {
    if (sa) {
      const double value=(sa>0 ? 1.0:-1.0)*
          right.data[abs(sa)-1+right.ld*column];
      const int diagonal=high*(high+1)/2+high;
      if (alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]) high_value+=value;
      else low_value+=value;
    }
    if (sb) {
      const double value=(sb>0 ? 1.0:-1.0)*
          right.data[a+right.ld*(rhs*n_beta+abs(sb)-1)];
      const int diagonal=high*(high+1)/2+high;
      if (beta[static_cast<std::int64_t>(diagonal)*n_beta+b]) high_value+=value;
      else low_value+=value;
    }
  }
  high_channels[index]=high_value;
  low_channels[index]=low_value;
}

__global__ void unpack_generator_moments(const double* moments,
    linalg::MatrixView orbital_adjoint,int n_orbitals,int n_channels) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=n_orbitals*n_orbitals) return;
  const int row=index%n_orbitals,column=index/n_orbitals;
  const int pair=packed_pair(row,column);
  orbital_adjoint.data[row+orbital_adjoint.ld*column]=
      row>=column ? moments[pair]:moments[n_channels+pair];
}

unsigned int grid_size(std::int64_t count) {
  constexpr std::int64_t threads = 256;
  const auto blocks = (count + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA direct-CI kernel grid is too large");
  return static_cast<unsigned int>(blocks);
}

int validate_adjoint_panels(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,int n_alpha,int n_beta) {
  if (left.rows!=n_alpha || right.rows!=n_alpha || left.cols<=0 ||
      left.cols!=right.cols || left.cols%n_beta!=0 ||
      left.ld<left.rows || right.ld<right.rows ||
      left.data==nullptr || right.data==nullptr)
    throw std::invalid_argument("invalid CUDA direct-CI adjoint product panels");
  if (left.cols>std::numeric_limits<int>::max()/n_alpha)
    throw std::overflow_error("CUDA direct-CI adjoint panel exceeds cuBLAS range");
  return static_cast<int>(left.cols/n_beta);
}

void validate_output(linalg::MatrixView output,std::int64_t rows,
    std::int64_t cols,const char* name) {
  if (output.rows!=rows || output.cols!=cols || output.ld<rows ||
      output.data==nullptr)
    throw std::invalid_argument(name);
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
      n_orbitals_(source.n_orbitals()),
      n_electrons_(source.alpha_connections().n_occupied()+
                   source.beta_connections_view().n_occupied()),
      n_channels_(static_cast<int>(source.channel_kernel().rows())),
      maximum_channel_tile_(maximum_channel_tile),
      shared_spin_graph_(source.shared_spin_connections()),
      kernel_(context, static_cast<std::size_t>(n_channels_)*n_channels_),
      effective_one_(context,n_channels_),
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
  return (kernel_.size()+effective_one_.size()+generator_moments_.size())*sizeof(double) +
         alpha_.resident_bytes() +
         (shared_spin_graph_ ? 0 : beta_.resident_bytes()) +
      sizeof(double)*(scattered_.rows()*scattered_.cols()+
          transformed_.rows()*transformed_.cols()+
          right_scattered_.rows()*right_scattered_.cols()+
          channel_adjoint_.rows()*channel_adjoint_.cols()+
          generator_high_.rows()*generator_high_.cols()+
          generator_low_.rows()*generator_low_.cols()+generator_left_.rows());
}

void DirectCiSigma::prepare(int block_width) const {
  if (block_width<=0) throw std::invalid_argument("invalid CUDA sigma block width");
  const auto products=static_cast<std::int64_t>(n_alpha_)*n_beta_*block_width;
  if (transformed_.rows()==products) return;
  scattered_=Matrix<double>{};
  transformed_=Matrix<double>{};
  right_scattered_=Matrix<double>{};
  channel_adjoint_=Matrix<double>{};
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

void DirectCiSigma::prepare_adjoint(int block_width,bool with_sigma,
                                   bool retain_channel_kernel) const {
  if (block_width<=0) throw std::invalid_argument("invalid CUDA adjoint block width");
  const auto products=static_cast<std::int64_t>(n_alpha_)*n_beta_*block_width;
  if (scattered_.rows()>0 && scattered_.cols()==n_channels_ &&
      right_scattered_.rows()==scattered_.rows() &&
      right_scattered_.cols()==n_channels_ &&
      (!with_sigma || (transformed_.rows()==products && transformed_.cols()>0)) &&
      (!retain_channel_kernel || channel_adjoint_.rows()==n_channels_)) return;

  scattered_=Matrix<double>{};
  transformed_=Matrix<double>{};
  right_scattered_=Matrix<double>{};
  channel_adjoint_=Matrix<double>{};
  const std::size_t available=context_->device_byte_budget()-
      context_->counters().current_device_bytes;
  const std::size_t kernel_bytes=retain_channel_kernel ?
      static_cast<std::size_t>(n_channels_)*n_channels_*sizeof(double):0;
  const std::size_t minimum_rows=2*static_cast<std::size_t>(n_channels_)*sizeof(double);
  if (kernel_bytes>available || minimum_rows>available-kernel_bytes)
    throw std::runtime_error("CUDA direct-CI budget cannot hold adjoint channel slabs");
  const std::size_t workspace=available-kernel_bytes;
  int tile=0;
  std::int64_t rows=0;
  if (with_sigma) {
    const auto transform_channel_bytes=static_cast<std::size_t>(products)*sizeof(double);
    if (transform_channel_bytes>workspace-minimum_rows)
      throw std::runtime_error("CUDA direct-CI budget cannot hold fused sigma slab");
    tile=static_cast<int>(std::min<std::size_t>({
        static_cast<std::size_t>(n_channels_),
        static_cast<std::size_t>(maximum_channel_tile_),
        (workspace-minimum_rows)/transform_channel_bytes}));
    rows=std::min<std::int64_t>(products,
        (workspace/sizeof(double)-static_cast<std::size_t>(products)*tile)/
        (2*static_cast<std::size_t>(n_channels_)));
  } else {
    rows=std::min<std::int64_t>(products,
        workspace/(2*static_cast<std::size_t>(n_channels_)*sizeof(double)));
  }
  // Match the sigma source-panel plan under the shared Context budget. There
  // is no private fixed byte cap competing with the accepted-point workspace.
  const auto panel_rows=std::max<std::int64_t>(1,
      products*std::min(n_channels_,maximum_channel_tile_)/n_channels_);
  rows=std::min(rows,panel_rows);
  if (rows<1) throw std::runtime_error("CUDA direct-CI adjoint row exceeds budget");
  if (rows>=32) rows=(rows/32)*32;
  Matrix<double> kernel_adjoint;
  if (retain_channel_kernel)
    kernel_adjoint=Matrix<double>(*context_,n_channels_,n_channels_);
  Matrix<double> scattered(*context_,rows,n_channels_);
  Matrix<double> right_scattered(*context_,rows,n_channels_);
  Matrix<double> transformed;
  if (with_sigma) transformed=Matrix<double>(*context_,products,tile);
  channel_adjoint_=std::move(kernel_adjoint);
  scattered_=std::move(scattered);
  right_scattered_=std::move(right_scattered);
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

void DirectCiSigma::contract_adjoint(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,linalg::MatrixView* image,
    linalg::MatrixView channel_kernel_adjoint) const {
  const nvtx3::scoped_range range{"ci.sigma_adjoint"};
  const std::int64_t products=left.rows*left.cols;
  const auto rows=scattered_.rows();
  const auto blocks=(products+rows-1)/rows;
  const int tile=image ? static_cast<int>(transformed_.cols()):1;
  const auto* alpha=alpha_.sources.data();
  const auto* beta=shared_spin_graph_ ? alpha:beta_.sources.data();
  std::int64_t retained_begin=-1;
  bool first_adjoint=true;
  for (int p=0;p<(image ? n_channels_:1);p+=tile) {
    const int p_count=image ? std::min(tile,n_channels_-p):0;
    for (std::int64_t step=0;step<blocks;++step) {
      const auto begin=(image && (p/tile)%2 ? blocks-1-step:step)*rows;
      const auto count=std::min(rows,products-begin);
      if (begin!=retained_begin) {
        scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
            left,scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_);
        check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint left scatter");
        retained_begin=begin;
      }
      const linalg::ConstMatrixView left_channels{
          scattered_.data(),count,n_channels_,count};
      if (p==0) {
        scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
            right,right_scattered_.data(),begin,count,n_alpha_,n_beta_,
            alpha,beta,n_channels_);
        check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint right scatter");
        const linalg::ConstMatrixView right_channels{
            right_scattered_.data(),count,n_channels_,count};
        gemm(*context_,channel_kernel_adjoint,left_channels,right_channels,
             1.0,first_adjoint ? 0.0:1.0,
             Operation::transpose,Operation::none);
        first_adjoint=false;
      }
      if (image) {
        const linalg::ConstMatrixView kernel_view{
            kernel_.data()+p,p_count,n_channels_,n_channels_};
        const linalg::MatrixView p_view{
            transformed_.data()+begin,count,p_count,products};
        gemm(*context_,p_view,left_channels,kernel_view,
             1.0,0.0,Operation::none,Operation::transpose);
      }
    }
    if (image) {
      gather_channels<<<grid_size(products),256,0,context_->stream()>>>(
          transformed_.data(),*image,products,n_alpha_,n_beta_,
          alpha,beta,p,p_count,p!=0);
      check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint sigma gather");
    }
  }
}

void DirectCiSigma::pullback_channel_kernel(
    linalg::ConstMatrixView channel_kernel_adjoint,
    linalg::MatrixView one_electron_adjoint,
    linalg::MatrixView pair_kernel_adjoint) const {
  // The right scatter has no remaining consumers after the K contraction.
  // Reuse its first n_channels slots for the small orbital pullback vector.
  auto* absorbed=right_scattered_.data();
  absorbed_one_adjoint<<<grid_size(n_channels_),256,0,context_->stream()>>>(
      channel_kernel_adjoint,absorbed,n_orbitals_,n_channels_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI absorbed-one pullback");
  pullback_one_electron<<<grid_size(static_cast<std::int64_t>(n_orbitals_)*n_orbitals_),
      256,0,context_->stream()>>>(absorbed,one_electron_adjoint,n_orbitals_,n_electrons_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI one-electron pullback");
  pullback_pair_kernel<<<grid_size(static_cast<std::int64_t>(n_channels_)*n_channels_),
      256,0,context_->stream()>>>(channel_kernel_adjoint,absorbed,
          pair_kernel_adjoint,n_channels_,n_electrons_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI pair-kernel pullback");
}

void DirectCiSigma::channel_adjoint(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,linalg::MatrixView channel_kernel_adjoint) const {
  const int block_width=validate_adjoint_panels(left,right,n_alpha_,n_beta_);
  validate_output(channel_kernel_adjoint,n_channels_,n_channels_,
      "invalid CUDA direct-CI channel adjoint output");
  if (channel_kernel_adjoint.data==left.data || channel_kernel_adjoint.data==right.data)
    throw std::invalid_argument("CUDA direct-CI channel adjoint output aliases input");
  prepare_adjoint(block_width,false,false);
  contract_adjoint(left,right,nullptr,channel_kernel_adjoint);
}

void DirectCiSigma::integral_adjoint(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,linalg::MatrixView one_electron_adjoint,
    linalg::MatrixView pair_kernel_adjoint) const {
  const int block_width=validate_adjoint_panels(left,right,n_alpha_,n_beta_);
  validate_output(one_electron_adjoint,n_orbitals_,n_orbitals_,
      "invalid CUDA direct-CI one-electron adjoint output");
  validate_output(pair_kernel_adjoint,n_channels_,n_channels_,
      "invalid CUDA direct-CI pair-kernel adjoint output");
  if (one_electron_adjoint.data==left.data || one_electron_adjoint.data==right.data ||
      pair_kernel_adjoint.data==left.data || pair_kernel_adjoint.data==right.data ||
      pair_kernel_adjoint.data==one_electron_adjoint.data)
    throw std::invalid_argument("CUDA direct-CI integral adjoint output aliases another view");
  prepare_adjoint(block_width,false,true);
  contract_adjoint(left,right,nullptr,channel_adjoint_.view());
  pullback_channel_kernel(channel_adjoint_.view(),
      one_electron_adjoint,pair_kernel_adjoint);
}

void DirectCiSigma::apply_with_adjoint(linalg::ConstMatrixView response,
    linalg::ConstMatrixView state,linalg::MatrixView image,
    linalg::MatrixView one_electron_adjoint,
    linalg::MatrixView pair_kernel_adjoint) const {
  const int block_width=validate_adjoint_panels(response,state,n_alpha_,n_beta_);
  validate_output(image,n_alpha_,response.cols,
      "invalid CUDA direct-CI response image");
  validate_output(one_electron_adjoint,n_orbitals_,n_orbitals_,
      "invalid CUDA direct-CI one-electron adjoint output");
  validate_output(pair_kernel_adjoint,n_channels_,n_channels_,
      "invalid CUDA direct-CI pair-kernel adjoint output");
  if (image.data==response.data || image.data==state.data ||
      one_electron_adjoint.data==response.data || one_electron_adjoint.data==state.data ||
      pair_kernel_adjoint.data==response.data || pair_kernel_adjoint.data==state.data ||
      image.data==one_electron_adjoint.data || image.data==pair_kernel_adjoint.data ||
      one_electron_adjoint.data==pair_kernel_adjoint.data)
    throw std::invalid_argument("CUDA direct-CI fused response output aliases another view");
  prepare_adjoint(block_width,true,true);
  contract_adjoint(response,state,&image,channel_adjoint_.view());
  pullback_channel_kernel(channel_adjoint_.view(),
      one_electron_adjoint,pair_kernel_adjoint);
}

void DirectCiSigma::diagonal(linalg::MatrixView values) const {
  validate_output(values,n_alpha_,n_beta_,
      "invalid CUDA direct-CI diagonal output");
  const auto* alpha=alpha_.sources.data();
  const auto* beta=shared_spin_graph_ ? alpha:beta_.sources.data();
  product_diagonal<<<grid_size(static_cast<std::int64_t>(n_alpha_)*n_beta_),
      256,0,context_->stream()>>>(values,kernel_.data(),alpha,beta,
          n_alpha_,n_beta_,n_orbitals_,n_channels_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI product diagonal");
}

void DirectCiSigma::update_integrals(linalg::ConstMatrixView one_electron,
    linalg::ConstMatrixView pair_kernel) {
  if (one_electron.rows!=n_orbitals_ || one_electron.cols!=n_orbitals_ ||
      one_electron.ld<n_orbitals_ || one_electron.data==nullptr ||
      pair_kernel.rows!=n_channels_ || pair_kernel.cols!=n_channels_ ||
      pair_kernel.ld<n_channels_ || pair_kernel.data==nullptr ||
      one_electron.data==pair_kernel.data)
    throw std::invalid_argument("invalid CUDA direct-CI integral update views");
  build_effective_one<<<grid_size(n_channels_),256,0,context_->stream()>>>(
      one_electron,pair_kernel,effective_one_.data(),
      n_orbitals_,n_channels_,n_electrons_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI effective one-electron update");
  rebuild_channel_kernel<<<grid_size(static_cast<std::int64_t>(n_channels_)*n_channels_),
      256,0,context_->stream()>>>(pair_kernel,effective_one_.data(),
          kernel_.data(),n_channels_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI channel-kernel update");
}

void DirectCiSigma::apply_generator(linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView orbital_generator,linalg::MatrixView image) const {
  validate_adjoint_panels(coefficients,coefficients,n_alpha_,n_beta_);
  validate_output(image,n_alpha_,coefficients.cols,
      "invalid CUDA direct-CI generator image");
  if (orbital_generator.rows!=n_orbitals_ ||
      orbital_generator.cols!=n_orbitals_ ||
      orbital_generator.ld<n_orbitals_ || orbital_generator.data==nullptr ||
      image.data==coefficients.data || image.data==orbital_generator.data)
    throw std::invalid_argument("invalid CUDA direct-CI orbital generator views");
  const auto* alpha=alpha_.sources.data();
  const auto* beta=shared_spin_graph_ ? alpha:beta_.sources.data();
  const std::int64_t products=coefficients.rows*coefficients.cols;
  apply_orbital_generator<<<grid_size(products),256,0,context_->stream()>>>(
      coefficients,orbital_generator,image,products,n_alpha_,n_beta_,
      n_orbitals_,alpha,beta);
  check_cuda(cudaGetLastError(),"CUDA direct-CI orbital generator action");
}

void DirectCiSigma::prepare_generator_adjoint(int block_width) const {
  if (block_width<=0) throw std::invalid_argument("invalid generator block width");
  if (generator_high_.rows()>0 && generator_low_.rows()==generator_high_.rows() &&
      generator_moments_.size()==2*static_cast<std::size_t>(n_channels_)) return;
  const std::size_t available=context_->device_byte_budget()-
      context_->counters().current_device_bytes;
  const std::size_t moments_bytes=2*static_cast<std::size_t>(n_channels_)*sizeof(double);
  const std::size_t row_bytes=(2*static_cast<std::size_t>(n_channels_)+1)*sizeof(double);
  if (moments_bytes>available || row_bytes>available-moments_bytes)
    throw std::runtime_error("CUDA direct-CI budget cannot hold generator adjoint row");
  const auto products=static_cast<std::int64_t>(n_alpha_)*n_beta_*block_width;
  const auto planned_rows=std::max<std::int64_t>(1,products*
      std::min(n_channels_,maximum_channel_tile_)/n_channels_);
  auto rows=std::min<std::int64_t>({products,planned_rows,
      static_cast<std::int64_t>((available-moments_bytes)/row_bytes)});
  if (rows>=32) rows=(rows/32)*32;
  Buffer<double> moments(*context_,2*static_cast<std::size_t>(n_channels_));
  Matrix<double> high(*context_,rows,n_channels_);
  Matrix<double> low(*context_,rows,n_channels_);
  generator_moments_=std::move(moments);
  generator_high_=std::move(high);
  generator_low_=std::move(low);
  generator_left_=Matrix<double>(*context_,rows,1);
}

void DirectCiSigma::one_body_generator_adjoint(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,linalg::MatrixView orbital_adjoint) const {
  const int block_width=validate_adjoint_panels(left,right,n_alpha_,n_beta_);
  validate_output(orbital_adjoint,n_orbitals_,n_orbitals_,
      "invalid CUDA direct-CI orbital adjoint output");
  if (orbital_adjoint.data==left.data || orbital_adjoint.data==right.data)
    throw std::invalid_argument("CUDA direct-CI orbital adjoint output aliases input");
  prepare_generator_adjoint(block_width);
  const auto* alpha=alpha_.sources.data();
  const auto* beta=shared_spin_graph_ ? alpha:beta_.sources.data();
  const int rows=static_cast<int>(generator_high_.rows());
  const auto products=left.rows*left.cols;
  bool first=true;
  for (std::int64_t begin=0;begin<products;begin+=rows) {
      const int count=static_cast<int>(std::min<std::int64_t>(rows,products-begin));
      scatter_oriented_generator<<<grid_size(static_cast<std::int64_t>(count)*n_channels_),
          256,0,context_->stream()>>>(left,right,generator_left_.data(),
              generator_high_.data(),generator_low_.data(),begin,count,
              n_alpha_,n_beta_,n_channels_,alpha,beta);
      check_cuda(cudaGetLastError(),"CUDA direct-CI oriented generator scatter");
      const linalg::ConstMatrixView left_column{
          generator_left_.data(),count,1,count};
      const linalg::ConstMatrixView high{
          generator_high_.data(),count,n_channels_,count};
      const linalg::ConstMatrixView low{
          generator_low_.data(),count,n_channels_,count};
      const linalg::MatrixView high_moments{
          generator_moments_.data(),n_channels_,1,n_channels_};
      const linalg::MatrixView low_moments{
          generator_moments_.data()+n_channels_,n_channels_,1,n_channels_};
      gemm(*context_,high_moments,high,left_column,
          1.0,first ? 0.0:1.0,Operation::transpose,Operation::none);
      gemm(*context_,low_moments,low,left_column,
          1.0,first ? 0.0:1.0,Operation::transpose,Operation::none);
      first=false;
  }
  unpack_generator_moments<<<grid_size(static_cast<std::int64_t>(n_orbitals_)*n_orbitals_),
      256,0,context_->stream()>>>(generator_moments_.data(),orbital_adjoint,
          n_orbitals_,n_channels_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI orbital generator pullback");
}

}  // namespace xmvb::cuda
