#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/ci/rows.cuh"
#include "mcscf/ci/sigma.hpp"

namespace xmvb::cuda {
namespace {

__global__ void scatter_channels(
    linalg::ConstMatrixView input, double* channels, std::int64_t begin,std::int64_t count,
    int n_alpha, int n_beta, const int* alpha, const int* beta,
    int channel_count,int parity,bool weighted) {
  const std::int64_t index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=count*channel_count) return;
  const auto row=product_row(begin+index%count,n_alpha,n_beta,parity);
  const int channel=index/count;
  const int a=row.alpha,b=row.beta;
  const auto column=row.column,rhs=row.rhs;
  const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
  const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
  double value=0;
  if (sa) value+=(sa>0 ? 1.0:-1.0)*input.data[abs(sa)-1+input.ld*column];
  if (sb) value+=(sb>0 ? 1.0:-1.0)*input.data[a+input.ld*(rhs*n_beta+abs(sb)-1)];
  channels[index]=value*(weighted ? row.weight:1.0);
}

// E_P is a Hermitian packed one-body channel. Its transpose therefore uses
// the same signed spin links, starting at each source row in this tile.
// Visit only count*channels links, not products*channels for every row tile.
// Atomic additions merge different channels without a full-carrier buffer.
__global__ void scatter_channel_adjoint(const double* channels,linalg::MatrixView image,
    std::int64_t begin,std::int64_t count,
    int n_alpha,int n_beta,const int* alpha,const int* beta,int n_channels,int parity) {
  const auto index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=count*n_channels) return;
  const auto row=product_row(begin+index%count,n_alpha,n_beta,parity);
  const int p=index/count;
  const int a=row.alpha,b=row.beta;
  const auto column=row.column,rhs=row.rhs;
  const int sa=alpha[static_cast<std::int64_t>(p)*n_alpha+a];
  const int sb=beta[static_cast<std::int64_t>(p)*n_beta+b];
  const double value=channels[index];
  if (sa)
    atomicAdd(image.data+abs(sa)-1+image.ld*column,(sa>0 ? 1.0:-1.0)*value);
  if (sb)
    atomicAdd(image.data+a+image.ld*(rhs*n_beta+abs(sb)-1),(sb>0 ? 1.0:-1.0)*value);
}

// D is needed on the state only after its unweighted integral-VJP consumer.
__global__ void weight_channels(double* data,std::int64_t begin,std::int64_t count,
    int na,int nb,int channels,int parity) {
  const auto index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index<count*channels)
    data[index]*=product_row(begin+index%count,na,nb,parity).weight;
}

// Each unordered pair has one owner; no full transpose workspace is needed.
__global__ void complete_exchange_image(linalg::MatrixView image,std::int64_t rows,
    int na,int nb,int parity,bool average) {
  const auto index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=rows) return;
  const auto row=product_row(index,na,nb,parity);
  auto* lower=image.data+row.alpha+image.ld*row.column;
  auto* upper=image.data+row.beta+image.ld*(row.rhs*nb+row.alpha);
  double value=*lower;
  if (row.alpha==row.beta) {
    if (parity<0) *lower=0.0;
    return;
  }
  if (average) value=0.5*(value+parity*(*upper));
  *lower=value;*upper=parity*value;
}

void clear_image(Context& context,linalg::MatrixView image) {
  check_cuda(cudaMemset2DAsync(image.data,image.ld*sizeof(double),0,
      image.rows*sizeof(double),image.cols,context.stream()),
      "clear CUDA source-scattered CI image");
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
    const int* alpha,const int* beta,int parity) {
  const std::int64_t product=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (product>=products) return;
  const auto row=product_row(product,n_alpha,n_beta,parity);
  const int a=row.alpha,b=row.beta;
  const auto column=row.column,rhs=row.rhs;
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

// Every live slab is [source row, all channels]. Channel equivalents bound
// its maximum size, but no full-carrier channel is allocated or reserved.
std::int64_t source_tile_rows(Context& context,std::int64_t products,
    int channels,int capacity,int panels,std::size_t extra=0) {
  const auto available=context.device_byte_budget()-context.counters().current_device_bytes;
  const auto row_bytes=static_cast<std::size_t>(panels)*channels*sizeof(double);
  if (extra>available || row_bytes>available-extra)
    throw std::runtime_error("CUDA direct-CI budget cannot hold one complete source row");
  const auto maximum=std::max<std::int64_t>(1,
      products*std::min(channels,capacity)/channels);
  auto rows=std::min({products,maximum,
      static_cast<std::int64_t>((available-extra)/row_bytes)});
  if (rows>=32) rows=(rows/32)*32;
  return rows;
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
    int channel_capacity,int exchange_parity)
    : context_(&context), n_alpha_(source.n_alpha_strings()),
      n_beta_(source.n_beta_strings()),
      n_orbitals_(source.n_orbitals()),
      n_electrons_(source.alpha_connections().n_occupied()+
                   source.beta_connections_view().n_occupied()),
      n_channels_(static_cast<int>(source.channel_kernel().rows())),
      channel_capacity_(channel_capacity),
      exchange_parity_(exchange_parity),
      shared_spin_graph_(source.shared_spin_connections()),
      kernel_(context, static_cast<std::size_t>(n_channels_)*n_channels_),
      effective_one_(context,n_channels_),
      alpha_(std::make_shared<SpinGraph>(context, source.alpha_connections(), n_channels_)),
      workspace_(std::make_shared<Workspace>()),scattered_(workspace_->scattered),
      transformed_(workspace_->transformed),right_scattered_(workspace_->right_scattered),
      channel_adjoint_(workspace_->channel_adjoint) {
  if (channel_capacity <= 0 || n_channels_ <= 0 ||
      source.channel_kernel().cols() != n_channels_ || exchange_parity < -1 ||
      exchange_parity > 1 || (exchange_parity && !shared_spin_graph_))
    throw std::invalid_argument("invalid CUDA direct-CI channel dimensions");
  context.copy_to_device(kernel_.data(), source.channel_kernel().data(),
                         kernel_.size()*sizeof(double));
  beta_=shared_spin_graph_ ? alpha_ :
      std::make_shared<SpinGraph>(context, source.beta_connections_view(), n_channels_);
}

DirectCiSigma::DirectCiSigma(Context& context,const DirectCiSigma& source)
    : context_(&context),n_alpha_(source.n_alpha_),n_beta_(source.n_beta_),
      n_orbitals_(source.n_orbitals_),n_electrons_(source.n_electrons_),
      n_channels_(source.n_channels_),channel_capacity_(source.channel_capacity_),
      exchange_parity_(source.exchange_parity_),
      shared_spin_graph_(source.shared_spin_graph_),
      kernel_(context,source.kernel_.size()),effective_one_(context,source.effective_one_.size()),
      alpha_(source.alpha_),beta_(source.beta_),workspace_(source.workspace_),
      scattered_(workspace_->scattered),transformed_(workspace_->transformed),
      right_scattered_(workspace_->right_scattered),channel_adjoint_(workspace_->channel_adjoint) {
  if (&context!=source.context_)
    throw std::invalid_argument("shared CI links require the same CUDA context");
}

std::size_t DirectCiSigma::resident_bytes() const noexcept {
  return (kernel_.size()+effective_one_.size())*sizeof(double) +
         alpha_->resident_bytes() +
         (shared_spin_graph_ ? 0 : beta_->resident_bytes()) +
      sizeof(double)*(scattered_.rows()*scattered_.cols()+
          transformed_.rows()*transformed_.cols()+
          right_scattered_.rows()*right_scattered_.cols()+
          channel_adjoint_.rows()*channel_adjoint_.cols());
}

void DirectCiSigma::reset_workspace() const {
  ++workspace_->generation;
  workspace_->products=0;
  scattered_=Matrix<double>{};
  transformed_=Matrix<double>{};
  right_scattered_=Matrix<double>{};
  channel_adjoint_=Matrix<double>{};
}

void DirectCiSigma::prepare(int block_width) const {
  if (block_width<=0) throw std::invalid_argument("invalid CUDA sigma block width");
  const auto products=contraction_rows(block_width);
  if (workspace_->products==products && scattered_.rows()>0 &&
      transformed_.rows()==scattered_.rows() && transformed_.cols()==n_channels_) return;
  reset_workspace();
  const std::size_t available=context_->device_byte_budget()-
      context_->counters().current_device_bytes;
  const std::size_t per_channel=2*static_cast<std::size_t>(products)*sizeof(double);
  int tile=static_cast<int>(std::min<std::size_t>(
      {static_cast<std::size_t>(n_channels_),static_cast<std::size_t>(channel_capacity_),
       available/per_channel}));
  tile=std::max<std::int64_t>(tile,(n_channels_+products-1)/products);
  if (static_cast<std::size_t>(tile)>available/per_channel)
    throw std::runtime_error("CUDA direct-CI budget cannot hold one complete channel row");
  if (tile<1)
    throw std::runtime_error("CUDA direct-CI budget cannot hold two single-channel slabs");
  // The ordinary action retains these panels for graph replay while its CI
  // caller grows other vectors. Keep that persistent-cache plan separate
  // from the fully consumed temporary adjoint panels below.
  auto rows=std::min(products,std::max<std::int64_t>(1,products*tile/n_channels_));
  constexpr int warp_rows=32;
  if (rows>=warp_rows) rows=(rows/warp_rows)*warp_rows;
  if ((static_cast<std::size_t>(rows)*n_channels_+
       static_cast<std::size_t>(products)*tile)*sizeof(double)>available)
    throw std::runtime_error("CUDA direct-CI budget cannot hold a complete channel row");
  Matrix<double> scattered(*context_,rows,n_channels_);
  Matrix<double> transformed(*context_,rows,n_channels_);
  scattered_=std::move(scattered);
  transformed_=std::move(transformed);
  workspace_->products=products;
}

void DirectCiSigma::prepare_adjoint(int block_width,bool with_sigma,
                                   bool retain_channel_kernel) const {
  if (block_width<=0) throw std::invalid_argument("invalid CUDA adjoint block width");
  const auto products=contraction_rows(block_width);
  if (workspace_->products==products && scattered_.rows()>0 && scattered_.cols()==n_channels_ &&
      right_scattered_.rows()==scattered_.rows() &&
      right_scattered_.cols()==n_channels_ &&
      (!with_sigma || (transformed_.rows()==scattered_.rows() &&
                      transformed_.cols()==n_channels_)) &&
      (!retain_channel_kernel || channel_adjoint_.rows()==n_channels_)) return;

  reset_workspace();
  const std::size_t kernel_bytes=retain_channel_kernel ?
      static_cast<std::size_t>(n_channels_)*n_channels_*sizeof(double):0;
  const auto rows=source_tile_rows(*context_,products,n_channels_,channel_capacity_,
      with_sigma ? 3:2,kernel_bytes);
  Matrix<double> kernel_adjoint;
  if (retain_channel_kernel)
    kernel_adjoint=Matrix<double>(*context_,n_channels_,n_channels_);
  Matrix<double> scattered(*context_,rows,n_channels_);
  Matrix<double> right_scattered(*context_,rows,n_channels_);
  Matrix<double> transformed;
  // Every GEMM writes count-by-channel output, just like its source slabs.
  // The full-carrier channel-equivalent capacity was only a planning bound;
  // its unconsumed tail must not become a physical device allocation.
  if (with_sigma) transformed=Matrix<double>(*context_,rows,n_channels_);
  channel_adjoint_=std::move(kernel_adjoint);
  scattered_=std::move(scattered);
  right_scattered_=std::move(right_scattered);
  transformed_=std::move(transformed);
  workspace_->products=products;
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
  const std::int64_t products=contraction_rows(static_cast<int>(coefficients.cols/n_beta_));
  prepare(static_cast<int>(coefficients.cols/n_beta_));
  if (!add) clear_image(*context_,image);
  const auto* alpha=alpha_->sources.data();
  const auto* beta=beta_->sources.data();

  // Each source row is generated once; all output channels are completed
  // before the same bounded storage is reused for the next row tile.
  const auto rows=scattered_.rows();
  for (std::int64_t begin=0;begin<products;begin+=rows) {
    const auto count=std::min(rows,products-begin);
    scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        coefficients,scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,
        exchange_parity_,true);
    check_cuda(cudaGetLastError(),"CUDA direct-CI channel scatter");
    const linalg::ConstMatrixView q_view{scattered_.data(),count,n_channels_,count};
    const linalg::ConstMatrixView kernel_view{kernel_.data(),n_channels_,n_channels_,n_channels_};
    const linalg::MatrixView p_view{transformed_.data(),count,n_channels_,count};
    gemm(*context_,p_view,q_view,kernel_view,1.0,0.0,Operation::none,Operation::transpose);
    scatter_channel_adjoint<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        transformed_.data(),image,begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_);
    check_cuda(cudaGetLastError(), "CUDA direct-CI channel gather");
  }
  complete_exchange(image,true);
}

void DirectCiSigma::contract_adjoint(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,linalg::MatrixView* image,
    linalg::MatrixView channel_kernel_adjoint) const {
  const nvtx3::scoped_range range{"ci.sigma_adjoint"};
  const std::int64_t products=contraction_rows(static_cast<int>(left.cols/n_beta_));
  const auto rows=scattered_.rows();
  const auto* alpha=alpha_->sources.data();
  const auto* beta=beta_->sources.data();
  const bool same_source=!exchange_parity_ && left.data==right.data && left.ld==right.ld;
  if (image) clear_image(*context_,*image);
  bool first_adjoint=true;
  for (std::int64_t begin=0;begin<products;begin+=rows) {
    const auto count=std::min(rows,products-begin);
    scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        left,scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_,true);
    check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint left scatter");
    if (!same_source) {
      scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          right,right_scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_,false);
      check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint right scatter");
    }
    const linalg::ConstMatrixView left_channels{scattered_.data(),count,n_channels_,count};
    const linalg::ConstMatrixView right_channels{
        same_source ? scattered_.data():right_scattered_.data(),count,n_channels_,count};
    gemm(*context_,channel_kernel_adjoint,left_channels,right_channels,
        1.0,first_adjoint ? 0.0:1.0,Operation::transpose,Operation::none);
    first_adjoint=false;
    if (image) {
      const linalg::ConstMatrixView kernel_view{kernel_.data(),n_channels_,n_channels_,n_channels_};
      // The integral VJP has consumed the right channels. Reuse their slab
      // for K*left; neither live GEMM input aliases this destination.
      const linalg::MatrixView p_view{right_scattered_.data(),count,n_channels_,count};
      gemm(*context_,p_view,left_channels,kernel_view,1.0,0.0,Operation::none,Operation::transpose);
      scatter_channel_adjoint<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          right_scattered_.data(),*image,begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_);
      check_cuda(cudaGetLastError(),"CUDA direct-CI adjoint sigma gather");
    }
  }
  if (image) complete_exchange(*image,true);
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
  prepare_adjoint(block_width,false,true);
  contract_adjoint(response,state,&image,channel_adjoint_.view());
  pullback_channel_kernel(channel_adjoint_.view(),
      one_electron_adjoint,pair_kernel_adjoint);
}

void DirectCiSigma::apply_direction_with_adjoint(
    linalg::ConstMatrixView response,linalg::ConstMatrixView state,
    const DirectCiSigma& direction,linalg::MatrixView combined_image,
    linalg::MatrixView one_electron_adjoint,
    linalg::MatrixView pair_kernel_adjoint,
    linalg::ConstMatrixView component,linalg::MatrixView component_image,
    linalg::MatrixView component_one_adjoint,
    linalg::MatrixView component_pair_adjoint) const {
  const nvtx3::scoped_range range{"ci.sigma_direction_adjoint"};
  const int block_width=validate_adjoint_panels(response,state,n_alpha_,n_beta_);
  validate_output(combined_image,n_alpha_,response.cols,
      "invalid CUDA direct-CI combined directional image");
  validate_output(one_electron_adjoint,n_orbitals_,n_orbitals_,
      "invalid CUDA direct-CI directional one-electron adjoint");
  validate_output(pair_kernel_adjoint,n_channels_,n_channels_,
      "invalid CUDA direct-CI directional pair-kernel adjoint");
  if (direction.context_!=context_ || direction.n_alpha_!=n_alpha_ ||
      direction.n_beta_!=n_beta_ || direction.n_orbitals_!=n_orbitals_ ||
      direction.n_channels_!=n_channels_ || direction.n_electrons_!=n_electrons_ ||
      direction.exchange_parity_!=exchange_parity_ ||
      direction.alpha_!=alpha_ || direction.beta_!=beta_)
    throw std::invalid_argument("CUDA direct-CI direction must share the spin graph");
  if (combined_image.data==response.data || combined_image.data==state.data ||
      one_electron_adjoint.data==response.data ||
      one_electron_adjoint.data==state.data ||
      pair_kernel_adjoint.data==response.data ||
      pair_kernel_adjoint.data==state.data ||
      combined_image.data==one_electron_adjoint.data ||
      combined_image.data==pair_kernel_adjoint.data ||
      one_electron_adjoint.data==pair_kernel_adjoint.data)
    throw std::invalid_argument("CUDA direct-CI directional outputs alias another view");
  Matrix<double> component_kernel;
  if (component.data) {
    validate_adjoint_panels(component,state,n_alpha_,n_beta_);
    validate_output(component_image,n_alpha_,response.cols,"invalid CI component image");
    validate_output(component_one_adjoint,n_orbitals_,n_orbitals_,"invalid CI component one adjoint");
    validate_output(component_pair_adjoint,n_channels_,n_channels_,"invalid CI component pair adjoint");
    component_kernel=Matrix<double>(*context_,n_channels_,n_channels_);
    clear_image(*context_,component_image);
  }
  prepare_adjoint(block_width,true,true);
  const std::int64_t products=contraction_rows(block_width);
  const auto rows=scattered_.rows();
  const auto* alpha=alpha_->sources.data();
  const auto* beta=beta_->sources.data();
  clear_image(*context_,combined_image);
  bool first_adjoint=true;
  for (std::int64_t begin=0;begin<products;begin+=rows) {
    const auto count=std::min(rows,products-begin);
    scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        response,scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_,true);
    check_cuda(cudaGetLastError(),"CUDA direct-CI directional response scatter");
    scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        state,right_scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_,false);
    check_cuda(cudaGetLastError(),"CUDA direct-CI directional state scatter");
    const linalg::ConstMatrixView response_channels{scattered_.data(),count,n_channels_,count};
    const linalg::ConstMatrixView state_channels{right_scattered_.data(),count,n_channels_,count};
    gemm(*context_,channel_adjoint_.view(),response_channels,state_channels,
        1.0,first_adjoint ? 0.0:1.0,Operation::transpose,Operation::none);
    first_adjoint=false;
    const linalg::MatrixView output{transformed_.data(),count,n_channels_,count};
    const linalg::ConstMatrixView fixed_kernel{kernel_.data(),n_channels_,n_channels_,n_channels_};
    gemm(*context_,output,response_channels,fixed_kernel,
        1.0,0.0,Operation::none,Operation::transpose);
    if (component.data) {
      // The total response channels have been consumed. Reuse that slab;
      // the common state scatter is still unweighted for the adjoint GEMM.
      scatter_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          component,scattered_.data(),begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_,true);
      check_cuda(cudaGetLastError(),"CUDA direct-CI component scatter");
      gemm(*context_,component_kernel.view(),response_channels,state_channels,
          1.0,begin==0 ? 0.0:1.0,Operation::transpose,Operation::none);
    }
    if (exchange_parity_) {
      weight_channels<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          right_scattered_.data(),begin,count,n_alpha_,n_beta_,n_channels_,exchange_parity_);
      check_cuda(cudaGetLastError(),"CUDA directional state exchange weights");
    }
    const linalg::ConstMatrixView directional_kernel{
        direction.kernel_.data(),n_channels_,n_channels_,n_channels_};
    gemm(*context_,output,state_channels,directional_kernel,
        1.0,1.0,Operation::none,Operation::transpose);
    scatter_channel_adjoint<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
        transformed_.data(),combined_image,begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_);
    check_cuda(cudaGetLastError(),"CUDA direct-CI combined directional gather");
    if (component.data) {
      gemm(*context_,output,response_channels,fixed_kernel,
          1.0,0.0,Operation::none,Operation::transpose);
      scatter_channel_adjoint<<<grid_size(count*n_channels_),256,0,context_->stream()>>>(
          transformed_.data(),component_image,begin,count,n_alpha_,n_beta_,alpha,beta,n_channels_,exchange_parity_);
      check_cuda(cudaGetLastError(),"CUDA direct-CI component gather");
    }
  }
  complete_exchange(combined_image,true);
  pullback_channel_kernel(channel_adjoint_.view(),
      one_electron_adjoint,pair_kernel_adjoint);
  if (component.data) {
    complete_exchange(component_image,true);
    pullback_channel_kernel(component_kernel.view(),component_one_adjoint,component_pair_adjoint);
  }
}

void DirectCiSigma::diagonal(linalg::MatrixView values) const {
  validate_output(values,n_alpha_,n_beta_,
      "invalid CUDA direct-CI diagonal output");
  const auto* alpha=alpha_->sources.data();
  const auto* beta=beta_->sources.data();
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
  const auto* alpha=alpha_->sources.data();
  const auto* beta=beta_->sources.data();
  const std::int64_t products=contraction_rows(static_cast<int>(coefficients.cols/n_beta_));
  apply_orbital_generator<<<grid_size(products),256,0,context_->stream()>>>(
      coefficients,orbital_generator,image,products,n_alpha_,n_beta_,
          n_orbitals_,alpha,beta,exchange_parity_);
  check_cuda(cudaGetLastError(),"CUDA direct-CI orbital generator action");
  complete_exchange(image,false);
}

void DirectCiSigma::complete_exchange(linalg::MatrixView image,bool average) const {
  if (!exchange_parity_) return;
  const auto rows=contraction_rows(static_cast<int>(image.cols/n_beta_));
  complete_exchange_image<<<grid_size(rows),256,0,context_->stream()>>>(
      image,rows,n_alpha_,n_beta_,exchange_parity_,average);
  check_cuda(cudaGetLastError(),"CUDA complete certified exchange image");
}
}  // namespace xmvb::cuda
