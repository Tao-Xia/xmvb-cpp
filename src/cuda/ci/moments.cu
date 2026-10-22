#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

namespace xmvb::cuda {
namespace {
constexpr int threads=256;
constexpr std::int64_t moment_chunk=4096;

struct StreamEvent {
  int source,local,slot,created,annihilated;
  double sign;
};
struct EventBatch {
  int begin,end;
  std::vector<int> sources;
};

// The metadata may cover the beta links in this small panel. Numerical
// spectator storage below is bounded by the panel width.
std::pair<std::vector<StreamEvent>,std::vector<EventBatch>> beta_event_batches(
    const mcscf::SpinConnections& graph,int first,int count) {
  std::vector<StreamEvent> events;
  for (int local=0;local<count;++local)
    for (const auto& link:graph.singles(first+local))
      events.push_back({link.source,local,0,link.created,link.annihilated,link.sign});
  std::stable_sort(events.begin(),events.end(),[](const auto& a,const auto& b) {
    return a.source<b.source;
  });
  std::vector<EventBatch> batches;
  for (std::size_t begin=0;begin<events.size();) {
    EventBatch batch{static_cast<int>(begin),0,{}};
    std::size_t end=begin;
    for (;end<events.size();++end) {
      if (end==begin || events[end].source!=events[end-1].source) {
        if (batch.sources.size()==static_cast<std::size_t>(count)) break;
        batch.sources.push_back(events[end].source);
      }
      events[end].slot=static_cast<int>(batch.sources.size())-1;
    }
    batch.end=static_cast<int>(end);
    batches.push_back(std::move(batch));
    begin=end;
  }
  return {std::move(events),std::move(batches)};
}

struct AddMoments {
  __device__ double2 operator()(double2 a,double2 b) const {
    return make_double2(a.x+b.x,a.y+b.y);
  }
};

// One packed orbital pair reads only the current target beta panel. The
// diagonal combines both spins; off-diagonals use alpha links only because
// beta source columns are delivered by a separate bounded gather.
__global__ void reduce_streamed_alpha(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,const int* alpha,const int* beta,
    int n_alpha,int n_beta,int first,bool alpha_only,double scale,
    bool transpose,linalg::MatrixView accumulator) {
  using Reduce=cub::BlockReduce<double2,threads>;
  __shared__ typename Reduce::TempStorage storage;
  const int channel=blockIdx.x;
  int high=0;
  while ((high+1)*(high+2)/2<=channel) ++high;
  const int low=channel-high*(high+1)/2;
  const int diagonal=high*(high+1)/2+high;
  double2 sum=make_double2(0.0,0.0);
  const std::int64_t size=static_cast<std::int64_t>(n_alpha)*left.cols;
  for (std::int64_t begin=static_cast<std::int64_t>(blockIdx.y)*moment_chunk;
       begin<size;begin+=static_cast<std::int64_t>(gridDim.y)*moment_chunk)
  for (std::int64_t index=begin+threadIdx.x;index<size && index<begin+moment_chunk;index+=threads) {
    const int a=static_cast<int>(index%n_alpha);
    const int local=static_cast<int>(index/n_alpha);
    const double weight=left.data[a+left.ld*local];
    if (weight==0.0) continue;
    if (high==low) {
      const int occ_alpha=alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]!=0;
      const int occ_beta=alpha_only ? 0:
          beta[static_cast<std::int64_t>(diagonal)*n_beta+first+local]!=0;
      sum.x+=weight*(occ_alpha+occ_beta)*right.data[a+right.ld*local];
    } else {
      const int signed_source=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
      if (!signed_source) continue;
      const int source=signed_source>0 ? signed_source-1:-signed_source-1;
      const double moment=weight*(signed_source>0 ? 1.0:-1.0)*
          right.data[source+right.ld*local];
      if (alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]) sum.x+=moment;
      else sum.y+=moment;
    }
  }
  sum=Reduce(storage).Reduce(sum,AddMoments{});
  if (threadIdx.x==0) {
    const int upper_row=transpose ? low:high;
    const int upper_col=transpose ? high:low;
    atomicAdd(accumulator.data+upper_row+accumulator.ld*upper_col,scale*sum.x);
    if (high!=low) {
      const int lower_row=transpose ? high:low;
      const int lower_col=transpose ? low:high;
      atomicAdd(accumulator.data+lower_row+accumulator.ld*lower_col,scale*sum.y);
    }
  }
}

__global__ void reduce_streamed_beta(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,const StreamEvent* events,int begin,
    double scale,bool transpose,linalg::MatrixView accumulator) {
  using Reduce=cub::BlockReduce<double,threads>;
  __shared__ typename Reduce::TempStorage storage;
  const auto e=events[begin+blockIdx.x];
  double sum=0.0;
  for (int a=threadIdx.x;a<left.rows;a+=threads)
    sum+=left.data[a+left.ld*e.local]*right.data[a+right.ld*e.slot];
  sum=Reduce(storage).Sum(sum);
  if (threadIdx.x==0) {
    const int row=transpose ? e.annihilated:e.created;
    const int col=transpose ? e.created:e.annihilated;
    atomicAdd(accumulator.data+row+accumulator.ld*col,scale*e.sign*sum);
  }
}

__global__ void finish_streamed(linalg::ConstMatrixView raw,
    linalg::MatrixView output,bool symmetric_sum,bool add) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=output.rows*output.cols) return;
  const int row=index%output.rows,col=index/output.rows;
  if (row>col) {
    if (!add) output.data[row+output.ld*col]=0.0;
    return;
  }
  double value=raw.data[row+raw.ld*col];
  if (symmetric_sum && row!=col) value+=raw.data[col+raw.ld*row];
  if (symmetric_sum && row==col) value*=2.0;
  if (add) output.data[row+output.ld*col]+=value;
  else output.data[row+output.ld*col]=value;
}

__global__ void apply_streamed_alpha(linalg::ConstMatrixView input,
    linalg::ConstMatrixView generator,const int* alpha,const int* beta,
    int n_alpha,int n_beta,int first,int orbitals,bool alpha_only,
    linalg::MatrixView output) {
  const std::int64_t index=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (index>=static_cast<std::int64_t>(n_alpha)*input.cols) return;
  const int a=static_cast<int>(index%n_alpha),local=static_cast<int>(index/n_alpha);
  const int b=first+local;
  const double self=input.data[a+input.ld*local];
  double value=0.0;
  for (int p=0;p<orbitals;++p) {
    const int diagonal=p*(p+1)/2+p;
    const int occupied=(alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]!=0)+
        (alpha_only ? 0:(beta[static_cast<std::int64_t>(diagonal)*n_beta+b]!=0));
    value+=occupied*generator.data[p+generator.ld*p]*self;
  }
  for (int high=1;high<orbitals;++high) {
    const int diagonal=high*(high+1)/2+high;
    const bool alpha_high=alpha[static_cast<std::int64_t>(diagonal)*n_alpha+a]!=0;
    for (int low=0;low<high;++low) {
      const int channel=high*(high+1)/2+low;
      const int signed_source=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
      if (!signed_source) continue;
      const int source=signed_source>0 ? signed_source-1:-signed_source-1;
      const int created=alpha_high ? high:low,annihilated=alpha_high ? low:high;
      value+=(signed_source>0 ? 1.0:-1.0)*
          generator.data[created+generator.ld*annihilated]*
          input.data[source+input.ld*local];
    }
  }
  output.data[a+output.ld*local]=value;
}

__global__ void apply_streamed_beta(linalg::ConstMatrixView input,
    linalg::ConstMatrixView generator,const StreamEvent* events,int begin,
    linalg::MatrixView output) {
  const int a=blockIdx.y*blockDim.x+threadIdx.x;
  if (a>=output.rows) return;
  const auto e=events[begin+blockIdx.x];
  const double factor=e.sign*generator.data[e.created+generator.ld*e.annihilated];
  if (factor!=0.0)
    atomicAdd(output.data+a+output.ld*e.local,
        factor*input.data[a+input.ld*e.slot]);
}

// One packed channel, both E_pq orientations. A partition owns beta
// columns and visits their contiguous alpha rows. The beta link and the
// representative triangle boundary are decoded once per column, not once
// per product and orbital channel. No product-sized index map is needed.
__global__ void reduce_moments(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right,const int* alpha,const int* beta,
    int n_alpha,int n_beta,int parity,double2* partials) {
  using Reduce=cub::BlockReduce<double2,threads>;
  __shared__ typename Reduce::TempStorage scratch;
  const int channel=blockIdx.x;
  int high=0;
  while ((high+1)*(high+2)/2<=channel) ++high;
  const int low=channel-high*(high+1)/2;
  const int diagonal=high*(high+1)/2+high;
  double2 sum=make_double2(0.0,0.0);
  for (std::int64_t column=blockIdx.y;column<right.cols;column+=gridDim.y) {
    const int b=static_cast<int>(column%n_beta);
    const int sb=beta[static_cast<std::int64_t>(channel)*n_beta+b];
    const bool beta_upper=beta[static_cast<std::int64_t>(diagonal)*n_beta+b]!=0;
    for (int a=(parity ? b:0)+threadIdx.x;a<n_alpha;a+=threads) {
      const int sa=alpha[static_cast<std::int64_t>(channel)*n_alpha+a];
      if (!sa && !sb) continue;
      const double weight=(parity && a!=b ? 2.0:1.0)*left.data[a+left.ld*column];
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
            right.data[a+right.ld*(column-b+abs(sb)-1)];
        if (beta_upper) upper+=value;
        else lower+=value;
      }
      sum.x+=weight*upper;
      sum.y+=weight*lower;
    }
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
  const int partitions=static_cast<int>(std::min<std::int64_t>(workspace_->moment_partitions,
      right.cols));
  const auto count=static_cast<std::size_t>(n_channels_)*partitions;
  Buffer<double2> partials(*context_,count);
  for (std::size_t i=0;i<left.size();++i) {
    reduce_moments<<<dim3(n_channels_,partitions),threads,0,context_->stream()>>>(
        left[i],right,alpha_->sources.data(),beta_->sources.data(),n_alpha_,n_beta_,
        exchange_parity_,partials.data());
    check_cuda(cudaGetLastError(),"CUDA direct link generator moments");
    const auto cells=static_cast<std::int64_t>(n_orbitals_)*n_orbitals_;
    store_moments<<<static_cast<unsigned int>((cells+threads-1)/threads),threads,0,
        context_->stream()>>>(partials.data(),partitions,outputs[i]);
    check_cuda(cudaGetLastError(),"CUDA generator moment partial reduction");
  }
}

void DirectCiSigma::generator_adjoint_streamed(int first,
    std::span<const linalg::ConstMatrixView> lefts,
    std::span<const CiStreamedGeneratorRequest> requests,
    std::span<const linalg::MatrixView> outputs,
    bool symmetric_sum,bool spin_exchange_reduced,bool add) const {
  const nvtx3::scoped_range range{"ci.generator_adjoint_streamed"};
  if (lefts.empty() || lefts.size()!=outputs.size() || requests.empty() ||
      first<0 || lefts.front().cols<=0 || lefts.front().cols>n_beta_ ||
      first>n_beta_-lefts.front().cols ||
      (spin_exchange_reduced && (!exchange_parity_ || !shared_spin_graph_)))
    throw std::invalid_argument("invalid CUDA streamed generator panel");
  const int count=static_cast<int>(lefts.front().cols);
  for (std::size_t i=0;i<lefts.size();++i) {
    const auto left=lefts[i];
    const auto output=outputs[i];
    if (!left.data || left.rows!=n_alpha_ || left.cols!=count || left.ld<left.rows ||
        !output.data || output.rows!=n_orbitals_ || output.cols!=n_orbitals_ ||
        output.ld<output.rows || output.data==left.data)
      throw std::invalid_argument("invalid CUDA streamed generator left or output");
  }
  for (const auto& request:requests)
    if (!request.read || !std::isfinite(request.scale))
      throw std::invalid_argument("invalid CUDA streamed generator reader");

  auto& context=*context_;
  std::vector<int> rows(n_alpha_),betas(count);
  std::iota(rows.begin(),rows.end(),0);
  std::iota(betas.begin(),betas.end(),first);
  Matrix<double> tile(context,n_alpha_,count);
  std::vector<Matrix<double>> reductions;
  reductions.reserve(lefts.size());
  const auto orbital_cells=static_cast<std::size_t>(n_orbitals_)*n_orbitals_;
  for (std::size_t i=0;i<lefts.size();++i) {
    reductions.emplace_back(context,n_orbitals_,n_orbitals_);
    check_cuda(cudaMemsetAsync(reductions.back().data(),0,orbital_cells*sizeof(double),
        context.stream()),"clear CUDA streamed generator reduction");
  }
  std::vector<StreamEvent> events;
  std::vector<EventBatch> batches;
  Buffer<StreamEvent> device_events;
  if (!spin_exchange_reduced) {
    auto plan=beta_event_batches(beta_->topology,first,count);
    events=std::move(plan.first);batches=std::move(plan.second);
    device_events=Buffer<StreamEvent>(context,events.size());
    context.copy_to_device(device_events.data(),events.data(),events.size()*sizeof(StreamEvent));
  }
  for (const auto& request:requests) {
    request.read(betas,rows,tile.view());
    for (std::size_t l=0;l<lefts.size();++l) {
      const auto cells=static_cast<std::int64_t>(n_alpha_)*count;
      const dim3 grid(n_channels_,static_cast<unsigned int>(std::min<std::int64_t>(
          65535,(cells+moment_chunk-1)/moment_chunk)));
      reduce_streamed_alpha<<<grid,threads,0,context.stream()>>>(
          lefts[l],tile.view(),alpha_->sources.data(),beta_->sources.data(),
          n_alpha_,n_beta_,first,spin_exchange_reduced,
          request.scale*(spin_exchange_reduced ? 2.0:1.0),
          request.transpose,reductions[l].view());
      check_cuda(cudaGetLastError(),"CUDA streamed alpha generator reduction");
    }
    if (spin_exchange_reduced) continue;
    for (const auto& batch:batches) {
      request.read(batch.sources,rows,tile.view().columns(0,batch.sources.size()));
      for (std::size_t l=0;l<lefts.size();++l) {
        reduce_streamed_beta<<<batch.end-batch.begin,threads,0,context.stream()>>>(
            lefts[l],tile.view(),device_events.data(),batch.begin,
            request.scale,request.transpose,reductions[l].view());
        check_cuda(cudaGetLastError(),"CUDA streamed beta generator reduction");
      }
    }
  }
  for (std::size_t l=0;l<lefts.size();++l) {
    finish_streamed<<<static_cast<unsigned int>((orbital_cells+threads-1)/threads),
        threads,0,context.stream()>>>(reductions[l].view(),outputs[l],symmetric_sum,add);
    check_cuda(cudaGetLastError(),"CUDA streamed generator finish");
  }
}

void DirectCiSigma::apply_one_body_generator_columns(int first,int count,
    const CiTileReader& columns,linalg::ConstMatrixView generator,
    linalg::MatrixView image,bool alpha_only) const {
  const nvtx3::scoped_range range{"ci.apply_generator_columns"};
  if (!columns || first<0 || count<=0 || count>n_beta_ || first>n_beta_-count ||
      !generator.data || generator.rows!=n_orbitals_ ||
      generator.cols!=n_orbitals_ || generator.ld<generator.rows ||
      !image.data || image.rows!=n_alpha_ || image.cols!=count || image.ld<image.rows ||
      image.data==generator.data)
    throw std::invalid_argument("invalid CUDA one-body generator columns");
  auto& context=*context_;
  std::vector<int> rows(n_alpha_),betas(count);
  std::iota(rows.begin(),rows.end(),0);
  std::iota(betas.begin(),betas.end(),first);
  Matrix<double> tile(context,n_alpha_,count);
  columns(betas,rows,tile.view());
  const auto cells=static_cast<std::size_t>(n_alpha_)*count;
  apply_streamed_alpha<<<static_cast<unsigned int>((cells+threads-1)/threads),
      threads,0,context.stream()>>>(tile.view(),generator,
      alpha_->sources.data(),beta_->sources.data(),n_alpha_,n_beta_,first,
      n_orbitals_,alpha_only,image);
  check_cuda(cudaGetLastError(),"CUDA one-body alpha generator columns");
  if (alpha_only) return;
  auto [events,batches]=beta_event_batches(beta_->topology,first,count);
  Buffer<StreamEvent> device_events(context,events.size());
  context.copy_to_device(device_events.data(),events.data(),events.size()*sizeof(StreamEvent));
  for (const auto& batch:batches) {
    columns(batch.sources,rows,tile.view().columns(0,batch.sources.size()));
    apply_streamed_beta<<<dim3(batch.end-batch.begin,(n_alpha_+threads-1)/threads),
        threads,0,context.stream()>>>(tile.view(),generator,device_events.data(),
        batch.begin,image);
    check_cuda(cudaGetLastError(),"CUDA one-body beta generator columns");
  }
}
} // namespace xmvb::cuda
