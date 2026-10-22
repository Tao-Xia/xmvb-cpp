#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/channel_product.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma_plan.hpp"

namespace xmvb::cuda {
namespace {

unsigned grid(std::size_t work) {
  return static_cast<unsigned>(std::min<std::size_t>(65535,(work+255)/256));
}

/** CPU plan maps only: no product-sized address table is uploaded. */
struct TileMap {
  const int *active,*betas,*groups,*supported,*halo_index,*column_index;
  const int *image,*local;
  const std::size_t* packed;
  int count,width,pairs,alpha,beta;
};

__global__ void scatter(TileMap map,linalg::ConstMatrixView input,
    const int* alpha,const int* beta,double* output) {
  const auto work=static_cast<std::size_t>(map.count)*map.width*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int row=i/map.pairs,p=i%map.pairs,t=row%map.count,v=row/map.count;
    const int channel=map.local[map.groups[t]*map.pairs+p];
    if (channel<0) continue;
    const int a=map.active[t],b=map.betas[t];
    const int sa=alpha[static_cast<std::size_t>(p)*map.alpha+a];
    const int sb=beta[static_cast<std::size_t>(p)*map.beta+b];
    const int h=map.halo_index[a],column=map.column_index[b];
    double value=0;
    if (sa && (abs(sa)-1!=a || map.supported[t]))
      value+=(sa>0 ? 1.0:-1.0)*input.data[
          map.halo_index[abs(sa)-1]+input.ld*(column*map.width+v)];
    if (sb && (abs(sb)-1!=b || map.supported[t]))
      value+=(sb>0 ? 1.0:-1.0)*input.data[
          h+input.ld*(map.column_index[abs(sb)-1]*map.width+v)];
    output[map.packed[row]+channel]=value;
  }
}

__global__ void pack_kernel(linalg::ConstMatrixView kernel,const int* channels,
    int count,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(output.rows)*count;
       i+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
    output.data[i]=kernel.data[i%output.rows+kernel.ld*channels[i/output.rows]];
}

__global__ void correct_image(double* values,int rows,int pairs,const int* diagonal,
    int count,int width,int targets,const int* images) {
  const auto work=static_cast<std::size_t>(count)*width*pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int p=i/(count*width),v=(i/count)%width,t=diagonal[i%count];
    values[images[t+v*targets]+static_cast<std::size_t>(p)*rows]*=0.5;
  }
}

// The small Gram is a cuBLAS product. Only the exact exchange fixed-point
// correction and global packed-channel placement require custom arithmetic.
__global__ void add_gram(linalg::ConstMatrixView gram,linalg::MatrixView total,
    const int* channels,const double* left,const double* right,
    const int* diagonal,const int* images,const std::size_t* packed,
    int diagonals,int width,int targets,int begin,int size,bool exchange) {
  const int k=gram.rows;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(k)*k;
       i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int p=i%k,q=i/k;
    double value=(exchange ? 2.0:1.0)*gram.data[p+gram.ld*q];
    if (exchange) for (int v=0;v<width;++v) for (int d=0;d<diagonals;++d) {
      const int row=diagonal[d]+v*targets,image=images[row];
      if (image>=begin && image<begin+size)
        value-=left[packed[row]+p]*right[packed[row]+q];
    }
    total.data[channels[p]+total.ld*channels[q]]+=value;
  }
}

/** Apply E_tile^T once while its channels are live. The input halo is dead
 * after the forward scatters and becomes the output accumulator. Each signed
 * link is visited once, as in the CPU incoming-link gather and the original
 * device source scatter; no output-column-by-channel rediscovery is needed.
 */
__global__ void scatter_image(TileMap map,const int* alpha,const int* beta,
    const double* image,linalg::MatrixView output,bool resident) {
  const auto work=static_cast<std::size_t>(map.count)*map.width*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int t=i%map.count,v=(i/map.count)%map.width,p=i/(map.count*map.width);
    const int a=map.active[t],b=map.betas[t];
    const double value=image[map.image[t+v*map.count]+
        static_cast<std::size_t>(p)*map.count*map.width];
    const int sa=alpha[static_cast<std::size_t>(p)*map.alpha+a];
    const int sb=beta[static_cast<std::size_t>(p)*map.beta+b];
    if (sa) atomicAdd(output.data+(resident ? abs(sa)-1:map.halo_index[abs(sa)-1])+
        output.ld*(resident ? v*map.beta+b:map.column_index[b]*map.width+v),
        (sa>0 ? value:-value));
    if (sb) atomicAdd(output.data+(resident ? a:map.halo_index[a])+
        output.ld*(resident ? v*map.beta+abs(sb)-1:map.column_index[abs(sb)-1]*map.width+v),
        (sb>0 ? value:-value));
  }
}

/** A consumer now reads the already contracted halo; it never revisits E. */
__global__ void gather(TileMap map,const int* rows,int nrows,
    linalg::ConstMatrixView halo,linalg::MatrixView output,int selected,bool add) {
  const auto work=static_cast<std::size_t>(nrows)*output.cols;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int r=i%nrows,column=i/nrows,v=column%map.width;
    const int h=map.halo_index[rows[r]],b=selected<0 ? column/map.width:selected;
    const double value=h<0 ? 0.0:halo.data[h+halo.ld*(b*map.width+v)];
    auto& target=output.data[r+output.ld*column];
    if (add) target+=value;else target=value;
  }
}

template<class T> Buffer<T> upload(Context& context,std::span<const T> values) {
  Buffer<T> result(context,values.size());
  context.copy_to_device(result.data(),values.data(),values.size_bytes());
  return result;
}

__global__ void read_carrier(linalg::ConstMatrixView source,const int* rows,const int* betas,
    int beta,int width,linalg::MatrixView output) {
  const auto work=static_cast<std::size_t>(output.rows)*output.cols;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const auto row=i%output.rows,col=i/output.rows;
    output.data[row+output.ld*col]=source.data[rows[row]+source.ld*((col%width)*beta+betas[col/width])];
  }
}

struct CarrierIndices {
  Context& context;
  std::vector<int> rows,columns;
  Buffer<int> indices;
  void bind(std::span<const int> rr,std::span<const int> cc) {
    if (std::equal(rr.begin(),rr.end(),rows.begin(),rows.end()) &&
        std::equal(cc.begin(),cc.end(),columns.begin(),columns.end())) return;
    rows.assign(rr.begin(),rr.end());columns.assign(cc.begin(),cc.end());
    std::vector<int> packed=rows;packed.insert(packed.end(),columns.begin(),columns.end());
    indices=Buffer<int>();indices=upload<int>(context,packed);
  }
};

} // namespace

void DirectCiSigma::apply_supported_responses(std::span<const CiCarrierRequest> requests,
    linalg::ConstMatrixView state,const mcscf::ProductSupport& support,std::size_t bytes,bool physical_image) const {
  if (state.rows!=n_alpha_ || state.cols<=0 || state.cols%n_beta_ || state.ld<state.rows)
    throw std::invalid_argument("invalid supported state carrier");
  auto& c=*context_;
  const int width=blas_int(state.cols/n_beta_);
  CarrierIndices indices{c};
  const auto reader=[&](linalg::ConstMatrixView source)->CiTileReader {
    if (source.rows!=state.rows || source.cols!=state.cols || source.ld<source.rows)
      throw std::invalid_argument("invalid supported response carrier");
    return [&,source](auto betas,auto rows,linalg::MatrixView output) {
      indices.bind(rows,betas);
      read_carrier<<<grid(output.rows*output.cols),256,0,c.stream()>>>(source,indices.indices.data(),
          indices.indices.data()+rows.size(),n_beta_,width,output);
    };
  };
  std::vector<CiTileRequest> tiles;
  tiles.reserve(requests.size());
  for (const auto& request:requests) {
    if (request.image.data==request.response.data || request.image.data==state.data)
      throw std::invalid_argument("supported sigma image aliases a live input");
    if (request.image.data) {
      if (request.image.rows!=state.rows || request.image.cols!=state.cols || request.image.ld<state.rows)
        throw std::invalid_argument("invalid supported image carrier");
      check_cuda(cudaMemset2DAsync(request.image.data,request.image.ld*sizeof(double),0,
          request.image.rows*sizeof(double),request.image.cols,c.stream()),"clear supported image carrier");
    }
    tiles.push_back({reader(request.response),{},request.direction,
        request.one_adjoint,request.pair_adjoint,request.image});
  }
  // Return an idle slab from the other executor before budgeting the halo.
  reset_workspace();
  const bool identical=requests.size()==1 && !requests.front().direction &&
      requests.front().response.data==state.data && requests.front().response.ld==state.ld;
  apply_tiled_responses(width,tiles,identical ? CiTileReader{}:reader(state),&support,bytes);
  // Unlike a structure projection, a resident endpoint promises a physical
  // image. Symmetrize the weighted exchange representative exactly once.
  if (physical_image) for (const auto& request:requests)
    if (request.image.data) complete_exchange(request.image,true);
}

void DirectCiSigma::apply_tiled_responses(int width,std::span<const CiTileRequest> requests,
    const CiTileReader& state,const mcscf::ProductSupport* support,
    std::size_t workspace_bytes,const CiColumnRelease& release) const {
  const nvtx3::scoped_range sweep_range{"ci.tile_sweep"};
  if (width<=0 || requests.empty()) throw std::invalid_argument("invalid device sigma tile sequence");
  bool collect=false,has_direction=false;
  for (const auto& request:requests) {
    const bool adjoint=request.one_adjoint.data!=nullptr;
    if (!request.response || (request.image && request.carrier_image.data) ||
        (!request.image && !request.carrier_image.data && !adjoint) ||
        adjoint!=(request.pair_adjoint.data!=nullptr) ||
        (request.direction && !state))
      throw std::invalid_argument("invalid device sigma tile consumers");
    if (adjoint && (request.one_adjoint.rows!=n_orbitals_ ||
        request.one_adjoint.cols!=n_orbitals_ || request.pair_adjoint.rows!=n_channels_ ||
        request.pair_adjoint.cols!=n_channels_))
      throw std::invalid_argument("invalid device sigma tile cotangent dimensions");
    if (request.direction && (request.direction->alpha_!=alpha_ ||
        request.direction->beta_!=beta_ || request.direction->context_!=context_ ||
        request.direction->exchange_parity_!=exchange_parity_))
      throw std::invalid_argument("direction requires the shared device spin graph and parity");
    collect=collect || adjoint; has_direction=has_direction || request.direction;
  }
  if (requests.size()>1 && !state)
    throw std::invalid_argument("multiple device responses require a common state");
  auto& ctx=*context_;
  const std::size_t n=n_orbitals_,pairs=n_channels_;
  const auto directions=std::count_if(requests.begin(),requests.end(),
      [](const auto& request) {return request.direction!=nullptr;});
  const auto envelopes=n*(n-1)*(n+1)*(n+2)/24;
  const auto kernel_bytes=(sizeof(double)*pairs*envelopes+
      sizeof(int)*(envelopes+pairs*n*(n-1)/2))*(1+directions);
  const bool restricted=support && support->has_channel_envelopes() && kernel_bytes<=workspace_bytes/8;
  if (restricted) workspace_bytes-=kernel_bytes;
  const mcscf::detail::ChannelSelectors selectors(n_channels_,n_orbitals_,restricted);
  const bool separate=static_cast<bool>(state) && (collect || has_direction || requests.size()>1);
  const int slabs=separate ? 3:2,packed_slabs=restricted ? (separate ? 2:1):0;
  mcscf::detail::SigmaTilePlan plan(alpha_->topology,beta_->topology,support,exchange_parity_,
      {width,n_channels_,slabs,workspace_bytes,packed_slabs},static_cast<bool>(release));
  mcscf::detail::ChannelLayout layout;
  std::vector<int> global,local(selectors.groups()*pairs,-1),selector_offsets(selectors.groups()+1);
  for (int group=0;group<selectors.groups();++group) {
    selector_offsets[group]=global.size();
    for (int p=0;p<selectors.channels(group);++p) {
      const int channel=selectors.global(group,p);
      global.push_back(channel);local[group*pairs+channel]=p;
    }
  }
  selector_offsets.back()=global.size();
  auto device_global=upload<int>(ctx,global),device_local=upload<int>(ctx,local);
  // Small orbital-space cotangents only; their size does not scale with U_alpha U_beta.
  std::vector<Matrix<double>> adjoints;
  adjoints.reserve(requests.size());
  for (const auto& request:requests) {
    adjoints.emplace_back(ctx,request.one_adjoint.data ? n_channels_:0,
        request.one_adjoint.data ? n_channels_:0);
    if (request.one_adjoint.data)
      check_cuda(cudaMemsetAsync(adjoints.back().data(),0,pairs*pairs*sizeof(double),ctx.stream()),
          "clear tiled channel cotangent");
  }
  Matrix<double> gram(ctx,collect ? n_channels_:0,collect ? n_channels_:0);
  std::vector<Matrix<double>> kernels;
  kernels.reserve(1+directions);
  // Reuse the same numerical device kernels. Host selectors contain no integral values.
  const auto pack=[&](const DirectCiSigma& action) {
    Matrix<double> panel(ctx,n_channels_,restricted ? global.size()-pairs:0);
    for (int g=1;g<selectors.groups();++g) if (selectors.channels(g)) {
      auto output=panel.view().columns(selector_offsets[g]-pairs,selectors.channels(g));
      pack_kernel<<<grid(output.rows*output.cols),256,0,ctx.stream()>>>(action.channel_kernel(),
          device_global.data()+selector_offsets[g],selectors.channels(g),output);
    }
    return panel;
  };
  kernels.push_back(pack(*this));
  std::vector<int> direction_ids(requests.size(),-1);
  for (std::size_t r=0;r<requests.size();++r) if (requests[r].direction) {
    direction_ids[r]=kernels.size();kernels.push_back(pack(*requests[r].direction));
  }
  Buffer<double> numerical;
  while (plan.next()) {
    const nvtx3::scoped_range tile_range{"ci.tile"};
    const int count=plan.active.size(),halo=plan.halo.size(),columns=plan.columns.size();
    layout.reset(selectors,plan.channel_groups,count,width);
    const auto packed_size=layout.entries(),channel_size=static_cast<std::size_t>(count)*width*pairs;
    const auto input_size=static_cast<std::size_t>(halo)*columns*width;
    const auto required=input_size+(separate ? 2:1)*packed_size+channel_size;
    if (required>numerical.size()) {
      numerical=Buffer<double>();numerical=Buffer<double>(ctx,required);
    }
    const auto input=linalg::MatrixView{numerical.data(),halo,static_cast<std::int64_t>(columns)*width,halo};
    double* response=numerical.data()+input_size;
    double* state_channels=separate ? response+packed_size:response;
    double* transformed=response+(separate ? 2:1)*packed_size;
    std::vector<int> metadata;
    const auto append=[&](std::span<const int> values) {
      const auto offset=metadata.size();metadata.insert(metadata.end(),values.begin(),values.end());return offset;
    };
    const auto active_offset=append(plan.active),halo_offset=append(plan.halo);
    const auto halo_index_offset=append(plan.halo_index),column_index_offset=append(plan.column_index);
    std::vector<int> betas(count),groups(count),supported(plan.active_support.begin(),plan.active_support.end());
    for (int i=0;i<count;++i) {
      betas[i]=plan.groups[plan.group_index[i]].beta;
      groups[i]=plan.channel_groups.empty() ? 0:plan.channel_groups[i];
    }
    const auto betas_offset=append(betas),groups_offset=append(groups),supported_offset=append(supported);
    std::vector<int> images(count*width);
    std::vector<std::size_t> packed(count*width);
    for (int row=0;row<count*width;++row) {images[row]=layout.image(row);packed[row]=layout.packed(row);}
    const auto image_offset=append(images);
    const auto diagonal_offset=append(plan.diagonal_rows);
    auto device=upload<int>(ctx,metadata);
    auto device_packed=upload<std::size_t>(ctx,packed);
    const auto ptr=[&](std::size_t offset) {return device.data()+offset;};
    const TileMap map{ptr(active_offset),ptr(betas_offset),ptr(groups_offset),ptr(supported_offset),
        ptr(halo_index_offset),ptr(column_index_offset),ptr(image_offset),device_local.data(),device_packed.data(),
        count,width,n_channels_,n_alpha_,n_beta_};
    const auto do_scatter=[&](const CiTileReader& reader,double* output) {
      {
        const nvtx3::scoped_range range{"ci.tile_read"};
        reader(plan.columns,plan.halo,input);
      }
      const nvtx3::scoped_range range{"ci.tile_scatter"};
      scatter<<<grid(channel_size),256,0,ctx.stream()>>>(map,input,alpha_->sources.data(),beta_->sources.data(),output);
    };
    bool state_ready=false;
    for (std::size_t r=0;r<requests.size();++r) {
      const auto& request=requests[r];
      do_scatter(request.response,response);
      if (separate && !state_ready) {do_scatter(state,state_channels);state_ready=true;}
      const auto multiply=[&](const DirectCiSigma& action,int kernel_id,const double* source,bool add) {
        const nvtx3::scoped_range range{"ci.tile_gemm"};
        for (int g=0;g<selectors.groups();++g) {
          const int size=layout.size(g),k=selectors.channels(g);
          if (!size) continue;
          const auto selected=g ? kernels[kernel_id].view().columns(selector_offsets[g]-pairs,k):action.channel_kernel();
          const linalg::ConstMatrixView channels{source+layout.offset(g),k,size,k};
          const linalg::MatrixView output{transformed+layout.begin(g),size,n_channels_,count*width};
          gemm(ctx,output,channels,selected,exchange_parity_ ? 2.0:1.0,add ? 1.0:0.0,
              Operation::transpose,Operation::transpose);
        }
      };
      if (request.one_adjoint.data) for (int g=0;g<selectors.groups();++g) {
        const int size=layout.size(g),k=selectors.channels(g);
        if (!size) continue;
        const nvtx3::scoped_range range{"ci.tile_gram"};
        const linalg::ConstMatrixView left{response+layout.offset(g),k,size,k},
            right{state_channels+layout.offset(g),k,size,k};
        const linalg::MatrixView tile{gram.data(),k,k,n_channels_};
        gemm(ctx,tile,left,right,1.0,0.0,Operation::none,Operation::transpose);
        add_gram<<<grid(static_cast<std::size_t>(k)*k),256,0,ctx.stream()>>>(tile,adjoints[r].view(),
            device_global.data()+selector_offsets[g],response,state_channels,ptr(diagonal_offset),
            ptr(image_offset),device_packed.data(),plan.diagonal_rows.size(),width,count,
            layout.begin(g),size,exchange_parity_!=0);
      }
      if (request.image || request.carrier_image.data) {
        multiply(*this,0,response,false);
        if (request.direction) multiply(*request.direction,direction_ids[r],state_channels,true);
        if (exchange_parity_ && !plan.diagonal_rows.empty())
          correct_image<<<grid(plan.diagonal_rows.size()*width*pairs),256,0,ctx.stream()>>>(
              transformed,count*width,n_channels_,ptr(diagonal_offset),plan.diagonal_rows.size(),
              width,count,ptr(image_offset));
        {
          const nvtx3::scoped_range range{"ci.tile_link_adjoint"};
          if (!request.carrier_image.data)
            check_cuda(cudaMemsetAsync(input.data,0,input_size*sizeof(double),ctx.stream()),
                "clear consumed halo for link adjoint");
          scatter_image<<<grid(channel_size),256,0,ctx.stream()>>>(map,alpha_->sources.data(),
              beta_->sources.data(),transformed,
              request.carrier_image.data ? request.carrier_image:input,request.carrier_image.data!=nullptr);
        }
        if (!request.image) continue;
        const CiTileGather gather_columns=[&](int source,std::span<const int> rows,
            linalg::MatrixView output,bool add,const int* borrowed_rows) {
          const bool whole=rows.empty();
          if (whole) rows=plan.halo;
          if (source < -1 || source>=columns || output.rows!=static_cast<std::int64_t>(rows.size()) ||
              output.cols!=(source<0 ? columns:1)*width || output.ld<output.rows)
            throw std::invalid_argument("invalid tiled sigma gather dimensions");
          if (rows.empty()) return;
          Buffer<int> device_rows;
          if (!whole && !borrowed_rows) {
            if (!std::is_sorted(rows.begin(),rows.end()) ||
                std::adjacent_find(rows.begin(),rows.end())!=rows.end() || rows.front()<0 || rows.back()>=n_alpha_)
              throw std::invalid_argument("tiled sigma gather requires sorted distinct alpha rows");
            device_rows=upload<int>(ctx,rows);
          }
          gather<<<grid(output.rows*output.cols),256,0,ctx.stream()>>>(map,
              whole ? ptr(halo_offset):borrowed_rows ? borrowed_rows:device_rows.data(),
              rows.size(),input,output,source,add);
        };
        const nvtx3::scoped_range range{"ci.tile_backproject"};
        request.image(plan.columns,plan.halo,gather_columns);
      }
    }
    if (release && !plan.released.empty()) release(plan.released);
    check_cuda(cudaGetLastError(),"device product-tile sigma sweep");
  }
  for (std::size_t r=0;r<requests.size();++r) if (requests[r].one_adjoint.data)
    pullback_channel_kernel(adjoints[r].view(),requests[r].one_adjoint,requests[r].pair_adjoint,gram.data());
}

} // namespace xmvb::cuda
