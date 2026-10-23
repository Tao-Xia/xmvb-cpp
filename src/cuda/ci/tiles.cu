#include "cuda/ci/sigma.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/upload.hpp"
#include "mcscf/ci/channel_product.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma_plan.hpp"

namespace xmvb::cuda {
namespace {

unsigned grid(std::size_t work) {
  return static_cast<unsigned>(std::min<std::size_t>(65535,(work+255)/256));
}

struct PackedGroup {
  std::size_t offset;
  int begin,channels;
  bool operator==(const PackedGroup&) const = default;
};

/** CPU plan maps only: no product-sized address table is uploaded. */
struct TileMap {
  const int *active,*betas,*groups,*supported,*halo_index,*column_index;
  const int* local;
  const PackedGroup* packed_groups;
  int count,width,pairs,alpha,beta;

  __device__ int group(int row) const { return groups ? groups[row%count]:0; }
  __device__ bool is_supported(int row) const {
    return groups ? group(row)==0:(!supported || supported[row%count]);
  }
  __device__ int halo_row(int row) const { return halo_index ? halo_index[row]:row; }
  __device__ int column(int col) const { return column_index ? column_index[col]:col; }
  __device__ std::size_t packed_row(int image,int group) const {
    if (!packed_groups) return static_cast<std::size_t>(image)*pairs;
    const auto& entry=packed_groups[group];
    return entry.offset+static_cast<std::size_t>(image-entry.begin)*entry.channels;
  }
};

__global__ void pack_kernel(linalg::ConstMatrixView kernel,const int* channels,
    int count,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(output.rows)*count;
       i+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
    output.data[i]=kernel.data[i%output.rows+kernel.ld*channels[i/output.rows]];
}


// Gram arithmetic belongs to cuBLAS; only packed-channel placement remains.
__global__ void add_gram(linalg::ConstMatrixView gram,linalg::MatrixView total,
    const int* channels) {
  const int k=gram.rows;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(k)*k;
       i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int p=i%k,q=i/k;
    total.data[channels[p]+total.ld*channels[q]]+=gram.data[p+gram.ld*q];
  }
}

struct GroupTile {
  std::size_t offset,scratch;
  int group,begin,size;
  std::size_t selected;
  int channels;
  bool operator==(const GroupTile&) const = default;
};

struct GroupedGemm {
  std::vector<cublasOperation_t> transa,transb;
  std::vector<int> m,n,k,lda,ldb,ldc,group_size;
  std::vector<double> alpha,beta;

  void reserve(std::size_t count) {
    transa.reserve(count);transb.reserve(count);
    m.reserve(count);n.reserve(count);k.reserve(count);
    lda.reserve(count);ldb.reserve(count);ldc.reserve(count);
    group_size.reserve(count);alpha.reserve(count);beta.reserve(count);
  }

  void prepare(std::span<const GroupTile> groups,int pairs,int image_rows,
      double scale,double previous,bool gram) {
    transa.clear();transb.clear();m.clear();n.clear();k.clear();
    lda.clear();ldb.clear();ldc.clear();group_size.clear();
    alpha.clear();beta.clear();
    for (const auto& group:groups) {
      transa.push_back(gram ? CUBLAS_OP_N:CUBLAS_OP_T);
      transb.push_back(CUBLAS_OP_T);
      m.push_back(gram ? group.channels:group.size);
      n.push_back(gram ? group.channels:pairs);
      k.push_back(gram ? group.size:group.channels);
      lda.push_back(group.channels);
      ldb.push_back(gram ? group.channels:pairs);
      ldc.push_back(gram ? group.channels:image_rows);
      group_size.push_back(1);
      alpha.push_back(scale);beta.push_back(previous);
    }
  }

  void run(Context& context,const double* const* left,
      const double* const* right,double* const* result) const {
    check_cublas(cublasSetPointerMode(context.blas(),CUBLAS_POINTER_MODE_HOST),
        "set grouped GEMM host scalars");
    check_cublas(cublasDgemmGroupedBatched(context.blas(),transa.data(),transb.data(),
        m.data(),n.data(),k.data(),alpha.data(),left,lda.data(),right,ldb.data(),
        beta.data(),result,ldc.data(),static_cast<int>(group_size.size()),
        group_size.data()),"cublasDgemmGroupedBatched");
  }
};

// Grouped GEMM pointers use the numerical panel's dedicated tiny scratch.
template<bool Gram>
__global__ void grouped_pointers(const GroupTile* groups,int first,int count,
    const double* source,const double* second,const double* full_kernel,
    double* output,const double** left,const double** right,double** result) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const auto entry=groups[first+i];
  left[i]=source+entry.offset;
  if constexpr (Gram) {
    right[i]=second+entry.offset;
    result[i]=output+entry.scratch;
  } else {
    right[i]=entry.group ? second+entry.selected:full_kernel;
    result[i]=output+entry.begin;
  }
}

// Each output entry has one owner. Iterate the original increasing group
// order so overlapping channel Grams never race or change accumulation order.
__global__ void add_grouped_gram(const double* scratch,linalg::MatrixView total,
    const GroupTile* groups,int first,int last,const int* local,int pairs) {
  const auto work=static_cast<std::size_t>(pairs)*pairs;
  for (auto entry=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       entry<work;entry+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int row=entry%pairs,column=entry/pairs;
    const auto target=row+total.ld*column;
    double value=total.data[target];
    for (int i=first;i<last;++i) {
      const auto group=groups[i];
      const int a=local[group.group*pairs+row];
      const int b=local[group.group*pairs+column];
      if (a>=0 && b>=0)
        value+=scratch[group.scratch+a+static_cast<std::size_t>(group.channels)*b];
    }
    total.data[target]=value;
  }
}


/** Inverse image rows are built once per symbolic tile. A panel visits only
 * its own rows, while its channel dimension remains the complete group. */
__global__ void scatter_row_panel(TileMap map,linalg::ConstMatrixView input,
    const int* alpha,const int* beta,const int* inverse,int first,int rows,
    std::size_t packed_origin,double* output) {
  const auto work=static_cast<std::size_t>(rows)*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int local_row=i/map.pairs,p=i%map.pairs;
    const int row=inverse ? inverse[first+local_row]:first+local_row;
    const int t=row%map.count,v=row/map.count;
    const int group=map.group(row),local=map.local[group*map.pairs+p];
    if (local<0) continue;
    const int a=map.active[t],b=map.betas[t];
    const int sa=alpha[static_cast<std::size_t>(p)*map.alpha+a];
    const int sb=beta[static_cast<std::size_t>(p)*map.beta+b];
    double value=0;
    if (sa && (abs(sa)-1!=a || map.is_supported(t)))
      value+=(sa>0 ? 1.0:-1.0)*input.data[
          map.halo_row(abs(sa)-1)+input.ld*(map.column(b)*map.width+v)];
    if (sb && (abs(sb)-1!=b || map.is_supported(t)))
      value+=(sb>0 ? 1.0:-1.0)*input.data[
          map.halo_row(a)+input.ld*(map.column(abs(sb)-1)*map.width+v)];
    output[map.packed_row(first+local_row,group)-packed_origin+local]=value;
  }
}

__global__ void weight_row_panel(TileMap map,const int* inverse,int first,int rows,
    std::size_t packed_origin,const double* source,double* weighted) {
  const auto work=static_cast<std::size_t>(rows)*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int local_row=i/map.pairs,p=i%map.pairs;
    const int row=inverse ? inverse[first+local_row]:first+local_row;
    const int t=row%map.count;
    const int group=map.group(row),local=map.local[group*map.pairs+p];
    if (local<0) continue;
    const auto position=map.packed_row(first+local_row,group)-packed_origin+local;
    weighted[position]=source[position]*(map.active[t]==map.betas[t] ? 0.5:1.0);
  }
}

__global__ void scatter_image_row_panel(TileMap map,const int* alpha,const int* beta,
    const int* inverse,int first,int rows,const double* image,
    linalg::MatrixView output,bool resident,bool exchange) {
  const auto work=static_cast<std::size_t>(rows)*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int local_row=i%rows,p=i/rows;
    const int row=inverse ? inverse[first+local_row]:first+local_row;
    const int t=row%map.count,v=row/map.count;
    const int a=map.active[t],b=map.betas[t];
    const double value=image[i]*(exchange && a==b ? 0.5:1.0);
    const int sa=alpha[static_cast<std::size_t>(p)*map.alpha+a];
    const int sb=beta[static_cast<std::size_t>(p)*map.beta+b];
    if (sa) atomicAdd(output.data+(resident ? abs(sa)-1:map.halo_row(abs(sa)-1))+
        output.ld*(resident ? v*map.beta+b:map.column(b)*map.width+v),
        (sa>0 ? value:-value));
    if (sb) atomicAdd(output.data+(resident ? a:map.halo_row(a))+
        output.ld*(resident ? v*map.beta+abs(sb)-1:map.column(abs(sb)-1)*map.width+v),
        (sb>0 ? value:-value));
  }
}

/** A consumer now reads the already contracted halo; it never revisits E. */
__global__ void gather(TileMap map,const int* halo_rows,const CiTileColumn* columns,
    linalg::ConstMatrixView halo,bool add) {
  const auto target=columns[blockIdx.x];
  const auto output=target.values;
  const int nrows=output.rows;
  const auto work=static_cast<std::size_t>(nrows)*output.cols;
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i%nrows,column=i/nrows,v=column%map.width;
    const int row=target.full_rows ? r:(target.rows ? target.rows:halo_rows)[r];
    const int h=map.halo_row(row),
        b=target.source<0 ? column/map.width:target.source;
    const double value=h<0 ? 0.0:halo.data[h+halo.ld*(b*map.width+v)];
    auto& target=output.data[r+output.ld*column];
    if (add) target+=value;else target=value;
  }
}

/** Same E^T halo as gather(), using the consumer's existing compact maps.
 * No host column descriptors, upload, or synchronization is needed. */
__global__ void gather_panel(TileMap map,const int* betas,int first,int source_first,int lane,const CiColumnMap* columns,
    linalg::MatrixView output,linalg::ConstMatrixView halo) {
  const int source=source_first+blockIdx.x;
  const auto column=columns[betas[source]-first];
  const auto work=static_cast<std::size_t>(column.size)*map.width;
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;i<work;
       i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i%column.size,v=i/column.size;
    const int a=column.rows ? column.rows[r]:r;
    const int h=map.halo_row(a);
    if (h>=0) output.data[column.offset+r+output.ld*(lane+v)]+=
        halo.data[h+halo.ld*(source*map.width+v)];
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

/** One bounded symbolic tile per clone family, independent of integrals and
 * amplitudes. Exact value comparison avoids support-pointer lifetime keys.
 * Multi-tile sweeps replace this slot rather than accumulating all halos.
 */
struct DirectCiSigma::TileWorkspace {
  /** Only scalar boundaries, never target rows, halos or channel layouts.
   * Keep one stream per algebraic source layout (primal, transition,
   * directional), each capped by the single-spin dimensions. */
  struct Boundaries {
    std::shared_ptr<const char> support;
    mcscf::detail::SigmaTileBudget budget{};
    bool release=false;
    std::vector<mcscf::detail::SigmaTileBoundary> entries;
    std::size_t workspace_peak=0,first_numeric=0;

    void clear() { entries.clear();workspace_peak=0;first_numeric=0; }

    void bind(const mcscf::ProductSupport* next_support,
        mcscf::detail::SigmaTileBudget next,bool next_release) {
      const auto identity=next_support ? next_support->identity():nullptr;
      if (support==identity && release==next_release && budget.width==next.width &&
          budget.pairs==next.pairs && budget.slabs==next.slabs &&
          budget.packed_slabs==next.packed_slabs &&
          budget.halo_slabs==next.halo_slabs) return;
      support=identity;budget=next;release=next_release;clear();
    }
  };
  std::array<Boundaries,3> boundaries;
  mcscf::detail::ChannelSelectors selectors;
  bool restricted;
  std::vector<int> global,local,offsets,metadata;
  std::vector<PackedGroup> packed_groups;
  Buffer<int> device_global,device_local;
  PinnedUpload device;
  std::size_t packed_offset=0;
  std::optional<mcscf::detail::SigmaTile> prepared;
  std::shared_ptr<const char> support_identity;
  mcscf::detail::SigmaTileBudget budget{};
  std::size_t payload=0,packed_entries=0;
  bool releases=false;

  bool matches(const mcscf::ProductSupport* support,mcscf::detail::SigmaTileBudget next,bool release) const {
    if (!prepared || support_identity!=(support ? support->identity():nullptr) ||
        budget.width!=next.width || budget.pairs!=next.pairs ||
        bool(budget.packed_slabs)!=bool(next.packed_slabs) || (release && !releases)) return false;
    // Slab counts affect admission, not symbolic paths. Reprice the exact
    // existing payload instead of rebuilding an identical complete schedule.
    // The row/halo/index terms are unchanged at the same RHS width.
    using Bytes=__int128_t;
    const Bytes channel_entries=Bytes(prepared->active.size())*next.width*next.pairs;
    const Bytes halo_entries=Bytes(prepared->halo.size())*prepared->columns.size()*next.width;
    const Bytes bytes=Bytes(payload)+sizeof(double)*(
        Bytes(next.packed_slabs-budget.packed_slabs)*packed_entries+
        Bytes(next.slabs-next.packed_slabs-budget.slabs+budget.packed_slabs)*channel_entries+
        Bytes(next.halo_slabs-budget.halo_slabs)*halo_entries);
    return bytes>=0 && bytes<=next.bytes;
  }

  TileWorkspace(Context& context,int pairs,int orbitals,bool restrict)
      :selectors(pairs,orbitals,restrict),restricted(restrict),
       local(selectors.groups()*pairs,-1),offsets(selectors.groups()+1),device(context) {
    for (int g=0;g<selectors.groups();++g) {
      offsets[g]=global.size();
      for (int p=0;p<selectors.channels(g);++p) {
        const int channel=selectors.global(g,p);
        global.push_back(channel);local[g*pairs+channel]=p;
      }
    }
    offsets.back()=global.size();
    device_global=upload<int>(context,global);device_local=upload<int>(context,local);
  }

  const int* indices() const noexcept { return reinterpret_cast<const int*>(device.data()); }
  const PackedGroup* packed() const noexcept {
    return packed_groups.empty() ? nullptr:
        reinterpret_cast<const PackedGroup*>(device.data()+packed_offset);
  }

  void bind(std::vector<int> indices,std::vector<PackedGroup> layout) {
    const nvtx3::scoped_range range{"ci.tile_index_upload"};
    if (metadata==indices && packed_groups==layout) return;
    metadata=std::move(indices);packed_groups=std::move(layout);
    const auto ints=metadata.size()*sizeof(int);
    packed_offset=(ints+alignof(PackedGroup)-1)/alignof(PackedGroup)*alignof(PackedGroup);
    const auto packed_bytes=packed_groups.size()*sizeof(PackedGroup);
    device.assign(packed_offset+packed_bytes,[&](std::span<std::byte> host) {
      if (ints) std::memcpy(host.data(),metadata.data(),ints);
      if (packed_bytes) std::memcpy(host.data()+packed_offset,packed_groups.data(),packed_bytes);
    });
  }
};

std::size_t DirectCiSigma::tile_resident_bytes() const noexcept {
  if (!workspace_->tiles) return 0;
  const auto& tile=*workspace_->tiles;
  return sizeof(int)*(tile.device_global.size()+tile.device_local.size())+tile.device.size();
}

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
  for (std::size_t r=0;r<requests.size();++r) {
    const auto& request=requests[r];
    const bool adjoint=request.one_adjoint.data!=nullptr;
    if (!request.response || (request.image && request.carrier_image.data) ||
        (!request.image && !request.carrier_image.data && !adjoint) ||
        adjoint!=(request.pair_adjoint.data!=nullptr) ||
        (request.direction && !state) ||
        (request.response_is_state && (r!=0 || !state)))
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
  const auto adjoint_count=std::count_if(requests.begin(),requests.end(),
      [](const auto& request) {return request.one_adjoint.data!=nullptr;});
  const auto alias=std::find_if(requests.begin(),requests.end(),
      [](const auto& request) {return request.response_is_state;});
  const bool separate=static_cast<bool>(state) &&
      (collect || has_direction || requests.size()>1);
  const int input_halos=static_cast<int>(requests.size())+
      (separate && alias==requests.end());
  const int output_halos=std::count_if(requests.begin(),requests.end(),
      [](const auto& request) {return static_cast<bool>(request.image);});
  const int halo_slabs=input_halos+output_halos;
  int device=0,l2_bytes=0;
  check_cuda(cudaGetDevice(&device),"query sigma panel device");
  check_cuda(cudaDeviceGetAttribute(&l2_bytes,cudaDevAttrL2CacheSize,device),
      "query sigma panel L2 size");
  const auto envelopes=n*(n-1)*(n+1)*(n+2)/24;
  const auto kernel_bytes=(sizeof(double)*pairs*envelopes+
      sizeof(int)*(envelopes+pairs*n*(n-1)/2))*(1+directions);
  const bool restricted=support && support->has_channel_envelopes() && kernel_bytes<=workspace_bytes/8;
  if (restricted) workspace_bytes-=kernel_bytes;
  if (!workspace_->tiles || workspace_->tiles->restricted!=restricted)
    workspace_->tiles=std::make_shared<TileWorkspace>(ctx,n_channels_,n_orbitals_,restricted);
  auto& symbols=*workspace_->tiles;
  const auto& selectors=symbols.selectors;
  const auto group_descriptor_bytes=static_cast<std::size_t>(selectors.groups())*sizeof(GroupTile);
  const bool allow_grouping=selectors.groups()>1;
  // Keep fixed panel scratch outside symbolic admission. The halo multiplier
  // is charged by SigmaTilePlan, while the physical check below also includes
  // the uploaded maps and selector tables.
  const auto replay_bytes=workspace_bytes;
  // Immutable selector tables are already resident in Context. This allowance
  // prices transient sweep storage, not a second copy of those shared tables.
  const auto fixed_map_bytes=2*sizeof(int)*(static_cast<std::size_t>(n_alpha_)+n_beta_)+
      (restricted ? static_cast<std::size_t>(selectors.groups())*sizeof(PackedGroup):0)+
      alignof(PackedGroup)-1;
  const auto panel_descriptor_bytes=allow_grouping ?
      group_descriptor_bytes+3*selectors.groups()*sizeof(double*):0;
  const auto fixed_numeric_bytes=sizeof(double)*pairs*pairs*(collect+adjoint_count)+
      panel_descriptor_bytes+fixed_map_bytes;
  if (fixed_numeric_bytes>=workspace_bytes)
    throw std::length_error("sigma panel fixed scratch exceeds tile workspace");
  workspace_bytes-=fixed_numeric_bytes;
  const int slabs=separate ? 3:2,packed_slabs=restricted ? (separate ? 2:1):0;
  const mcscf::detail::SigmaTileBudget budget{
      width,n_channels_,slabs,workspace_bytes,packed_slabs,halo_slabs};
  const bool replay=symbols.matches(support,budget,static_cast<bool>(release));
  auto& boundaries=symbols.boundaries[has_direction ? 2:separate ? 1:0];
  boundaries.bind(support,budget,static_cast<bool>(release));
  // Context residency changes as CIAH vectors grow. Reuse need not reproduce
  // a new greedy partition: the old target partition is exact whenever its
  // actual device storage still fits. The CPU admission estimate also prices
  // gather tables that this executor does not allocate. Never reuse that
  // estimate as a physical device-allocation certificate.
  const auto cotangent_bytes=sizeof(double)*pairs*pairs*adjoint_count+
      panel_descriptor_bytes;
  const auto old_index_bytes=symbols.device.size();
  if (!boundaries.entries.empty() &&
      (boundaries.workspace_peak+cotangent_bytes>replay_bytes ||
       boundaries.first_numeric+old_index_bytes+cotangent_bytes>replay_bytes))
    boundaries.clear();
  std::unique_ptr<mcscf::detail::SigmaTilePlan> iterator;
  if (!replay) {
    const nvtx3::scoped_range range{"ci.tile_plan"};
    symbols.prepared.reset();
    iterator=std::make_unique<mcscf::detail::SigmaTilePlan>(alpha_->topology,beta_->topology,
        support,exchange_parity_,budget,static_cast<bool>(release));
  }
  mcscf::detail::ChannelLayout layout;
  const auto& global=symbols.global;
  const auto& selector_offsets=symbols.offsets;
  const auto& device_global=symbols.device_global;
  const auto& device_local=symbols.device_local;
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
  Matrix<double> gram;
  std::vector<Matrix<double>> kernels;
  kernels.reserve(1+directions);
  // Reuse the same numerical device kernels. Host selectors contain no integral values.
  const auto pack=[&](const DirectCiSigma& action) {
    Matrix<double> panel(ctx,n_channels_,restricted ? global.size()-pairs:0);
    // Group panels are already consecutive in global-channel order. Pack the
    // concatenation once, not one tiny launch for every envelope.
    if (panel.cols()) pack_kernel<<<grid(panel.rows()*panel.cols()),256,0,ctx.stream()>>>(
        action.channel_kernel(),device_global.data()+pairs,panel.cols(),panel.view());
    return panel;
  };
  kernels.push_back(pack(*this));
  std::vector<int> direction_ids(requests.size(),-1);
  for (std::size_t r=0;r<requests.size();++r) if (requests[r].direction) {
    direction_ids[r]=kernels.size();kernels.push_back(pack(*requests[r].direction));
  }
  Buffer<double> numerical;
  PinnedUpload panel_upload(ctx);
  GroupedGemm grouped_gemm;
  if (allow_grouping) {
    grouped_gemm.reserve(selectors.groups());
    panel_upload.assign(group_descriptor_bytes,[](std::span<std::byte> host) {
      std::fill(host.begin(),host.end(),std::byte{});
    });
  }
  bool first=true;
  std::size_t ordinal=0;
  const auto boundary_limit=static_cast<std::size_t>(n_alpha_)+n_beta_;
  const auto next=[&] {
    const nvtx3::scoped_range range{"ci.tile_plan_next"};
    if (replay) return first;
    if (ordinal<boundaries.entries.size()) {
      const nvtx3::scoped_range known{"ci.tile_boundary_replay"};
      iterator->next_known(boundaries.entries[ordinal++]);
      return true;
    }
    if (!iterator->next()) return false;
    if (ordinal<boundary_limit) boundaries.entries.push_back(iterator->boundary());
    ++ordinal;
    return true;
  };
  while (next()) {
    const auto previous_index_bytes=symbols.device.size();
    const auto& plan=replay ? *symbols.prepared:static_cast<const mcscf::detail::SigmaTile&>(*iterator);
    const nvtx3::scoped_range tile_range{"ci.tile"};
    const int count=plan.active.size(),halo=plan.halo.size(),columns=plan.columns.size();
    layout.reset(selectors,plan.channel_groups,count,width);
    const auto input_size=static_cast<std::size_t>(halo)*columns*width;
    std::vector<int> metadata;
    const auto append=[&](std::span<const int> values) {
      const auto offset=metadata.size();metadata.insert(metadata.end(),values.begin(),values.end());return offset;
    };
    const auto active_offset=append(plan.active),halo_offset=append(plan.halo);
    const auto columns_offset=append(plan.columns);
    const bool full_halo=halo==n_alpha_,full_columns=columns==n_beta_;
    const auto halo_index_offset=full_halo ? 0:append(plan.halo_index);
    const auto column_index_offset=full_columns ? 0:append(plan.column_index);
    const bool identity=plan.channel_groups.empty() ||
        std::all_of(plan.channel_groups.begin(),plan.channel_groups.end(),
            [](int group) {return group==0;});
    std::vector<int> betas(count),groups(identity ? 0:count),supported;
    // Packed covalent labels already certify membership: precisely group 0.
    // Other supports need explicit membership only when the tile has a frontier.
    if (identity && plan.unsupported_targets)
      supported.assign(plan.active_support.begin(),plan.active_support.end());
    for (int i=0;i<count;++i) {
      betas[i]=plan.groups[plan.group_index[i]].beta;
      if (!identity) groups[i]=plan.channel_groups[i];
    }
    const auto betas_offset=append(betas),groups_offset=append(groups),supported_offset=append(supported);
    // Map each live image row back to its original target/RHS lane once.
    // Numerical panels then visit only their own rows.
    std::vector<int> inverse(identity ? 0:static_cast<std::size_t>(count)*width);
    if (!identity) for (int row=0;row<count*width;++row)
      inverse[layout.image(row)]=row;
    std::vector<PackedGroup> packed_groups(identity ? 0:selectors.groups());
    if (!identity) for (int g=0;g<selectors.groups();++g)
      packed_groups[g]={layout.offset(g),layout.begin(g),selectors.channels(g)};
    const auto inverse_offset=append(inverse);
    symbols.bind(std::move(metadata),std::move(packed_groups));
    const auto retained_size=static_cast<std::size_t>(halo_slabs)*input_size;
    const auto adjoint_bytes=sizeof(double)*pairs*pairs*adjoint_count;
    const auto fixed_bytes=symbols.device.size()+adjoint_bytes+
        panel_descriptor_bytes+
        (collect ? sizeof(double)*pairs*pairs:0);
    const auto row_bytes=sizeof(double)*pairs*(separate ? 3:2);
    if (fixed_bytes>=replay_bytes || retained_size*sizeof(double)>=replay_bytes-fixed_bytes ||
        row_bytes>replay_bytes-fixed_bytes-retained_size*sizeof(double))
      throw std::length_error("sigma tile retained halos leave no numerical row panel");
    const auto available_rows=(replay_bytes-fixed_bytes-retained_size*sizeof(double))/row_bytes;
    const auto cache_rows=l2_bytes>0 ?
        std::max<std::size_t>(1,static_cast<std::size_t>(l2_bytes)/row_bytes):available_rows;
    const int panel_limit=static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(count)*width,std::min(available_rows,cache_rows)));
    const auto panel_capacity=static_cast<std::size_t>(panel_limit)*pairs;
    const auto pointer_capacity=allow_grouping ? 3*static_cast<std::size_t>(selectors.groups()):0;
    const auto required=retained_size+(separate ? 3:2)*panel_capacity+
        pointer_capacity+(collect ? pairs*pairs:0);
    if (required!=numerical.size()) {
      numerical=Buffer<double>();numerical=Buffer<double>(ctx,required);
    }
    if (!replay) {
      const auto numeric_bytes=numerical.size()*sizeof(double);
      const auto index_bytes=symbols.device.size();
      // Repricing includes the preceding metadata packet while its
      // stream-ordered release may still be pending.
      boundaries.workspace_peak=std::max(boundaries.workspace_peak,
          numeric_bytes+std::max(previous_index_bytes,index_bytes));
      if (ordinal==1) boundaries.first_numeric=numeric_bytes;
    }
    const auto ptr=[&](std::size_t offset) {return symbols.indices()+offset;};
    const TileMap map{ptr(active_offset),ptr(betas_offset),
        identity ? nullptr:ptr(groups_offset),supported.empty() ? nullptr:ptr(supported_offset),
        full_halo ? nullptr:ptr(halo_index_offset),full_columns ? nullptr:ptr(column_index_offset),
        device_local.data(),symbols.packed(),
        count,width,n_channels_,n_alpha_,n_beta_};
    const auto halo_view=[&](int slot) {
      return linalg::MatrixView{numerical.data()+static_cast<std::size_t>(slot)*input_size,
          halo,static_cast<std::int64_t>(columns)*width,halo};
    };
    std::vector<int> output_slot(requests.size(),-1);
    int next_slot=input_halos;
    for (std::size_t r=0;r<requests.size();++r) if (requests[r].image)
      output_slot[r]=next_slot++;
    const int state_slot=separate ?
        (alias==requests.end() ? static_cast<int>(requests.size()):
            static_cast<int>(alias-requests.begin())):0;
    {
      const nvtx3::scoped_range range{"ci.tile_read"};
      for (std::size_t r=0;r<requests.size();++r)
        requests[r].response(plan.columns,plan.halo,halo_view(r));
      if (separate && alias==requests.end())
        state(plan.columns,plan.halo,halo_view(state_slot));
    }
    for (const int slot:output_slot) if (slot>=0)
      check_cuda(cudaMemsetAsync(halo_view(slot).data,0,input_size*sizeof(double),ctx.stream()),
          "clear tiled output halo");
    const int* inverse_rows=identity ? nullptr:ptr(inverse_offset);
    const auto total_rows=count*width;
    std::vector<GroupTile> live_groups;
    live_groups.reserve(selectors.groups());
    std::vector<int> gram_ends;
    gram_ends.reserve(selectors.groups()+1);
    for (int panel_first=0;panel_first<total_rows;) {
      const nvtx3::scoped_range panel_range{"ci.tile_row_panel"};
      const int panel_rows=std::min(panel_limit,total_rows-panel_first);
      live_groups.clear();
      std::size_t packed_rows=0;
      for (int g=0;g<selectors.groups();++g) {
        const int first_row=std::max(panel_first,layout.begin(g));
        const int last_row=std::min(panel_first+panel_rows,layout.begin(g)+layout.size(g));
        if (first_row>=last_row) continue;
        const int k=selectors.channels(g);
        live_groups.push_back({packed_rows,0,g,first_row-panel_first,last_row-first_row,
            g ? static_cast<std::size_t>(pairs)*
                (selector_offsets[g]-static_cast<int>(pairs)):0,k});
        packed_rows+=static_cast<std::size_t>(last_row-first_row)*k;
      }
      const auto& first_group=live_groups.front();
      const auto packed_origin=layout.offset(first_group.group)+
          static_cast<std::size_t>(panel_first-layout.begin(first_group.group))*first_group.channels;
      double* response_channels=numerical.data()+retained_size;
      double* state_channels=separate ? response_channels+panel_capacity:response_channels;
      double* transformed=response_channels+(separate ? 2:1)*panel_capacity;
      auto* pointer_bytes=transformed+panel_capacity;
      const double** left_ptrs=allow_grouping ?
          reinterpret_cast<const double**>(pointer_bytes):nullptr;
      const double** right_ptrs=allow_grouping ? left_ptrs+selectors.groups():nullptr;
      double** result_ptrs=allow_grouping ?
          reinterpret_cast<double**>(pointer_bytes+2*selectors.groups()):nullptr;
      double* gram_values=pointer_bytes+pointer_capacity;
      const bool grouped=allow_grouping && live_groups.size()>1;
      gram_ends.clear();gram_ends.push_back(0);
      std::size_t gram_used=0;
      for (int i=0;i<static_cast<int>(live_groups.size());++i) {
        auto& group=live_groups[i];
        const auto area=static_cast<std::size_t>(group.channels)*group.channels;
        const bool after_full=i>0 && live_groups[i-1].group==0;
        if (gram_used && (after_full || gram_used+area>pairs*pairs)) {
          gram_ends.push_back(i);gram_used=0;
        }
        group.scratch=gram_used;gram_used+=area;
      }
      gram_ends.push_back(static_cast<int>(live_groups.size()));
      if (grouped) panel_upload.assign(group_descriptor_bytes,[&](std::span<std::byte> bytes) {
        std::memcpy(bytes.data(),live_groups.data(),live_groups.size()*sizeof(GroupTile));
      });
      const auto* device_groups=reinterpret_cast<const GroupTile*>(panel_upload.data());
      const auto scatter_panel=[&](linalg::ConstMatrixView input,double* output) {
        const nvtx3::scoped_range range{"ci.tile_scatter"};
        scatter_row_panel<<<grid(static_cast<std::size_t>(panel_rows)*pairs),256,0,ctx.stream()>>>(
            map,input,alpha_->sources.data(),beta_->sources.data(),inverse_rows,
            panel_first,panel_rows,packed_origin,output);
      };
      const auto grouped_call=[&](bool gram_call,int begin,int end,const double* source,
          const double* second,const double* full_kernel,double* output,double previous) {
        const int batch=end-begin;
        grouped_gemm.prepare({live_groups.data()+begin,static_cast<std::size_t>(batch)},
            n_channels_,panel_rows,exchange_parity_ ? 2.0:1.0,previous,gram_call);
        if (gram_call)
          grouped_pointers<true><<<grid(batch),256,0,ctx.stream()>>>(
              device_groups,begin,batch,source,second,full_kernel,
              output,left_ptrs,right_ptrs,result_ptrs);
        else
          grouped_pointers<false><<<grid(batch),256,0,ctx.stream()>>>(
              device_groups,begin,batch,source,second,full_kernel,
              output,left_ptrs,right_ptrs,result_ptrs);
        check_cuda(cudaGetLastError(),"prepare grouped row-panel GEMM pointers");
        grouped_gemm.run(ctx,left_ptrs,right_ptrs,result_ptrs);
      };
      if (separate) scatter_panel(halo_view(state_slot),state_channels);
      for (std::size_t r=0;r<requests.size();++r) {
        const auto& request=requests[r];
        double* response=separate && request.response_is_state ? state_channels:response_channels;
        if (response!=state_channels || !separate)
          scatter_panel(halo_view(r),response);
        const double* gram_right=separate ? state_channels:response;
        if (request.one_adjoint.data) {
          if (exchange_parity_) {
            weight_row_panel<<<grid(static_cast<std::size_t>(panel_rows)*pairs),256,0,ctx.stream()>>>(
                map,inverse_rows,panel_first,panel_rows,packed_origin,gram_right,transformed);
            gram_right=transformed;
          }
          for (std::size_t part=0;part+1<gram_ends.size();++part) {
            const int begin=gram_ends[part],end=gram_ends[part+1];
            if (grouped && end-begin>1) {
              const nvtx3::scoped_range range{"ci.tile_gram"};
              grouped_call(true,begin,end,response,gram_right,nullptr,gram_values,0.0);
              add_grouped_gram<<<grid(pairs*pairs),256,0,ctx.stream()>>>(
                  gram_values,adjoints[r].view(),device_groups,
                  begin,end,device_local.data(),n_channels_);
            } else for (int index=begin;index<end;++index) {
              const auto& group=live_groups[index];
              const int k=group.channels;
              const nvtx3::scoped_range range{"ci.tile_gram"};
              const linalg::ConstMatrixView left{response+group.offset,k,group.size,k},
                  right{gram_right+group.offset,k,group.size,k};
              const linalg::MatrixView tile{gram_values,k,k,k};
              gemm(ctx,tile,left,right,exchange_parity_ ? 2.0:1.0,0.0,
                  Operation::none,Operation::transpose);
              add_gram<<<grid(static_cast<std::size_t>(k)*k),256,0,ctx.stream()>>>(
                  tile,adjoints[r].view(),device_global.data()+selector_offsets[group.group]);
            }
          }
        }
        if (!(request.image || request.carrier_image.data)) continue;
        const auto multiply=[&](const DirectCiSigma& action,int kernel_id,
            const double* source,bool add) {
          const nvtx3::scoped_range range{"ci.tile_gemm"};
          const auto regular=[&](const GroupTile& group) {
            const int g=group.group,k=group.channels;
            const auto selected=g ? kernels[kernel_id].view().columns(selector_offsets[g]-pairs,k):
                action.channel_kernel();
            const linalg::ConstMatrixView channels{source+group.offset,k,group.size,k};
            const linalg::MatrixView output{transformed+group.begin,group.size,n_channels_,panel_rows};
            gemm(ctx,output,channels,selected,exchange_parity_ ? 2.0:1.0,
                add ? 1.0:0.0,Operation::transpose,Operation::transpose);
          };
          const int first_restricted=live_groups.front().group==0 ? 1:0;
          if (first_restricted) regular(live_groups.front());
          const int rest=static_cast<int>(live_groups.size())-first_restricted;
          if (rest>1) grouped_call(false,first_restricted,live_groups.size(),source,
              kernels[kernel_id].data(),action.channel_kernel().data,transformed,
              add ? 1.0:0.0);
          else if (rest==1) regular(live_groups[first_restricted]);
        };
        multiply(*this,0,response,false);
        if (request.direction) multiply(*request.direction,direction_ids[r],state_channels,true);
        {
          const nvtx3::scoped_range range{"ci.tile_link_adjoint"};
          scatter_image_row_panel<<<grid(static_cast<std::size_t>(panel_rows)*pairs),256,0,ctx.stream()>>>(
              map,alpha_->sources.data(),beta_->sources.data(),inverse_rows,panel_first,
              panel_rows,transformed,request.carrier_image.data ? request.carrier_image:
                  halo_view(output_slot[r]),request.carrier_image.data!=nullptr,exchange_parity_!=0);
        }
      }
      panel_first+=panel_rows;
    }
    for (std::size_t r=0;r<requests.size();++r) {
      const auto& request=requests[r];
      if (request.image) {
        const auto output_halo=halo_view(output_slot[r]);
        const auto gather_columns=[&](std::span<const CiTileColumn> targets,bool add) {
          if (targets.empty()) return;
          std::size_t work=0;
          for (const auto& target:targets) {
            const auto& output=target.values;
            if (target.source < -1 || target.source>=columns || output.rows<0 ||
                (target.full_rows ? output.rows!=n_alpha_:
                    (!target.rows && output.rows!=halo)) ||
                output.cols!=(target.source<0 ? columns:1)*width || output.ld<output.rows)
              throw std::invalid_argument("invalid tiled sigma gather dimensions");
            work=std::max(work,static_cast<std::size_t>(output.rows)*output.cols);
          }
          if (!work) return;
          const auto device_targets=upload<CiTileColumn>(ctx,targets);
          gather<<<dim3(targets.size(),grid(work)),256,0,ctx.stream()>>>(map,
              ptr(halo_offset),device_targets.data(),output_halo,add);
        };
        const auto compact_panel=[&](const CiTilePanel& target) {
          const auto [first,source_first,count,lane,max_rows,maps,output]=target;
          if (first<0 || first>=n_beta_ || count<0 || source_first<0 || source_first+count>columns ||
              lane<0 || lane+width>output.cols || max_rows<0 || max_rows>n_alpha_ ||
              !maps || output.ld<output.rows)
            throw std::invalid_argument("invalid compact sigma gather panel");
          if (!count || !max_rows) return;
          gather_panel<<<dim3(count,grid(static_cast<std::size_t>(max_rows)*width)),256,0,ctx.stream()>>>(
              map,ptr(columns_offset),first,source_first,lane,maps,output,output_halo);
        };
        const CiTileGather gather_target{gather_columns,compact_panel};
        const nvtx3::scoped_range range{"ci.tile_backproject"};
        request.image(plan.columns,plan.halo,gather_target);
      }
    }
    if (release && !plan.released.empty()) release(plan.released);
    check_cuda(cudaGetLastError(),"device product-tile sigma sweep");
    if (!replay && first && iterator->exhausted()) {
      // Move only a complete one-tile schedule. Larger sweeps remain streamed;
      // no cursor, borrowed support pointer, or list of all tiles is retained.
      symbols.prepared.emplace(std::move(static_cast<mcscf::detail::SigmaTile&>(*iterator)));
      symbols.support_identity=support ? support->identity():nullptr;
      symbols.budget=budget;symbols.payload=iterator->payload_bytes();
      symbols.packed_entries=layout.entries();symbols.releases=static_cast<bool>(release);
      break;
    }
    first=false;
  }
  // Integral pullback needs only K scratch values after the sweep.
  if (collect && numerical.size()<pairs) gram=Matrix<double>(ctx,n_channels_,1);
  double* absorbed=numerical.size()>=pairs ? numerical.data():gram.data();
  for (std::size_t r=0;r<requests.size();++r) if (requests[r].one_adjoint.data)
    pullback_channel_kernel(adjoints[r].view(),requests[r].one_adjoint,requests[r].pair_adjoint,absorbed);
}

} // namespace xmvb::cuda
