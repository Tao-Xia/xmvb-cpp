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
  int begin,channels,rows;
  bool operator==(const PackedGroup&) const = default;
};

/** CPU plan maps only: no product-sized address table is uploaded. */
struct TileMap {
  const int *active,*betas,*groups,*supported,*halo_index,*column_index;
  const int *image,*local;
  const PackedGroup* packed_groups;
  int count,width,pairs,alpha,beta;

  __device__ int group(int row) const { return groups ? groups[row%count]:0; }
  __device__ int image_row(int row) const {
    return image ? image[row%count]+(row/count)*packed_groups[group(row)].rows:row;
  }
  __device__ bool is_supported(int row) const {
    return packed_groups ? group(row)==0:(!supported || supported[row%count]);
  }
  __device__ int halo_row(int row) const { return halo_index ? halo_index[row]:row; }
  __device__ int column(int col) const { return column_index ? column_index[col]:col; }
  __device__ std::size_t packed_row(int row) const {
    if (!packed_groups) return static_cast<std::size_t>(row)*pairs;
    const auto& entry=packed_groups[group(row)];
    return entry.offset+static_cast<std::size_t>(image_row(row)-entry.begin)*entry.channels;
  }
};

__global__ void scatter(TileMap map,linalg::ConstMatrixView input,
    const int* alpha,const int* beta,double* output) {
  const auto work=static_cast<std::size_t>(map.count)*map.width*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int row=i/map.pairs,p=i%map.pairs,t=row%map.count,v=row/map.count;
    const int channel=map.local[map.group(row)*map.pairs+p];
    if (channel<0) continue;
    const int a=map.active[t],b=map.betas[t];
    const int sa=alpha[static_cast<std::size_t>(p)*map.alpha+a];
    const int sb=beta[static_cast<std::size_t>(p)*map.beta+b];
    const int h=map.halo_row(a),column=map.column(b);
    double value=0;
    if (sa && (abs(sa)-1!=a || map.is_supported(t)))
      value+=(sa>0 ? 1.0:-1.0)*input.data[
          map.halo_row(abs(sa)-1)+input.ld*(column*map.width+v)];
    if (sb && (abs(sb)-1!=b || map.is_supported(t)))
      value+=(sb>0 ? 1.0:-1.0)*input.data[
          h+input.ld*(map.column(abs(sb)-1)*map.width+v)];
    output[map.packed_row(row)+channel]=value;
  }
}

__global__ void pack_kernel(linalg::ConstMatrixView kernel,const int* channels,
    int count,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(output.rows)*count;
       i+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
    output.data[i]=kernel.data[i%output.rows+kernel.ld*channels[i/output.rows]];
}

__global__ void correct_image(double* values,TileMap map,const int* diagonal,int count) {
  const auto work=static_cast<std::size_t>(count)*map.width*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int p=i/(count*map.width),v=(i/count)%map.width,t=diagonal[i%count];
    values[map.image_row(t+v*map.count)+
        static_cast<std::size_t>(p)*map.count*map.width]*=0.5;
  }
}

/** Halve only exchange-fixed source rows. The subsequent Gram has BLAS
 * alpha=2: off-diagonal orbits contribute twice and fixed points once.
 * Scratch borrows the not-yet-written sigma image, never a new carrier.
 */
__global__ void weight_diagonal(TileMap map,const int* diagonal,int count,double* values) {
  const auto work=static_cast<std::size_t>(count)*map.width*map.pairs;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<work;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int row=diagonal[i%count]+((i/count)%map.width)*map.count,p=i/(count*map.width);
    const int local=map.local[map.group(row)*map.pairs+p];
    if (local>=0) values[map.packed_row(row)+local]*=0.5;
  }
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

// The input halo has already been scattered when grouped GEMM begins. Borrow
// it for the three device pointer arrays; all consumers share this stream.
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
    const double value=image[map.image_row(t+v*map.count)+
        static_cast<std::size_t>(p)*map.count*map.width];
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
          budget.packed_slabs==next.packed_slabs) return;
      support=identity;budget=next;release=next_release;clear();
    }
  };
  std::array<Boundaries,3> boundaries;
  mcscf::detail::ChannelSelectors selectors;
  bool restricted;
  std::vector<int> global,local,offsets,metadata;
  std::vector<PackedGroup> packed_groups;
  std::vector<GroupTile> groups;
  Buffer<int> device_global,device_local;
  PinnedUpload device;
  std::size_t packed_offset=0,group_offset=0;
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
    const Bytes bytes=Bytes(payload)+sizeof(double)*(
        Bytes(next.packed_slabs-budget.packed_slabs)*packed_entries+
        Bytes(next.slabs-next.packed_slabs-budget.slabs+budget.packed_slabs)*channel_entries);
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
  const GroupTile* batches() const noexcept {
    return groups.empty() ? nullptr:
        reinterpret_cast<const GroupTile*>(device.data()+group_offset);
  }

  void bind(std::vector<int> indices,std::vector<PackedGroup> layout,
      std::span<const GroupTile> batches) {
    const nvtx3::scoped_range range{"ci.tile_index_upload"};
    if (metadata==indices && packed_groups==layout &&
        groups.size()==batches.size() && std::equal(groups.begin(),groups.end(),batches.begin())) return;
    metadata=std::move(indices);packed_groups=std::move(layout);
    groups.assign(batches.begin(),batches.end());
    const auto align=[](std::size_t n,std::size_t a) {return (n+a-1)/a*a;};
    const auto ints=metadata.size()*sizeof(int),packed_bytes=packed_groups.size()*sizeof(PackedGroup);
    packed_offset=align(ints,alignof(PackedGroup));
    group_offset=align(packed_offset+packed_bytes,alignof(GroupTile));
    device.assign(group_offset+groups.size()*sizeof(GroupTile),[&](std::span<std::byte> host) {
      if (ints) std::memcpy(host.data(),metadata.data(),ints);
      if (packed_bytes) std::memcpy(host.data()+packed_offset,packed_groups.data(),packed_bytes);
      if (!groups.empty()) std::memcpy(host.data()+group_offset,groups.data(),groups.size()*sizeof(GroupTile));
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
  if (!workspace_->tiles || workspace_->tiles->restricted!=restricted)
    workspace_->tiles=std::make_shared<TileWorkspace>(ctx,n_channels_,n_orbitals_,restricted);
  auto& symbols=*workspace_->tiles;
  const auto& selectors=symbols.selectors;
  const auto group_descriptor_bytes=static_cast<std::size_t>(selectors.groups())*sizeof(GroupTile);
  const bool allow_grouping=selectors.groups()>1 && workspace_bytes>group_descriptor_bytes;
  // Replay certificates include the complete device packet, including group
  // descriptors. Greedy numeric admission reserves those descriptors below;
  // do not subtract them a second time from the physical replay allowance.
  const auto replay_bytes=workspace_bytes;
  if (allow_grouping) workspace_bytes-=group_descriptor_bytes;
  const bool separate=static_cast<bool>(state) && (collect || has_direction || requests.size()>1);
  const int slabs=separate ? 3:2,packed_slabs=restricted ? (separate ? 2:1):0;
  const mcscf::detail::SigmaTileBudget budget{width,n_channels_,slabs,workspace_bytes,packed_slabs};
  const bool replay=symbols.matches(support,budget,static_cast<bool>(release));
  auto& boundaries=symbols.boundaries[has_direction ? 2:separate ? 1:0];
  boundaries.bind(support,budget,static_cast<bool>(release));
  // Context residency changes as CIAH vectors grow. Reuse need not reproduce
  // a new greedy partition: the old target partition is exact whenever its
  // actual device storage still fits. The CPU admission estimate also prices
  // gather tables that this executor does not allocate. Never reuse that
  // estimate as a physical device-allocation certificate.
  const auto cotangent_bytes=sizeof(double)*pairs*pairs*(collect+
      std::count_if(requests.begin(),requests.end(),
          [](const auto& request) {return request.one_adjoint.data!=nullptr;}));
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
  GroupedGemm grouped_gemm;
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
    std::vector<GroupTile> live_groups;
    for (int g=0;g<selectors.groups();++g) {
      const int size=layout.size(g),k=selectors.channels(g);
      if (!size) continue;
      live_groups.push_back({layout.offset(g),0,g,layout.begin(g),size,
          g ? static_cast<std::size_t>(pairs)*
              (selector_offsets[g]-static_cast<int>(pairs)):0,k});
    }
    static_assert(sizeof(void*)==sizeof(double));
    const bool grouped=allow_grouping && live_groups.size()>1 &&
        input_size>=3*live_groups.size();
    const double** left_ptrs=nullptr;
    const double** right_ptrs=nullptr;
    double** result_ptrs=nullptr;
    if (grouped) {
      left_ptrs=reinterpret_cast<const double**>(input.data);
      right_ptrs=left_ptrs+live_groups.size();
      result_ptrs=reinterpret_cast<double**>(input.data+2*live_groups.size());
    }
    // After the forward scatters, response/state channels own every live
    // operand. The halo is dead until link backprojection, so its tail can
    // hold independent group Grams after the three GEMM pointer arrays.
    // Preserve increasing group order and use only already allocated bytes.
    std::size_t gram_capacity=static_cast<std::size_t>(pairs)*pairs;
    double* gram_values=nullptr;
    if (collect) {
      std::size_t largest=0;
      for (const auto& group:live_groups)
        largest=std::max(largest,static_cast<std::size_t>(group.channels)*group.channels);
      const auto pointers=3*live_groups.size();
      if (grouped && input_size-pointers>=largest) {
        gram_capacity=input_size-pointers;
        gram_values=input.data+pointers;
      } else {
        if (!gram.data()) gram=Matrix<double>(ctx,n_channels_,n_channels_);
        gram_values=gram.data();
      }
    }
    std::vector<int> gram_ends{0};
    std::size_t gram_used=0;
    for (int i=0;i<static_cast<int>(live_groups.size());++i) {
      auto& group=live_groups[i];
      const auto area=static_cast<std::size_t>(group.channels)*group.channels;
      // Group zero is the unrestricted, full-channel product. Preserve its
      // ordinary cuBLAS GEMM (including split-K selection); batch only the
      // independent restricted channel envelopes in the borrowed tail.
      const bool after_full=i>0 && live_groups[i-1].group==0;
      if (gram_used && (after_full || gram_used+area>gram_capacity)) {
        gram_ends.push_back(i);gram_used=0;
      }
      group.scratch=gram_used;gram_used+=area;
    }
    gram_ends.push_back(static_cast<int>(live_groups.size()));
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
    // CPU ChannelLayout places RHS lanes consecutively inside each group.
    // One target permutation plus the exact per-lane group size determines
    // every lane's image; never upload width copies of the same permutation.
    std::vector<int> images(identity ? 0:count);
    if (!identity) for (int row=0;row<count;++row) images[row]=layout.image(row);
    std::vector<PackedGroup> packed_groups(identity ? 0:selectors.groups());
    if (!identity) for (int g=0;g<selectors.groups();++g)
      packed_groups[g]={layout.offset(g),layout.begin(g),selectors.channels(g),layout.size(g)/width};
    const auto image_offset=append(images);
    const auto diagonal_offset=append(plan.diagonal_rows);
    symbols.bind(std::move(metadata),std::move(packed_groups),
        grouped ? std::span<const GroupTile>{live_groups}:std::span<const GroupTile>{});
    if (!replay) {
      const auto numeric_bytes=numerical.size()*sizeof(double);
      const auto index_bytes=symbols.device.size();
      // The numerical slab only grows within a sweep. Include the preceding
      // indices, still live when that slab grows, and both packed-map states.
      boundaries.workspace_peak=std::max(boundaries.workspace_peak,
          numeric_bytes+std::max(previous_index_bytes,index_bytes));
      if (ordinal==1) boundaries.first_numeric=numeric_bytes;
    }
    const auto ptr=[&](std::size_t offset) {return symbols.indices()+offset;};
    const TileMap map{ptr(active_offset),ptr(betas_offset),
        identity ? nullptr:ptr(groups_offset),supported.empty() ? nullptr:ptr(supported_offset),
        full_halo ? nullptr:ptr(halo_index_offset),full_columns ? nullptr:ptr(column_index_offset),
        identity ? nullptr:ptr(image_offset),device_local.data(),
        symbols.packed(),
        count,width,n_channels_,n_alpha_,n_beta_};
    const auto do_scatter=[&](const CiTileReader& reader,double* output) {
      {
        const nvtx3::scoped_range range{"ci.tile_read"};
        reader(plan.columns,plan.halo,input);
      }
      const nvtx3::scoped_range range{"ci.tile_scatter"};
      scatter<<<grid(channel_size),256,0,ctx.stream()>>>(map,input,alpha_->sources.data(),beta_->sources.data(),output);
    };
    const auto grouped_call=[&](bool gram_call,int begin,int end,
        const double* source,const double* second,const double* full_kernel,
        double* output,double previous) {
      const int batch=end-begin;
      grouped_gemm.prepare({live_groups.data()+begin,static_cast<std::size_t>(batch)},
          n_channels_,count*width,exchange_parity_ ? 2.0:1.0,previous,gram_call);
      if (gram_call)
        grouped_pointers<true><<<grid(batch),256,0,ctx.stream()>>>(
            symbols.batches(),begin,batch,source,second,full_kernel,
            output,left_ptrs,right_ptrs,result_ptrs);
      else
        grouped_pointers<false><<<grid(batch),256,0,ctx.stream()>>>(
            symbols.batches(),begin,batch,source,second,full_kernel,
            output,left_ptrs,right_ptrs,result_ptrs);
      check_cuda(cudaGetLastError(),"prepare grouped channel GEMM pointers");
      grouped_gemm.run(ctx,left_ptrs,right_ptrs,result_ptrs);
    };
    bool state_ready=false;
    for (std::size_t r=0;r<requests.size();++r) {
      const auto& request=requests[r];
      do_scatter(request.response,response);
      if (separate && !state_ready) {do_scatter(state,state_channels);state_ready=true;}
      const auto multiply=[&](const DirectCiSigma& action,int kernel_id,const double* source,bool add) {
        const nvtx3::scoped_range range{"ci.tile_gemm"};
        if (grouped) {
          grouped_call(false,0,static_cast<int>(live_groups.size()),source,
              kernels[kernel_id].data(),action.channel_kernel().data,transformed,
              add ? 1.0:0.0);
          return;
        }
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
      const double* gram_right=state_channels;
      if (request.one_adjoint.data && exchange_parity_ && !plan.diagonal_rows.empty()) {
        check_cuda(cudaMemcpyAsync(transformed,state_channels,packed_size*sizeof(double),
            cudaMemcpyDeviceToDevice,ctx.stream()),"stage exchange-weighted Gram rows");
        weight_diagonal<<<grid(plan.diagonal_rows.size()*width*pairs),256,0,ctx.stream()>>>(
            map,ptr(diagonal_offset),plan.diagonal_rows.size(),transformed);
        gram_right=transformed;
      }
      if (request.one_adjoint.data)
        for (std::size_t part=0;part+1<gram_ends.size();++part) {
          const int begin=gram_ends[part],end=gram_ends[part+1];
          if (grouped && end-begin>1) {
            const nvtx3::scoped_range range{"ci.tile_gram"};
            grouped_call(true,begin,end,response,gram_right,nullptr,gram_values,0.0);
            add_grouped_gram<<<grid(pairs*pairs),256,0,ctx.stream()>>>(
                gram_values,adjoints[r].view(),symbols.batches(),
                begin,end,device_local.data(),n_channels_);
          } else for (int index=begin;index<end;++index) {
            const int g=live_groups[index].group;
            const int size=layout.size(g),k=selectors.channels(g);
            const nvtx3::scoped_range range{"ci.tile_gram"};
            const linalg::ConstMatrixView left{response+layout.offset(g),k,size,k},
                right{gram_right+layout.offset(g),k,size,k};
            const linalg::MatrixView tile{gram_values,k,k,k};
            gemm(ctx,tile,left,right,exchange_parity_ ? 2.0:1.0,0.0,
                Operation::none,Operation::transpose);
            add_gram<<<grid(static_cast<std::size_t>(k)*k),256,0,ctx.stream()>>>(
                tile,adjoints[r].view(),device_global.data()+selector_offsets[g]);
          }
        }
      if (request.image || request.carrier_image.data) {
        multiply(*this,0,response,false);
        if (request.direction) multiply(*request.direction,direction_ids[r],state_channels,true);
        if (exchange_parity_ && !plan.diagonal_rows.empty())
          correct_image<<<grid(plan.diagonal_rows.size()*width*pairs),256,0,ctx.stream()>>>(
              transformed,map,ptr(diagonal_offset),plan.diagonal_rows.size());
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
              ptr(halo_offset),device_targets.data(),input,add);
        };
        const auto compact_panel=[&](const CiTilePanel& target) {
          const auto [first,source_first,count,lane,max_rows,maps,output]=target;
          if (first<0 || first>=n_beta_ || count<0 || source_first<0 || source_first+count>columns ||
              lane<0 || lane+width>output.cols || max_rows<0 || max_rows>n_alpha_ ||
              !maps || output.ld<output.rows)
            throw std::invalid_argument("invalid compact sigma gather panel");
          if (!count || !max_rows) return;
          gather_panel<<<dim3(count,grid(static_cast<std::size_t>(max_rows)*width)),256,0,ctx.stream()>>>(
              map,ptr(columns_offset),first,source_first,lane,maps,output,input);
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
      symbols.packed_entries=packed_size;symbols.releases=static_cast<bool>(release);
      break;
    }
    first=false;
  }
  // The complete sweep has consumed every numerical slab. Integral pullback
  // needs only K scratch values, not the former standalone K-by-K Gram.
  if (collect && numerical.size()<pairs && !gram.data()) gram=Matrix<double>(ctx,n_channels_,1);
  double* absorbed=numerical.size()>=pairs ? numerical.data():gram.data();
  for (std::size_t r=0;r<requests.size();++r) if (requests[r].one_adjoint.data)
    pullback_channel_kernel(adjoints[r].view(),requests[r].one_adjoint,requests[r].pair_adjoint,absorbed);
}

} // namespace xmvb::cuda
