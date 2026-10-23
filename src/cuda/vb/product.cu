#include "cuda/vb/product.hpp"
#include "cuda/vb/columns.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <list>
#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/column_cache.hpp"
#include "mcscf/ci/product_support.hpp"
#include "cuda/vb/panel.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"

namespace xmvb::cuda {
namespace {
unsigned grid(std::size_t n) {return static_cast<unsigned>(std::min<std::size_t>(65535,(n+255)/256));}
template<class Tile>
void free_cache_for_allocation(Context& c,mcscf::ColumnCache<Tile>& cache,std::size_t bytes) {
  const auto available=[&] {
    const auto resident=c.counters().current_device_bytes;
    const auto budget=c.device_byte_budget();
    // The shared allocator already accounts for live payload. Device-wide
    // free memory excludes reusable pool storage and is not this budget.
    return resident<=budget ? budget-resident:std::size_t{0};
  };
  while (bytes>available() && cache.evict_one()) {}
}
void sparse_check(cusparseStatus_t result,const char* operation) {
  if (result!=CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation)+": "+cusparseGetErrorString(result));
}
template<class T> Buffer<T> upload(Context& c,std::span<const T> values) {
  Buffer<T> result(c,values.size());c.copy_to_device(result.data(),values.data(),values.size_bytes());return result;
}
void clear(Context& c,linalg::MatrixView panel) {
  if (!panel.rows || !panel.cols) return;
  check_cuda(cudaMemset2DAsync(panel.data,panel.ld*sizeof(double),0,panel.rows*sizeof(double),panel.cols,c.stream()),
      "clear product tile");
}

/** Exact-zero vendor conversion of one requested beta-factor block. The CSR
 * is reused at the orbital point; the dense single-spin panel is then dead.
 */
struct Factors {
  Buffer<int> offsets,indices;
  Buffer<double> values;
  template<class Reserve,class FreeForScratch>
  Factors(Context& c,cusparseHandle_t handle,linalg::ConstMatrixView dense,
      Reserve&& reserve,FreeForScratch&& free_for_scratch)
      :offsets(c,dense.rows+1) {
    const nvtx3::scoped_range range{"vb.factor_build"};
    cusparseDnMatDescr_t input=nullptr;
    cusparseSpMatDescr_t output=nullptr;
    try {
      sparse_check(cusparseCreateDnMat(&input,dense.rows,dense.cols,dense.ld,
          const_cast<double*>(dense.data),CUDA_R_64F,CUSPARSE_ORDER_COL),"create beta-factor panel");
      sparse_check(cusparseCreateCsr(&output,dense.rows,dense.cols,0,offsets.data(),nullptr,nullptr,
          CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F),"create beta-factor CSR");
      std::size_t bytes=0;
      sparse_check(cusparseDenseToSparse_bufferSize(handle,input,output,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,&bytes),
          "query beta-factor CSR scratch");
      free_for_scratch(bytes);
      Buffer<std::byte> scratch(c,bytes);
      sparse_check(cusparseDenseToSparse_analysis(handle,input,output,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,scratch.data()),
          "analyze beta-factor exact zeros");
      c.synchronize();
      std::int64_t rows,columns,nnz;
      sparse_check(cusparseSpMatGetSize(output,&rows,&columns,&nnz),"count beta-factor exact nonzeros");
      if (nnz>std::numeric_limits<int>::max()) throw std::overflow_error("beta-factor CSR exceeds integer range");
      reserve(offsets.size()*sizeof(int)+static_cast<std::size_t>(nnz)*(sizeof(int)+sizeof(double)));
      free_for_scratch(static_cast<std::size_t>(nnz)*(sizeof(int)+sizeof(double)));
      indices=Buffer<int>(c,nnz);values=Buffer<double>(c,nnz);
      sparse_check(cusparseCsrSetPointers(output,offsets.data(),indices.data(),values.data()),"set beta-factor CSR payload");
      sparse_check(cusparseDenseToSparse_convert(handle,input,output,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,scratch.data()),
          "pack beta-factor exact nonzeros");
    } catch (...) {
      if (input) cusparseDestroyDnMat(input);
      if (output) cusparseDestroySpMat(output);
      throw;
    }
    cusparseDestroyDnMat(input);cusparseDestroySpMat(output);
  }
  std::size_t bytes() const {return (offsets.size()+indices.size())*sizeof(int)+values.size()*sizeof(double);}
};

__global__ void generate_selected(const int* starts,const int* raw_beta,
    ProductIncidence incidence,linalg::ConstMatrixView coefficients,
    const int* offsets,const int* columns,const double* factors,
    linalg::MatrixView output) {
  for (auto a=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       a<output.rows;a+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
    for (int g=starts[a];g<starts[a+1];++g) {
      double value=0.0;
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
        value+=incidence.weights[t]*coefficients.data[incidence.structures[t]];
      for (int p=offsets[raw_beta[g]];p<offsets[raw_beta[g]+1];++p)
        output.data[a+output.ld*columns[p]]+=value*factors[p];
    }
}

/** CPU source-group sweep: primal, beta-direction and independent CI lanes
 * share their incidence sum and fixed beta-factor traversal. */
__global__ void generate_selected_jet(const int* starts,const int* raw_beta,
    ProductIncidence incidence,linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView response,const int* offsets,const int* columns,
    const double* factors,const int* delta_offsets,const int* delta_columns,
    const double* delta_factors,int width,bool joint,linalg::MatrixView output) {
  for (auto a=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       a<output.rows;a+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
    for (int g=starts[a];g<starts[a+1];++g) {
      double value=0.0,direction=0.0;
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t) {
        const auto index=incidence.structures[t];
        value+=incidence.weights[t]*coefficients.data[index];
        if (response.data) direction+=incidence.weights[t]*response.data[index];
      }
      const int b=raw_beta[g];
      for (int p=offsets[b];p<offsets[b+1];++p) {
        const auto index=a+output.ld*columns[p];
        output.data[index]+=value*factors[p];
        output.data[index+output.ld*width]+=direction*factors[p];
        if (joint) output.data[index+2*output.ld*width]+=direction*factors[p];
      }
      if (delta_offsets) for (int p=delta_offsets[b];p<delta_offsets[b+1];++p)
        output.data[a+output.ld*(width+delta_columns[p])]+=value*delta_factors[p];
    }
}

__global__ void project_groups(const int* raw_alpha,const int* raw_beta,int count,
    const int* offsets,const int* columns,const double* factors,
    linalg::ConstMatrixView image,int width,int lanes,int target0,int target1,
    linalg::MatrixView output) {
  for (auto g=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       g<count;g+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int b=raw_beta[g];double first=0.0,second=0.0;
    for (int p=offsets[b];p<offsets[b+1];++p) {
      const auto index=raw_alpha[g]+image.ld*columns[p];
      first+=image.data[index]*factors[p];
      if (lanes==2) second+=image.data[index+image.ld*width]*factors[p];
    }
    output.data[g+output.ld*target0]+=first;
    if (lanes==2) output.data[g+output.ld*target1]+=second;
  }
}

__global__ void contract_groups(ProductIncidence incidence,int begin,int count,
    const double* values,double* target) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<count;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int g=begin+i;
    for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
      atomicAdd(target+incidence.structures[t],incidence.weights[t]*values[i]);
  }
}

/** Sparse structure incidence against independent bounded beta tiles. Each
 * group has one raw product coordinate; no contiguous product carrier exists.
 */
__global__ void ordered_products_tiles(ProductIncidence incidence,const int* aa,
    const int* bb,int groups,linalg::ConstMatrixView source,
    double* const* tiles,int alpha,int extent,linalg::MatrixView target,bool adjoint) {
  for (auto g=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       g<groups;g+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    double* const entry=tiles[bb[g]/extent]+aa[g]+static_cast<std::size_t>(alpha)*(bb[g]%extent);
    if (adjoint) {
      const double value=*entry;
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
        atomicAdd(target.data+incidence.structures[t],incidence.weights[t]*value);
    } else {
      double value=0.0;
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
        value+=incidence.weights[t]*source.data[incidence.structures[t]];
      *entry=value;
    }
  }
}

__global__ void restrict_maximum_tile(linalg::MatrixView values,
    const std::uint64_t* allowed,int words) {
  const auto size=static_cast<std::size_t>(values.rows)*values.cols;
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<size;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const auto row=i%values.rows,column=i/values.rows;
    if (!(allowed[column*words+row/64]&(std::uint64_t{1}<<(row%64))))
      values.data[row+values.ld*column]=0.0;
  }
}

void add(Context& c,linalg::MatrixView target,linalg::ConstMatrixView source,double scale=1.0) {
  if (target.ld==target.rows && source.ld==source.rows) {
    check_cublas(cublasDaxpy(c.blas(),blas_int(target.rows*target.cols),&scale,
        source.data,1,target.data,1),"product panel accumulation");
    return;
  }
  for (std::int64_t j=0;j<target.cols;++j)
    check_cublas(cublasDaxpy(c.blas(),blas_int(target.rows),&scale,
        source.data+source.ld*j,1,target.data+target.ld*j,1),"product tile accumulation");
}

void multiply(Context& c,linalg::MatrixView target,double scale) {
  if (target.ld==target.rows) {
    check_cublas(cublasDscal(c.blas(),blas_int(target.rows*target.cols),&scale,
        target.data,1),"product panel scaling");
    return;
  }
  for (std::int64_t j=0;j<target.cols;++j)
    check_cublas(cublasDscal(c.blas(),blas_int(target.rows),&scale,
        target.data+target.ld*j,1),"product tile scaling");
}

struct ColumnReadIndex {int local,destination;};

/** One beta column of an ordered, independently allocated source tile. The
 * dense values borrow the tile; rows still describe the exact CI support. */
struct ResidentColumn {
  const double* values;
  const int* rows;
  int size;
};

struct ResidentMaps {
  std::vector<std::shared_ptr<const ProductColumnPattern>> patterns;
  Buffer<ResidentColumn> columns;
  std::size_t charged_bytes=0;
  std::size_t bytes() const {return charged_bytes;}
};

__global__ void read_resident_columns(const ResidentColumn* sources,
    const ColumnReadIndex* columns,const int* requested,int count,
    linalg::MatrixView output) {
  const auto column=columns[blockIdx.x];
  const auto source=sources[column.local];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(count);i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i,a=requested[r];
    bool present=a>=0 && a<source.size;
    if (source.rows) {
      int first=0,last=source.size;
      while (first<last) {
        const int mid=first+(last-first)/2;
        if (source.rows[mid]<a) first=mid+1;else last=mid;
      }
      present=first<source.size && source.rows[first]==a;
    }
    output.data[r+output.ld*column.destination]=present && source.values ?
        source.values[a]:0.0;
  }
}

__global__ void pack_resident_column(const CiColumnMap* maps,
    linalg::ConstMatrixView resident,linalg::MatrixView compact) {
  const int b=blockIdx.x;
  const auto column=maps[b];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(column.size);
       i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int a=column.rows ? column.rows[i]:static_cast<int>(i);
    compact.data[column.offset+i]=resident.data[a+resident.ld*b];
  }
}

__global__ void read_columns(const CiColumnMap* maps,linalg::ConstMatrixView packed,
    const ColumnReadIndex* columns,const int* requested,int count,int lane,
    linalg::MatrixView output) {
  const auto column=columns[blockIdx.x];
  const auto source=maps[column.local];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(count);i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i,a=requested[r];
    // A contiguous beta panel may include an exactly empty column. Its null
    // index pointer is not a full identity map and must never read the next
    // packed column (or the end of the allocation).
    int index=a>=0 && a<source.size ? a:-1;
    if (source.rows) {
      int first=0,last=source.size;
      while (first<last) {const int mid=first+(last-first)/2;if (source.rows[mid]<a) first=mid+1;else last=mid;}
      index=first<source.size && source.rows[first]==a ? first:-1;
    }
    output.data[r+output.ld*column.destination]=index<0 ? 0.0:
        packed.data[source.offset+packed.ld*lane+index];
  }
}

struct RetainedPattern {
  std::shared_ptr<const ProductColumnPattern> pattern;
  // This LRU owns a device-memory allowance. Host row metadata is reflected
  // in RSS, not charged a second time to Context's device budget.
  std::size_t bytes() const {return pattern->device_rows.size()*sizeof(int);}
};
struct SupportMask {
  Buffer<std::uint64_t> values;
  std::size_t bytes() const {return values.size()*sizeof(std::uint64_t);}
};
} // namespace

/** Orbital-frame-independent data uploaded once per StructureAction. The
 * numerical beta factors and extent-dependent masks stay with each executor.
 */
struct ProductSymbolic {
  Context& c;
  int alpha,beta,groups;
  const mcscf::ProductSupport* support;
  Buffer<int> starts,raw_alpha,raw_beta;
  std::vector<int> reachable;
  std::vector<std::weak_ptr<const ProductColumnPattern>> patterns;
  std::shared_ptr<const ProductColumnPattern> full_pattern;
  std::unique_ptr<mcscf::ColumnCache<RetainedPattern>> pattern_cache;
  std::unique_ptr<ProductExchangeSupport> exchange;

  ProductSymbolic(Context& ctx,const vb::StructureAction::OrthogonalView& view,
      int alpha_dimension,int beta_dimension,const mcscf::ProductSupport* exact_support)
      :c(ctx),alpha(alpha_dimension),beta(beta_dimension),groups(view.products.size()),
       support(exact_support),patterns(std::max(beta,0)) {
    if (alpha<=0 || beta<=0)
      throw std::invalid_argument("product dimensions must be positive");
    if (support) reachable=support->columns();
    else {reachable.resize(beta);std::iota(reachable.begin(),reachable.end(),0);}
    std::vector<int> offsets(alpha+1,0),aa(groups),bb(groups);
    for (int g=0;g<groups;++g) {
      aa[g]=view.products[g]/beta;bb[g]=view.products[g]%beta;++offsets[aa[g]+1];
    }
    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
    starts=upload<int>(c,offsets);raw_alpha=upload<int>(c,aa);raw_beta=upload<int>(c,bb);
    if (view.exchange_parity) {
      if (view.alpha.occupations()!=view.beta.occupations())
        throw std::invalid_argument("exchange support requires identical spin ordering");
      auto seeds=std::vector<mcscf::ProductIndex>(view.products.begin(),view.products.end());
      std::sort(seeds.begin(),seeds.end());
      for (const auto seed:seeds)
        if (!std::binary_search(seeds.begin(),seeds.end(),(seed%beta)*beta+seed/beta))
          throw std::invalid_argument("exchange source seeds are not transpose closed");
      // ProductPaths applies the same prefix rules to both spin lists.
      // Swapping a seed and its two deficit budgets therefore gives a witness
      // for every transposed support entry, without scanning the product grid.
      exchange.reset(new ProductExchangeSupport);
    }
  }

  void project_exchange(std::span<ProductColumns* const> panels,int extent,int parity) {
    if (!exchange) throw std::logic_error("uncertified product exchange support");
    for (std::size_t p=0;p<panels.size();++p) if (panels[p])
      for (std::size_t j=0;j<panels[p]->patterns.size();++j)
        if (panels[p]->patterns[j]!=pattern(static_cast<int>(p)*extent+j))
          throw std::logic_error("exchange panel does not belong to its symbolic support");
    exchange->project(c,panels,extent,parity);
  }

  void prepare_patterns(std::size_t budget) {
    if (!pattern_cache) pattern_cache=std::make_unique<mcscf::ColumnCache<RetainedPattern>>(budget,
        [](int)->RetainedPattern {throw std::logic_error("exact product pattern not generated");});
  }

  std::shared_ptr<const ProductColumnPattern> pattern(int b) {
    const bool full=!support || support->full();
    if (full && full_pattern) return full_pattern;
    if (pattern_cache->contains(b)) return pattern_cache->get(b).pattern;
    if (auto found=patterns[b].lock()) return found;
    auto result=std::make_shared<ProductColumnPattern>();result->full=full;
    if (!full) support->append_column_rows(b,result->rows);
    result->size=full ? alpha:result->rows.size();
    if (!full) pattern_cache->reserve(result->rows.size()*sizeof(int));
    if (!full) result->device_rows=upload<int>(c,result->rows);
    patterns[b]=result;
    if (!full) pattern_cache->insert(b,{result});else full_pattern=result;
    return result;
  }
};

struct ProductContraction::Impl {
  Context& c;
  const ExteriorTransform& alpha_transform;
  const ExteriorTransform& beta_transform;
  int alpha,beta,groups,factor_extent=0,mask_extent=0;
  std::size_t beta_shears;
  const mcscf::ProductSupport* support;
  std::shared_ptr<ProductSymbolic> symbolic;
  cusparseHandle_t sparse=nullptr;
  std::unique_ptr<mcscf::ColumnCache<Factors>> factor_cache;
  std::unique_ptr<mcscf::ColumnCache<SupportMask>> mask_cache;
  struct ReadGroup {int id,offset,count;};
  struct ReadPlan {
    int extent=0;
    std::vector<int> betas,rows;
    std::vector<ReadGroup> groups;
    Buffer<int> indices;
    bool complete=false;
    std::size_t bytes() const noexcept {
      return (betas.size()+rows.size()+indices.size())*sizeof(int)+
          groups.size()*sizeof(ReadGroup);
    }
  };
  // Exact request indices only. No borrowed numerical panel or sigma graph is
  // retained; the LRU is capped by the current sweep's metadata allowance.
  std::list<ReadPlan> read_plans;
  std::size_t read_plan_bytes=0;

  Impl(Context& ctx,const vb::StructureAction::OrthogonalView& view,
      const ExteriorTransform& a,const ExteriorTransform& b,const mcscf::ProductSupport* exact_support)
      :Impl(ctx,a,b,std::make_shared<ProductSymbolic>(ctx,view,a.dimension(),b.dimension(),exact_support),
          view.beta.shear_pair_count()) {}

  Impl(Context& ctx,const ExteriorTransform& a,const ExteriorTransform& b,
      std::shared_ptr<ProductSymbolic> shared,std::size_t shears)
      :c(ctx),alpha_transform(a),beta_transform(b),alpha(shared->alpha),beta(shared->beta),
       groups(shared->groups),beta_shears(shears),support(shared->support),symbolic(std::move(shared)) {
    if (&symbolic->c!=&c || a.dimension()!=alpha || b.dimension()!=beta)
      throw std::invalid_argument("product symbolic support does not match executor frame");
    sparse_check(cusparseCreate(&sparse),"create product contraction sparse handle");
    sparse_check(cusparseSetStream(sparse,c.stream()),"set product contraction stream");
  }
  ~Impl() {if (sparse) cusparseDestroy(sparse);}

  std::shared_ptr<const ProductColumnPattern> pattern(int b) {
    return symbolic->pattern(b);
  }

  ProductColumns allocate(std::span<const int> keys,int width) {
    std::vector<std::shared_ptr<const ProductColumnPattern>> columns;
    columns.reserve(keys.size());
    for (int key:keys) columns.push_back(pattern(key));
    return ProductColumns(c,std::move(columns),width);
  }

  std::size_t panel_bytes(std::span<const int> keys,int width) {
    std::size_t rows=0;
    for (int key:keys) rows+=pattern(key)->size;
    return keys.size()*sizeof(CiColumnMap)+rows*width*sizeof(double);
  }

  using OrderedTiles=std::vector<Matrix<double>>;

  OrderedTiles ordered_tiles(int extent) {
    OrderedTiles tiles;
    tiles.reserve((beta+extent-1)/extent);
    for (int first=0;first<beta;first+=extent) {
      tiles.emplace_back(c,alpha,std::min(extent,beta-first));
      clear(c,tiles.back().view());
    }
    return tiles;
  }

  std::vector<linalg::MatrixView> ordered_views(OrderedTiles& tiles) {
    std::vector<linalg::MatrixView> views;
    views.reserve(tiles.size());
    for (auto& tile:tiles) views.push_back(tile.view());
    return views;
  }

  Buffer<double*> ordered_pointers(OrderedTiles& tiles) {
    std::vector<double*> pointers;
    pointers.reserve(tiles.size());
    for (auto& tile:tiles) pointers.push_back(tile.data());
    return upload<double*>(c,pointers);
  }

  /** Full rows already have dense column-major layout. Borrow their owner
   * rather than allocate, pack, and retain a second numerical panel. */
  linalg::MatrixView dense_panel(ProductColumns& block,Matrix<double>& scratch) {
    if (std::all_of(block.patterns.begin(),block.patterns.end(),
        [&](const auto& pattern) {return pattern->full && pattern->size==alpha;}))
      return {block.values.data(),alpha,
          static_cast<std::int64_t>(block.patterns.size())*block.values.cols(),alpha};
    scratch=Matrix<double>(c,alpha,block.patterns.size()*block.values.cols());
    return scratch.view();
  }

  void transform(linalg::MatrixView values,int first,int columns,bool adjoint) {
    const nvtx3::scoped_range range{adjoint ? "vb.product_alpha_adjoint":"vb.product_alpha"};
    if (!support || support->full()) {alpha_transform.apply_left(values,adjoint);return;}
    alpha_transform.apply_supported_left(values,mask_cache->get(first).values.data(),columns,adjoint);
  }

  void prepare_masks(int extent,std::size_t budget) {
    if (mask_extent!=extent) {mask_cache.reset();mask_extent=extent;}
    if (!mask_cache) mask_cache=std::make_unique<mcscf::ColumnCache<SupportMask>>(budget,[this](int first) {
      const int count=std::min(mask_extent,beta-first),words=(alpha+63)/64;
      std::vector<std::uint64_t> bits(static_cast<std::size_t>(count)*words,0);
      for (int b=0;b<count;++b) {
        const auto column=pattern(first+b);
        for (int row:column->rows)
          bits[static_cast<std::size_t>(b)*words+row/64]|=std::uint64_t{1}<<(row%64);
      }
      return SupportMask{upload<std::uint64_t>(c,bits)};
    },mcscf::ColumnCache<SupportMask>::Consume{},[this](int first) {
      return static_cast<std::size_t>(std::min(mask_extent,beta-first))*
          ((alpha+63)/64)*sizeof(std::uint64_t);
    });
  }

  void prepare_factors(int extent,std::size_t budget) {
    if (factor_extent!=extent) {
      factor_cache.reset();factor_extent=extent;
    }
    if (!factor_cache) factor_cache=std::make_unique<mcscf::ColumnCache<Factors>>(budget,[this](int id) {
      const nvtx3::scoped_range range{"vb.factor_sample",nvtx3::payload{static_cast<std::uint64_t>(id)}};
      const int first=id*factor_extent,count=std::min(factor_extent,beta-first);
      free_cache_for_allocation(c,*factor_cache,static_cast<std::size_t>(beta)*count*sizeof(double));
      Matrix<double> values(c,beta,count);
      beta_transform.sample_rows(first,values.view());
      free_cache_for_allocation(c,*factor_cache,(static_cast<std::size_t>(beta)+1)*sizeof(int));
      return Factors(c,sparse,values.view(),
          [this](std::size_t bytes) {factor_cache->reserve(bytes);},
          [this](std::size_t bytes) {free_cache_for_allocation(c,*factor_cache,bytes);});
    },mcscf::ColumnCache<Factors>::Consume{},[this](int id) {
      return (static_cast<std::size_t>(beta)+1)*sizeof(int);
    });
  }

  void generate_source(ProductIncidence incidence,linalg::ConstMatrixView source,
      const Factors& f,linalg::MatrixView values) {
    const nvtx3::scoped_range range{"vb.source_generate"};
    clear(c,values);
    if (source.data) generate_selected<<<grid(alpha),256,0,c.stream()>>>(symbolic->starts.data(),symbolic->raw_beta.data(),
        incidence,source,f.offsets.data(),f.indices.data(),f.values.data(),values);
  }

  // CPU's cooperative panel width, constrained by the current device budget.
  // A one-column beta space is indivisible; otherwise no panel spans beta.
  int panel_extent(std::size_t allowance) const {
    const auto row_bytes=sizeof(double)*(12ULL*alpha+8ULL*beta);
    return static_cast<int>(std::max<std::size_t>(1,std::min<std::size_t>(
        {beta>1 ? static_cast<std::size_t>(beta-1):std::size_t{1},
            static_cast<std::size_t>(product_panel_width()),allowance/row_bytes})));
  }

  ProductJet jet(ProductIncidence incidence,const DirectCiSigma& sigma,
      ProductEndpoint& endpoint,linalg::ConstMatrixView upper,
      linalg::ConstMatrixView response,linalg::ConstMatrixView delta_upper,
      const DirectCiSigma* delta_sigma,std::size_t bytes,ProductJet* ci,
      linalg::ConstMatrixView relative_direction,bool selected_only=false,bool overlap_only=false) {
    const nvtx3::scoped_range range{"vb.product_jet"};
    const bool tangent=response.data || delta_upper.data || delta_sigma;
    const bool joint=ci!=nullptr;
    // Accepted images have their own panel geometry. Their reader accepts
    // arbitrary beta requests, so changing the current workspace extent does
    // not invalidate a complete accepted Y/(H-E)Y endpoint.
    const auto quarter=std::max<std::size_t>(1,bytes/5);
    const int extent=panel_extent(quarter);
    const bool cached=tangent && endpoint.images && endpoint.images->complete();
    const nvtx3::scoped_range cache_range{cached ? "vb.jet.cached":"vb.jet.uncached"};
    if (cached && delta_upper.data && !relative_direction.data)
      throw std::invalid_argument("cached product jet requires its relative frame direction");
    if (endpoint.coefficients.rows()!=incidence.size || endpoint.coefficients.cols()!=1 ||
        (response.data && (response.rows!=incidence.size || response.cols!=1)) ||
        (joint && !response.data))
      throw std::invalid_argument("invalid selected product jet source");
    const int n=sigma.n_orbitals(),d=sigma.n_channels();
    const auto make_result=[&] {
      ProductJet result{overlap_only ? Matrix<double>{}:Matrix<double>(c,incidence.size,1),Matrix<double>(c,incidence.size,1),
          selected_only ? Matrix<double>{}:Matrix<double>(c,n,n),
          selected_only ? Matrix<double>{}:Matrix<double>(c,d,d),
          selected_only ? Matrix<double>{}:Matrix<double>(c,n,n)};
      clear(c,result.hamiltonian.view());clear(c,result.overlap.view());
      clear(c,result.generator.view());return result;
    };
    auto result=make_result();
    if (joint) *ci=make_result();
    // CPU input/output/sigma owners plus bounded reverse/factor panels.
    // Device index mirrors also count against this process-wide allowance.
    const int width=tangent ? (joint ? 3:2):1;
    const int input_width=cached ? width-1:width;
    const int panels=(beta+extent-1)/extent;
    // Host-only diagnostics: no device synchronization or numerical changes.
    const bool audit=std::getenv("CUVB_PANEL_AUDIT")!=nullptr;
    enum SourceStage { sigma_stage, reverse_stage, terminal_stage };
    int source_stage=sigma_stage;
    std::array<std::size_t,3> builds{},rebuilds{},reads{},hits{};
    std::vector<unsigned> generations(audit ? panels:0,0);
    const nvtx3::scoped_range plan_range{"vb.jet.plan",nvtx3::payload{
        (static_cast<std::uint64_t>(extent)<<32)|static_cast<unsigned>(panels)}};
    std::vector<int> all_beta(beta),all_rows(alpha);
    std::iota(all_beta.begin(),all_beta.end(),0);std::iota(all_rows.begin(),all_rows.end(),0);
    const auto selected=[&](int id) {
      return std::span<const int>{all_beta}.subspan(id*extent,std::min(extent,beta-id*extent));
    };
    std::vector<bool> reachable_panel(panels,false);
    for (int b:symbolic->reachable) reachable_panel[b/extent]=true;
    int sampled_beta=0;
    for (int id=0;id<panels;++id) if (reachable_panel[id]) sampled_beta+=selected(id).size();
    const bool ordered_beta=vb::detail::prefer_ordered_beta(
        groups,alpha,beta,beta_shears,sampled_beta);
    const auto products=static_cast<std::size_t>(alpha)<=
        std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(beta) ?
        static_cast<std::size_t>(alpha)*beta:std::numeric_limits<std::size_t>::max();
    const auto source_slots=static_cast<std::size_t>(input_width)+
        (delta_upper.data && !cached ? 2U:0U);
    const auto source_descriptors=static_cast<std::size_t>(panels)*input_width*sizeof(double*)+
        static_cast<std::size_t>(beta)*input_width*sizeof(ResidentColumn);
    const bool resident_input=ordered_beta && source_descriptors<quarter/2 &&
        products<=(quarter/2-source_descriptors)/(sizeof(double)*source_slots);
    symbolic->prepare_patterns(quarter/4);
    // The fixed-point beta rows and the directional beta rows follow exactly
    // the CPU ColumnPanels lifetime; no product-sized factor is introduced.
    prepare_factors(extent,quarter/4);
    prepare_masks(extent,quarter/4);
    auto& factors=*factor_cache;
    ExteriorDirection da,db;
    if (delta_upper.data && !cached) {
      da=alpha_transform.direction(upper,delta_upper);
      db=beta_transform.direction(upper,delta_upper);
    }
    mcscf::ColumnCache<Factors> delta_factors(quarter/4,[&](int id) {
      const nvtx3::scoped_range range{"vb.factor_direction",nvtx3::payload{static_cast<std::uint64_t>(id)}};
      auto keys=selected(id);
      free_cache_for_allocation(c,delta_factors,2*static_cast<std::size_t>(beta)*keys.size()*sizeof(double));
      Matrix<double> values(c,beta,keys.size()),delta(c,beta,keys.size());
      beta_transform.sample_directional_rows(db,id*extent,values.view(),delta.view());
      free_cache_for_allocation(c,delta_factors,(static_cast<std::size_t>(beta)+1)*sizeof(int));
      return Factors(c,sparse,delta.view(),
          [&](std::size_t bytes) {delta_factors.reserve(bytes);},
          [&](std::size_t bytes) {free_cache_for_allocation(c,delta_factors,bytes);});
    },{},[&](int id) {
      return (static_cast<std::size_t>(beta)+1)*sizeof(int);
    });
    const auto generate=[&](linalg::ConstMatrixView source,const Factors& f,
        linalg::MatrixView values) {
      generate_source(incidence,source,f,values);
    };
    // The CPU ordered path is retained when all independent narrow tiles fit.
    // The extra two slots in the source budget cover the directional beta
    // transform's bounded transpose stripes; no full product array is formed.
    OrderedTiles resident_y,resident_dy,resident_q;
    const auto expand=[&](linalg::ConstMatrixView source,OrderedTiles& tiles) {
      tiles=ordered_tiles(extent);
      if (!source.data || !groups) return;
      auto pointers=ordered_pointers(tiles);
      ordered_products_tiles<<<grid(groups),256,0,c.stream()>>>(incidence,
          symbolic->raw_alpha.data(),symbolic->raw_beta.data(),groups,source,
          pointers.data(),alpha,extent,{},false);
      check_cuda(cudaGetLastError(),"scatter ordered product tiles");
    };
    if (resident_input) {
      if (!cached) {
        expand(endpoint.coefficients.view(),resident_y);
        if (tangent) {
          expand(response,resident_dy);
          auto y_views=ordered_views(resident_y),dy_views=ordered_views(resident_dy);
          if (delta_upper.data) {
            for (int id=0;id<panels;++id)
              alpha_transform.apply_directional_left(da,y_views[id],dy_views[id]);
            beta_transform.apply_directional_right_tiles(db,y_views,dy_views);
          } else {
            for (int id=0;id<panels;++id) {
              alpha_transform.apply_left(y_views[id]);
              alpha_transform.apply_left(dy_views[id]);
            }
            beta_transform.apply_right_tiles(y_views);
            beta_transform.apply_right_tiles(dy_views);
          }
        } else {
          auto y_views=ordered_views(resident_y);
          for (auto view:y_views) alpha_transform.apply_left(view);
          beta_transform.apply_right_tiles(y_views);
        }
      }
      if (joint || cached) {
        expand(response,resident_q);
        auto q_views=ordered_views(resident_q);
        for (auto view:q_views) alpha_transform.apply_left(view);
        beta_transform.apply_right_tiles(q_views);
      }
    }
    if (cached && !endpoint.readers) {
      std::vector<const ProductColumns*> accepted(endpoint.images->size(),nullptr);
      for (int id=0;id<endpoint.images->size();++id) {
        const auto& tile=endpoint.images->at(id);
        if (!tile.patterns.empty()) accepted[id]=&tile;
      }
      endpoint.readers=std::make_unique<ProductColumnReadCache>(c,accepted,
          endpoint.extent,quarter/8);
    }
    const CiTileReader accepted_y=[&](auto betas,auto rows,linalg::MatrixView values) {
      endpoint.readers->read(0,betas,rows,values);
    };
    const CiTileReader accepted_r=[&](auto betas,auto rows,linalg::MatrixView values) {
      endpoint.readers->read(1,betas,rows,values);
    };
    if (resident_input && cached) {
      // Cached dy is transformed q plus the orbital generator. In a joint
      // response q remains an independent lane; otherwise it becomes dy.
      if (joint) {
        resident_dy=ordered_tiles(extent);
        for (int id=0;id<panels;++id) if (reachable_panel[id])
          copy_panel(resident_dy[id].view(),resident_q[id].view());
      }
      if (delta_upper.data) {
        auto& dy_tiles=joint ? resident_dy:resident_q;
        for (int id=0;id<panels;++id) if (reachable_panel[id]) {
          const int count=selected(id).size();
          Matrix<double> orbital(c,alpha,count);
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_y,
              relative_direction,orbital.view());
          add(c,dy_tiles[id].view(),orbital.view());
        }
      }
    }
    mcscf::ColumnCache<ResidentMaps> resident_maps(quarter/8,[&](int id) {
      const auto keys=selected(id);
      ResidentMaps result;
      result.patterns.reserve(keys.size());
      std::vector<ResidentColumn> columns(static_cast<std::size_t>(input_width)*keys.size());
      for (int local=0;local<keys.size();++local) {
        const auto& column=result.patterns.emplace_back(pattern(keys[local]));
        result.charged_bytes+=column->device_rows.size()*sizeof(int);
        for (int lane=0;lane<input_width;++lane) {
          const auto& tiles=cached ? (lane==0 ? (joint ? resident_dy:resident_q):resident_q):
              (lane==0 ? resident_y:(lane==1 ? resident_dy:resident_q));
          columns[static_cast<std::size_t>(lane)*keys.size()+local]={
              tiles[id].data()+static_cast<std::size_t>(alpha)*local,
              column->full ? nullptr:column->device_rows.data(),column->size};
        }
      }
      result.columns=upload<ResidentColumn>(c,columns);
      result.charged_bytes+=columns.size()*sizeof(ResidentColumn);
      return result;
    },{},[&](int id) {
      // Reserve without generating patterns twice. A sparse column owns at
      // most alpha exact row indices; actual bytes() is charged on insertion.
      return static_cast<std::size_t>(selected(id).size())*
          (input_width*sizeof(ResidentColumn)+
              (support && !support->full() ? static_cast<std::size_t>(alpha)*sizeof(int):0));
    });
    mcscf::ColumnCache<ProductColumns> input(quarter,[&](int id) {
      if (resident_input) throw std::logic_error("resident product source must bypass compact cache");
      if (audit) {
        ++builds[source_stage];
        if (generations[id]++) ++rebuilds[source_stage];
      }
      const nvtx3::scoped_range range{"vb.source_panel",nvtx3::payload{static_cast<std::uint64_t>(id)}};
      const auto keys=selected(id);const int count=keys.size();
      auto block=allocate(keys,input_width);
      Matrix<double> scratch;
      const auto values=dense_panel(block,scratch);
      if (cached) {
        auto dy=values.columns(0,count);
        generate(response,factors.get(id),dy);transform(dy,id*extent,count,false);
        if (joint) copy_panel(values.columns(count,count),dy);
        if (delta_upper.data) {
          Matrix<double> orbital(c,alpha,count);
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_y,
              relative_direction,orbital.view());
          add(c,dy,orbital.view());
        }
        if (scratch.data()) block.pack(c,values,false);
        return block;
      }
      auto y=values.columns(0,count);
      if (tangent) {
        auto dy=values.columns(count,count);
        const auto& f=factors.get(id);
        const auto* df=delta_upper.data ? &delta_factors.get(id):nullptr;
        clear(c,values);
        {
          const nvtx3::scoped_range range{"vb.source_generate"};
          generate_selected_jet<<<grid(alpha),256,0,c.stream()>>>(symbolic->starts.data(),symbolic->raw_beta.data(),
              incidence,endpoint.coefficients.view(),response,f.offsets.data(),f.indices.data(),f.values.data(),
              df ? df->offsets.data():nullptr,df ? df->indices.data():nullptr,df ? df->values.data():nullptr,
              count,joint,values);
        }
        check_cuda(cudaGetLastError(),"generate fused product source lanes");
        if (delta_upper.data) {
          alpha_transform.apply_directional_left(da,y,dy);
          if (joint) transform(values.columns(2*count,count),id*extent,count,false);
        } else transform(values,id*extent,count,false);
      } else {generate(endpoint.coefficients.view(),factors.get(id),y);transform(y,id*extent,count,false);}
      if (scratch.data()) block.pack(c,values,false);
      return block;
    },{},[&](int id) {return panel_bytes(selected(id),input_width);});
    const auto metadata_budget=quarter/8;
    while (!read_plans.empty() && read_plan_bytes>metadata_budget) {
      read_plan_bytes-=read_plans.back().bytes();read_plans.pop_back();
    }
    ReadPlan temporary_plan;
    std::size_t read_plan_hits=0,read_plan_uploads=0;
    const auto get_read_plan=[&](auto betas,auto rows) -> ReadPlan& {
      const auto matches=[&](const ReadPlan& plan) {
        return plan.extent==extent && plan.betas.size()==betas.size() &&
            plan.rows.size()==rows.size() &&
            std::equal(plan.betas.begin(),plan.betas.end(),betas.begin()) &&
            std::equal(plan.rows.begin(),plan.rows.end(),rows.begin());
      };
      auto found=std::find_if(read_plans.begin(),read_plans.end(),matches);
      if (found!=read_plans.end()) {
        ++read_plan_hits;
        read_plans.splice(read_plans.begin(),read_plans,found);
        return read_plans.front();
      }
      if (matches(temporary_plan)) {++read_plan_hits;return temporary_plan;}
      // The maximum plan size is linear in requested indices. Evict before
      // uploading a new plan so repeat calls can reuse its device index map.
      const auto bound=2*rows.size()*sizeof(int)+
          betas.size()*(3*sizeof(int)+sizeof(ReadGroup));
      const bool retain=bound<=metadata_budget;
      if (retain) while (!read_plans.empty() && read_plan_bytes>metadata_budget-bound) {
        read_plan_bytes-=read_plans.back().bytes();read_plans.pop_back();
      }
      ReadPlan plan;
      plan.extent=extent;
      plan.betas.assign(betas.begin(),betas.end());
      plan.rows.assign(rows.begin(),rows.end());
      plan.complete=true;
      std::vector<int> indices(rows.begin(),rows.end());
      for (int j=0;j<betas.size();++j) {
        const int b=betas[j];
        if (b<0 || b>=beta) throw std::out_of_range("invalid product reader beta column");
        const int id=b/extent;
        if (!reachable_panel[id]) {plan.complete=false;continue;}
        const int offset=(indices.size()-rows.size())/2;
        if (plan.groups.empty() || plan.groups.back().id!=id)
          plan.groups.push_back({id,offset,0});
        ++plan.groups.back().count;
        indices.push_back(b%extent);indices.push_back(j);
      }
      plan.indices=upload<int>(c,indices);
      ++read_plan_uploads;
      if (!retain) {temporary_plan=std::move(plan);return temporary_plan;}
      read_plan_bytes+=plan.bytes();
      read_plans.push_front(std::move(plan));
      return read_plans.front();
    };
    const auto reader=[&](int lane) -> CiTileReader {
      if (cached && lane==0) return accepted_y;
      const int input_lane=cached ? lane-1:lane;
      return [&,input_lane](auto betas,auto rows,linalg::MatrixView values) {
        if (betas.empty() || rows.empty()) return;
        auto& read_plan=get_read_plan(betas,rows);
        // Input panels can be evicted or rebuilt. Their device maps and
        // values are borrowed afresh before each same-stream kernel launch.
        // Every requested row is overwritten by the device gather. Clear only
        // when an entire, unreachable beta panel has no reader at all.
        if (!read_plan.complete) clear(c,values);
        static_assert(sizeof(ColumnReadIndex)==2*sizeof(int) && alignof(ColumnReadIndex)==alignof(int));
        const auto* columns=reinterpret_cast<const ColumnReadIndex*>(read_plan.indices.data()+rows.size());
        for (const auto& group:read_plan.groups) {
          if (resident_input) {
            auto& maps=resident_maps.get(group.id);
            read_resident_columns<<<dim3(group.count,grid(rows.size())),256,0,c.stream()>>>(
                maps.columns.data()+static_cast<std::size_t>(input_lane)*selected(group.id).size(),
                columns+group.offset,read_plan.indices.data(),rows.size(),values);
            check_cuda(cudaGetLastError(),"read resident product tiles");
          } else {
            if (audit) {++reads[source_stage];hits[source_stage]+=input.contains(group.id);}
            auto& block=input.get(group.id);
            read_columns<<<dim3(group.count,grid(rows.size())),256,0,c.stream()>>>(
                block.maps.data(),block.values.view(),columns+group.offset,
                read_plan.indices.data(),rows.size(),input_lane,values);
          }
        }
      };
    };
    const auto y=reader(0),dy=reader(tangent ? 1:0),q=joint ? reader(2):CiTileReader{};
    const int targets=joint ? 4:2;
    const auto projection_slots=static_cast<std::size_t>(targets)*
        (delta_upper.data && !cached ? 2U:1U)+
        (delta_upper.data && !cached ? 2U:0U);
    const auto projection_descriptors=static_cast<std::size_t>(panels)*targets*
        (delta_upper.data && !cached ? 2U:1U)*sizeof(double*);
    const bool resident_projection=ordered_beta && projection_descriptors<quarter &&
        products<=(quarter-projection_descriptors)/(sizeof(double)*projection_slots);
    std::vector<OrderedTiles> projected(targets),left_projected(targets);
    const std::array<linalg::MatrixView,4> destinations{result.hamiltonian.view(),result.overlap.view(),
        joint ? ci->hamiltonian.view():linalg::MatrixView{},joint ? ci->overlap.view():linalg::MatrixView{}};
    const int group_extent=std::max<std::size_t>(1,std::min<std::size_t>(groups,
        quarter/(4*sizeof(double)*targets)));
    struct Projection {
      Matrix<double> values;
      std::size_t bytes() const {return values.rows()*values.cols()*sizeof(double);}
    };
    // Reverse group tiles and reader plans have separate bounded allowances.
    mcscf::ColumnCache<Projection> projections(quarter/(cached ? 8:4),[&](int id) {
      Projection value{Matrix<double>(c,std::min<std::int64_t>(group_extent,groups-id*group_extent),targets)};
      clear(c,value.values.view());return value;
    },[&](int id,Projection& tile) {
      for (int k=0;k<targets;++k) if (destinations[k].data)
        contract_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(incidence,
            id*group_extent,tile.values.rows(),tile.values.view().column(k).data,destinations[k].data);
    },[&](int id) {
      return static_cast<std::size_t>(std::min<std::int64_t>(group_extent,groups-id*group_extent))*
          targets*sizeof(double);
    });
    // Match the CPU product jet: independent RHS share one alpha transform
    // and one beta-row traversal, then accumulate into distinct targets.
    const auto project_panel=[&](int id,linalg::MatrixView values,
        std::initializer_list<int> destinations,bool differentiate) {
      const nvtx3::scoped_range range{"vb.source_project",nvtx3::payload{
          (static_cast<std::uint64_t>(differentiate)<<32)|static_cast<unsigned>(id)}};
      const auto keys=selected(id);const int count=keys.size();
      const int lanes=destinations.size();
      const std::array<int,2> target{*destinations.begin(),lanes==2 ? *(destinations.begin()+1):0};
      Matrix<double> delta;
      if (differentiate) {
        delta=Matrix<double>(c,values.rows,values.cols);clear(c,delta.view());
        alpha_transform.apply_directional_left(da,values,delta.view(),true);
      } else {
        // Input and reverse factors borrow the same immutable support mask.
        transform(values,id*extent,count,true);
      }
      if (resident_projection) {
        for (int lane=0;lane<lanes;++lane) {
          const int destination=target[lane];
          if (projected[destination].empty()) projected[destination]=ordered_tiles(extent);
          add(c,projected[destination][id].view(),
              (differentiate ? delta.view():values).columns(lane*count,count));
          if (differentiate) {
            if (left_projected[destination].empty())
              left_projected[destination]=ordered_tiles(extent);
            add(c,left_projected[destination][id].view(),values.columns(lane*count,count));
          }
        }
        return;
      }
      const auto& f=factors.get(id);
      for (std::int64_t begin=0;begin<groups;begin+=group_extent) {
        auto& tile=projections.get(begin/group_extent);
        project_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(symbolic->raw_alpha.data()+begin,symbolic->raw_beta.data()+begin,
            tile.values.rows(),f.offsets.data(),f.indices.data(),f.values.data(),
            differentiate ? delta.view():values,count,lanes,target[0],target[1],tile.values.view());
        if (differentiate) {
          const auto& df=delta_factors.get(id);
          project_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(symbolic->raw_alpha.data()+begin,symbolic->raw_beta.data()+begin,
              tile.values.rows(),df.offsets.data(),df.indices.data(),df.values.data(),values,
              count,lanes,target[0],target[1],tile.values.view());
        }
      }
    };
    bool exchange_reduced=false,retain=false,retaining=true;
    mcscf::ColumnCache<ProductColumns> output(quarter,[&](int id) {
      auto block=allocate(selected(id),input_width);clear(c,block.values.view());return block;
    },[&](int id,ProductColumns& block) {
      const int previous_stage=source_stage;
      source_stage=reverse_stage;
      const nvtx3::scoped_range range{"vb.reverse_panel",nvtx3::payload{static_cast<std::uint64_t>(id)}};
      const int count=selected(id).size(),first=id*extent;
      Matrix<double> scratch;
      const auto values=dense_panel(block,scratch);
      if (scratch.data()) {clear(c,values);block.pack(c,values,true);}
      const auto hy=values.columns(0,count);
      if (!tangent) {
        if (retain && retaining) {
          const auto state_bytes=block.patterns.size()*sizeof(CiColumnMap)+
              static_cast<std::size_t>(block.values.rows())*2*sizeof(double);
          if (!endpoint.images->prepare(id,state_bytes)) retaining=false;
          if (retaining) {
            ProductColumns state(c,block.patterns,2);
            if (resident_input) {
              pack_resident_column<<<dim3(count,grid(alpha)),256,0,c.stream()>>>(
                  state.maps.data(),static_cast<linalg::ConstMatrixView>(resident_y[id].view()),
                  state.values.view());
              check_cuda(cudaGetLastError(),"pack resident accepted product source");
            } else {
              auto& source=input.get(id);
              copy_panel(state.values.view().columns(0,1),source.values.view().columns(0,1));
            }
            copy_panel(state.values.view().columns(1,1),block.values.view().columns(0,1));
            retaining=endpoint.images->retain(id,std::move(state));
          }
        }
        const std::array terms{CiStreamedGeneratorTerm{
            static_cast<linalg::ConstMatrixView>(hy),0}};
        const std::array batches{CiStreamedGeneratorBatch{y,
            {terms.data(),terms.size()}}};
        const std::array moments{result.generator.view()};
        if (!selected_only)
          sigma.generator_adjoint_streamed(first,batches,moments,false,
              exchange_reduced,true);
        project_panel(id,hy,{0},false);
      } else {
        const int z=cached ? 0:count,qs=cached ? count:2*count;
        const std::array lefts{static_cast<linalg::ConstMatrixView>(values.columns(z,count)),
            joint ? static_cast<linalg::ConstMatrixView>(values.columns(qs,count)):linalg::ConstMatrixView{}};
        const std::array moments{result.generator.view(),joint ? ci->generator.view():linalg::MatrixView{}};
        if (cached) {
          const std::array terms{CiStreamedGeneratorTerm{lefts[0],0},
              CiStreamedGeneratorTerm{lefts[1],1}};
          const std::array batches{CiStreamedGeneratorBatch{y,
              {terms.data(),joint ? 2U:1U}}};
          sigma.generator_adjoint_streamed(first,batches,
              {moments.data(),joint ? 2U:1U},false,exchange_reduced,true);
        } else {
          // Pair each right reader with precisely the left/output consumers
          // that need it. The y pair shares its beta-neighbor tile and event
          // traversal; dy and q add into distinct orbital outputs.
          const auto hleft=static_cast<linalg::ConstMatrixView>(hy);
          const std::array yterms{CiStreamedGeneratorTerm{lefts[0],0},
              CiStreamedGeneratorTerm{lefts[1],1}};
          const std::array dterms{CiStreamedGeneratorTerm{hleft,0}};
          const std::array qterms{CiStreamedGeneratorTerm{hleft,1}};
          const std::array batches{
              CiStreamedGeneratorBatch{y,{yterms.data(),joint ? 2U:1U}},
              CiStreamedGeneratorBatch{dy,{dterms.data(),dterms.size()}},
              CiStreamedGeneratorBatch{q,{qterms.data(),qterms.size()}}};
          sigma.generator_adjoint_streamed(first,
              {batches.data(),joint ? 3U:2U},
              {moments.data(),joint ? 2U:1U},false,exchange_reduced,true);
        }
        project_panel(id,values.columns(z,(joint ? 2:1)*count),
            joint ? std::initializer_list<int>{0,2}:std::initializer_list<int>{0},false);
        if (delta_upper.data && !cached) project_panel(id,hy,{0},true);
      }
      source_stage=previous_stage;
    },[&](int id) {return panel_bytes(selected(id),input_width);});
    const auto writer=[&](int lane) -> CiTileWriter {
      return [&,lane](auto betas,auto,const CiTileGather& gather) {
        for (int begin=0;begin<betas.size();) {
          const int id=betas[begin]/extent;
          const int first=begin;
          while (begin<betas.size() && betas[begin]/extent==id) ++begin;
          if (!reachable_panel[id]) continue;
          auto& block=output.get(id);
          const int slot=cached ? lane-1:lane;
          // Queue the consumer before the next get() can evict this panel.
          gather.panel({.first_beta=id*extent,.source_begin=first,.count=begin-first,
              .lane=slot,.max_rows=block.max_rows,.maps=block.maps.data(),.values=block.values.view()});
        }
      };
    };
    std::vector<CiTileRequest> requests;
    if (!cached) requests.push_back({y,writer(0),nullptr,tangent ? linalg::MatrixView{}:result.one.view(),
        tangent ? linalg::MatrixView{}:result.pair.view(),{},tangent});
    if (tangent) requests.push_back({dy,writer(1),delta_sigma,result.one.view(),result.pair.view()});
    if (joint) requests.push_back({q,writer(2),nullptr,ci->one.view(),ci->pair.view()});
    if (!overlap_only)
      sigma.apply_tiled_responses(1,requests,tangent ? y:CiTileReader{},support,quarter);
    if (!overlap_only && (sigma.exchange_parity() || !tangent) && !output.evictions) {
      std::vector<ProductColumns*> panels_view(panels,nullptr);
      for (int id=0;id<panels;++id) if (reachable_panel[id]) panels_view[id]=&output.get(id);
      if (sigma.exchange_parity() && !output.evictions) {
        while (!panels_view.empty() && !panels_view.back()) panels_view.pop_back();
        symbolic->project_exchange(panels_view,extent,sigma.exchange_parity());exchange_reduced=true;
      }
    }
    if (!selected_only && !tangent && !output.evictions) {
      endpoint.readers.reset();
      endpoint.extent=extent;endpoint.images=std::make_unique<ProductImages>(quarter);
      endpoint.images->begin(panels);
      for (int id=0;id<panels;++id) if (!reachable_panel[id]) endpoint.images->retain(id,{});
      retain=true;
    }
    output.finish();
    source_stage=terminal_stage;
    if (retain) endpoint.images->finish();
    Matrix<double> transpose;
    if (cached && delta_upper.data) {
      transpose=Matrix<double>(c,n,n);
      const double one=1.0,zero=0.0;
      check_cublas(cublasDgeam(c.blas(),CUBLAS_OP_T,CUBLAS_OP_T,n,n,&one,
          relative_direction.data,relative_direction.ld,&zero,relative_direction.data,
          relative_direction.ld,transpose.data(),n),"product relative direction transpose");
    }
    for (int id=0;id<panels;++id) if (reachable_panel[id]) {
      const int count=selected(id).size();
      Matrix<double> values(c,alpha,(tangent && joint ? 2:1)*count);
      auto dv=values.view().columns(0,count);
      auto dq=joint ? values.view().columns(count,count):linalg::MatrixView{};
      if (!tangent) {y(selected(id),all_rows,dv);project_panel(id,dv,{1},false);}
      else {
        dy(selected(id),all_rows,dv);
        if (joint) q(selected(id),all_rows,dq);
        if (cached) {
          const std::array terms{CiStreamedGeneratorTerm{
              static_cast<linalg::ConstMatrixView>(dv),0},
              CiStreamedGeneratorTerm{static_cast<linalg::ConstMatrixView>(dq),1}};
          const std::array batches{
              CiStreamedGeneratorBatch{accepted_y,{terms.data(),joint ? 2U:1U},
                  -endpoint.energy,false},
              CiStreamedGeneratorBatch{accepted_r,{terms.data(),joint ? 2U:1U},
                  1.0,true}};
          const std::array moments{result.generator.view(),joint ? ci->generator.view():linalg::MatrixView{}};
          sigma.generator_adjoint_streamed(id*extent,batches,
              {moments.data(),joint ? 2U:1U},false,sigma.exchange_parity()!=0,true);
        }
        project_panel(id,values.view(),joint ? std::initializer_list<int>{1,3}:
            std::initializer_list<int>{1},false);
        if (delta_upper.data && !cached) {y(selected(id),all_rows,dv);project_panel(id,dv,{1},true);}
        if (delta_upper.data && cached) {
          // G^T(HY) = G^T R + E G^T Y. Consume the two retained
          // accepted lanes once, then share their final reverse transform.
          values=Matrix<double>{};
          Matrix<double> terminal(c,alpha,2*count);
          auto gy=terminal.view().columns(0,count),gr=terminal.view().columns(count,count);
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_y,transpose.view(),gy,
              sigma.exchange_parity()!=0);
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_r,transpose.view(),gr,
              sigma.exchange_parity()!=0);
          if (sigma.exchange_parity()) multiply(c,terminal.view(),2.0);
          add(c,gr,gy,endpoint.energy);
          project_panel(id,terminal.view(),{1,0},false);
        }
      }
    }
    projections.finish();
    if (resident_projection) for (int k=0;k<targets;++k) if (!projected[k].empty()) {
      auto views=ordered_views(projected[k]);
      if (!left_projected[k].empty()) {
        auto left=ordered_views(left_projected[k]);
        beta_transform.apply_directional_right_tiles(db,left,views,true);
      } else beta_transform.apply_right_tiles(views,true);
      auto pointers=ordered_pointers(projected[k]);
      if (groups) {
        ordered_products_tiles<<<grid(groups),256,0,c.stream()>>>(incidence,
            symbolic->raw_alpha.data(),symbolic->raw_beta.data(),groups,{},
            pointers.data(),alpha,extent,destinations[k],true);
        check_cuda(cudaGetLastError(),"contract ordered product tiles");
      }
    }
    if (!selected_only) multiply(c,result.generator.view(),2.0);
    if (tangent && !cached) add_upper(c,result.generator.view(),result.one.view(),-4*endpoint.energy);
    if (joint) {multiply(c,ci->generator.view(),2.0);if (!cached) add_upper(c,ci->generator.view(),ci->one.view(),-4*endpoint.energy);}
    if (audit) {
      std::fprintf(stderr,"panel_audit tangent=%d joint=%d cached=%d resident=%d panels=%d input_evictions=%zu output_evictions=%zu input_peak=%zu\n",
          tangent,joint,cached,resident_input,panels,input.evictions,output.evictions,input.peak_bytes);
      std::fprintf(stderr,"panel_ordered input=%d projection=%d products=%zu sampled_beta=%d\n",
          resident_input,resident_projection,products,sampled_beta);
      const char* names[]={"sigma","reverse","terminal"};
      for (int s=0;s<3;++s) std::fprintf(stderr,
          "panel_source stage=%s builds=%zu rebuilds=%zu reads=%zu hits=%zu\n",
          names[s],builds[s],rebuilds[s],reads[s],hits[s]);
      std::fprintf(stderr,"panel_reader_plan hits=%zu uploads=%zu retained_bytes=%zu\n",
          read_plan_hits,read_plan_uploads,read_plan_bytes);
    }
    return result;
  }

  void copy_panel(linalg::MatrixView out,linalg::ConstMatrixView in) {
    check_cuda(cudaMemcpy2DAsync(out.data,out.ld*sizeof(double),in.data,in.ld*sizeof(double),
        in.rows*sizeof(double),in.cols,cudaMemcpyDeviceToDevice,c.stream()),"copy product panel");
  }

  double maximum(ProductIncidence incidence,linalg::ConstMatrixView source,std::size_t bytes) {
    const nvtx3::scoped_range range{"vb.product_maximum"};
    if (source.rows!=incidence.size || source.cols!=1 || source.ld<source.rows)
      throw std::invalid_argument("invalid selected product maximum source");
    const auto allowance=std::max<std::size_t>(1,bytes/5);
    const int extent=panel_extent(allowance);
    const auto reduce=[&](linalg::ConstMatrixView values) {
      int index=0;
      check_cublas(cublasSetPointerMode(c.blas(),CUBLAS_POINTER_MODE_HOST),
          "product panel maximum pointer mode");
      check_cublas(cublasIdamax(c.blas(),blas_int(values.rows*values.cols),
          values.data,1,&index),"product panel maximum index");
      if (index<=0) throw std::runtime_error("empty product panel maximum");
      double value=0.0;c.copy_to_host(&value,values.data+index-1,sizeof(double));
      if (!std::isfinite(value)) throw std::runtime_error("non-finite product panel maximum");
      return std::abs(value);
    };
    const auto alpha_size=static_cast<std::size_t>(alpha);
    const auto beta_size=static_cast<std::size_t>(beta);
    const auto tile_count=(beta_size-1)/extent+1;
    const auto pointer_bytes=(tile_count+2*beta_size)*sizeof(double*);
    const auto stripe_bytes=beta_size*std::min<std::size_t>(
        product_panel_width(),alpha>1 ? alpha_size-1:0)*sizeof(double);
    // Leave workspace for support metadata and the large-spin stripe fallback.
    const bool fits=alpha_size<=(allowance/2)/(beta_size*sizeof(double)) &&
        pointer_bytes+stripe_bytes<=allowance/4 &&
        alpha_size*beta_size*sizeof(double)<=c.available_device_bytes()/2;
    if (fits) {
      if (support && !support->full()) {
        symbolic->prepare_patterns(allowance/4);
        prepare_masks(extent,allowance/4);
      }
      auto tiles=ordered_tiles(extent);
      if (source.data && groups) {
        auto pointers=ordered_pointers(tiles);
        ordered_products_tiles<<<grid(groups),256,0,c.stream()>>>(incidence,
            symbolic->raw_alpha.data(),symbolic->raw_beta.data(),groups,source,
            pointers.data(),alpha,extent,{},false);
        check_cuda(cudaGetLastError(),"scatter maximum ordered products");
      }
      auto views=ordered_views(tiles);
      for (int id=0;id<static_cast<int>(views.size());++id)
        transform(views[id],id*extent,views[id].cols,false);
      beta_transform.apply_right_tiles(views);
      double maximum=0.0;
      for (int id=0;id<static_cast<int>(views.size());++id) {
        const int first=id*extent,count=views[id].cols;
        const auto touched=std::lower_bound(symbolic->reachable.begin(),symbolic->reachable.end(),first);
        if (touched==symbolic->reachable.end() || *touched>=first+count) continue;
        if (support && !support->full()) {
          // Match the sampled path's exact output-column support after beta mixing.
          const int words=(alpha+63)/64;
          restrict_maximum_tile<<<grid(static_cast<std::size_t>(alpha)*count),256,0,c.stream()>>>(
              views[id],mask_cache->get(first).values.data(),words);
          check_cuda(cudaGetLastError(),"restrict maximum ordered tile");
        }
        maximum=std::max(maximum,reduce(views[id]));
      }
      return maximum;
    }
    symbolic->prepare_patterns(allowance/4);
    prepare_factors(extent,allowance/4);
    prepare_masks(extent,allowance/4);
    double maximum=0.0;
    for (int first=0;first<beta;first+=extent) {
      const int count=std::min(extent,beta-first);
      const auto touched=std::lower_bound(symbolic->reachable.begin(),symbolic->reachable.end(),first);
      if (touched==symbolic->reachable.end() || *touched>=first+count) continue;
      Matrix<double> values(c,alpha,count);
      generate_source(incidence,source,factor_cache->get(first/extent),values.view());
      transform(values.view(),first,count,false);
      maximum=std::max(maximum,reduce(values.view()));
    }
    return maximum;
  }
};

ProductContraction::ProductContraction(Context& c,const vb::StructureAction::OrthogonalView& view,
    const ExteriorTransform& alpha,const ExteriorTransform& beta,const mcscf::ProductSupport* support)
    :impl_(std::make_unique<Impl>(c,view,alpha,beta,support)) {}
ProductContraction::ProductContraction(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
ProductContraction::~ProductContraction()=default;
std::unique_ptr<ProductContraction> ProductContraction::clone() const {
  return std::unique_ptr<ProductContraction>(new ProductContraction(std::make_unique<Impl>(
      impl_->c,impl_->alpha_transform,impl_->beta_transform,impl_->symbolic,impl_->beta_shears)));
}
void ProductContraction::update_frame() {impl_->factor_cache.reset();}
void ProductContraction::release_cached_workspace(std::size_t bytes) const {
  auto& p=*impl_;
  while (bytes>p.c.available_device_bytes()) {
    if (p.factor_cache && p.factor_cache->evict_one()) continue;
    if (p.mask_cache && p.mask_cache->evict_one()) continue;
    if (!p.read_plans.empty()) {
      p.read_plan_bytes-=p.read_plans.back().bytes();p.read_plans.pop_back();continue;
    }
    break;
  }
}
void ProductContraction::accept(ProductEndpoint& endpoint,double energy) const {
  endpoint.energy=energy;
  if (!endpoint.images || !endpoint.images->complete()) return;
  for (int id=0;id<endpoint.images->size();++id) {
    auto& panel=endpoint.images->at(id);
    if (!panel.values.rows()) continue;
    add(impl_->c,panel.values.view().columns(1,1),panel.values.view().columns(0,1),-energy);
  }
}
void ProductContraction::apply_selected(ProductIncidence incidence,linalg::ConstMatrixView source,
    const DirectCiSigma& sigma,linalg::MatrixView h,linalg::MatrixView m,
    std::size_t bytes,bool overlap_only) const {
  if (source.rows!=incidence.size || source.cols<=0 || m.rows!=source.rows ||
      m.cols!=source.cols || (!overlap_only &&
      (h.rows!=source.rows || h.cols!=source.cols || !h.data)))
    throw std::invalid_argument("invalid selected product action block");
  ProductEndpoint endpoint{Matrix<double>(impl_->c,incidence.size,1)};
  for (int rhs=0;rhs<source.cols;++rhs) {
    impl_->copy_panel(endpoint.coefficients.view(),source.column(rhs));
    auto result=impl_->jet(incidence,sigma,endpoint,{},{},{},nullptr,bytes,nullptr,{},true,overlap_only);
    if (!overlap_only) impl_->copy_panel(h.column(rhs),result.hamiltonian.view());
    impl_->copy_panel(m.column(rhs),result.overlap.view());
  }
}
double ProductContraction::maximum_orthogonal_component(ProductIncidence incidence,
    linalg::ConstMatrixView source,std::size_t bytes) const {
  return impl_->maximum(incidence,source,bytes);
}
ProductJet ProductContraction::jet(ProductIncidence incidence,const DirectCiSigma& sigma,
    ProductEndpoint& endpoint,linalg::ConstMatrixView upper,linalg::ConstMatrixView response,
    linalg::ConstMatrixView delta_upper,const DirectCiSigma* delta_sigma,
    std::size_t bytes,ProductJet* ci,linalg::ConstMatrixView relative_direction) const {
  return impl_->jet(incidence,sigma,endpoint,upper,response,delta_upper,delta_sigma,bytes,ci,relative_direction);
}

} // namespace xmvb::cuda
