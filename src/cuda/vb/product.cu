#include "cuda/vb/product.hpp"
#include "cuda/vb/columns.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <vector>
#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/column_cache.hpp"
#include "mcscf/ci/product_support.hpp"

namespace xmvb::cuda {
namespace {
unsigned grid(std::size_t n) {return static_cast<unsigned>(std::min<std::size_t>(65535,(n+255)/256));}
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
  Factors(Context& c,cusparseHandle_t handle,linalg::ConstMatrixView dense):offsets(c,dense.rows+1) {
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
      Buffer<std::byte> scratch(c,bytes);
      sparse_check(cusparseDenseToSparse_analysis(handle,input,output,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,scratch.data()),
          "analyze beta-factor exact zeros");
      c.synchronize();
      std::int64_t rows,columns,nnz;
      sparse_check(cusparseSpMatGetSize(output,&rows,&columns,&nnz),"count beta-factor exact nonzeros");
      if (nnz>std::numeric_limits<int>::max()) throw std::overflow_error("beta-factor CSR exceeds integer range");
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

__global__ void generate(const int* starts,const int* raw_beta,linalg::ConstMatrixView coefficients,
    const int* offsets,const int* columns,const double* factors,int count,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(output.rows)*coefficients.cols;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int a=i%output.rows,v=i/output.rows;
    for (int group=starts[a];group<starts[a+1];++group) {
      const double value=coefficients.data[group+coefficients.ld*v];
      for (int p=offsets[raw_beta[group]];p<offsets[raw_beta[group]+1];++p)
        output.data[a+output.ld*(v*count+columns[p])]+=value*factors[p];
    }
  }
}

__global__ void project(const int* raw_alpha,const int* raw_beta,int groups,
    const int* offsets,const int* columns,const double* factors,int count,
    linalg::ConstMatrixView image,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(groups)*output.cols;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int g=i%groups,v=i/groups,b=raw_beta[g];double value=0;
    for (int p=offsets[b];p<offsets[b+1];++p)
      value+=image.data[raw_alpha[g]+image.ld*(v*count+columns[p])]*factors[p];
    output.data[g+output.ld*v]+=value;
  }
}

struct ColumnRead { const int* rows; linalg::ConstMatrixView values; int destination; };

__global__ void read_columns(const ColumnRead* columns,const int* requested,int count,
    int width,linalg::MatrixView output) {
  const auto source=columns[blockIdx.x];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(count)*width;i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i%count,v=i/count,a=requested[r];int index=a;
    if (source.rows) {
      int first=0,last=source.values.rows;
      while (first<last) {const int mid=first+(last-first)/2;if (source.rows[mid]<a) first=mid+1;else last=mid;}
      index=first<source.values.rows && source.rows[first]==a ? first:-1;
    }
    output.data[r+output.ld*(source.destination*width+v)]=index<0 ? 0.0:source.values.data[index+source.values.ld*v];
  }
}

struct RetainedPattern {
  std::shared_ptr<const ProductColumnPattern> pattern;
  std::size_t bytes() const {return 2*pattern->rows.size()*sizeof(int);}
};
} // namespace

struct ProductContraction::Impl {
  Context& c;
  const ExteriorTransform& alpha_transform;
  const ExteriorTransform& beta_transform;
  int alpha,beta,groups,factor_extent=0;
  std::shared_ptr<const mcscf::ProductSupport> support;
  Buffer<int> starts,raw_alpha,raw_beta;
  cusparseHandle_t sparse=nullptr;
  std::vector<int> reachable,beta_positions;
  std::vector<std::weak_ptr<const ProductColumnPattern>> patterns;
  std::shared_ptr<const ProductColumnPattern> full_pattern;
  std::unique_ptr<mcscf::ColumnCache<RetainedPattern>> pattern_cache;
  std::unique_ptr<mcscf::ColumnCache<Factors>> factor_cache;

  Impl(Context& ctx,const vb::StructureAction::OrthogonalView& view,
      const ExteriorTransform& a,const ExteriorTransform& b,std::shared_ptr<const mcscf::ProductSupport> exact_support)
      :c(ctx),alpha_transform(a),beta_transform(b),alpha(a.dimension()),beta(b.dimension()),
       groups(view.products.size()),support(std::move(exact_support)),beta_positions(beta,-1),patterns(beta) {
    if (support) reachable=support->columns();
    else {reachable.resize(beta);std::iota(reachable.begin(),reachable.end(),0);}
    for (int i=0;i<reachable.size();++i) beta_positions[reachable[i]]=i;
    std::vector<int> offsets(alpha+1,0),aa(groups),bb(groups);
    for (int g=0;g<groups;++g) {
      aa[g]=view.products[g]/beta;bb[g]=view.products[g]%beta;++offsets[aa[g]+1];
    }
    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
    starts=upload<int>(c,offsets);raw_alpha=upload<int>(c,aa);raw_beta=upload<int>(c,bb);
    sparse_check(cusparseCreate(&sparse),"create product contraction sparse handle");
    sparse_check(cusparseSetStream(sparse,c.stream()),"set product contraction stream");
  }
  ~Impl() {if (sparse) cusparseDestroy(sparse);}

  std::span<const int> keys(int block,int extent) const {
    const std::size_t first=static_cast<std::size_t>(block)*extent;
    return std::span<const int>{reachable}.subspan(first,std::min<std::size_t>(extent,reachable.size()-first));
  }

  std::shared_ptr<const ProductColumnPattern> pattern(int b) {
    if (!support && full_pattern) return full_pattern;
    if (pattern_cache->contains(b)) return pattern_cache->get(b).pattern;
    if (auto found=patterns[b].lock()) return found;
    auto result=std::make_shared<ProductColumnPattern>();result->full=!support;
    if (support) support->append_column_rows(b,result->rows);
    else {result->rows.resize(alpha);std::iota(result->rows.begin(),result->rows.end(),0);}
    result->size=result->rows.size();result->device_rows=upload<int>(c,result->rows);
    patterns[b]=result;
    if (support) pattern_cache->insert(b,{result});else full_pattern=result;
    return result;
  }

  ProductColumns allocate(std::span<const int> keys,int width) {
    std::vector<std::shared_ptr<const ProductColumnPattern>> columns;
    columns.reserve(keys.size());
    for (int key:keys) columns.push_back(pattern(key));
    return ProductColumns(c,std::move(columns),width);
  }

  void transform(linalg::MatrixView values,const ProductColumns& block,bool adjoint) {
    const nvtx3::scoped_range range{adjoint ? "vb.product_alpha_adjoint":"vb.product_alpha"};
    if (!support) {alpha_transform.apply_left(values,adjoint);return;}
    std::vector<int> offsets{0},rows;
    for (const auto& p:block.patterns) {
      rows.insert(rows.end(),p->rows.begin(),p->rows.end());offsets.push_back(rows.size());
    }
    alpha_transform.apply_supported_left(values,offsets,rows,adjoint);
  }

  void apply(linalg::ConstMatrixView coefficients,const DirectCiSigma& sigma,
      linalg::MatrixView h,linalg::MatrixView m,std::size_t bytes,bool overlap_only) {
    const nvtx3::scoped_range range{"vb.product_tiles"};
    if (coefficients.rows!=groups || coefficients.cols<=0 || h.rows!=groups || m.rows!=groups ||
        h.cols!=coefficients.cols || m.cols!=coefficients.cols)
      throw std::invalid_argument("invalid product contraction coordinates");
    const int width=blas_int(coefficients.cols);
    const auto quarter=std::max<std::size_t>(1,bytes/4);
    if (!pattern_cache) pattern_cache=std::make_unique<mcscf::ColumnCache<RetainedPattern>>(quarter/4,
        [](int)->RetainedPattern {throw std::logic_error("support rows require exact generation");});
    const auto row_bytes=sizeof(double)*(static_cast<std::size_t>(alpha)*width+beta);
    const int extent=std::min<std::size_t>(reachable.size(),std::max<std::size_t>(1,quarter/(8*row_bytes)));
    if (!extent) throw std::logic_error("empty reachable product support");
    if (factor_extent!=extent) {factor_cache.reset();factor_extent=extent;}
    if (!factor_cache) factor_cache=std::make_unique<mcscf::ColumnCache<Factors>>(quarter/2,[this](int block) {
      const nvtx3::scoped_range range{"vb.product_beta_rows"};
      const auto selected=keys(block,factor_extent);
      Matrix<double> values(c,beta,selected.size());
      beta_transform.sample_rows(selected,values.view());
      return Factors(c,sparse,values.view());
    });
    clear(c,h);clear(c,m);
    const int blocks=(reachable.size()+extent-1)/extent;
    std::vector<unsigned char> overlap_consumed(blocks,0);
    const auto project_block=[&](int id,ProductColumns& block,linalg::MatrixView image,linalg::MatrixView target) {
      const auto& factors=factor_cache->get(id);
      transform(image,block,true);
      project<<<grid(static_cast<std::size_t>(groups)*width),256,0,c.stream()>>>(raw_alpha.data(),raw_beta.data(),groups,
          factors.offsets.data(),factors.indices.data(),factors.values.data(),block.patterns.size(),image,target);
    };
    const auto build=[&](int id) {
      const nvtx3::scoped_range range{"vb.product_generate"};
      const auto selected=keys(id,extent);
      auto block=allocate(selected,width);
      const auto& factors=factor_cache->get(id);
      Matrix<double> values(c,alpha,selected.size()*width);clear(c,values.view());
      generate<<<grid(static_cast<std::size_t>(alpha)*width),256,0,c.stream()>>>(starts.data(),raw_beta.data(),coefficients,
          factors.offsets.data(),factors.indices.data(),factors.values.data(),selected.size(),values.view());
      transform(values.view(),block,false);block.pack(c,values.view(),false);
      if (!overlap_consumed[id]) {
        project_block(id,block,values.view(),m);overlap_consumed[id]=1;
      }
      return block;
    };
    mcscf::ColumnCache<ProductColumns> input(quarter,build);
    mcscf::ColumnCache<ProductColumns> output(quarter,[&](int id) {
      auto block=allocate(keys(id,extent),width);clear(c,block.values.view());return block;
    },[&](int id,ProductColumns& block) {
      const nvtx3::scoped_range range{"vb.product_project"};
      Matrix<double> image(c,alpha,block.patterns.size()*width);clear(c,image.view());
      block.pack(c,image.view(),true);project_block(id,block,image.view(),h);
    });
    // A batch has one owner and byte charge. Early column retirement only
    // releases its block after the final member; LRU eviction may project a
    // partial block earlier. Both are exact by linearity of the projection.
    using Entries=std::vector<std::pair<int,int>>;
    const auto partition=[&](std::span<const int> betas) {
      std::map<int,Entries> result;
      for (int j=0;j<betas.size();++j) {
        const int pos=beta_positions[betas[j]];
        if (pos>=0) result[pos/extent].emplace_back(pos%extent,j);
      }
      return result;
    };
    const CiTileReader read=[&](auto betas,auto rows,linalg::MatrixView values) {
      clear(c,values);const auto device_rows=upload<int>(c,rows);
      const auto partitions=partition(betas);
      const auto copy=[&](ProductColumns& block,const Entries& entries) {
        std::vector<ColumnRead> columns;columns.reserve(entries.size());
        for (auto [pos,destination]:entries) {
          const auto& p=block.patterns[pos];
          columns.push_back({p->full ? nullptr:p->device_rows.data(),block.column(pos),destination});
        }
        const auto device_columns=upload<ColumnRead>(c,columns);
        if (!rows.empty()) read_columns<<<dim3(columns.size(),grid(rows.size()*width)),256,0,c.stream()>>>(
            device_columns.data(),device_rows.data(),rows.size(),width,values);
      };
      // Borrowed hits are consumed before any insertion can evict their block.
      std::vector<int> missing;
      for (const auto& [id,entries]:partitions) {
        if (input.contains(id)) copy(input.get(id),entries);else missing.push_back(id);
      }
      for (int id:missing) {
        auto block=build(id);copy(block,partitions.at(id));input.insert(id,std::move(block));
      }
    };
    const CiTileWriter write=[&](auto betas,auto,const CiTileGather& gather) {
      for (const auto& [id,entries]:partition(betas)) {
        auto& block=output.get(id);
        std::vector<CiTileColumn> columns;columns.reserve(entries.size());
        for (auto [pos,source]:entries) if (block.patterns[pos]->size)
          columns.push_back({source,block.patterns[pos]->device_rows.data(),block.column(pos)});
        gather(columns,true);
      }
    };
    if (!overlap_only) {
      if (quarter/sizeof(double)/alpha/beta>=static_cast<std::size_t>(width))
        for (int id=0;id<blocks;++id) input.get(id);
      std::vector<int> remaining(blocks);
      for (int id=0;id<blocks;++id) remaining[id]=keys(id,extent).size();
      const CiTileRequest request{read,write};
      sigma.apply_tiled_responses(width,{&request,1},{},support.get(),quarter,[&](auto columns) {
        for (int b:columns) if (beta_positions[b]>=0) {
          const int id=beta_positions[b]/extent;
          if (!--remaining[id]) {input.release(id);output.release(id);}
        }
      });
      output.finish();
    }
    for (int id=0;id<blocks;++id) if (!overlap_consumed[id]) build(id);
    check_cuda(cudaGetLastError(),"broad-support product contraction");
  }
};

ProductContraction::ProductContraction(Context& c,const vb::StructureAction::OrthogonalView& view,
    const ExteriorTransform& alpha,const ExteriorTransform& beta,std::shared_ptr<const mcscf::ProductSupport> support)
    :impl_(std::make_unique<Impl>(c,view,alpha,beta,std::move(support))) {}
ProductContraction::~ProductContraction()=default;
void ProductContraction::update_frame() {impl_->factor_cache.reset();}
void ProductContraction::apply(linalg::ConstMatrixView coefficients,
    const DirectCiSigma& sigma,linalg::MatrixView h,linalg::MatrixView m,std::size_t bytes,bool overlap_only) const {
  impl_->apply(coefficients,sigma,h,m,bytes,overlap_only);
}

} // namespace xmvb::cuda
