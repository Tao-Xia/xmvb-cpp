#include "cuda/vb/product.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/column_cache.hpp"
#include "mcscf/ci/product_support.hpp"
#include "parallel/openmp.hpp"

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
  check_cuda(cudaMemset2DAsync(panel.data,panel.ld*sizeof(double),0,panel.rows*sizeof(double),panel.cols,c.stream()),
      "clear product tile");
}

/** Official exact-zero dense-to-CSR conversion. Rows are raw beta strings;
 * columns are requested transformed beta rows. No cutoff or numerical host
 * staging is used. The temporary dense factor is a single-spin panel only.
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
      if (input) cusparseDestroyDnMat(input);if (output) cusparseDestroySpMat(output);throw;
    }
    cusparseDestroyDnMat(input);cusparseDestroySpMat(output);
  }
};

__global__ void generate(const int* starts,const int* raw_beta,linalg::ConstMatrixView coefficients,
    const int* offsets,const int* columns,const double* factors,int count,linalg::MatrixView output) {
  // Same CPU ownership: one raw alpha/RHS row walks its sparse beta factors.
  // Each output has one owner; no atomics, determinant vector or raw product grid.
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(output.rows)*coefficients.cols;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int a=i%output.rows,v=i/output.rows;
    for (int group=starts[a];group<starts[a+1];++group) {
      const double value=coefficients.data[group+coefficients.ld*v];
      const int b=raw_beta[group];
      for (int p=offsets[b];p<offsets[b+1];++p)
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

/** Copy one compact column to/from a multi-column full-row staging panel. */
__global__ void pack_column(const int* rows,int size,int beta,int count,
    linalg::MatrixView full,linalg::MatrixView compact,bool unpack) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(size)*compact.cols;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int r=i%size,v=i/size,a=rows ? rows[r]:r;
    auto& destination=unpack ? full.data[a+full.ld*(v*count+beta)]:compact.data[r+compact.ld*v];
    destination=unpack ? compact.data[r+compact.ld*v]:full.data[a+full.ld*(v*count+beta)];
  }
}

__global__ void copy_halo(const int* source_rows,int size,const int* requested,int count,
    int column,int width,linalg::ConstMatrixView source,linalg::MatrixView output) {
  for (auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(count)*width;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int r=i%count,v=i/count,a=requested[r];int index=a;
    if (source_rows) {
      int first=0,last=size;
      while (first<last) {const int mid=first+(last-first)/2;if (source_rows[mid]<a) first=mid+1;else last=mid;}
      index=first<size && source_rows[first]==a ? first:-1;
    }
    output.data[r+output.ld*(column*width+v)]=index<0 ? 0.0:source.data[index+source.ld*v];
  }
}

struct Pattern {
  bool full=false;
  int size=0;
  std::vector<int> rows;
  Buffer<int> device_rows;
};
struct Column {
  std::shared_ptr<const Pattern> pattern;
  Matrix<double> values;
  std::size_t bytes() const {
    return values.rows()*values.cols()*sizeof(double)+pattern->rows.size()*sizeof(int);
  }
};
struct FactorRow {
  Matrix<double> values;
  std::size_t bytes() const {return values.rows()*sizeof(double);}
};
} // namespace

struct ProductContraction::Impl {
  Context& c;
  const ExteriorTransform& alpha_transform;
  const ExteriorTransform& beta_transform;
  int alpha,beta,groups;
  std::shared_ptr<const mcscf::ProductSupport> support;
  Buffer<int> starts,raw_alpha,raw_beta;
  cusparseHandle_t sparse=nullptr;
  std::vector<std::weak_ptr<const Pattern>> patterns;
  std::vector<unsigned char> column_support;
  mutable std::unique_ptr<mcscf::ColumnCache<FactorRow>> row_cache;

  Impl(Context& ctx,const vb::StructureAction::OrthogonalView& view,
      const ExteriorTransform& a,const ExteriorTransform& b,std::shared_ptr<const mcscf::ProductSupport> exact_support)
      :c(ctx),alpha_transform(a),beta_transform(b),alpha(a.dimension()),beta(b.dimension()),
       groups(view.products.size()),patterns(beta),column_support(beta,1) {
    support=std::move(exact_support);
    if (support) {
      std::fill(column_support.begin(),column_support.end(),0);
      for (int b:support->columns()) column_support[b]=1;
    }
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

  std::shared_ptr<const Pattern> pattern(int b) {
    if (auto found=patterns[b].lock()) return found;
    auto result=std::make_shared<Pattern>();result->full=!support;
    if (support) {
      support->append_column_rows(b,result->rows);
      result->size=result->rows.size();result->device_rows=upload<int>(c,result->rows);
    } else result->size=alpha;
    patterns[b]=result;return result;
  }

  Matrix<double> sample(std::span<const int> keys,linalg::ConstMatrixView upper) const {
    Matrix<double> output(c,beta,keys.size());
    std::vector<int> missing,positions(keys.size(),-1),pending(beta,-1);
    for (int i=0;i<keys.size();++i) {
      if (!row_cache->contains(keys[i])) {
        if (pending[keys[i]]<0) {pending[keys[i]]=missing.size();missing.push_back(keys[i]);}
        positions[i]=pending[keys[i]];continue;
      }
      check_cuda(cudaMemcpyAsync(output.view().column(i).data,row_cache->get(keys[i]).values.data(),
          beta*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()),"read retained beta factor");
    }
    if (!missing.empty()) {
      Matrix<double> generated(c,beta,missing.size());
      beta_transform.generate_rows(upper,missing,generated.view());
      for (int i=0;i<keys.size();++i) if (positions[i]>=0)
        check_cuda(cudaMemcpyAsync(output.view().column(i).data,generated.view().column(positions[i]).data,
            beta*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()),"stage requested beta factor");
      for (int j=0;j<missing.size();++j) {
        FactorRow row{Matrix<double>(c,beta,1)};
        check_cuda(cudaMemcpyAsync(row.values.data(),generated.view().column(j).data,
            beta*sizeof(double),cudaMemcpyDeviceToDevice,c.stream()),"retain requested beta factor");
        row_cache->insert(missing[j],std::move(row));
      }
    }
    return output;
  }

  void transform(linalg::MatrixView values,const std::vector<std::shared_ptr<const Pattern>>& columns,bool adjoint) {
    if (!support) {alpha_transform.apply_left(values,adjoint);return;}
    std::vector<int> offsets{0},rows;
    for (const auto& column:columns) {
      rows.insert(rows.end(),column->rows.begin(),column->rows.end());offsets.push_back(rows.size());
    }
    alpha_transform.apply_supported_left(values,offsets,rows,adjoint);
  }

  void apply(linalg::ConstMatrixView coefficients,linalg::ConstMatrixView upper,
      const DirectCiSigma& sigma,linalg::MatrixView h,linalg::MatrixView m,std::size_t bytes,bool overlap_only) {
    const nvtx3::scoped_range range{"vb.product_tiles"};
    if (coefficients.rows!=groups || coefficients.cols<=0 || h.rows!=groups || m.rows!=groups ||
        h.cols!=coefficients.cols || m.cols!=coefficients.cols)
      throw std::invalid_argument("invalid product contraction coordinates");
    const int width=blas_int(coefficients.cols);
    const auto quarter=std::max<std::size_t>(1,bytes/4);
    const auto row_bytes=sizeof(double)*(static_cast<std::size_t>(alpha)*width+beta);
    const auto parallel_extent=static_cast<std::size_t>(cooperative_openmp_thread_count())*8;
    const int extent=std::min<std::size_t>(beta,std::max<std::size_t>(1,std::min(parallel_extent,quarter/(8*row_bytes))));
    if (!row_cache) row_cache=std::make_unique<mcscf::ColumnCache<FactorRow>>(quarter/2,
        [](int)->FactorRow {throw std::logic_error("beta factors require batched generation");});
    clear(c,h);clear(c,m);
    std::vector<unsigned char> overlap_consumed(beta,0);
    const auto project_panel=[&](std::span<const int> keys,
        const std::vector<std::shared_ptr<const Pattern>>& columns,linalg::MatrixView image,
        const Factors& factors,linalg::MatrixView target) {
      transform(image,columns,true);
      project<<<grid(static_cast<std::size_t>(groups)*width),256,0,c.stream()>>>(raw_alpha.data(),raw_beta.data(),groups,
          factors.offsets.data(),factors.indices.data(),factors.values.data(),keys.size(),image,target);
    };
    const auto build=[&](std::span<const int> keys) {
      std::vector<std::shared_ptr<const Pattern>> columns;
      for (int key:keys) columns.push_back(pattern(key));
      auto sampled=sample(keys,upper);Factors factors(c,sparse,sampled.view());
      Matrix<double> values(c,alpha,keys.size()*width);clear(c,values.view());
      generate<<<grid(static_cast<std::size_t>(alpha)*width),256,0,c.stream()>>>(starts.data(),raw_beta.data(),coefficients,
          factors.offsets.data(),factors.indices.data(),factors.values.data(),keys.size(),values.view());
      transform(values.view(),columns,false);
      std::vector<Column> result;result.reserve(keys.size());
      for (int b=0;b<keys.size();++b) {
        const auto& p=columns[b];result.push_back({p,Matrix<double>(c,p->size,width)});
        if (p->size) pack_column<<<grid(static_cast<std::size_t>(p->size)*width),256,0,c.stream()>>>(
            p->device_rows.data(),p->size,b,keys.size(),values.view(),result.back().values.view(),false);
      }
      // Consume overlap once per newly requested column. The raw panel is
      // dead after compact packing; its storage becomes the overlap pullback.
      if (std::any_of(keys.begin(),keys.end(),[&](int key) {return !overlap_consumed[key];})) {
        for (int b=0;b<keys.size();++b) if (overlap_consumed[keys[b]])
          for (int v=0;v<width;++v) clear(c,values.view().column(v*keys.size()+b));
        project_panel(keys,columns,values.view(),factors,m);
        for (int key:keys) overlap_consumed[key]=1;
      }
      return result;
    };
    mcscf::ColumnCache<Column> input(quarter,[](int)->Column {
      throw std::logic_error("product inputs require exact-demand batches");});
    std::vector<int> completed;
    std::vector<Column> completed_images;
    const auto flush=[&] {
      if (completed.empty()) return;
      const int count=completed.size();
      Matrix<double> image(c,alpha,count*width);clear(c,image.view());
      std::vector<std::shared_ptr<const Pattern>> columns;
      for (int b=0;b<count;++b) {
        const auto& p=completed_images[b].pattern;columns.push_back(p);
        if (p->size) pack_column<<<grid(static_cast<std::size_t>(p->size)*width),256,0,c.stream()>>>(
            p->device_rows.data(),p->size,b,count,image.view(),completed_images[b].values.view(),true);
      }
      auto sampled=sample(completed,upper);Factors factors(c,sparse,sampled.view());
      project_panel(completed,columns,image.view(),factors,h);
      completed.clear();completed_images.clear();
    };
    mcscf::ColumnCache<Column> output(quarter,[&](int key) {
      auto p=pattern(key);Column column{p,Matrix<double>(c,p->size,width)};
      if (p->size) clear(c,column.values.view());return column;
    },[&](int key,Column& column) {
      completed.push_back(key);completed_images.push_back(std::move(column));
      if (completed.size()==extent) flush();
    });
    const auto produce=[&](std::span<const int> missing,const auto& consume) {
      for (std::size_t first=0;first<missing.size();first+=extent) {
        const auto keys=missing.subspan(first,std::min<std::size_t>(extent,missing.size()-first));
        auto tiles=build(keys);consume(keys,tiles);
        for (int j=0;j<keys.size();++j) input.insert(keys[j],std::move(tiles[j]));
      }
    };
    const CiTileReader read=[&](auto betas,auto rows,linalg::MatrixView values) {
      clear(c,values);
      const auto device_rows=upload<int>(c,rows);
      std::vector<int> missing,positions;
      const auto copy=[&](int column,const Column& panel) {
        copy_halo<<<grid(rows.size()*width),256,0,c.stream()>>>(panel.pattern->device_rows.data(),panel.pattern->size,
            device_rows.data(),rows.size(),column,width,panel.values.view(),values);
      };
      // All hits are consumed before any insertion can evict a borrowed owner.
      for (int j=0;j<betas.size();++j) {
        if (!column_support[betas[j]]) continue;
        if (input.contains(betas[j])) copy(j,input.get(betas[j]));
        else {missing.push_back(betas[j]);positions.push_back(j);}
      }
      std::size_t cursor=0;
      produce(missing,[&](auto keys,const auto& tiles) {
        for (int j=0;j<keys.size();++j) copy(positions[cursor++],tiles[j]);
      });
    };
    const CiTileWriter write=[&](auto betas,auto,const CiTileGather& gather) {
      for (int j=0;j<betas.size();++j) {
        if (!column_support[betas[j]]) continue;
        const auto p=pattern(betas[j]);if (!p->size) continue;
        auto& column=output.get(betas[j]);
        if (p->full) {
          std::vector<int> all(alpha);std::iota(all.begin(),all.end(),0);
          gather(j,all,column.values.view(),true);
        } else gather(j,p->rows,column.values.view(),true);
      }
    };
    if (!overlap_only) {
      // Same CPU residency rule: generate a fitting input once, before sigma
      // starts requesting halos. A larger input remains exact-demand tiled.
      if (quarter/sizeof(double)/alpha/beta>=static_cast<std::size_t>(width)) {
        std::vector<int> reachable;
        if (support) reachable=support->columns();
        else {reachable.resize(beta);std::iota(reachable.begin(),reachable.end(),0);}
        produce(reachable,[](auto,const auto&) {});
      }
      const CiTileRequest request{read,write};
      sigma.apply_tiled_responses(width,{&request,1},{},support.get(),quarter,[&](auto keys) {
        for (int key:keys) {input.release(key);output.release(key);}
      });
      output.finish();flush();
    }
    // Quotient sigma need not touch all columns of the full overlap action.
    std::vector<int> remaining;
    if (support) {
      for (int key:support->columns()) if (!overlap_consumed[key]) remaining.push_back(key);
    } else for (int key=0;key<beta;++key) if (!overlap_consumed[key]) remaining.push_back(key);
    produce(remaining,[](auto,const auto&) {});
    check_cuda(cudaGetLastError(),"broad-support product contraction");
  }
};

ProductContraction::ProductContraction(Context& c,const vb::StructureAction::OrthogonalView& view,
    const ExteriorTransform& alpha,const ExteriorTransform& beta,std::shared_ptr<const mcscf::ProductSupport> support)
    :impl_(std::make_unique<Impl>(c,view,alpha,beta,std::move(support))) {}
ProductContraction::~ProductContraction()=default;
void ProductContraction::update_frame() {impl_->row_cache.reset();}
void ProductContraction::apply(linalg::ConstMatrixView coefficients,linalg::ConstMatrixView upper,
    const DirectCiSigma& sigma,linalg::MatrixView h,linalg::MatrixView m,std::size_t bytes,bool overlap_only) const {
  impl_->apply(coefficients,upper,sigma,h,m,bytes,overlap_only);
}

} // namespace xmvb::cuda
