#include "cuda/vb/product.hpp"
#include "cuda/vb/columns.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <vector>
#include <array>
#include <cmath>
#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/column_cache.hpp"
#include "mcscf/ci/product_support.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"

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

__global__ void project_groups(const int* raw_alpha,const int* raw_beta,int count,
    const int* offsets,const int* columns,const double* factors,
    linalg::ConstMatrixView image,double* output) {
  for (auto g=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       g<count;g+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    const int b=raw_beta[g];double value=0.0;
    for (int p=offsets[b];p<offsets[b+1];++p)
      value+=image.data[raw_alpha[g]+image.ld*columns[p]]*factors[p];
    output[g]+=value;
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

/** The same sparse incidence used by the panel generator, in ordered form. */
__global__ void ordered_products(ProductIncidence incidence,const int* aa,
    const int* bb,int groups,linalg::ConstMatrixView source,
    linalg::MatrixView image,bool adjoint) {
  for (auto g=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       g<groups;g+=static_cast<std::size_t>(gridDim.x)*blockDim.x) {
    if (adjoint) {
      const double value=source.data[aa[g]+source.ld*bb[g]];
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
        atomicAdd(image.data+incidence.structures[t],incidence.weights[t]*value);
    } else {
      double value=0.0;
      for (auto t=incidence.offsets[g];t<incidence.offsets[g+1];++t)
        value+=incidence.weights[t]*source.data[incidence.structures[t]];
      image.data[aa[g]+image.ld*bb[g]]=value;
    }
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

struct ColumnRead { const int* rows; linalg::ConstMatrixView values; int destination; };

__global__ void read_columns(const ColumnRead* columns,const int* requested,int count,
    int width,linalg::MatrixView output) {
  const auto source=columns[blockIdx.x];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(count)*width;i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i%count,v=i/count,a=requested[r];
    // A contiguous beta panel may include an exactly empty column. Its null
    // index pointer is not a full identity map and must never read the next
    // packed column (or the end of the allocation).
    int index=a>=0 && a<source.values.rows ? a:-1;
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
  // This LRU owns a device-memory allowance. Host row metadata is reflected
  // in RSS, not charged a second time to Context's device budget.
  std::size_t bytes() const {return pattern->device_rows.size()*sizeof(int);}
};
struct SupportMask {
  Buffer<std::uint64_t> values;
  std::size_t bytes() const {return values.size()*sizeof(std::uint64_t);}
};
} // namespace

struct ProductContraction::Impl {
  Context& c;
  const ExteriorTransform& alpha_transform;
  const ExteriorTransform& beta_transform;
  int alpha,beta,groups,factor_extent=0,mask_extent=0;
  std::size_t beta_shears;
  const mcscf::ProductSupport* support;
  Buffer<int> starts,raw_alpha,raw_beta;
  cusparseHandle_t sparse=nullptr;
  std::vector<int> reachable;
  std::vector<std::weak_ptr<const ProductColumnPattern>> patterns;
  std::shared_ptr<const ProductColumnPattern> full_pattern;
  std::unique_ptr<mcscf::ColumnCache<RetainedPattern>> pattern_cache;
  std::unique_ptr<mcscf::ColumnCache<Factors>> factor_cache;
  std::unique_ptr<mcscf::ColumnCache<SupportMask>> mask_cache;

  Impl(Context& ctx,const vb::StructureAction::OrthogonalView& view,
      const ExteriorTransform& a,const ExteriorTransform& b,const mcscf::ProductSupport* exact_support)
      :c(ctx),alpha_transform(a),beta_transform(b),alpha(a.dimension()),beta(b.dimension()),
       groups(view.products.size()),beta_shears(view.beta.shear_pair_count()),
       support(exact_support),patterns(beta) {
    if (support) reachable=support->columns();
    else {reachable.resize(beta);std::iota(reachable.begin(),reachable.end(),0);}
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

  std::shared_ptr<const ProductColumnPattern> pattern(int b) {
    const bool full=!support || support->full();
    if (full && full_pattern) return full_pattern;
    if (pattern_cache->contains(b)) return pattern_cache->get(b).pattern;
    if (auto found=patterns[b].lock()) return found;
    auto result=std::make_shared<ProductColumnPattern>();result->full=full;
    if (!full) support->append_column_rows(b,result->rows);
    result->size=full ? alpha:result->rows.size();
    if (!full) result->device_rows=upload<int>(c,result->rows);
    patterns[b]=result;
    if (!full) pattern_cache->insert(b,{result});else full_pattern=result;
    return result;
  }

  ProductColumns allocate(std::span<const int> keys,int width) {
    std::vector<std::shared_ptr<const ProductColumnPattern>> columns;
    columns.reserve(keys.size());
    for (int key:keys) columns.push_back(pattern(key));
    return ProductColumns(c,std::move(columns),width);
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
    });
  }

  void prepare_factors(int extent,std::size_t budget) {
    if (factor_extent!=extent) {
      factor_cache.reset();factor_extent=extent;
    }
    if (!factor_cache) factor_cache=std::make_unique<mcscf::ColumnCache<Factors>>(budget,[this](int id) {
      const int first=id*factor_extent,count=std::min(factor_extent,beta-first);
      std::vector<int> selected(count);std::iota(selected.begin(),selected.end(),first);
      Matrix<double> values(c,beta,count);
      beta_transform.sample_rows(selected,values.view());return Factors(c,sparse,values.view());
    });
  }

  void generate_source(ProductIncidence incidence,linalg::ConstMatrixView source,
      const Factors& f,linalg::MatrixView values) {
    clear(c,values);
    if (source.data) generate_selected<<<grid(alpha),256,0,c.stream()>>>(starts.data(),raw_beta.data(),
        incidence,source,f.offsets.data(),f.indices.data(),f.values.data(),values);
  }

  ProductJet jet(ProductIncidence incidence,const DirectCiSigma& sigma,
      ProductEndpoint& endpoint,linalg::ConstMatrixView upper,
      linalg::ConstMatrixView response,linalg::ConstMatrixView delta_upper,
      const DirectCiSigma* delta_sigma,std::size_t bytes,ProductJet* ci,
      linalg::ConstMatrixView relative_direction,bool selected_only=false,bool overlap_only=false) {
    const nvtx3::scoped_range range{"vb.product_jet"};
    const bool tangent=response.data || delta_upper.data || delta_sigma;
    const bool joint=ci!=nullptr;
    const bool cached=tangent && endpoint.images && endpoint.images->complete();
    if (cached && delta_upper.data && !relative_direction.data)
      throw std::invalid_argument("cached product jet requires its relative frame direction");
    if (endpoint.coefficients.rows()!=incidence.size || endpoint.coefficients.cols()!=1 ||
        (response.data && (response.rows!=incidence.size || response.cols!=1)) ||
        (joint && !response.data))
      throw std::invalid_argument("invalid selected product jet source");
    const int n=sigma.n_orbitals(),d=sigma.n_channels();
    const auto make_result=[&] {
      ProductJet result{Matrix<double>(c,incidence.size,1),Matrix<double>(c,incidence.size,1),
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
    const auto quarter=std::max<std::size_t>(1,bytes/5);
    const int width=tangent ? (joint ? 3:2):1;
    const int input_width=cached ? width-1:width;
    const auto row_bytes=sizeof(double)*(12ULL*alpha+8ULL*beta);
    const int extent=cached ? endpoint.extent:std::max<std::size_t>(1,std::min<std::size_t>(beta,quarter/row_bytes));
    const int panels=(beta+extent-1)/extent;
    std::vector<int> all_beta(beta),all_rows(alpha);
    std::iota(all_beta.begin(),all_beta.end(),0);std::iota(all_rows.begin(),all_rows.end(),0);
    const auto selected=[&](int id) {
      return std::span<const int>{all_beta}.subspan(id*extent,std::min(extent,beta-id*extent));
    };
    std::vector<bool> reachable_panel(panels,false);
    for (int b:reachable) reachable_panel[b/extent]=true;
    int sampled_beta=0;
    for (int id=0;id<panels;++id) if (reachable_panel[id]) sampled_beta+=selected(id).size();
    const bool ordered_beta=vb::detail::prefer_ordered_beta(groups,alpha,beta,beta_shears,sampled_beta);
    const auto products=static_cast<std::size_t>(alpha)*beta;
    if (!pattern_cache) pattern_cache=std::make_unique<mcscf::ColumnCache<RetainedPattern>>(quarter/4,
        [](int)->RetainedPattern {throw std::logic_error("exact product pattern not generated");});
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
      auto keys=selected(id);Matrix<double> values(c,beta,keys.size()),delta(c,beta,keys.size());
      beta_transform.sample_directional_rows(db,keys,values.view(),delta.view());
      return Factors(c,sparse,delta.view());
    });
    const auto generate=[&](linalg::ConstMatrixView source,const Factors& f,
        linalg::MatrixView values) {
      generate_source(incidence,source,f,values);
    };
    // CPU's ordered source map is a temporary alternative to G-by-row-panel
    // contraction, not an endpoint. Both orders have the same exact support.
    Matrix<double> resident_y,resident_dy,resident_q;
    const bool resident_input=ordered_beta && products<=quarter/
        (2*sizeof(double)*(input_width+(delta_upper.data && !cached ? 2:0)));
    const auto expand=[&](linalg::ConstMatrixView source,Matrix<double>& image) {
      image=Matrix<double>(c,alpha,beta);clear(c,image.view());
      if (source.data) ordered_products<<<grid(groups),256,0,c.stream()>>>(
          incidence,raw_alpha.data(),raw_beta.data(),groups,source,image.view(),false);
    };
    if (resident_input) {
      if (!cached) {
        expand(endpoint.coefficients.view(),resident_y);
        if (tangent) {
          expand(response,resident_dy);
          if (delta_upper.data) {
            alpha_transform.apply_directional_left(da,resident_y.view(),resident_dy.view());
            // Reuse the existing directional-right implementation via a
            // pair of transposed temporaries, never retained at the point.
            Matrix<double> x(c,beta,alpha),dx(c,beta,alpha);
            const double one=1.0,zero=0.0;
            const auto transpose=[&](linalg::ConstMatrixView in,linalg::MatrixView out) {
              check_cublas(cublasDgeam(c.blas(),CUBLAS_OP_T,CUBLAS_OP_T,out.rows,out.cols,
                  &one,in.data,in.ld,&zero,in.data,in.ld,out.data,out.ld),"ordered source transpose");
            };
            transpose(resident_y.view(),x.view());transpose(resident_dy.view(),dx.view());
            beta_transform.apply_directional_left(db,x.view(),dx.view());
            transpose(x.view(),resident_y.view());transpose(dx.view(),resident_dy.view());
          } else {
            alpha_transform.apply_left(resident_y.view());beta_transform.apply_right(resident_y.view());
            alpha_transform.apply_left(resident_dy.view());beta_transform.apply_right(resident_dy.view());
          }
        } else {alpha_transform.apply_left(resident_y.view());beta_transform.apply_right(resident_y.view());}
      }
      if (joint || cached) {
        expand(response,resident_q);
        alpha_transform.apply_left(resident_q.view());beta_transform.apply_right(resident_q.view());
      }
    }
    std::vector<const ProductColumns*> accepted(panels,nullptr);
    if (cached) for (int id=0;id<panels;++id)
      if (reachable_panel[id]) accepted[id]=&endpoint.images->at(id);
    const CiTileReader accepted_y=[&](auto betas,auto rows,linalg::MatrixView values) {
      read_product_columns(c,accepted,extent,0,betas,rows,values);
    };
    const CiTileReader accepted_r=[&](auto betas,auto rows,linalg::MatrixView values) {
      read_product_columns(c,accepted,extent,1,betas,rows,values);
    };
    mcscf::ColumnCache<ProductColumns> input(resident_input ? quarter/2:quarter,[&](int id) {
      const auto keys=selected(id);const int count=keys.size();
      auto block=allocate(keys,input_width);
      Matrix<double> values(c,alpha,count*input_width);
      if (cached) {
        auto dy=values.view().columns(0,count);
        if (resident_input) copy_panel(dy,resident_q.view().columns(id*extent,count));
        else {generate(response,factors.get(id),dy);transform(dy,id*extent,count,false);}
        if (joint) copy_panel(values.view().columns(count,count),dy);
        if (delta_upper.data) {
          Matrix<double> orbital(c,alpha,count);
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_y,
              relative_direction,orbital.view());
          add(c,dy,orbital.view());
        }
        block.pack(c,values.view(),false);return block;
      }
      if (resident_input) {
        copy_panel(values.view().columns(0,count),resident_y.view().columns(id*extent,count));
        if (tangent) copy_panel(values.view().columns(count,count),resident_dy.view().columns(id*extent,count));
        if (joint) copy_panel(values.view().columns(2*count,count),resident_q.view().columns(id*extent,count));
        block.pack(c,values.view(),false);return block;
      }
      auto y=values.view().columns(0,count);
      generate(endpoint.coefficients.view(),factors.get(id),y);
      if (tangent) {
        auto dy=values.view().columns(count,count);
        if (delta_upper.data) generate(endpoint.coefficients.view(),delta_factors.get(id),dy);
        else clear(c,dy);
        if (response.data) {
          Matrix<double> q(c,alpha,count);
          generate(response,factors.get(id),q.view());add(c,dy,q.view());
          if (joint) copy_panel(values.view().columns(2*count,count),q.view());
        }
        if (delta_upper.data) alpha_transform.apply_directional_left(da,y,dy);
        else {transform(y,id*extent,count,false);transform(dy,id*extent,count,false);}
        if (joint) transform(values.view().columns(2*count,count),id*extent,count,false);
      } else transform(y,id*extent,count,false);
      block.pack(c,values.view(),false);return block;
    });
    const auto reader=[&](int lane) -> CiTileReader {
      if (cached && lane==0) return accepted_y;
      const int input_lane=cached ? lane-1:lane;
      return [&,input_lane](auto betas,auto rows,linalg::MatrixView values) {
        clear(c,values);auto indices=upload<int>(c,rows);
        std::map<int,std::vector<std::pair<int,int>>> blocks;
        for (int j=0;j<betas.size();++j)
          if (reachable_panel[betas[j]/extent]) blocks[betas[j]/extent].emplace_back(betas[j]%extent,j);
        for (const auto& [id,entries]:blocks) {
          auto& block=input.get(id);
          std::vector<ColumnRead> columns;
          for (auto [b,destination]:entries) {
            auto column=block.column(b);
            const linalg::ConstMatrixView view{column.data+column.ld*input_lane,column.rows,1,column.ld};
            columns.push_back({block.patterns[b]->full ? nullptr:block.patterns[b]->device_rows.data(),view,destination});
          }
          auto metadata=upload<ColumnRead>(c,columns);
          if (!rows.empty()) read_columns<<<dim3(columns.size(),grid(rows.size())),256,0,c.stream()>>>(
              metadata.data(),indices.data(),rows.size(),1,values);
        }
      };
    };
    const auto y=reader(0),dy=reader(tangent ? 1:0),q=joint ? reader(2):CiTileReader{};
    const int targets=joint ? 4:2;
    const bool resident_projection=ordered_beta &&
        products<=quarter/(sizeof(double)*(targets*(delta_upper.data && !cached ? 2:1)+
            (delta_upper.data && !cached ? 2:0)));
    std::vector<Matrix<double>> projected(targets),left_projected(targets);
    const std::array<linalg::MatrixView,4> destinations{result.hamiltonian.view(),result.overlap.view(),
        joint ? ci->hamiltonian.view():linalg::MatrixView{},joint ? ci->overlap.view():linalg::MatrixView{}};
    const int group_extent=std::max<std::size_t>(1,std::min<std::size_t>(groups,
        quarter/(4*sizeof(double)*targets)));
    struct Projection {
      Matrix<double> values;
      std::size_t bytes() const {return values.rows()*values.cols()*sizeof(double);}
    };
    mcscf::ColumnCache<Projection> projections(quarter/4,[&](int id) {
      Projection value{Matrix<double>(c,std::min<std::int64_t>(group_extent,groups-id*group_extent),targets)};
      clear(c,value.values.view());return value;
    },[&](int id,Projection& tile) {
      for (int k=0;k<targets;++k) contract_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(incidence,
          id*group_extent,tile.values.rows(),tile.values.view().column(k).data,destinations[k].data);
    });
    const auto project_panel=[&](int id,linalg::MatrixView values,int destination,bool differentiate) {
      const auto keys=selected(id);const int count=keys.size();
      Matrix<double> delta;
      if (differentiate) {
        delta=Matrix<double>(c,values.rows,values.cols);clear(c,delta.view());
        alpha_transform.apply_directional_left(da,values,delta.view(),true);
      } else {
        // Input and reverse factors borrow the same immutable support mask.
        transform(values,id*extent,count,true);
      }
      if (resident_projection) {
        if (!projected[destination].data()) {
          projected[destination]=Matrix<double>(c,alpha,beta);clear(c,projected[destination].view());
        }
        add(c,projected[destination].view().columns(id*extent,count),
            differentiate ? delta.view():static_cast<linalg::ConstMatrixView>(values));
        if (differentiate) {
          if (!left_projected[destination].data()) {
            left_projected[destination]=Matrix<double>(c,alpha,beta);clear(c,left_projected[destination].view());
          }
          add(c,left_projected[destination].view().columns(id*extent,count),values);
        }
        return;
      }
      const auto& f=factors.get(id);
      for (std::int64_t begin=0;begin<groups;begin+=group_extent) {
        auto& tile=projections.get(begin/group_extent);
        project_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(raw_alpha.data()+begin,raw_beta.data()+begin,
            tile.values.rows(),f.offsets.data(),f.indices.data(),f.values.data(),
            differentiate ? delta.view():values,tile.values.view().column(destination).data);
        if (differentiate) {
          const auto& df=delta_factors.get(id);
          project_groups<<<grid(tile.values.rows()),256,0,c.stream()>>>(raw_alpha.data()+begin,raw_beta.data()+begin,
              tile.values.rows(),df.offsets.data(),df.indices.data(),df.values.data(),values,
              tile.values.view().column(destination).data);
        }
      }
    };
    bool exchange_reduced=false,retain=false,retaining=true;
    mcscf::ColumnCache<ProductColumns> output(quarter,[&](int id) {
      auto block=allocate(selected(id),input_width);clear(c,block.values.view());return block;
    },[&](int id,ProductColumns& block) {
      const int count=selected(id).size(),first=id*extent;
      Matrix<double> values(c,alpha,count*input_width);clear(c,values.view());block.pack(c,values.view(),true);
      const auto hy=values.view().columns(0,count);
      if (!tangent) {
        if (retain && retaining) {
          auto& source=input.get(id);
          ProductColumns state(c,block.patterns,2);
          copy_panel(state.values.view().columns(0,1),source.values.view().columns(0,1));
          copy_panel(state.values.view().columns(1,1),block.values.view().columns(0,1));
          retaining=endpoint.images->retain(id,std::move(state));
        }
        const std::array lefts{static_cast<linalg::ConstMatrixView>(hy)};
        const std::array requests{CiStreamedGeneratorRequest{y}};
        const std::array moments{result.generator.view()};
        if (!selected_only)
          sigma.generator_adjoint_streamed(first,lefts,requests,moments,false,exchange_reduced,true);
        project_panel(id,hy,0,false);
      } else {
        const int z=cached ? 0:count,qs=cached ? count:2*count;
        const std::array lefts{static_cast<linalg::ConstMatrixView>(values.view().columns(z,count)),
            joint ? static_cast<linalg::ConstMatrixView>(values.view().columns(qs,count)):linalg::ConstMatrixView{}};
        const std::array moments{result.generator.view(),joint ? ci->generator.view():linalg::MatrixView{}};
        const std::array requests{CiStreamedGeneratorRequest{y}};
        sigma.generator_adjoint_streamed(first,{lefts.data(),joint ? 2U:1U},requests,
            {moments.data(),joint ? 2U:1U},false,exchange_reduced,true);
        if (!cached) {
          const std::array hleft{static_cast<linalg::ConstMatrixView>(hy)};
          const std::array drequests{CiStreamedGeneratorRequest{dy}};
          sigma.generator_adjoint_streamed(first,hleft,drequests,{moments.data(),1},false,exchange_reduced,true);
          if (joint) {
            const std::array qrequests{CiStreamedGeneratorRequest{q}};
            sigma.generator_adjoint_streamed(first,hleft,qrequests,{moments.data()+1,1},false,exchange_reduced,true);
          }
        }
        project_panel(id,values.view().columns(z,count),0,false);
        if (joint) project_panel(id,values.view().columns(qs,count),2,false);
        if (delta_upper.data && !cached) project_panel(id,hy,0,true);
      }
    });
    const auto writer=[&](int lane) -> CiTileWriter {
      return [&,lane](auto betas,auto,const CiTileGather& gather) {
        std::map<int,std::vector<std::pair<int,int>>> blocks;
        for (int j=0;j<betas.size();++j)
          if (reachable_panel[betas[j]/extent]) blocks[betas[j]/extent].emplace_back(betas[j]%extent,j);
        for (const auto& [id,entries]:blocks) {
          auto& block=output.get(id);std::vector<CiTileColumn> columns;
          for (auto [b,source]:entries) {
            auto column=block.column(b);
            const int slot=cached ? lane-1:lane;
            if (column.rows) columns.push_back({source,block.patterns[b]->full ? nullptr:block.patterns[b]->device_rows.data(),
                {column.data+column.ld*slot,column.rows,1,column.ld},block.patterns[b]->full});
          }
          gather(columns,true);
        }
      };
    };
    std::vector<CiTileRequest> requests;
    if (!cached) requests.push_back({y,writer(0),nullptr,tangent ? linalg::MatrixView{}:result.one.view(),
        tangent ? linalg::MatrixView{}:result.pair.view()});
    if (tangent) requests.push_back({dy,writer(1),delta_sigma,result.one.view(),result.pair.view()});
    if (joint) requests.push_back({q,writer(2),nullptr,ci->one.view(),ci->pair.view()});
    if (!overlap_only)
      sigma.apply_tiled_responses(1,requests,tangent ? y:CiTileReader{},support,quarter);
    if (!overlap_only && (sigma.exchange_parity() || !tangent) && !output.evictions) {
      std::vector<ProductColumns*> panels_view(panels,nullptr);
      for (int id=0;id<panels;++id) if (reachable_panel[id]) panels_view[id]=&output.get(id);
      if (sigma.exchange_parity() && !output.evictions) {
        while (!panels_view.empty() && !panels_view.back()) panels_view.pop_back();
        project_product_exchange(c,panels_view,extent,sigma.exchange_parity());exchange_reduced=true;
      }
    }
    if (!selected_only && !tangent && !output.evictions) {
      endpoint.extent=extent;endpoint.images=std::make_unique<ProductImages>(quarter);
      endpoint.images->begin(panels);
      for (int id=0;id<panels;++id) if (!reachable_panel[id]) endpoint.images->retain(id,{});
      retain=true;
    }
    output.finish();
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
      const int count=selected(id).size();Matrix<double> values(c,alpha,count);
      if (!tangent) {y(selected(id),all_rows,values.view());project_panel(id,values.view(),1,false);}
      else {
        dy(selected(id),all_rows,values.view());
        if (cached) {
          Matrix<double> q_values;
          if (joint) {q_values=Matrix<double>(c,alpha,count);q(selected(id),all_rows,q_values.view());}
          const std::array lefts{static_cast<linalg::ConstMatrixView>(values.view()),
              static_cast<linalg::ConstMatrixView>(q_values.view())};
          const std::array requests{CiStreamedGeneratorRequest{accepted_y,-endpoint.energy,false},
              CiStreamedGeneratorRequest{accepted_r,1.0,true}};
          const std::array moments{result.generator.view(),joint ? ci->generator.view():linalg::MatrixView{}};
          sigma.generator_adjoint_streamed(id*extent,{lefts.data(),joint ? 2U:1U},requests,
              {moments.data(),joint ? 2U:1U},false,sigma.exchange_parity()!=0,true);
          if (joint) project_panel(id,q_values.view(),3,false);
        } else if (joint) {
          Matrix<double> q_values(c,alpha,count);q(selected(id),all_rows,q_values.view());
          project_panel(id,q_values.view(),3,false);
        }
        project_panel(id,values.view(),1,false);
        if (delta_upper.data && !cached) {y(selected(id),all_rows,values.view());project_panel(id,values.view(),1,true);}
        if (delta_upper.data && cached) {
          sigma.apply_one_body_generator_columns(id*extent,count,accepted_y,transpose.view(),values.view(),
              sigma.exchange_parity()!=0);
          if (sigma.exchange_parity()) multiply(c,values.view(),2.0);
          project_panel(id,values.view(),1,false);
          const CiTileReader hy_reader=[&](auto betas,auto rows,linalg::MatrixView target) {
            accepted_r(betas,rows,target);
            Matrix<double> primal(c,target.rows,target.cols);accepted_y(betas,rows,primal.view());
            add(c,target,primal.view(),endpoint.energy);
          };
          sigma.apply_one_body_generator_columns(id*extent,count,hy_reader,transpose.view(),values.view(),
              sigma.exchange_parity()!=0);
          if (sigma.exchange_parity()) multiply(c,values.view(),2.0);
          project_panel(id,values.view(),0,false);
        }
      }
    }
    projections.finish();
    if (resident_projection) for (int k=0;k<targets;++k) if (projected[k].data()) {
      if (left_projected[k].data()) {
        Matrix<double> x(c,beta,alpha),dx(c,beta,alpha);
        const double one=1.0,zero=0.0;
        const auto transpose=[&](linalg::ConstMatrixView in,linalg::MatrixView out) {
          check_cublas(cublasDgeam(c.blas(),CUBLAS_OP_T,CUBLAS_OP_T,out.rows,out.cols,
              &one,in.data,in.ld,&zero,in.data,in.ld,out.data,out.ld),"ordered projection transpose");
        };
        transpose(left_projected[k].view(),x.view());transpose(projected[k].view(),dx.view());
        beta_transform.apply_directional_left(db,x.view(),dx.view(),true);
        transpose(dx.view(),projected[k].view());
      } else beta_transform.apply_right(projected[k].view(),true);
      ordered_products<<<grid(groups),256,0,c.stream()>>>(incidence,raw_alpha.data(),raw_beta.data(),
          groups,projected[k].view(),destinations[k],true);
    }
    if (!selected_only) multiply(c,result.generator.view(),2.0);
    if (tangent && !cached) add_upper(c,result.generator.view(),result.one.view(),-4*endpoint.energy);
    if (joint) {multiply(c,ci->generator.view(),2.0);if (!cached) add_upper(c,ci->generator.view(),ci->one.view(),-4*endpoint.energy);}
    return result;
  }

  void copy_panel(linalg::MatrixView out,linalg::ConstMatrixView in) {
    check_cuda(cudaMemcpy2DAsync(out.data,out.ld*sizeof(double),in.data,in.ld*sizeof(double),
        in.rows*sizeof(double),in.cols,cudaMemcpyDeviceToDevice,c.stream()),"copy product panel");
  }

  double maximum(ProductIncidence incidence,linalg::ConstMatrixView source,std::size_t bytes) {
    if (source.rows!=incidence.size || source.cols!=1 || source.ld<source.rows)
      throw std::invalid_argument("invalid selected product maximum source");
    const auto allowance=std::max<std::size_t>(1,bytes/5);
    const auto row_bytes=sizeof(double)*(12ULL*alpha+8ULL*beta);
    const int extent=std::max<std::size_t>(1,std::min<std::size_t>(beta,allowance/row_bytes));
    if (!pattern_cache) pattern_cache=std::make_unique<mcscf::ColumnCache<RetainedPattern>>(allowance/4,
        [](int)->RetainedPattern {throw std::logic_error("exact product pattern not generated");});
    prepare_factors(extent,allowance/4);
    prepare_masks(extent,allowance/4);
    double maximum=0.0;
    for (int first=0;first<beta;first+=extent) {
      const int count=std::min(extent,beta-first);
      const auto touched=std::lower_bound(reachable.begin(),reachable.end(),first);
      if (touched==reachable.end() || *touched>=first+count) continue;
      Matrix<double> values(c,alpha,count);
      generate_source(incidence,source,factor_cache->get(first/extent),values.view());
      transform(values.view(),first,count,false);
      int index=0;
      check_cublas(cublasSetPointerMode(c.blas(),CUBLAS_POINTER_MODE_HOST),
          "product panel maximum pointer mode");
      check_cublas(cublasIdamax(c.blas(),blas_int(values.rows()*values.cols()),
          values.data(),1,&index),"product panel maximum index");
      if (index<=0) throw std::runtime_error("empty product panel maximum");
      double value=0.0;c.copy_to_host(&value,values.data()+index-1,sizeof(double));
      if (!std::isfinite(value)) throw std::runtime_error("non-finite product panel maximum");
      maximum=std::max(maximum,std::abs(value));
    }
    return maximum;
  }
};

ProductContraction::ProductContraction(Context& c,const vb::StructureAction::OrthogonalView& view,
    const ExteriorTransform& alpha,const ExteriorTransform& beta,const mcscf::ProductSupport* support)
    :impl_(std::make_unique<Impl>(c,view,alpha,beta,support)) {}
ProductContraction::~ProductContraction()=default;
void ProductContraction::update_frame() {impl_->factor_cache.reset();}
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
  if (source.rows!=incidence.size || source.cols<=0 || h.rows!=source.rows || m.rows!=source.rows ||
      h.cols!=source.cols || m.cols!=source.cols)
    throw std::invalid_argument("invalid selected product action block");
  ProductEndpoint endpoint{Matrix<double>(impl_->c,incidence.size,1)};
  for (int rhs=0;rhs<source.cols;++rhs) {
    impl_->copy_panel(endpoint.coefficients.view(),source.column(rhs));
    auto result=impl_->jet(incidence,sigma,endpoint,{},{},{},nullptr,bytes,nullptr,{},true,overlap_only);
    impl_->copy_panel(h.column(rhs),result.hamiltonian.view());
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
