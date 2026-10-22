#include "cuda/vb/action.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/ci/sigma.hpp"
#include "cuda/ci/spin_self.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/memory/graph.hpp"
#include "cuda/vb/transform.hpp"
#include "cuda/vb/product.hpp"
#include "mcscf/ci/product_support.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"

namespace xmvb::cuda {
namespace {

__global__ void divide_diagonal(linalg::MatrixView carrier,
    const double* diagonal, linalg::ConstMatrixView energies,
    std::int64_t products, double shift, std::int64_t elements) {
  const std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=elements) return;
  const auto rhs=i/products, product=i%products;
  double denominator=diagonal[product]-energies.data[rhs*energies.ld]-shift;
  // Exactly the existing production FULL inverse convention, not a new tolerance.
  if (fabs(denominator)<1.0e-8) denominator=1.0e-8;
  const auto row=product%carrier.rows;
  const auto column=product/carrier.rows+rhs*(products/carrier.rows);
  carrier.data[row+column*carrier.ld]/=denominator;
}

__global__ void assemble_spin_self_model(const std::int64_t* row_offsets,
    const std::int64_t* products,const std::int64_t* columns,
    const double* weights,linalg::ConstMatrixView projected_alpha,
    linalg::ConstMatrixView beta_moments,const int* alpha_ids,const int* beta_ids,
    const double* alpha_energy,const double* beta_energy,
    const double* alpha_overlap,const double* beta_overlap,
    double* hamiltonian,double* overlap,std::int64_t groups,
    int n_alpha,int n_beta) {
  const std::int64_t group=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (group>=groups) return;
  if (row_offsets[group]==row_offsets[group+1]) return;
  const auto product=products[group];
  const int a=alpha_ids[product%n_alpha];
  const int b=beta_ids[product/n_alpha];
  const double norm=alpha_overlap[a]*beta_overlap[b];
  double energy=alpha_energy[a]+beta_energy[b];
  for (int channel=0;channel<projected_alpha.cols;++channel)
    energy+=projected_alpha.data[a+projected_alpha.ld*channel]*
        beta_moments.data[b+beta_moments.ld*channel];
  for (auto term=row_offsets[group];term<row_offsets[group+1];++term) {
    const auto structure=columns[term];
    bool first=true;
    for (auto prior=row_offsets[group];prior<term;++prior)
      if (columns[prior]==structure) { first=false; break; }
    if (!first) continue;
    double combined=0.0;
    for (auto matching=term;matching<row_offsets[group+1];++matching)
      if (columns[matching]==structure) combined+=weights[matching];
    const double square=norm*combined*combined;
    atomicAdd(hamiltonian+structure,square*energy);
    atomicAdd(overlap+structure,square);
  }
}

unsigned int tile_grid(std::int64_t work) {
  return static_cast<unsigned int>(std::min<std::int64_t>(
      65535,(work+255)/256));
}

__global__ void spin_gram_identity(const std::int64_t* product_indices,
    std::int64_t first_group,int count,int alpha,int beta,
    linalg::MatrixView alpha_columns,linalg::MatrixView beta_columns) {
  const auto alpha_size=static_cast<std::int64_t>(alpha)*count;
  const auto total=alpha_size+static_cast<std::int64_t>(beta)*count;
  for (auto index=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       index<total;index+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
    if (index<alpha_size) {
      const int column=static_cast<int>(index/alpha);
      const int row=static_cast<int>(index%alpha);
      const auto product=product_indices[first_group+column];
      alpha_columns.data[row+alpha_columns.ld*column]=
          row==product%alpha ? 1.0:0.0;
    } else {
      const auto local=index-alpha_size;
      const int column=static_cast<int>(local/beta);
      const int row=static_cast<int>(local%beta);
      const auto product=product_indices[first_group+column];
      beta_columns.data[row+beta_columns.ld*column]=
          row==product/alpha ? 1.0:0.0;
    }
  }
}

__global__ void spin_gram_product(const std::int64_t* product_indices,
    std::int64_t first_group,std::int64_t groups,int count,int alpha,
    linalg::ConstMatrixView alpha_gram,linalg::ConstMatrixView beta_gram,
    linalg::MatrixView product_gram) {
  const auto work=groups*count;
  for (auto index=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       index<work;index+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
    const auto source=index/groups;
    const auto target=index%groups;
    const auto q=product_indices[target];
    product_gram.data[target+product_gram.ld*source]=
        alpha_gram.data[q%alpha+alpha_gram.ld*source]*
        beta_gram.data[q/alpha+beta_gram.ld*source];
  }
}

__global__ void selected_source_terms(const std::int64_t* row_offsets,
    const std::int64_t* columns,const double* weights,
    std::int64_t first_group,int count,linalg::MatrixView source_terms) {
  for (auto local=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       local<count;local+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
    const auto group=first_group+local;
    // One owner per source row also sums duplicate structure terms exactly.
    for (auto term=row_offsets[group];term<row_offsets[group+1];++term)
      source_terms.data[local+source_terms.ld*columns[term]]+=weights[term];
  }
}

void check_sparse(cusparseStatus_t status, const char* operation) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + cusparseGetErrorString(status));
}

/** Dense descriptors borrow caller-owned views and never own numerical data. */
class DenseDescriptor {
 public:
  explicit DenseDescriptor(linalg::ConstMatrixView view) {
    check_sparse(cusparseCreateDnMat(&value_, view.rows, view.cols, view.ld,
        const_cast<double*>(view.data), CUDA_R_64F, CUSPARSE_ORDER_COL),
        "cusparseCreateDnMat");
  }
  ~DenseDescriptor() { cusparseDestroyDnMat(value_); }
  operator cusparseDnMatDescr_t() const noexcept { return value_; }
 private:
  cusparseDnMatDescr_t value_ = nullptr;
};

} // namespace

struct StructureAction::Impl {
  Context& context;
  int structures, alpha, beta;
  std::int64_t groups, products;
  vb::StructureDiagonal diagonal;
  DirectCiSigma sigma;
  ExteriorTransform alpha_transform;
  std::unique_ptr<ExteriorTransform> beta_storage;
  const ExteriorTransform* beta_transform;
  std::unique_ptr<SpinSelf> alpha_self;
  std::unique_ptr<SpinSelf> beta_self_storage;
  const SpinSelf* beta_self;
  Buffer<int> alpha_self_ids,beta_self_ids;
  Matrix<double> model_upper;
  std::uint64_t model_version=std::numeric_limits<std::uint64_t>::max();
  Buffer<std::int64_t> row_offsets, product_indices, columns;
  Buffer<double> weights;
  cusparseHandle_t sparse = nullptr;
  cusparseSpMatDescr_t incidence = nullptr;
  mutable Matrix<double> carriers, support;
  linalg::MatrixView coefficients, image;
  mutable Buffer<std::byte> sparse_workspace;
  mutable Matrix<double> input, hamiltonian, overlap;
  mutable Matrix<double> carrier_diagonal;
  mutable std::unique_ptr<Graph> graph;
  mutable double graph_seconds=0;
  std::unique_ptr<ProductContraction> product_contraction;
  std::shared_ptr<const mcscf::ProductSupport> exact_support;
  mutable std::uint64_t graph_workspace_generation=0;
  std::uint64_t version=0;

  Impl(Context& ctx, const vb::StructureAction& source)
      : context(ctx), structures(source.n_structures()),
        alpha(source.n_alpha_strings()), beta(source.n_beta_strings()),
        groups(source.orthogonal_view().products.size()),
        products(static_cast<std::int64_t>(alpha)*beta),
        diagonal(source.preconditioner_diagonal()),
        sigma(ctx, source.orthogonal_view().sigma,16,source.orthogonal_view().exchange_parity),
        alpha_transform(ctx, source.orthogonal_view().alpha),
        beta_storage(&source.orthogonal_view().alpha == &source.orthogonal_view().beta
            ? nullptr : std::make_unique<ExteriorTransform>(ctx, source.orthogonal_view().beta)),
        beta_transform(beta_storage ? beta_storage.get() : &alpha_transform),
        beta_self(nullptr),alpha_self_ids(ctx,alpha),beta_self_ids(ctx,beta),
        model_upper(ctx,source.orthogonal_view().alpha.n_orbitals(),source.orthogonal_view().alpha.n_orbitals()),
        row_offsets(ctx, groups+1), product_indices(ctx, groups),
        columns(ctx, source.orthogonal_view().terms.size()),
        weights(ctx, source.orthogonal_view().terms.size()),
        carriers(ctx,alpha,2*static_cast<std::int64_t>(beta)),support(ctx,groups,1),
        coefficients(carriers.view().columns(0,beta)),image(carriers.view().columns(beta,beta)),
        input(ctx,structures,1),hamiltonian(ctx,structures,1),overlap(ctx,structures,1) {
    const auto view=source.orthogonal_view();
    copy_to_device(context,model_upper.view(),view.alpha.orbital_transform());
    std::vector<unsigned char> alpha_touched(alpha),beta_touched(beta);
    for (std::size_t group=0;group<view.products.size();++group) {
      if (view.offsets[group]==view.offsets[group+1]) continue;
      alpha_touched[view.products[group]/beta]=1;
      beta_touched[view.products[group]%beta]=1;
    }
    if (!beta_storage)
      for (int i=0;i<alpha;++i) alpha_touched[i]=beta_touched[i]=alpha_touched[i]|beta_touched[i];
    const auto make_self=[&](const auto& transform,const auto& touched,auto& device_ids) {
      std::vector<std::uint64_t> masks;
      std::vector<int> ids(touched.size(),-1);
      for (std::size_t i=0;i<touched.size();++i) if (touched[i]) {
        ids[i]=static_cast<int>(masks.size());masks.push_back(transform.occupations()[i]);
      }
      context.copy_to_device(device_ids.data(),ids.data(),ids.size()*sizeof(int));
      return std::make_unique<SpinSelf>(ctx,transform.n_orbitals(),masks);
    };
    alpha_self=make_self(view.alpha,alpha_touched,alpha_self_ids);
    if (beta_storage) beta_self_storage=make_self(view.beta,beta_touched,beta_self_ids);
    if (!beta_storage)
      check_cuda(cudaMemcpyAsync(beta_self_ids.data(),alpha_self_ids.data(),alpha*sizeof(int),
          cudaMemcpyDeviceToDevice,context.stream()),"share spin-self row map");
    beta_self=beta_self_storage ? beta_self_storage.get():alpha_self.get();
    std::vector<std::int64_t> offsets(view.offsets.begin(), view.offsets.end());
    std::vector<std::int64_t> addresses;
    addresses.reserve(groups);
    for (const auto p:view.products) {
      const std::int64_t a=p/beta, b=p%beta;
      addresses.push_back(a+static_cast<std::int64_t>(alpha)*b);
    }
    std::vector<std::int64_t> indices;
    std::vector<double> values;
    indices.reserve(view.terms.size()); values.reserve(view.terms.size());
    for (const auto& term:view.terms) {
      indices.push_back(term.structure); values.push_back(term.coefficient);
    }
    context.copy_to_device(row_offsets.data(), offsets.data(), offsets.size()*sizeof(std::int64_t));
    context.copy_to_device(product_indices.data(), addresses.data(), addresses.size()*sizeof(std::int64_t));
    context.copy_to_device(columns.data(), indices.data(), indices.size()*sizeof(std::int64_t));
    context.copy_to_device(weights.data(), values.data(), values.size()*sizeof(double));
    try {
      check_sparse(cusparseCreate(&sparse), "cusparseCreate");
      check_sparse(cusparseSetStream(sparse, context.stream()), "cusparseSetStream");
      check_sparse(cusparseCreateCsr(&incidence, groups, structures, values.size(),
          row_offsets.data(), columns.data(), weights.data(), CUSPARSE_INDEX_64I,
          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "cusparseCreateCsr");
      // Query both orientations once; reuse one device workspace in every action.
      Matrix<double> structure_vector(context, structures, 1);
      const auto bytes=std::max(spmm_bytes(false,structure_vector.view(),support.view()),
          spmm_bytes(true,support.view(),structure_vector.view()));
      sparse_workspace=Buffer<std::byte>(context, bytes);
      if (view.support) exact_support=std::make_shared<mcscf::ProductSupport>(*view.support);
      const int sampled_columns=view.support ? view.support->columns().size():beta;
      if (!vb::detail::prefer_ordered_beta(groups,alpha,beta,view.beta.shear_pair_count(),sampled_columns))
        product_contraction=std::make_unique<ProductContraction>(context,view,alpha_transform,*beta_transform,exact_support);
      // Numerical workspace is planned at first apply, after the physical
      // point exists. Eager preparation here can consume its carrier storage
      // while the topology still holds only placeholder integrals.
    } catch (...) {
      if (incidence) cusparseDestroySpMat(incidence);
      if (sparse) cusparseDestroy(sparse);
      throw;
    }
  }

  ~Impl() {
    graph.reset();
    if (incidence) cusparseDestroySpMat(incidence);
    if (sparse) cusparseDestroy(sparse);
  }

  std::size_t spmm_bytes(bool transpose, linalg::ConstMatrixView input,
                         linalg::MatrixView output) const {
    DenseDescriptor x(input), y(output);
    std::size_t bytes=0;
    const double one=1, zero=0;
    check_sparse(cusparseSpMM_bufferSize(sparse,
        transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, incidence, x, &zero, y,
        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes), "cusparseSpMM_bufferSize");
    return bytes;
  }

  void spmm_with_workspace(bool transpose,linalg::ConstMatrixView input,
             linalg::MatrixView output,void* workspace) const {
    DenseDescriptor x(input), y(output);
    const double one=1, zero=0;
    check_sparse(cusparseSpMM(sparse,
        transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, incidence, x, &zero, y,
        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, workspace), "cusparseSpMM");
  }

  void spmm(bool transpose, linalg::ConstMatrixView input,
             linalg::MatrixView output) const {
    spmm_with_workspace(transpose,input,output,sparse_workspace.data());
  }

  void project(linalg::ConstMatrixView carrier,linalg::MatrixView output) const {
    const nvtx3::scoped_range range{"vb.backproject"};
    thrust::gather(thrust::cuda::par_nosync.on(context.stream()), product_indices.data(),
        product_indices.data()+groups, carrier.data, support.data());
    spmm(true,support.view(),output);
  }

  // Called exactly once during capture; there is one maintained contraction path.
  void emit(linalg::ConstMatrixView vector,linalg::MatrixView h,linalg::MatrixView m) const {
      {
      const nvtx3::scoped_range range{"vb.expand"};
      spmm(false,vector,support.view());
      check_cuda(cudaMemsetAsync(coefficients.data,0,products*sizeof(double),context.stream()),
          "clear product input");
      thrust::scatter(thrust::cuda::par_nosync.on(context.stream()),support.data(),support.data()+groups,
          product_indices.data(),coefficients.data);
      }
      {
      const nvtx3::scoped_range range{"vb.transform"};
      alpha_transform.apply_left(coefficients);
      beta_transform->apply_right(coefficients);
      }
      if (exact_support) {
        const CiCarrierRequest request{coefficients,image};
        sigma.apply_supported_responses({&request,1},coefficients,*exact_support,
            std::min(workspace::available(),context.device_byte_budget()-
                context.counters().current_device_bytes)/4,false);
      } else sigma.apply(coefficients,image);
      {
        const nvtx3::scoped_range range{"vb.transform_adjoint"};
        // The input is the overlap image after sigma consumed it. Transform H/M
        // together, in the same two panels; neither a carrier copy nor new storage.
        beta_transform->apply_right(carriers.view(),true);
        alpha_transform.apply_left(carriers.view(),true);
      }
      project(coefficients,m);
      project(image,h);
  }

  void apply(linalg::ConstMatrixView vectors,linalg::MatrixView h,linalg::MatrixView m) const {
    if (vectors.rows!=structures || h.rows!=structures || m.rows!=structures ||
        vectors.cols<=0 || h.cols!=vectors.cols || m.cols!=vectors.cols)
      throw std::invalid_argument("CUDA structure action block dimensions");
    if (product_contraction) {
      sigma.reset_workspace();
      Matrix<double> h_products(context,groups,1),m_products(context,groups,1);
      const auto bytes=std::min(workspace::available(),
          context.device_byte_budget()-context.counters().current_device_bytes);
      for (std::int64_t rhs=0;rhs<vectors.cols;++rhs) {
        spmm(false,vectors.column(rhs),support.view());
        product_contraction->apply(support.view(),sigma,
            h_products.view(),m_products.view(),bytes);
        spmm(true,h_products.view(),h.column(rhs));
        spmm(true,m_products.view(),m.column(rhs));
      }
      return;
    }
    if (exact_support) {
      // CPU's ordered-beta alternative: transform once, execute the same
      // supported sigma plan, and pull H/M back together. It is not a switch
      // back to a full-product sigma merely because the input is resident.
      for (std::int64_t rhs=0;rhs<vectors.cols;++rhs) emit(vectors.column(rhs),h.column(rhs),m.column(rhs));
      return;
    }
    sigma.prepare(1);
    if (!graph || graph_workspace_generation!=sigma.workspace_generation()) {
      // Another point may grow/repartition the shared scratch. Recapture the
      // same physical action before replay, never launch stale device pointers.
      context.synchronize();
      graph.reset();
      const auto start=std::chrono::steady_clock::now();
      graph=std::make_unique<Graph>(context,[&] {
        emit(input.view(),hamiltonian.view(),overlap.view());
      });
      graph_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      graph_workspace_generation=sigma.workspace_generation();
    }
    const auto bytes=static_cast<std::size_t>(structures)*sizeof(double);
    for (std::int64_t rhs=0;rhs<vectors.cols;++rhs) {
      // Fixed graph addresses need three selected-space vectors, not new carriers.
      check_cuda(cudaMemcpyAsync(input.data(),vectors.column(rhs).data,bytes,
          cudaMemcpyDeviceToDevice,context.stream()),"stage graph input");
      graph->launch();
      check_cuda(cudaMemcpyAsync(h.column(rhs).data,hamiltonian.data(),bytes,
          cudaMemcpyDeviceToDevice,context.stream()),"copy graph H image");
      check_cuda(cudaMemcpyAsync(m.column(rhs).data,overlap.data(),bytes,
          cudaMemcpyDeviceToDevice,context.stream()),"copy graph M image");
    }
  }
};

StructureAction::StructureAction(Context& context,const vb::StructureAction& source)
    : impl_(std::make_unique<Impl>(context,source)) {}
StructureAction::~StructureAction()=default;
int StructureAction::size() const noexcept { return impl_->structures; }
double StructureAction::graph_build_seconds() const noexcept { return impl_->graph_seconds; }
std::size_t StructureAction::graph_nodes() const noexcept {
  return impl_->graph ? impl_->graph->nodes():0;
}
const vb::StructureDiagonal& StructureAction::preconditioner_diagonal() const noexcept {
  return impl_->diagonal;
}
std::uint64_t StructureAction::version() const noexcept { return impl_->version; }
void StructureAction::diagonal_model(linalg::MatrixView h,
    linalg::MatrixView m) const {
  auto& p=*impl_;
  if (h.rows!=p.structures || m.rows!=p.structures || h.cols!=1 || m.cols!=1 ||
      h.ld<h.rows || m.ld<m.rows || !h.data || !m.data || h.data==m.data)
    throw std::invalid_argument("CUDA carrier Jacobi dimensions");
  const nvtx3::scoped_range range{"vb.nonorthogonal_jacobi"};
  // Only supported spin rows are prepared. Cross-spin contractions are
  // consumed at touched product groups, never expanded to a carrier grid.
  p.sigma.reset_workspace();
  const auto kernel=p.sigma.channel_kernel();
  Matrix<double> symmetric(p.context,kernel.rows,kernel.cols),
      projected(p.context,p.alpha_self->moments().rows,kernel.cols);
  const double one=1.0;
  check_cublas(cublasDgeam(p.context.blas(),CUBLAS_OP_N,CUBLAS_OP_T,
      blas_int(kernel.rows),blas_int(kernel.cols),&one,kernel.data,blas_int(kernel.ld),
      &one,kernel.data,blas_int(kernel.ld),symmetric.data(),blas_int(kernel.rows)),
      "CUDA spin-self cross kernel");
  if (p.model_version!=p.version) {
    p.alpha_self->update(p.model_upper.view(),kernel);
    if (p.beta_self_storage) p.beta_self_storage->update(p.model_upper.view(),kernel);
    p.model_version=p.version;
  }
  gemm(p.context,projected.view(),p.alpha_self->moments(),symmetric.view());
  check_cuda(cudaMemsetAsync(h.data,0,p.structures*sizeof(double),p.context.stream()),
      "clear carrier Jacobi H");
  check_cuda(cudaMemsetAsync(m.data,0,p.structures*sizeof(double),p.context.stream()),
      "clear carrier Jacobi M");
  const auto blocks=(p.groups+255)/256;
  if (blocks>std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA carrier Jacobi grid overflow");
  assemble_spin_self_model<<<static_cast<unsigned int>(blocks),256,0,p.context.stream()>>>(
      p.row_offsets.data(),p.product_indices.data(),p.columns.data(),
      p.weights.data(),projected.view(),p.beta_self->moments(),p.alpha_self_ids.data(),p.beta_self_ids.data(),
      p.alpha_self->energy().data,p.beta_self->energy().data,
      p.alpha_self->overlap().data,p.beta_self->overlap().data,
      h.data,m.data,p.groups,p.alpha,p.beta);
  check_cuda(cudaGetLastError(),"CUDA carrier Jacobi assembly");
}
void StructureAction::build_selected_metric(linalg::MatrixView output) const {
  const nvtx3::scoped_range range{"vb.selected_metric_gram"};
  const auto& p=*impl_;
  if (output.rows!=p.structures || output.cols!=p.structures ||
      output.ld<output.rows || !output.data)
    throw std::invalid_argument("CUDA selected metric output dimensions");
  if (p.groups==0) {
    for (int column=0;column<p.structures;++column)
      check_cuda(cudaMemsetAsync(output.data+output.ld*column,0,
          p.structures*sizeof(double),p.context.stream()),
          "clear empty selected metric");
    return;
  }
  const auto& counters=p.context.counters();
  const auto budget=p.context.device_byte_budget();
  if (counters.current_device_bytes>budget)
    throw std::runtime_error("CUDA selected metric accounting exceeds budget");
  std::size_t device_free=0,device_total=0;
  check_cuda(cudaMemGetInfo(&device_free,&device_total),
             "query CUDA selected metric free memory");
  const auto available=std::min(budget-counters.current_device_bytes,device_free);
  const auto elements_per_column=static_cast<std::size_t>(p.alpha)+p.beta+
      static_cast<std::size_t>(p.groups)+2*static_cast<std::size_t>(p.structures);
  if (elements_per_column>std::numeric_limits<std::size_t>::max()/sizeof(double))
    throw std::overflow_error("CUDA selected metric tile size overflow");
  const auto bytes_per_column=elements_per_column*sizeof(double);
  int width=static_cast<int>(std::min<std::size_t>({
      static_cast<std::size_t>(p.groups),static_cast<std::size_t>(p.structures),
      available/bytes_per_column}));
  Buffer<double> spin_grams;
  Matrix<double> product_gram,selected_product,source_terms;
  Buffer<std::byte> extra_workspace;
  for (;width>0;width/=2) {
    Buffer<double> proposed_grams(p.context,
        static_cast<std::size_t>(p.alpha+p.beta)*width);
    Matrix<double> proposed_product(p.context,p.groups,width);
    Matrix<double> proposed_selected(p.context,p.structures,width);
    Matrix<double> proposed_terms(p.context,width,p.structures);
    auto scratch=p.spmm_bytes(true,proposed_product.view(),proposed_selected.view());
    const int tail=static_cast<int>(p.groups%width);
    if (tail)
      scratch=std::max(scratch,p.spmm_bytes(true,
          proposed_product.view().columns(0,tail),
          proposed_selected.view().columns(0,tail)));
    if (scratch>p.sparse_workspace.size()) {
      std::size_t free_after=0,total_after=0;
      check_cuda(cudaMemGetInfo(&free_after,&total_after),
                 "query CUDA selected metric scratch memory");
      const auto budget_after=budget-p.context.counters().current_device_bytes;
      if (scratch>std::min(budget_after,free_after)) continue;
      extra_workspace=Buffer<std::byte>(p.context,scratch);
    }
    spin_grams=std::move(proposed_grams);
    product_gram=std::move(proposed_product);
    selected_product=std::move(proposed_selected);
    source_terms=std::move(proposed_terms);
    break;
  }
  if (width==0)
    throw std::runtime_error("CUDA selected metric budget cannot hold one source-product column");
  void* workspace=extra_workspace.empty() ?
      static_cast<void*>(p.sparse_workspace.data()):
      static_cast<void*>(extra_workspace.data());
  for (std::int64_t begin=0;begin<p.groups;begin+=width) {
    const int count=static_cast<int>(std::min<std::int64_t>(width,p.groups-begin));
    const auto alpha_gram=linalg::MatrixView{spin_grams.data(),p.alpha,count,p.alpha};
    const auto beta_gram=linalg::MatrixView{
        spin_grams.data()+static_cast<std::int64_t>(p.alpha)*count,
        p.beta,count,p.beta};
    const auto identity_elements=static_cast<std::int64_t>(p.alpha+p.beta)*count;
    spin_gram_identity<<<tile_grid(identity_elements),256,0,p.context.stream()>>>(
        p.product_indices.data(),begin,count,p.alpha,p.beta,alpha_gram,beta_gram);
    check_cuda(cudaGetLastError(),"CUDA selected spin Gram identities");
    if (p.beta_transform==&p.alpha_transform) {
      const linalg::MatrixView both{spin_grams.data(),p.alpha,2*count,p.alpha};
      p.alpha_transform.apply_left(both);
      p.alpha_transform.apply_left(both,true);
    } else {
      p.alpha_transform.apply_left(alpha_gram);
      p.alpha_transform.apply_left(alpha_gram,true);
      p.beta_transform->apply_left(beta_gram);
      p.beta_transform->apply_left(beta_gram,true);
    }
    const auto gram=product_gram.view().columns(0,count);
    const auto product_work=p.groups*static_cast<std::int64_t>(count);
    spin_gram_product<<<tile_grid(product_work),256,0,p.context.stream()>>>(
        p.product_indices.data(),begin,p.groups,count,p.alpha,
        alpha_gram,beta_gram,gram);
    check_cuda(cudaGetLastError(),"CUDA selected product Gram tile");
    const auto projected=selected_product.view().columns(0,count);
    p.spmm_with_workspace(true,gram,projected,workspace);
    const linalg::MatrixView terms{source_terms.data(),count,p.structures,width};
    for (int column=0;column<p.structures;++column)
      check_cuda(cudaMemsetAsync(terms.data+terms.ld*column,0,
          count*sizeof(double),p.context.stream()),
          "clear selected source term tile");
    selected_source_terms<<<tile_grid(count),256,0,p.context.stream()>>>(
        p.row_offsets.data(),p.columns.data(),p.weights.data(),
        begin,count,terms);
    check_cuda(cudaGetLastError(),"CUDA selected source terms");
    gemm(p.context,output,projected,terms,1.0,begin==0 ? 0.0:1.0);
  }
}
const DirectCiSigma& StructureAction::carrier_sigma() const noexcept { return impl_->sigma; }
const mcscf::ProductSupport* StructureAction::product_support() const noexcept {
  return impl_->exact_support.get();
}
std::array<linalg::MatrixView,2> StructureAction::carrier_workspace() const noexcept {
  return {impl_->coefficients,impl_->image};
}
void StructureAction::update_frame(linalg::ConstMatrixView upper) {
  auto& p=*impl_;
  if (upper.rows!=p.model_upper.rows() || upper.cols!=p.model_upper.cols() || upper.ld<upper.rows)
    throw std::invalid_argument("CUDA spin-self orbital frame shape mismatch");
  check_cuda(cudaMemcpy2DAsync(p.model_upper.data(),p.model_upper.rows()*sizeof(double),
      upper.data,upper.ld*sizeof(double),upper.rows*sizeof(double),upper.cols,
      cudaMemcpyDeviceToDevice,p.context.stream()),"CUDA spin-self frame update");
  impl_->alpha_transform.update_upper(upper);
  if (impl_->beta_storage) impl_->beta_storage->update_upper(upper);
  if (p.product_contraction) p.product_contraction->update_frame();
  ++impl_->version;
}
void StructureAction::update_integrals(linalg::ConstMatrixView h,
    linalg::ConstMatrixView g,linalg::ConstMatrixView upper) {
  impl_->sigma.update_integrals(h,g);
  update_frame(upper);
  if (impl_->carrier_diagonal.rows()) impl_->sigma.diagonal(impl_->carrier_diagonal.view());
}
void StructureAction::apply(linalg::ConstMatrixView vectors,
    linalg::MatrixView h,linalg::MatrixView m) const { impl_->apply(vectors,h,m); }

void StructureAction::apply_overlap(linalg::ConstMatrixView vectors,
    linalg::MatrixView overlap) const {
  const auto& p=*impl_;
  if (vectors.rows!=p.structures || overlap.rows!=p.structures || vectors.cols<=0 ||
      overlap.cols!=vectors.cols)
    throw std::invalid_argument("CUDA overlap block dimensions");
  if (p.product_contraction) {
    Matrix<double> h_products(p.context,p.groups,1),m_products(p.context,p.groups,1);
    const auto bytes=std::min(workspace::available(),
        p.context.device_byte_budget()-p.context.counters().current_device_bytes);
    for (std::int64_t rhs=0;rhs<vectors.cols;++rhs) {
      p.spmm(false,vectors.column(rhs),p.support.view());
      p.product_contraction->apply(p.support.view(),p.sigma,
          h_products.view(),m_products.view(),bytes,true);
      p.spmm(true,m_products.view(),overlap.column(rhs));
    }
    return;
  }
  for (std::int64_t rhs=0;rhs<vectors.cols;++rhs) {
    expand_spin(vectors.column(rhs),p.coefficients);
    apply_orbital_transform(p.coefficients);
    apply_adjoint_orbital_transform(p.coefficients);
    contract_spin(p.coefficients,overlap.column(rhs));
  }
}

int StructureAction::n_alpha_strings() const noexcept { return impl_->alpha; }
int StructureAction::n_beta_strings() const noexcept { return impl_->beta; }

void StructureAction::expand_spin(linalg::ConstMatrixView structures,
    linalg::MatrixView carrier) const {
  const auto& p=*impl_;
  if (structures.rows!=p.structures || structures.cols<=0 || carrier.rows!=p.alpha ||
      carrier.cols!=p.beta*structures.cols || carrier.ld!=p.alpha)
    throw std::invalid_argument("CUDA spin expansion dimensions");
  for (std::int64_t rhs=0;rhs<structures.cols;++rhs) {
    p.spmm(false,structures.column(rhs),p.support.view());
    auto panel=carrier.columns(rhs*p.beta,p.beta);
    check_cuda(cudaMemsetAsync(panel.data,0,p.products*sizeof(double),p.context.stream()),
        "clear expanded spin panel");
    thrust::scatter(thrust::cuda::par_nosync.on(p.context.stream()),p.support.data(),
        p.support.data()+p.groups,p.product_indices.data(),panel.data);
  }
}

void StructureAction::contract_spin(linalg::ConstMatrixView carrier,
    linalg::MatrixView structures) const {
  const auto& p=*impl_;
  if (structures.rows!=p.structures || structures.cols<=0 || carrier.rows!=p.alpha ||
      carrier.cols!=p.beta*structures.cols || carrier.ld!=p.alpha)
    throw std::invalid_argument("CUDA spin contraction dimensions");
  for (std::int64_t rhs=0;rhs<structures.cols;++rhs)
    p.project(carrier.columns(rhs*p.beta,p.beta),structures.column(rhs));
}

void StructureAction::apply_orbital_transform(linalg::MatrixView carrier) const {
  impl_->alpha_transform.apply_left(carrier);
  impl_->beta_transform->apply_right(carrier);
}
void StructureAction::apply_inverse_orbital_transform(linalg::MatrixView carrier) const {
  impl_->beta_transform->apply_right(carrier,false,true);
  impl_->alpha_transform.apply_left(carrier,false,true);
}
void StructureAction::apply_inverse_adjoint_orbital_transform(
    linalg::MatrixView carrier) const {
  impl_->beta_transform->apply_right(carrier,true,true);
  impl_->alpha_transform.apply_left(carrier,true,true);
}
void StructureAction::apply_adjoint_orbital_transform(linalg::MatrixView carrier) const {
  impl_->beta_transform->apply_right(carrier,true);
  impl_->alpha_transform.apply_left(carrier,true);
}
void StructureAction::apply_carrier_hamiltonian(linalg::ConstMatrixView carrier,
    linalg::MatrixView image) const { impl_->sigma.apply(carrier,image); }

double StructureAction::maximum_orthogonal_component(
    linalg::ConstMatrixView structure) const {
  auto& p=*impl_;
  if (structure.rows!=p.structures || structure.cols!=1 ||
      structure.ld<structure.rows || !structure.data)
    throw std::invalid_argument("CUDA orthogonal CI maximum dimensions");
  expand_spin(structure,p.coefficients);
  apply_orbital_transform(p.coefficients);
  int index=0;
  check_cublas(cublasSetPointerMode(p.context.blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA orthogonal CI maximum scalar pointer mode");
  check_cublas(cublasIdamax(p.context.blas(),blas_int(p.products),
      p.coefficients.data,1,&index),"CUDA orthogonal CI maximum index");
  if (index<1 || index>p.products)
    throw std::runtime_error("CUDA orthogonal CI maximum index is invalid");
  double value=0.0;
  p.context.copy_to_host(&value,p.coefficients.data+index-1,sizeof(double));
  if (!std::isfinite(value))
    throw std::runtime_error("CUDA orthogonal CI maximum is non-finite");
  return std::abs(value);
}

void StructureAction::divide_carrier_diagonal(linalg::MatrixView carrier,
    linalg::ConstMatrixView energies,double eigenvalue) const {
  auto& p=*impl_;
  if (carrier.rows!=p.alpha || carrier.cols<=0 || carrier.cols%p.beta!=0 ||
      energies.rows!=1 || energies.cols!=carrier.cols/p.beta)
    throw std::invalid_argument("CUDA carrier inverse dimensions");
  if (p.carrier_diagonal.rows()==0) {
    p.carrier_diagonal=Matrix<double>(p.context,p.alpha,p.beta);
    p.sigma.diagonal(p.carrier_diagonal.view());
  }
  const auto elements=p.products*energies.cols;
  const auto blocks=(elements+255)/256;
  if (blocks>std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA diagonal inverse grid overflow");
  divide_diagonal<<<static_cast<unsigned int>(blocks),256,0,p.context.stream()>>>(
      carrier,p.carrier_diagonal.data(),energies,p.products,eigenvalue,elements);
  check_cuda(cudaGetLastError(),"CUDA carrier diagonal inverse");
}

vb::StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  auto& context=impl_->context;
  const Eigen::MatrixXd packed=vectors;
  Matrix<double> x(context,packed.rows(),packed.cols()),h(context,packed.rows(),packed.cols()),
      m(context,packed.rows(),packed.cols());
  context.copy_to_device(x.data(),packed.data(),packed.size()*sizeof(double));
  apply(x.view(),h.view(),m.view());
  vb::StructureActionResult result{Eigen::MatrixXd(packed.rows(),packed.cols()),
      Eigen::MatrixXd(packed.rows(),packed.cols())};
  context.copy_to_host(result.hamiltonian.data(),h.data(),packed.size()*sizeof(double));
  context.copy_to_host(result.overlap.data(),m.data(),packed.size()*sizeof(double));
  return result;
}

} // namespace xmvb::cuda
