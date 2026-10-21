#include "cuda/vb/action.hpp"

#include <cstdint>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/ci/sigma.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/memory/graph.hpp"
#include "cuda/vb/transform.hpp"

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

__global__ void assemble_carrier_jacobi(const std::int64_t* row_offsets,
    const std::int64_t* products,const std::int64_t* columns,
    const double* weights,const double* carrier_diagonal,
    const double* alpha_scales,const double* beta_scales,
    double* hamiltonian,double* overlap,std::int64_t groups,
    int n_alpha,int n_beta) {
  const std::int64_t group=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (group>=groups) return;
  const auto product=products[group];
  const int a=static_cast<int>(product%n_alpha);
  const int b=static_cast<int>(product/n_alpha);
  const double scaled=alpha_scales[a]*beta_scales[b];
  const double energy=carrier_diagonal[product];
  for (auto term=row_offsets[group];term<row_offsets[group+1];++term) {
    const auto structure=columns[term];
    bool first=true;
    for (auto prior=row_offsets[group];prior<term;++prior)
      if (columns[prior]==structure) { first=false; break; }
    if (!first) continue;
    double combined=0.0;
    for (auto matching=term;matching<row_offsets[group+1];++matching)
      if (columns[matching]==structure) combined+=weights[matching];
    const double coefficient=scaled*combined;
    const double square=coefficient*coefficient;
    atomicAdd(hamiltonian+structure,square*energy);
    atomicAdd(overlap+structure,square);
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
  Buffer<std::int64_t> row_offsets, product_indices, columns;
  Buffer<double> weights;
  cusparseHandle_t sparse = nullptr;
  cusparseSpMatDescr_t incidence = nullptr;
  mutable Matrix<double> carriers, support;
  linalg::MatrixView coefficients, image;
  mutable Buffer<std::byte> sparse_workspace;
  mutable Matrix<double> input, hamiltonian, overlap;
  mutable Matrix<double> carrier_diagonal;
  std::unique_ptr<Graph> graph;
  double graph_seconds=0;
  std::uint64_t version=0;

  Impl(Context& ctx, const vb::StructureAction& source)
      : context(ctx), structures(source.n_structures()),
        alpha(source.n_alpha_strings()), beta(source.n_beta_strings()),
        groups(source.orthogonal_view().products.size()),
        products(static_cast<std::int64_t>(alpha)*beta),
        diagonal(source.preconditioner_diagonal()),
        sigma(ctx, source.orthogonal_view().sigma),
        alpha_transform(ctx, source.orthogonal_view().alpha),
        beta_storage(&source.orthogonal_view().alpha == &source.orthogonal_view().beta
            ? nullptr : std::make_unique<ExteriorTransform>(ctx, source.orthogonal_view().beta)),
        beta_transform(beta_storage ? beta_storage.get() : &alpha_transform),
        row_offsets(ctx, groups+1), product_indices(ctx, groups),
        columns(ctx, source.orthogonal_view().terms.size()),
        weights(ctx, source.orthogonal_view().terms.size()),
        carriers(ctx,alpha,2*static_cast<std::int64_t>(beta)),support(ctx,groups,1),
        coefficients(carriers.view().columns(0,beta)),image(carriers.view().columns(beta,beta)),
        input(ctx,structures,1),hamiltonian(ctx,structures,1),overlap(ctx,structures,1) {
    const auto view=source.orthogonal_view();
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
      sigma.prepare(1);
      context.synchronize();
      const auto start=std::chrono::steady_clock::now();
      graph=std::make_unique<Graph>(context,[&] {
        emit(input.view(),hamiltonian.view(),overlap.view());
      });
      context.synchronize();
      graph_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
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

  void spmm(bool transpose, linalg::ConstMatrixView input,
             linalg::MatrixView output) const {
    DenseDescriptor x(input), y(output);
    const double one=1, zero=0;
    check_sparse(cusparseSpMM(sparse,
        transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, incidence, x, &zero, y,
        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, sparse_workspace.data()), "cusparseSpMM");
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
      sigma.apply(coefficients,image);
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
std::size_t StructureAction::graph_nodes() const noexcept { return impl_->graph->nodes(); }
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
  if (p.carrier_diagonal.rows()==0) {
    p.carrier_diagonal=Matrix<double>(p.context,p.alpha,p.beta);
    p.sigma.diagonal(p.carrier_diagonal.view());
  }
  check_cuda(cudaMemsetAsync(h.data,0,p.structures*sizeof(double),p.context.stream()),
      "clear carrier Jacobi H");
  check_cuda(cudaMemsetAsync(m.data,0,p.structures*sizeof(double),p.context.stream()),
      "clear carrier Jacobi M");
  const auto blocks=(p.groups+255)/256;
  if (blocks>std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA carrier Jacobi grid overflow");
  assemble_carrier_jacobi<<<static_cast<unsigned int>(blocks),256,0,p.context.stream()>>>(
      p.row_offsets.data(),p.product_indices.data(),p.columns.data(),
      p.weights.data(),p.carrier_diagonal.data(),
      p.alpha_transform.scale_data(),p.beta_transform->scale_data(),
      h.data,m.data,p.groups,p.alpha,p.beta);
  check_cuda(cudaGetLastError(),"CUDA carrier Jacobi assembly");
}
const DirectCiSigma& StructureAction::carrier_sigma() const noexcept { return impl_->sigma; }
void StructureAction::update_frame(linalg::ConstMatrixView upper) {
  impl_->alpha_transform.update_upper(upper);
  if (impl_->beta_storage) impl_->beta_storage->update_upper(upper);
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
