#include "cuda/vb/action.hpp"

#include <cstdint>
#include <limits>
#include <vector>

#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/ci/sigma.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/vb/transform.hpp"

namespace xmvb::cuda {
namespace {

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
  mutable Matrix<double> coefficients, image, support;
  mutable Buffer<std::byte> sparse_workspace;

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
        coefficients(ctx, alpha, beta), image(ctx, alpha, beta), support(ctx, groups, 1) {
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
    } catch (...) {
      if (incidence) cusparseDestroySpMat(incidence);
      if (sparse) cusparseDestroy(sparse);
      throw;
    }
  }

  ~Impl() {
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

  void project(linalg::MatrixView output) const {
    const nvtx3::scoped_range range{"vb.backproject"};
    beta_transform->apply_right(image.view(), true);
    alpha_transform.apply_left(image.view(), true);
    thrust::gather(thrust::cuda::par_nosync.on(context.stream()), product_indices.data(),
        product_indices.data()+groups, image.data(), support.data());
    spmm(true,support.view(),output);
  }

  void apply(linalg::ConstMatrixView vectors,linalg::MatrixView h,linalg::MatrixView m) const {
    if (vectors.rows!=structures || h.rows!=structures || m.rows!=structures ||
        vectors.cols<=0 || h.cols!=vectors.cols || m.cols!=vectors.cols)
      throw std::invalid_argument("CUDA structure action block dimensions");
    for (std::int64_t rhs=0; rhs<vectors.cols; ++rhs) {
      {
      const nvtx3::scoped_range range{"vb.expand"};
      spmm(false,vectors.column(rhs),support.view());
      check_cuda(cudaMemsetAsync(coefficients.data(),0,products*sizeof(double),context.stream()),
          "clear product input");
      thrust::scatter(thrust::cuda::par_nosync.on(context.stream()),support.data(),support.data()+groups,
          product_indices.data(),coefficients.data());
      }
      {
      const nvtx3::scoped_range range{"vb.transform"};
      alpha_transform.apply_left(coefficients.view());
      beta_transform->apply_right(coefficients.view());
      }
      check_cuda(cudaMemcpyAsync(image.data(),coefficients.data(),products*sizeof(double),
          cudaMemcpyDeviceToDevice,context.stream()),"copy overlap image");
      project(m.column(rhs));
      sigma.apply(coefficients.view(),image.view());
      project(h.column(rhs));
    }
  }
};

StructureAction::StructureAction(Context& context,const vb::StructureAction& source)
    : impl_(std::make_unique<Impl>(context,source)) {}
StructureAction::~StructureAction()=default;
int StructureAction::size() const noexcept { return impl_->structures; }
const vb::StructureDiagonal& StructureAction::preconditioner_diagonal() const noexcept {
  return impl_->diagonal;
}
void StructureAction::apply(linalg::ConstMatrixView vectors,
    linalg::MatrixView h,linalg::MatrixView m) const { impl_->apply(vectors,h,m); }

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
