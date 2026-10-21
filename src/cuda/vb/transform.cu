#include "cuda/vb/transform.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

using Edge = vb::ExteriorOrbitalTransform::ShearEdge;

// The ordered elementary-factor coefficients are a small triangular problem,
// not a product-space contraction. Use exactly the CPU elimination ordering.
__global__ void factor_upper(linalg::ConstMatrixView upper,
    linalg::MatrixView unit,double* coefficients) {
  if (blockIdx.x || threadIdx.x) return;
  const int n=upper.rows;
  for (int j=0;j<n;++j) for (int i=0;i<n;++i)
    unit.data[i+unit.ld*j]=upper.data[i+upper.ld*j]/upper.data[j+upper.ld*j];
  int step=0;
  for (int removed=1;removed<n;++removed)
    for (int inserted=removed-1;inserted>=0;--inserted) {
      const double coefficient=unit.data[inserted+unit.ld*removed];
      coefficients[step++]=coefficient;
      for (int i=0;i<n;++i)
        unit.data[i+unit.ld*removed]-=coefficient*unit.data[i+unit.ld*inserted];
    }
}

__global__ void update_scales(linalg::ConstMatrixView upper,
    const std::uint64_t* occupations,double* scales,double* inverse,int dimension) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=dimension) return;
  double value=1.0;
  for (int p=0;p<upper.rows;++p)
    if (occupations[i]&(std::uint64_t{1}<<p)) value*=upper.data[p+upper.ld*p];
  scales[i]=value;
  inverse[i]=1.0/value;
}

__global__ void inverse_scale(const double* scales, double* inverse, int n) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) inverse[i] = 1.0/scales[i];
}

unsigned int grid_size(std::int64_t count) {
  constexpr std::int64_t threads = 256;
  const auto blocks = (count + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA exterior kernel grid is too large");
  return static_cast<unsigned int>(blocks);
}

// An elementary orbital shear has disjoint determinant source/target pairs.
// Thus each updated element has one owner, while the complete ordered product
// still follows the CPU factorization exactly.
__global__ void shear_left(linalg::MatrixView coefficients,
                           const Edge* edges, int edge_count,
                           const double* coefficient, double orientation, bool adjoint,
                           std::int64_t elements) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  const Edge edge = edges[index % edge_count];
  const std::int64_t column = index / edge_count;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  coefficients.data[target + coefficients.ld*column] +=
      orientation*(*coefficient)*edge.sign*coefficients.data[source + coefficients.ld*column];
}

__global__ void shear_right(linalg::MatrixView coefficients,
                            const Edge* edges, int edge_count,
                            const double* coefficient, double orientation, bool adjoint,
                            int spin_dimension, std::int64_t elements) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  // Adjacent lanes update adjacent rows in one column, not strided columns.
  const std::int64_t row = index % coefficients.rows;
  const std::int64_t side = index / coefficients.rows;
  const Edge edge = edges[side % edge_count];
  const std::int64_t rhs = side / edge_count;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  const std::int64_t source_column = rhs*spin_dimension + source;
  const std::int64_t target_column = rhs*spin_dimension + target;
  coefficients.data[row + coefficients.ld*target_column] +=
      orientation*(*coefficient)*edge.sign*
      coefficients.data[row + coefficients.ld*source_column];
}

}  // namespace

ExteriorTransform::ExteriorTransform(
    Context& context, const vb::ExteriorOrbitalTransform& source)
    : context_(&context), dimension_(source.dimension()),n_orbitals_(source.n_orbitals()),
      scales_(context, static_cast<std::size_t>(source.dimension())),
      inverse_scales_(context, static_cast<std::size_t>(source.dimension())),
      shear_coefficients_(context,source.shear_count()),
      occupations_(context,source.occupations().size()),
      unit_upper_(context,source.n_orbitals(),source.n_orbitals()) {
  if (dimension_ <= 0 || source.determinant_scales().size() != dimension_)
    throw std::invalid_argument("invalid CUDA exterior transform dimensions");
  context.copy_to_device(scales_.data(), source.determinant_scales().data(),
                         scales_.size()*sizeof(double));
  inverse_scale<<<grid_size(dimension_),256,0,context.stream()>>>(
      scales_.data(), inverse_scales_.data(), dimension_);
  check_cuda(cudaGetLastError(), "inverse exterior scaling");
  std::vector<Edge> all_edges;
  std::vector<double> coefficients;
  coefficients.reserve(source.shear_count());
  shears_.reserve(source.shear_count());
  for (std::size_t i = 0; i < source.shear_count(); ++i) {
    const auto edges = source.shear_edges(i);
    if (edges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("CUDA exterior shear is too large");
    shears_.push_back({all_edges.size(), static_cast<int>(edges.size())});
    coefficients.push_back(source.shear_coefficient(i));
    all_edges.insert(all_edges.end(), edges.begin(), edges.end());
  }
  edges_ = Buffer<Edge>(context, all_edges.size());
  context.copy_to_device(edges_.data(), all_edges.data(),
                         all_edges.size()*sizeof(Edge));
  context.copy_to_device(shear_coefficients_.data(),coefficients.data(),
                         coefficients.size()*sizeof(double));
  context.copy_to_device(occupations_.data(),source.occupations().data(),
                         occupations_.size()*sizeof(std::uint64_t));
}

void ExteriorTransform::update_upper(linalg::ConstMatrixView upper) {
  if (upper.rows!=n_orbitals_ || upper.cols!=n_orbitals_ ||
      upper.ld<upper.rows || !upper.data)
    throw std::invalid_argument("CUDA exterior frame dimensions");
  factor_upper<<<1,1,0,context_->stream()>>>(upper,unit_upper_.view(),
      shear_coefficients_.data());
  update_scales<<<grid_size(dimension_),256,0,context_->stream()>>>(upper,
      occupations_.data(),scales_.data(),inverse_scales_.data(),dimension_);
  check_cuda(cudaGetLastError(),"CUDA exterior frame update");
}

void ExteriorTransform::apply_left(linalg::MatrixView coefficients,
                                    bool adjoint, bool inverse) const {
  if (coefficients.rows != dimension_ || coefficients.cols <= 0 ||
      coefficients.ld < coefficients.rows || coefficients.data == nullptr)
    throw std::invalid_argument("invalid CUDA exterior left panel");
  const auto scale = [&] {
    // cuBLAS permits in-place dgmm when lda == ldc, as here.
    check_cublas(cublasDdgmm(context_->blas(), CUBLAS_SIDE_LEFT,
        blas_int(coefficients.rows), blas_int(coefficients.cols),
        coefficients.data, blas_int(coefficients.ld),
        inverse ? inverse_scales_.data() : scales_.data(), 1,
        coefficients.data, blas_int(coefficients.ld)),
        "cublasDdgmm exterior left scaling");
  };
  if (adjoint == inverse) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const auto index=adjoint != inverse ? shears_.size()-1-step : step;
    const Shear& shear = shears_[index];
    if (shear.count == 0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*coefficients.cols;
    shear_left<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear_coefficients_.data()+index,inverse ? -1.0:1.0,adjoint,work);
    check_cuda(cudaGetLastError(), "CUDA exterior left shear");
  }
  if (adjoint != inverse) {
    scale();
  }
}

void ExteriorTransform::apply_right(linalg::MatrixView coefficients,
                                     bool adjoint, bool inverse) const {
  if (coefficients.rows <= 0 || coefficients.cols <= 0 ||
      coefficients.cols % dimension_ != 0 ||
      coefficients.ld < coefficients.rows || coefficients.data == nullptr)
    throw std::invalid_argument("invalid CUDA exterior right panel");
  const std::int64_t rhs = coefficients.cols/dimension_;
  const auto scale = [&] {
    for (std::int64_t block = 0; block < rhs; ++block) {
      const auto panel = coefficients.columns(block*dimension_, dimension_);
      check_cublas(cublasDdgmm(context_->blas(), CUBLAS_SIDE_RIGHT,
          blas_int(panel.rows), blas_int(panel.cols), panel.data,
          blas_int(panel.ld), inverse ? inverse_scales_.data() : scales_.data(), 1, panel.data,
          blas_int(panel.ld)), "cublasDdgmm exterior right scaling");
    }
  };
  if (adjoint == inverse) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const auto index=adjoint != inverse ? shears_.size()-1-step : step;
    const Shear& shear = shears_[index];
    if (shear.count == 0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*
                              coefficients.rows*rhs;
    shear_right<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear_coefficients_.data()+index,inverse ? -1.0:1.0,adjoint,dimension_,work);
    check_cuda(cudaGetLastError(), "CUDA exterior right shear");
  }
  if (adjoint != inverse) {
    scale();
  }
}

}  // namespace xmvb::cuda
