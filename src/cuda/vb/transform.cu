#include "cuda/vb/transform.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

using Edge = vb::ExteriorOrbitalTransform::ShearEdge;

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
                           double coefficient, bool adjoint,
                           std::int64_t elements) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  const Edge edge = edges[index % edge_count];
  const std::int64_t column = index / edge_count;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  coefficients.data[target + coefficients.ld*column] +=
      coefficient*edge.sign*coefficients.data[source + coefficients.ld*column];
}

__global__ void shear_right(linalg::MatrixView coefficients,
                            const Edge* edges, int edge_count,
                            double coefficient, bool adjoint,
                            int spin_dimension, std::int64_t elements) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  const Edge edge = edges[index % edge_count];
  const std::int64_t side = index / edge_count;
  const std::int64_t row = side % coefficients.rows;
  const std::int64_t rhs = side / coefficients.rows;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  const std::int64_t source_column = rhs*spin_dimension + source;
  const std::int64_t target_column = rhs*spin_dimension + target;
  coefficients.data[row + coefficients.ld*target_column] +=
      coefficient*edge.sign*
      coefficients.data[row + coefficients.ld*source_column];
}

}  // namespace

ExteriorTransform::ExteriorTransform(
    Context& context, const vb::ExteriorOrbitalTransform& source)
    : context_(&context), dimension_(source.dimension()),
      scales_(context, static_cast<std::size_t>(source.dimension())) {
  if (dimension_ <= 0 || source.determinant_scales().size() != dimension_)
    throw std::invalid_argument("invalid CUDA exterior transform dimensions");
  context.copy_to_device(scales_.data(), source.determinant_scales().data(),
                         scales_.size()*sizeof(double));
  std::vector<Edge> all_edges;
  shears_.reserve(source.shear_count());
  for (std::size_t i = 0; i < source.shear_count(); ++i) {
    const auto edges = source.shear_edges(i);
    if (edges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("CUDA exterior shear is too large");
    shears_.push_back({all_edges.size(), static_cast<int>(edges.size()),
                       source.shear_coefficient(i)});
    all_edges.insert(all_edges.end(), edges.begin(), edges.end());
  }
  edges_ = Buffer<Edge>(context, all_edges.size());
  context.copy_to_device(edges_.data(), all_edges.data(),
                         all_edges.size()*sizeof(Edge));
}

void ExteriorTransform::apply_left(linalg::MatrixView coefficients,
                                    bool adjoint) const {
  if (coefficients.rows != dimension_ || coefficients.cols <= 0 ||
      coefficients.ld < coefficients.rows || coefficients.data == nullptr)
    throw std::invalid_argument("invalid CUDA exterior left panel");
  const auto scale = [&] {
    // cuBLAS permits in-place dgmm when lda == ldc, as here.
    check_cublas(cublasDdgmm(context_->blas(), CUBLAS_SIDE_LEFT,
        blas_int(coefficients.rows), blas_int(coefficients.cols),
        coefficients.data, blas_int(coefficients.ld), scales_.data(), 1,
        coefficients.data, blas_int(coefficients.ld)),
        "cublasDdgmm exterior left scaling");
  };
  if (!adjoint) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const Shear& shear = shears_[adjoint ? shears_.size()-1-step : step];
    if (shear.count == 0 || shear.coefficient == 0.0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*coefficients.cols;
    shear_left<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear.coefficient, adjoint, work);
    check_cuda(cudaGetLastError(), "CUDA exterior left shear");
  }
  if (adjoint) {
    scale();
  }
}

void ExteriorTransform::apply_right(linalg::MatrixView coefficients,
                                     bool adjoint) const {
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
          blas_int(panel.ld), scales_.data(), 1, panel.data,
          blas_int(panel.ld)), "cublasDdgmm exterior right scaling");
    }
  };
  if (!adjoint) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const Shear& shear = shears_[adjoint ? shears_.size()-1-step : step];
    if (shear.count == 0 || shear.coefficient == 0.0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*
                              coefficients.rows*rhs;
    shear_right<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear.coefficient, adjoint, dimension_, work);
    check_cuda(cudaGetLastError(), "CUDA exterior right shear");
  }
  if (adjoint) {
    scale();
  }
}

}  // namespace xmvb::cuda
