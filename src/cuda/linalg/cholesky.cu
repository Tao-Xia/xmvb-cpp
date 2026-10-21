#include "cuda/linalg/cholesky.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace xmvb::cuda {
namespace {

int checked_dimension(linalg::ConstMatrixView matrix) {
  if (matrix.data == nullptr || matrix.rows <= 0 ||
      matrix.rows != matrix.cols || matrix.ld < matrix.rows)
    throw std::invalid_argument("CUDA Cholesky requires a square device matrix");
  if (matrix.rows > std::numeric_limits<int>::max())
    throw std::overflow_error("CUDA Cholesky dimension exceeds cuSOLVER range");
  return static_cast<int>(matrix.rows);
}

void copy_columns(Context& context, linalg::MatrixView destination,
                  linalg::ConstMatrixView source) {
  check_cuda(cudaMemcpy2DAsync(destination.data,destination.ld*sizeof(double),
      source.data,source.ld*sizeof(double),source.rows*sizeof(double),source.cols,
      cudaMemcpyDeviceToDevice,context.stream()),"CUDA Cholesky matrix copy");
}

std::size_t span_bytes(linalg::ConstMatrixView matrix) {
  const auto columns = static_cast<std::uint64_t>(matrix.cols - 1);
  const auto leading = static_cast<std::uint64_t>(matrix.ld);
  const auto rows = static_cast<std::uint64_t>(matrix.rows);
  if (columns > (std::numeric_limits<std::size_t>::max() / sizeof(double) - rows) /
                    leading)
    throw std::overflow_error("CUDA Cholesky view span overflows size_t");
  return static_cast<std::size_t>(columns * leading + rows) * sizeof(double);
}

bool overlaps(linalg::ConstMatrixView left, linalg::ConstMatrixView right) {
  const auto a = reinterpret_cast<std::uintptr_t>(left.data);
  const auto b = reinterpret_cast<std::uintptr_t>(right.data);
  const auto a_bytes = span_bytes(left), b_bytes = span_bytes(right);
  if (a_bytes > std::numeric_limits<std::uintptr_t>::max() - a ||
      b_bytes > std::numeric_limits<std::uintptr_t>::max() - b)
    throw std::overflow_error("CUDA Cholesky device view address overflows");
  return a < b + b_bytes && b < a + a_bytes;
}

void check_info(Context& context, const Buffer<int>& info,
                const char* operation, bool factorization) {
  int status = 0;
  context.copy_to_host(&status, info.data(), sizeof(status));
  if (status < 0)
    throw std::runtime_error(std::string(operation) +
                             " rejected argument " + std::to_string(-status));
  if (status > 0)
    throw std::runtime_error(factorization ?
        "CUDA Cholesky matrix is not positive definite" :
        "CUDA Cholesky solve encountered a singular factor");
}

}  // namespace

Cholesky::Cholesky(Context& context, linalg::ConstMatrixView matrix)
    : context_(&context), dimension_(checked_dimension(matrix)),
      factor_(context, dimension_, dimension_), info_(context, 1) {
  const auto solver=context.solver();
  copy_columns(context, factor_.view(), matrix);
  int workspace_size = 0;
  check_cusolver(cusolverDnDpotrf_bufferSize(
      solver, CUBLAS_FILL_MODE_LOWER, dimension_, factor_.data(),
      dimension_, &workspace_size), "cusolverDnDpotrf_bufferSize");
  if (workspace_size < 0)
    throw std::runtime_error("CUDA Cholesky reported negative workspace size");
  Buffer<double> workspace(context, static_cast<std::size_t>(workspace_size));
  check_cusolver(cusolverDnDpotrf(
      solver, CUBLAS_FILL_MODE_LOWER, dimension_, factor_.data(),
      dimension_, workspace.data(), workspace_size, info_.data()),
      "cusolverDnDpotrf");
  check_info(context, info_, "cusolverDnDpotrf", true);
}

void Cholesky::solve(linalg::ConstMatrixView rhs, linalg::MatrixView out) const {
  if (rhs.data == nullptr || out.data == nullptr ||
      rhs.rows != dimension_ || out.rows != dimension_ ||
      rhs.cols <= 0 || out.cols != rhs.cols ||
      rhs.ld < rhs.rows || out.ld < out.rows)
    throw std::invalid_argument("CUDA Cholesky RHS/output dimensions disagree");
  if (rhs.cols > std::numeric_limits<int>::max() ||
      out.ld > std::numeric_limits<int>::max())
    throw std::overflow_error("CUDA Cholesky RHS exceeds cuSOLVER range");

  // Any partial overlap is staged before writing out. Exact aliases already
  // contain the RHS and can be passed directly to Dpotrs.
  if (rhs.data != out.data || rhs.ld != out.ld) {
    Matrix<double> staged;
    if (overlaps(rhs, out)) {
      staged = Matrix<double>(*context_, dimension_, rhs.cols);
      copy_columns(*context_, staged.view(), rhs);
      copy_columns(*context_, out, staged.view());
    } else {
      copy_columns(*context_, out, rhs);
    }
  }
  check_cusolver(cusolverDnDpotrs(
      context_->solver(), CUBLAS_FILL_MODE_LOWER, dimension_, static_cast<int>(rhs.cols),
      factor_.data(), dimension_, out.data, static_cast<int>(out.ld), info_.data()),
      "cusolverDnDpotrs");
  check_info(*context_, info_, "cusolverDnDpotrs", false);
}

}  // namespace xmvb::cuda
