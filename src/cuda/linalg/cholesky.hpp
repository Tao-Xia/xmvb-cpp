#pragma once

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** Exact device Cholesky solve of one positive definite selected metric.
 * Owns an immutable lower factor. Construction certifies the factorization;
 * every solve checks the cuSOLVER result without downloading the matrix.
 * Calls use Context's stream and are serialized by the caller on that stream.
 */
class Cholesky {
 public:
  Cholesky(Context& context, linalg::ConstMatrixView matrix);
  Cholesky(const Cholesky&) = delete;
  Cholesky& operator=(const Cholesky&) = delete;

  /** Solve matrix * out = rhs. Exact in-place and overlapping views are valid. */
  void solve(linalg::ConstMatrixView rhs, linalg::MatrixView out) const;

  int dimension() const noexcept { return dimension_; }

 private:
  Context* context_;
  int dimension_;
  Matrix<double> factor_;
  mutable Buffer<int> info_;
};

}  // namespace xmvb::cuda
