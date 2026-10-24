#pragma once

#include <vector>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

/** Full right singular vectors and descending singular values.
 * Tall inputs are reduced by QR before Jacobi SVD of the square R factor.
 * No normal equations, numerical cutoff, truncation or host factorization.
 * Rank decisions belong to the caller and retain the original input shape.
 */
struct RightSvd {
    Matrix<double> right;
    std::vector<double> values;
};

RightSvd right_svd(Context&, linalg::ConstMatrixView);

} // namespace xmvb::cuda
