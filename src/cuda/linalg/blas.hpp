#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

enum class Operation { none, transpose };

/** Add only the independent upper entries of a matrix cotangent. */
void add_upper(Context&, linalg::MatrixView target, linalg::ConstMatrixView source, double scale);
/** Same accumulation with a stream-ordered device scalar. */
void add_upper_device(Context&,
                      linalg::MatrixView target,
                      linalg::ConstMatrixView source,
                      const double* scale);

inline int blas_int(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::overflow_error("cuBLAS dimension exceeds int range");
    }
    return static_cast<int>(value);
}

inline void gemm(Context& context,
                 linalg::MatrixView out,
                 linalg::ConstMatrixView a,
                 linalg::ConstMatrixView b,
                 double alpha = 1.0,
                 double beta = 0.0,
                 Operation op_a = Operation::none,
                 Operation op_b = Operation::none) {
    const auto a_rows = op_a == Operation::none ? a.rows : a.cols;
    const auto a_cols = op_a == Operation::none ? a.cols : a.rows;
    const auto b_rows = op_b == Operation::none ? b.rows : b.cols;
    const auto b_cols = op_b == Operation::none ? b.cols : b.rows;
    if (a_cols != b_rows || out.rows != a_rows || out.cols != b_cols ||
        (a.cols > 0 && a.ld < a.rows) || (b.cols > 0 && b.ld < b.rows) ||
        (out.cols > 0 && out.ld < out.rows)) {
        throw std::invalid_argument("cuBLAS GEMM matrix dimensions");
    }
    if (out.empty())
        return;
    const auto trans_a = op_a == Operation::none ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto trans_b = op_b == Operation::none ? CUBLAS_OP_N : CUBLAS_OP_T;
    check_cublas(cublasDgemm_64(context.blas(),
                                trans_a,
                                trans_b,
                                out.rows,
                                out.cols,
                                a_cols,
                                &alpha,
                                a.data,
                                a.ld,
                                b.data,
                                b.ld,
                                &beta,
                                out.data,
                                out.ld),
                 "cublasDgemm");
}

} // namespace xmvb::cuda
