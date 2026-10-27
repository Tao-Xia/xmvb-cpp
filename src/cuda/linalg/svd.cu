#include <cstdint>
#include "cuda/linalg/svd.hpp"

#include <algorithm>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

__global__ void upper_factor(linalg::ConstMatrixView qr, linalg::MatrixView r) {
    const auto k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (k < r.rows * r.cols) {
        const auto i = k % r.rows, j = k / r.rows;
        r.data[k] = i <= j ? qr.data[i + qr.ld * j] : 0.0;
    }
}

} // namespace

RightSvd right_svd(Context& context, linalg::ConstMatrixView input) {
    const nvtx3::scoped_range range{"linalg.right_svd"};
    const int m = blas_int(input.rows), n = blas_int(input.cols);
    if (!input.data || m <= 0 || n <= 0 || input.ld < m)
        throw std::invalid_argument("right SVD requires a nonempty device matrix");
    const bool compressed = m > n;
    const int rows = compressed ? n : m;
    Matrix<double> a(context, m, n), square, tau;
    check_cuda(cudaMemcpy2DAsync(a.data(),
                                 m * sizeof(double),
                                 input.data,
                                 input.ld * sizeof(double),
                                 m * sizeof(double),
                                 n,
                                 cudaMemcpyDeviceToDevice,
                                 context.stream()),
               "CUDA right SVD input copy");
    if (compressed) {
        square = Matrix<double>(context, n, n);
        tau = Matrix<double>(context, n, 1);
    }
    const auto factor = compressed ? square.view() : a.view();
    const int economy = rows >= n ? 1 : 0;
    Matrix<double> values(context, std::min(rows, n), 1), left(context, rows, economy ? n : rows);
    RightSvd result{Matrix<double>(context, n, n), {}};
    Buffer<int> info(context, 2);
    check_cuda(cudaMemsetAsync(info.data(), 0, 2 * sizeof(int), context.stream()),
               "CUDA right SVD status initialization");
    const auto solver = context.solver();
    gesvdjInfo_t params = nullptr;
    check_cusolver(cusolverDnCreateGesvdjInfo(&params), "cusolverDnCreateGesvdjInfo");
    try {
        int qr_size = 0, svd_size = 0;
        if (compressed)
            check_cusolver(cusolverDnDgeqrf_bufferSize(solver, m, n, a.data(), m, &qr_size),
                           "cusolverDnDgeqrf_bufferSize right SVD");
        check_cusolver(cusolverDnDgesvdj_bufferSize(solver,
                                                    CUSOLVER_EIG_MODE_VECTOR,
                                                    economy,
                                                    rows,
                                                    n,
                                                    factor.data,
                                                    rows,
                                                    values.data(),
                                                    left.data(),
                                                    rows,
                                                    result.right.data(),
                                                    n,
                                                    &svd_size,
                                                    params),
                       "cusolverDnDgesvdj_bufferSize right SVD");
        if (qr_size < 0 || svd_size < 0)
            throw std::runtime_error("negative right SVD workspace");
        Buffer<double> work(context, std::max(qr_size, svd_size));
        if (compressed) {
            check_cusolver(
                cusolverDnDgeqrf(
                    solver, m, n, a.data(), m, tau.data(), work.data(), qr_size, info.data()),
                "cusolverDnDgeqrf right SVD");
            upper_factor<<<(static_cast<std::int64_t>(n) * n + 127) / 128,
                           128,
                           0,
                           context.stream()>>>(a.view(), square.view());
            check_cuda(cudaGetLastError(), "CUDA right SVD upper factor");
        }
        check_cusolver(cusolverDnDgesvdj(solver,
                                         CUSOLVER_EIG_MODE_VECTOR,
                                         economy,
                                         rows,
                                         n,
                                         factor.data,
                                         rows,
                                         values.data(),
                                         left.data(),
                                         rows,
                                         result.right.data(),
                                         n,
                                         work.data(),
                                         svd_size,
                                         info.data() + 1,
                                         params),
                       "cusolverDnDgesvdj right SVD");
        int status[2] = {};
        context.copy_to_host(status, info.data(), sizeof(status));
        if (status[0] || status[1])
            throw std::runtime_error("CUDA right SVD did not converge");
        result.values.resize(std::min(m, n));
        context.copy_to_host(
            result.values.data(), values.data(), result.values.size() * sizeof(double));
        check_cusolver(cusolverDnDestroyGesvdjInfo(params), "cusolverDnDestroyGesvdjInfo");
    } catch (...) {
        cusolverDnDestroyGesvdjInfo(params);
        throw;
    }
    return result;
}

} // namespace xmvb::cuda
