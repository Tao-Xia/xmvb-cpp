#include <cstdint>
#include "cuda/linalg/spectrum.hpp"

#include <stdexcept>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {
void solver_check(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(operation);
}

__global__ void symmetrize(double* values, int n) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n * n) {
        const int i = k % n, j = k / n;
        if (i < j) {
            const double v = 0.5 * (values[k] + values[j + i * n]);
            values[k] = v;
            values[j + i * n] = v;
        }
    }
}

__global__ void powers(const double* eigenvalues, double* weights, int n, double exponent) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        weights[i] = pow(eigenvalues[i], exponent);
}
} // namespace

SymmetricSpectrum
symmetric_spectrum(Context& ctx, cusolverDnHandle_t solver, linalg::ConstMatrixView input) {
    const int n = blas_int(input.rows);
    if (n <= 0 || input.cols != n || input.ld < n || !input.data)
        throw std::invalid_argument("symmetric spectrum requires a square device matrix");
    SymmetricSpectrum out{Matrix<double>(ctx, n, n), Matrix<double>(ctx, n, 1), {}};
    check_cuda(cudaMemcpy2DAsync(out.vectors.data(),
                                 n * sizeof(double),
                                 input.data,
                                 input.ld * sizeof(double),
                                 n * sizeof(double),
                                 n,
                                 cudaMemcpyDeviceToDevice,
                                 ctx.stream()),
               "CUDA symmetric spectrum copy");
    symmetrize<<<(static_cast<std::int64_t>(n) * n + 127) / 128, 128, 0, ctx.stream()>>>(
        out.vectors.data(), n);
    check_cuda(cudaGetLastError(), "CUDA symmetric spectrum symmetrize");
    int size = 0;
    solver_check(cusolverDnDsyevd_bufferSize(solver,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER,
                                             n,
                                             out.vectors.data(),
                                             n,
                                             out.values.data(),
                                             &size),
                 "cusolverDnDsyevd_bufferSize");
    Buffer<double> work(ctx, size);
    Buffer<int> info(ctx, 1);
    solver_check(cusolverDnDsyevd(solver,
                                  CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER,
                                  n,
                                  out.vectors.data(),
                                  n,
                                  out.values.data(),
                                  work.data(),
                                  size,
                                  info.data()),
                 "cusolverDnDsyevd");
    int status = 0;
    ctx.copy_to_host(&status, info.data(), sizeof(int));
    if (status)
        throw std::runtime_error("CUDA symmetric eigensolver did not converge");
    out.host_values.resize(n);
    ctx.copy_to_host(out.host_values.data(), out.values.data(), n * sizeof(double));
    return out;
}

Matrix<double> spectral_power(Context& ctx, const SymmetricSpectrum& eig, double exponent) {
    const int n = blas_int(eig.values.rows());
    Matrix<double> weights(ctx, n, 1), scaled(ctx, n, n), result(ctx, n, n);
    powers<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
        eig.values.data(), weights.data(), n, exponent);
    check_cuda(cudaGetLastError(), "CUDA spectral powers");
    check_cublas(cublasDdgmm_64(ctx.blas(),
                                CUBLAS_SIDE_RIGHT,
                                n,
                                n,
                                eig.vectors.data(),
                                n,
                                weights.data(),
                                1,
                                scaled.data(),
                                n),
                 "CUDA spectral column scaling");
    gemm(ctx,
         result.view(),
         scaled.view(),
         eig.vectors.view(),
         1.0,
         0.0,
         Operation::none,
         Operation::transpose);
    return result;
}

AoMetric::AoMetric(Context& ctx, const Eigen::Ref<const Eigen::MatrixXd>& overlap)
    : context_(&ctx), overlap_(ctx, overlap.rows(), overlap.cols()) {
    const nvtx3::scoped_range range{"ao.metric_factorization"};
    if (overlap.rows() <= 0 || overlap.rows() != overlap.cols())
        throw std::invalid_argument("AO metric must be square");
    copy_to_device(ctx, overlap_.view(), overlap);
    auto eig = symmetric_spectrum(ctx, ctx.solver(), overlap_.view());
    if (!(eig.host_values.front() > 0.0))
        throw std::runtime_error("AO metric is not positive definite");
    root_ = spectral_power(ctx, eig, 0.5);
    inverse_root_ = spectral_power(ctx, eig, -0.5);
}
} // namespace xmvb::cuda
