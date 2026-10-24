#include "cuda/linalg/davidson.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"
#include "linalg/davidson_core.hpp"

namespace xmvb::cuda {
namespace {

__global__ void sign_seeds(linalg::MatrixView vectors, int first) {
    if (blockIdx.x || threadIdx.x)
        return;
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int column = first; column < vectors.cols; ++column)
        for (int row = 0; row < vectors.rows; ++row) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors.data[row + column * vectors.ld] = (state & 1ULL) ? 1.0 : -1.0;
        }
}

__global__ void jacobi_correction_kernel(double* correction,
                                         const double* residual,
                                         const double* diagonal_h,
                                         const double* diagonal_s,
                                         std::int64_t dimension,
                                         double lambda) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dimension)
        return;
    const double h = diagonal_h[index];
    const double lambda_s = lambda * diagonal_s[index];
    double denominator = h - lambda_s;
    const double scale = fmax(1.0, fmax(fabs(h), fabs(lambda_s)));
    const double floor = sqrt(2.2204460492503131e-16) * scale;
    if (fabs(denominator) < floor) {
        denominator = copysign(floor, denominator == 0.0 ? 1.0 : denominator);
    }
    correction[index] = -residual[index] / denominator;
}

__global__ void finite_matrix_kernel(
    const double* values, std::int64_t rows, std::int64_t cols, std::int64_t ld, int* finite) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols)
        return;
    if (!isfinite(values[(index / rows) * ld + index % rows])) {
        atomicExch(finite, 0);
    }
}

__global__ void
valid_diagonal_kernel(linalg::ConstMatrixView h, linalg::ConstMatrixView s, int* valid) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < h.rows && (!isfinite(h.data[i]) || !isfinite(s.data[i]) || !(s.data[i] > 0.0)))
        atomicExch(valid, 0);
}

struct DeviceBackend {
    using Matrix = cuda::Matrix<double>;
    static constexpr bool supports_host_callbacks = false;
    Context& context;
    const DeviceEigenAction& action;
    const DeviceEigenPreconditioner& inverse;
    Buffer<double> diagonal_h;
    Buffer<double> diagonal_s;
    linalg::ConstMatrixView h_diagonal, s_diagonal;

    DeviceBackend(Context& context_,
                  const DeviceEigenAction& action_,
                  const Eigen::Ref<const Eigen::VectorXd>& h,
                  const Eigen::Ref<const Eigen::VectorXd>& s,
                  const DeviceEigenPreconditioner& inverse_)
        : context(context_), action(action_), inverse(inverse_),
          diagonal_h(context_, static_cast<std::size_t>(h.size())),
          diagonal_s(context_, static_cast<std::size_t>(s.size())) {
        context.copy_to_device(
            diagonal_h.data(), h.data(), static_cast<std::size_t>(h.size()) * sizeof(double));
        context.copy_to_device(
            diagonal_s.data(), s.data(), static_cast<std::size_t>(s.size()) * sizeof(double));
        h_diagonal = {diagonal_h.data(), h.size(), 1, h.size()};
        s_diagonal = {diagonal_s.data(), s.size(), 1, s.size()};
        check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
                     "cublasSetPointerMode");
    }

    DeviceBackend(Context& c,
                  const DeviceEigenAction& a,
                  linalg::ConstMatrixView h,
                  linalg::ConstMatrixView s,
                  const DeviceEigenPreconditioner& p)
        : context(c), action(a), inverse(p), h_diagonal(h), s_diagonal(s) {
        check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
                     "cublasSetPointerMode");
    }
    void validate_diagonals(linalg::ConstMatrixView h, linalg::ConstMatrixView s) const {
        if (!h.data || !s.data || h.rows <= 0 || h.rows > std::numeric_limits<int>::max() ||
            h.cols != 1 || s.cols != 1 || s.rows != h.rows || h.ld < h.rows || s.ld < s.rows)
            throw std::invalid_argument("invalid device Davidson diagonals");
        Buffer<int> valid(context, 1);
        const int initial = 1;
        context.copy_to_device(valid.data(), &initial, sizeof(int));
        valid_diagonal_kernel<<<blas_int((h.rows + 255) / 256), 256, 0, context.stream()>>>(
            h, s, valid.data());
        check_cuda(cudaGetLastError(), "validate device Davidson diagonals");
        int result = 0;
        context.copy_to_host(&result, valid.data(), sizeof(int));
        if (!result)
            throw std::invalid_argument("invalid device Davidson diagonal values");
    }

    Matrix make(std::int64_t rows, std::int64_t cols) const {
        return Matrix(context, rows, cols);
    }
    Matrix upload(const Eigen::Ref<const Eigen::MatrixXd>& host) const {
        Matrix output(context, host.rows(), host.cols());
        upload_matrix(output.view(), host);
        return output;
    }
    Matrix upload(linalg::ConstMatrixView source) const {
        Matrix output(context, source.rows, source.cols);
        copy(output.view(), source);
        return output;
    }
    bool all_finite(linalg::ConstMatrixView source) const {
        Buffer<int> finite(context, 1);
        const int initial = 1;
        context.copy_to_device(finite.data(), &initial, sizeof(int));
        const std::int64_t count = source.rows * source.cols;
        constexpr int threads = 256;
        const int blocks = blas_int((count + threads - 1) / threads);
        finite_matrix_kernel<<<blocks, threads, 0, context.stream()>>>(
            source.data, source.rows, source.cols, source.ld, finite.data());
        check_cuda(cudaGetLastError(), "Davidson initial finite kernel launch");
        int host_finite = 0;
        context.copy_to_host(&host_finite, finite.data(), sizeof(int));
        return host_finite != 0;
    }
    linalg::MatrixView view(Matrix& matrix) const {
        return matrix.view();
    }
    linalg::ConstMatrixView cview(const Matrix& matrix) const {
        return matrix.view();
    }
    void upload_matrix(linalg::MatrixView destination,
                       const Eigen::Ref<const Eigen::MatrixXd>& host) const {
        if (destination.rows != host.rows() || destination.cols != host.cols()) {
            throw std::invalid_argument("CUDA Davidson upload dimensions");
        }
        for (std::int64_t column = 0; column < destination.cols; ++column) {
            context.copy_to_device(destination.column(column).data,
                                   host.col(column).data(),
                                   static_cast<std::size_t>(destination.rows) * sizeof(double));
        }
    }

    Eigen::MatrixXd download(linalg::ConstMatrixView source) const {
        Eigen::MatrixXd host(source.rows, source.cols);
        for (std::int64_t column = 0; column < source.cols; ++column) {
            context.copy_to_host(host.col(column).data(),
                                 source.column(column).data,
                                 static_cast<std::size_t>(source.rows) * sizeof(double));
        }
        return host;
    }
    void copy(linalg::MatrixView destination, linalg::ConstMatrixView source) const {
        if (destination.rows != source.rows || destination.cols != source.cols) {
            throw std::invalid_argument("CUDA Davidson copy dimensions");
        }
        for (std::int64_t column = 0; column < source.cols; ++column) {
            check_cuda(cudaMemcpyAsync(destination.column(column).data,
                                       source.column(column).data,
                                       static_cast<std::size_t>(source.rows) * sizeof(double),
                                       cudaMemcpyDeviceToDevice,
                                       context.stream()),
                       "Davidson device copy");
        }
    }
    void axpy(linalg::MatrixView destination, double factor, linalg::ConstMatrixView source) const {
        if (destination.rows != source.rows || destination.cols != source.cols) {
            throw std::invalid_argument("CUDA Davidson AXPY dimensions");
        }
        for (std::int64_t column = 0; column < source.cols; ++column) {
            check_cublas(cublasDaxpy(context.blas(),
                                     blas_int(source.rows),
                                     &factor,
                                     source.column(column).data,
                                     1,
                                     destination.column(column).data,
                                     1),
                         "cublasDaxpy");
        }
    }
    void scal(linalg::MatrixView destination, double factor) const {
        for (std::int64_t column = 0; column < destination.cols; ++column) {
            check_cublas(cublasDscal(context.blas(),
                                     blas_int(destination.rows),
                                     &factor,
                                     destination.column(column).data,
                                     1),
                         "cublasDscal");
        }
    }
    double dot(linalg::ConstMatrixView left, linalg::ConstMatrixView right) const {
        double result = 0.0;
        check_cublas(
            cublasDdot(context.blas(), blas_int(left.rows), left.data, 1, right.data, 1, &result),
            "cublasDdot");
        return result;
    }
    double norm(linalg::ConstMatrixView vector) const {
        double result = 0.0;
        check_cublas(cublasDnrm2(context.blas(), blas_int(vector.rows), vector.data, 1, &result),
                     "cublasDnrm2");
        return result;
    }
    void combine(linalg::MatrixView destination,
                 linalg::ConstMatrixView basis,
                 const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        Matrix device_coefficients = upload(coefficients);
        gemm(context, destination, basis, device_coefficients.view());
    }
    Eigen::MatrixXd project(linalg::ConstMatrixView basis, linalg::ConstMatrixView images) const {
        Matrix projected(context, basis.cols, images.cols);
        gemm(context, projected.view(), basis, images, 1.0, 0.0, Operation::transpose);
        return download(projected.view());
    }
    void subtract_projection(linalg::MatrixView destination,
                             linalg::ConstMatrixView basis,
                             const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        if (destination.cols != 1 || destination.rows != basis.rows ||
            coefficients.rows() != basis.cols || coefficients.cols() != 1) {
            throw std::invalid_argument("CUDA Davidson projection dimensions");
        }
        Matrix device_coefficients = upload(coefficients);
        constexpr double minus_one = -1.0;
        constexpr double one = 1.0;
        check_cublas(cublasDgemv(context.blas(),
                                 CUBLAS_OP_N,
                                 blas_int(basis.rows),
                                 blas_int(basis.cols),
                                 &minus_one,
                                 basis.data,
                                 blas_int(basis.ld),
                                 device_coefficients.data(),
                                 1,
                                 &one,
                                 destination.data,
                                 1),
                     "cublasDgemv");
    }
    template <class H, class S>
    void precondition(linalg::MatrixView correction,
                      linalg::ConstMatrixView residual,
                      const H&,
                      const S&,
                      double lambda) const {
        if (inverse) {
            inverse(residual, lambda, correction);
            constexpr double minus_one = -1.0;
            check_cublas(
                cublasDscal(
                    context.blas(), blas_int(correction.rows), &minus_one, correction.data, 1),
                "Davidson inverse sign");
            return;
        }
        const auto count = correction.rows;
        constexpr int threads = 256;
        const int blocks = blas_int((count + threads - 1) / threads);
        jacobi_correction_kernel<<<blocks, threads, 0, context.stream()>>>(
            correction.data, residual.data, h_diagonal.data, s_diagonal.data, count, lambda);
        check_cuda(cudaGetLastError(), "Davidson Jacobi kernel launch");
    }
    void apply(linalg::ConstMatrixView vectors, linalg::MatrixView h, linalg::MatrixView s) const {
        if (!action) {
            throw std::invalid_argument("generalized eigenvalue action is empty");
        }
        action(vectors, h, s);
    }
    void finish_device_result(DeviceDavidsonResult& result,
                              Matrix&& roots,
                              Matrix&& hamiltonian_roots,
                              Matrix&& overlap_roots,
                              bool identity_overlap) const {
        result.identity_overlap = identity_overlap;
        result.eigenpairs.eigenvector_matrix = std::move(roots);
        result.hamiltonian_eigenvectors = std::move(hamiltonian_roots);
        result.overlap_eigenvectors = std::move(overlap_roots);
    }
};

} // namespace

void fill_davidson_sign_seeds(Context& context, linalg::MatrixView vectors, int first_column) {
    if (!vectors.data || vectors.rows <= 0 || vectors.ld < vectors.rows || first_column < 0 ||
        first_column > vectors.cols)
        throw std::invalid_argument("invalid Davidson sign seed dimensions");
    if (first_column == vectors.cols)
        return;
    sign_seeds<<<1, 1, 0, context.stream()>>>(vectors, first_column);
    check_cuda(cudaGetLastError(), "Davidson fixed sign seeds");
}

linalg::DavidsonResult solve_davidson(Context& context,
                                      const DeviceEigenAction& action,
                                      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                                      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                                      const linalg::DavidsonOptions& options,
                                      const DeviceEigenPreconditioner& preconditioner) {
    const int dimension = static_cast<int>(hamiltonian_diagonal.size());
    linalg::detail::validate_davidson_diagonals(hamiltonian_diagonal, overlap_diagonal);
    linalg::detail::validate_davidson_options(dimension, options);
    const int n_initial = std::min(options.max_subspace_dimension, 2 * options.n_roots);
    const Eigen::MatrixXd initial =
        linalg::make_davidson_initial_vectors(hamiltonian_diagonal, overlap_diagonal, n_initial);
    return solve_davidson(
        context, action, hamiltonian_diagonal, overlap_diagonal, initial, options, preconditioner);
}

linalg::DavidsonResult solve_davidson(Context& context,
                                      const DeviceEigenAction& action,
                                      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                                      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                                      const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
                                      const linalg::DavidsonOptions& options,
                                      const DeviceEigenPreconditioner& preconditioner) {
    linalg::detail::validate_davidson_diagonals(hamiltonian_diagonal, overlap_diagonal);
    linalg::detail::validate_davidson_options(static_cast<int>(hamiltonian_diagonal.size()),
                                              options);
    if (initial_vectors.rows() != hamiltonian_diagonal.size() ||
        initial_vectors.cols() < options.n_roots ||
        initial_vectors.cols() > options.max_subspace_dimension || !initial_vectors.allFinite()) {
        throw std::invalid_argument("invalid Davidson initial vector block");
    }
    if (options.physical_residual_norm || options.residual_preconditioner) {
        throw std::invalid_argument("CUDA Davidson does not support host residual callbacks");
    }
    DeviceBackend backend(context, action, hamiltonian_diagonal, overlap_diagonal, preconditioner);
    return linalg::detail::solve_davidson(
        backend, hamiltonian_diagonal, overlap_diagonal, initial_vectors, options);
}

DeviceDavidsonResult
solve_device_davidson(Context& context,
                      const DeviceEigenAction& action,
                      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                      const linalg::DavidsonOptions& options,
                      const DeviceEigenPreconditioner& preconditioner) {
    const int dimension = static_cast<int>(hamiltonian_diagonal.size());
    linalg::detail::validate_davidson_diagonals(hamiltonian_diagonal, overlap_diagonal);
    linalg::detail::validate_davidson_options(dimension, options);
    const int n_initial = std::min(options.max_subspace_dimension, 2 * options.n_roots);
    const Eigen::MatrixXd initial =
        linalg::make_davidson_initial_vectors(hamiltonian_diagonal, overlap_diagonal, n_initial);
    return solve_device_davidson(
        context, action, hamiltonian_diagonal, overlap_diagonal, initial, options, preconditioner);
}

DeviceDavidsonResult
solve_device_davidson(Context& context,
                      const DeviceEigenAction& action,
                      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                      const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
                      const linalg::DavidsonOptions& options,
                      const DeviceEigenPreconditioner& preconditioner) {
    DeviceBackend backend(context, action, hamiltonian_diagonal, overlap_diagonal, preconditioner);
    return linalg::detail::solve_davidson<DeviceDavidsonResult>(
        backend, hamiltonian_diagonal, overlap_diagonal, initial_vectors, options);
}

DeviceDavidsonResult
solve_device_davidson(Context& context,
                      const DeviceEigenAction& action,
                      const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                      const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                      linalg::ConstMatrixView initial_vectors,
                      const linalg::DavidsonOptions& options,
                      const DeviceEigenPreconditioner& preconditioner) {
    DeviceBackend backend(context, action, hamiltonian_diagonal, overlap_diagonal, preconditioner);
    return linalg::detail::solve_davidson<DeviceDavidsonResult>(
        backend, hamiltonian_diagonal, overlap_diagonal, initial_vectors, options);
}

DeviceDavidsonResult solve_device_davidson(Context& context,
                                           const DeviceEigenAction& action,
                                           linalg::ConstMatrixView hamiltonian_diagonal,
                                           linalg::ConstMatrixView overlap_diagonal,
                                           linalg::ConstMatrixView initial_vectors,
                                           const linalg::DavidsonOptions& options,
                                           const DeviceEigenPreconditioner& preconditioner) {
    DeviceBackend backend(context, action, hamiltonian_diagonal, overlap_diagonal, preconditioner);
    return linalg::detail::solve_davidson<DeviceDavidsonResult>(
        backend, hamiltonian_diagonal, overlap_diagonal, initial_vectors, options);
}

} // namespace xmvb::cuda
