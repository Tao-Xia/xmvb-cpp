#include "core/index.hpp"
#include "linalg/eigensolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <lapacke.h>

#include "linalg/davidson_core.hpp"

namespace xmvb::linalg {
namespace {

void validate_generalized_eigenproblem_inputs(std::span<const double> hamiltonian_matrix,
                                              std::span<const double> overlap_matrix,
                                              Eigen::Index dimension) {
    if (dimension <= 0) {
        throw std::invalid_argument("dimension must be positive");
    }
    const std::size_t expected_size = core::checked_multiply(
        core::checked_index<std::size_t>(dimension), core::checked_index<std::size_t>(dimension));
    if (hamiltonian_matrix.size() != expected_size || overlap_matrix.size() != expected_size) {
        throw std::invalid_argument(
            "hamiltonian_matrix and overlap_matrix must be square matrices");
    }
}

void check_generalized_eigensolver_info(lapack_int info) {
    if (info < 0) {
        throw std::runtime_error("LAPACKE_dsygvd failed: illegal argument " +
                                 std::to_string(-info));
    }
    if (info > 0) {
        throw std::runtime_error("LAPACKE_dsygvd failed: overlap_matrix is not positive definite"
                                 " (info=" +
                                 std::to_string(info) + ")");
    }
}

struct CpuBackend {
    using Matrix = Eigen::MatrixXd;
    static constexpr bool supports_host_callbacks = true;
    const GeneralizedEigenAction& action;
    bool identity_overlap;

    Matrix make(std::int64_t rows, std::int64_t cols) const {
        return Matrix(rows, cols);
    }
    Matrix upload(const Eigen::Ref<const Eigen::MatrixXd>& host) const {
        return Matrix(host);
    }
    MatrixView view(Matrix& matrix) const {
        return {matrix.data(), matrix.rows(), matrix.cols(), matrix.outerStride()};
    }
    ConstMatrixView cview(const Matrix& matrix) const {
        return {matrix.data(), matrix.rows(), matrix.cols(), matrix.outerStride()};
    }
    Matrix download(ConstMatrixView source) const {
        Matrix result(source.rows, source.cols);
        for (std::int64_t column = 0; column < source.cols; ++column) {
            std::copy_n(source.column(column).data, source.rows, result.col(column).data());
        }
        return result;
    }
    void upload_into(MatrixView destination, const Eigen::Ref<const Eigen::VectorXd>& host) const {
        if (destination.cols != 1 || destination.rows != host.size()) {
            throw std::invalid_argument("CPU Davidson upload dimensions");
        }
        std::copy_n(host.data(), host.size(), destination.data);
    }
    void copy(MatrixView destination, ConstMatrixView source) const {
        if (destination.rows != source.rows || destination.cols != source.cols) {
            throw std::invalid_argument("CPU Davidson copy dimensions");
        }
        for (std::int64_t column = 0; column < source.cols; ++column) {
            std::copy_n(source.column(column).data, source.rows, destination.column(column).data);
        }
    }
    void axpy(MatrixView destination, double factor, ConstMatrixView source) const {
        for (std::int64_t column = 0; column < source.cols; ++column) {
            const auto src = source.column(column);
            const auto dst = destination.column(column);
            Eigen::Map<Eigen::VectorXd>(dst.data, dst.rows).noalias() +=
                factor * Eigen::Map<const Eigen::VectorXd>(src.data, src.rows);
        }
    }
    void scal(MatrixView destination, double factor) const {
        for (std::int64_t column = 0; column < destination.cols; ++column) {
            const auto dst = destination.column(column);
            Eigen::Map<Eigen::VectorXd>(dst.data, dst.rows) *= factor;
        }
    }
    double dot(ConstMatrixView left, ConstMatrixView right) const {
        return Eigen::Map<const Eigen::VectorXd>(left.data, left.rows)
            .dot(Eigen::Map<const Eigen::VectorXd>(right.data, right.rows));
    }
    double norm(ConstMatrixView vector) const {
        return Eigen::Map<const Eigen::VectorXd>(vector.data, vector.rows).norm();
    }
    void combine(MatrixView destination,
                 ConstMatrixView basis,
                 const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<Eigen::MatrixXd, 0, Stride> output(
            destination.data, destination.rows, destination.cols, Stride(destination.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        output.noalias() = input * coefficients;
    }
    Matrix project(ConstMatrixView basis, ConstMatrixView images) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input_basis(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input_images(
            images.data, images.rows, images.cols, Stride(images.ld));
        return input_basis.transpose() * input_images;
    }
    void subtract_projection(MatrixView destination,
                             ConstMatrixView basis,
                             const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<Eigen::MatrixXd, 0, Stride> output(
            destination.data, destination.rows, destination.cols, Stride(destination.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        output.noalias() -= input * coefficients;
    }
    void precondition(MatrixView correction,
                      ConstMatrixView residual,
                      const Eigen::Ref<const Eigen::VectorXd>& diagonal_h,
                      const Eigen::Ref<const Eigen::VectorXd>& diagonal_s,
                      double lambda) const {
        for (Eigen::Index i = 0; i < diagonal_h.size(); ++i) {
            double denominator = diagonal_h[i] - lambda * diagonal_s[i];
            const double floor =
                std::sqrt(std::numeric_limits<double>::epsilon()) *
                std::max({1.0, std::abs(diagonal_h[i]), std::abs(lambda * diagonal_s[i])});
            if (std::abs(denominator) < floor) {
                denominator = std::copysign(floor, denominator == 0.0 ? 1.0 : denominator);
            }
            correction.data[i] = -residual.data[i] / denominator;
        }
    }
    void apply(ConstMatrixView vectors, MatrixView h, MatrixView s) const {
        if (!action) {
            throw std::invalid_argument("generalized eigenvalue action is empty");
        }
        Matrix input = download(vectors);
        GeneralizedEigenActionResult images = action(input);
        if (images.hamiltonian.rows() != h.rows || images.hamiltonian.cols() != h.cols ||
            !images.hamiltonian.allFinite() ||
            (identity_overlap
                 ? images.overlap.size() != 0
                 : images.overlap.rows() != s.rows || images.overlap.cols() != s.cols ||
                       !images.overlap.allFinite())) {
            throw std::runtime_error("generalized eigenvalue action returned invalid block images");
        }
        copy(h, cview(images.hamiltonian));
        if (!identity_overlap)
            copy(s, cview(images.overlap));
    }
};

} // namespace

Eigen::MatrixXd
make_davidson_initial_vectors(const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                              const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                              int n_vectors) {
    const Eigen::Index dimension = hamiltonian_diagonal.size();
    std::vector<int> diagonal_order(dimension);
    for (int index = 0; index < dimension; ++index)
        diagonal_order[index] = index;
    std::partial_sort(diagonal_order.begin(),
                      diagonal_order.begin() + n_vectors / 2,
                      diagonal_order.end(),
                      [&](int left, int right) {
                          return hamiltonian_diagonal[left] / overlap_diagonal[left] <
                                 hamiltonian_diagonal[right] / overlap_diagonal[right];
                      });
    Eigen::MatrixXd vectors = Eigen::MatrixXd::Zero(dimension, n_vectors);
    const int n_diagonal_vectors = n_vectors / 2;
    for (int vector = 0; vector < n_diagonal_vectors; ++vector) {
        vectors(diagonal_order[vector], vector) = 1.0;
    }
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int vector = n_diagonal_vectors; vector < n_vectors; ++vector) {
        for (int row = 0; row < dimension; ++row) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors(row, vector) = (state & 1ULL) == 0ULL ? -1.0 : 1.0;
        }
    }
    return vectors;
}

DavidsonOptions make_davidson_options(Eigen::Index dimension,
                                      int n_roots,
                                      double energy_tolerance,
                                      double relative_residual_tolerance,
                                      double stationarity_tolerance) {
    if (dimension <= 0 || n_roots <= 0 || n_roots > dimension) {
        throw std::invalid_argument("invalid Davidson eigenproblem dimensions");
    }
    if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
        !std::isfinite(relative_residual_tolerance) || relative_residual_tolerance <= 0.0 ||
        !std::isfinite(stationarity_tolerance) || stationarity_tolerance <= 0.0) {
        throw std::invalid_argument("Davidson accuracy tolerances must be positive");
    }
    constexpr long double kBasisBudgetDoubles =
        static_cast<long double>(512ULL * 1024ULL * 1024ULL) / sizeof(double);
    const long double n = static_cast<long double>(dimension);
    const long double capacity = 2.0L * kBasisBudgetDoubles /
                                 (std::sqrt(9.0L * n * n + 4.0L * kBasisBudgetDoubles) + 3.0L * n);
    const int max_subspace = core::checked_index<int>(
        std::min<Eigen::Index>(dimension, static_cast<int>(std::floor(capacity))));
    if (max_subspace < std::min<Eigen::Index>(dimension, 2LL * n_roots)) {
        throw std::runtime_error("Davidson selected-root block exceeds the basis workspace budget");
    }
    return DavidsonOptions{n_roots,
                           core::checked_index<Eigen::Index>(core::checked_multiply(
                               2, core::checked_index<std::size_t>(dimension))),
                           max_subspace,
                           energy_tolerance,
                           relative_residual_tolerance,
                           stationarity_tolerance};
}

GeneralizedEigenResult
GeneralizedEigensolver::solve_dense(std::span<const double> hamiltonian_matrix,
                                    std::span<const double> overlap_matrix,
                                    Eigen::Index dimension) const {
    validate_generalized_eigenproblem_inputs(hamiltonian_matrix, overlap_matrix, dimension);
    const auto lapack_dimension = core::checked_index<lapack_int>(dimension);
    GeneralizedEigenResult result;
    std::vector<double> h_copy(hamiltonian_matrix.begin(), hamiltonian_matrix.end());
    std::vector<double> s_copy(overlap_matrix.begin(), overlap_matrix.end());
    result.eigenvalues.resize(static_cast<std::size_t>(dimension));
    const lapack_int info = LAPACKE_dsygvd(LAPACK_COL_MAJOR,
                                           1,
                                           'V',
                                           'U',
                                           lapack_dimension,
                                           h_copy.data(),
                                           lapack_dimension,
                                           s_copy.data(),
                                           lapack_dimension,
                                           result.eigenvalues.data());
    check_generalized_eigensolver_info(info);
    result.eigenvector_matrix = std::move(h_copy);
    return result;
}

std::vector<double>
GeneralizedEigensolver::solve_dense_eigenvalues(std::span<const double> hamiltonian_matrix,
                                                std::span<const double> overlap_matrix,
                                                Eigen::Index dimension) const {
    validate_generalized_eigenproblem_inputs(hamiltonian_matrix, overlap_matrix, dimension);
    const auto lapack_dimension = core::checked_index<lapack_int>(dimension);
    std::vector<double> h_copy(hamiltonian_matrix.begin(), hamiltonian_matrix.end());
    std::vector<double> s_copy(overlap_matrix.begin(), overlap_matrix.end());
    std::vector<double> eigenvalues(static_cast<std::size_t>(dimension));
    const lapack_int info = LAPACKE_dsygvd(LAPACK_COL_MAJOR,
                                           1,
                                           'N',
                                           'U',
                                           lapack_dimension,
                                           h_copy.data(),
                                           lapack_dimension,
                                           s_copy.data(),
                                           lapack_dimension,
                                           eigenvalues.data());
    check_generalized_eigensolver_info(info);
    return eigenvalues;
}

DavidsonResult GeneralizedEigensolver::solve_davidson(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const DavidsonOptions& options) const {
    const Eigen::Index dimension = hamiltonian_diagonal.size();
    detail::validate_davidson_diagonals(hamiltonian_diagonal, overlap_diagonal);
    detail::validate_davidson_options(dimension, options);
    const int n_initial = std::min(options.max_subspace_dimension, 2 * options.n_roots);
    const Eigen::MatrixXd initial =
        make_davidson_initial_vectors(hamiltonian_diagonal, overlap_diagonal, n_initial);
    return solve_davidson(action, hamiltonian_diagonal, overlap_diagonal, initial, options);
}

DavidsonResult GeneralizedEigensolver::solve_davidson(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
    const DavidsonOptions& options) const {
    CpuBackend backend{action, options.identity_overlap};
    return detail::solve_davidson(
        backend, hamiltonian_diagonal, overlap_diagonal, initial_vectors, options);
}

} // namespace xmvb::linalg
