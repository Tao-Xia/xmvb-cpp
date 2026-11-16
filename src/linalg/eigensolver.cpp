#include "core/index.hpp"
#include "linalg/eigensolver.hpp"

#include "core/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <lapacke.h>

#include "linalg/davidson_core.hpp"
#include "parallel/openmp.hpp"

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

int vector_workers(std::int64_t rows) {
    const int workers = effective_openmp_thread_count();
    const auto cache = Eigen::l1CacheSize();
    // Small CI vectors stay in one cache. Large vectors share row slabs;
    // the threshold follows the current team's cache capacity, not a molecule.
    const auto cache_values =
        static_cast<std::int64_t>(cache) / static_cast<std::int64_t>(sizeof(double));
    return cache > 0 && rows > workers * cache_values ? workers : 1;
}

template <class Function> void for_vector_rows(std::int64_t rows, const Function& action) {
    const int workers = vector_workers(rows);
    if (workers == 1) {
        action(0, rows);
        return;
    }
#pragma omp parallel num_threads(workers)
    {
        std::int64_t rank = 0, team = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const auto first = (rows / team) * rank + std::min(rows % team, rank);
        const auto next = rank + 1;
        const auto last = (rows / team) * next + std::min(rows % team, next);
        action(first, last - first);
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
        copy(view(result), source);
        return result;
    }
    void copy(MatrixView destination, ConstMatrixView source) const {
        if (destination.rows != source.rows || destination.cols != source.cols) {
            throw std::invalid_argument("CPU Davidson copy dimensions");
        }
        for_vector_rows(source.rows, [&](auto first, auto count) {
            for (std::int64_t column = 0; column < source.cols; ++column)
                std::copy_n(source.column(column).data + first,
                            count,
                            destination.column(column).data + first);
        });
    }
    void axpy(MatrixView destination, double factor, ConstMatrixView source) const {
        for_vector_rows(source.rows, [&](auto first, auto count) {
            for (std::int64_t column = 0; column < source.cols; ++column)
                Eigen::Map<Eigen::VectorXd>(destination.column(column).data + first, count)
                    .noalias() += factor * Eigen::Map<const Eigen::VectorXd>(
                                               source.column(column).data + first, count);
        });
    }
    void scal(MatrixView destination, double factor) const {
        for_vector_rows(destination.rows, [&](auto first, auto count) {
            for (std::int64_t column = 0; column < destination.cols; ++column)
                Eigen::Map<Eigen::VectorXd>(destination.column(column).data + first, count) *=
                    factor;
        });
    }
    double dot(ConstMatrixView left, ConstMatrixView right) const {
        const int workers = vector_workers(left.rows);
        if (workers == 1)
            return Eigen::Map<const Eigen::VectorXd>(left.data, left.rows)
                .dot(Eigen::Map<const Eigen::VectorXd>(right.data, right.rows));
        std::vector<double> partials(workers, 0.0);
        for_vector_rows(left.rows, [&](auto first, auto count) {
            std::size_t worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            partials[worker] =
                Eigen::Map<const Eigen::VectorXd>(left.data + first, count)
                    .dot(Eigen::Map<const Eigen::VectorXd>(right.data + first, count));
        });
        return std::accumulate(partials.begin(), partials.end(), 0.0);
    }
    double norm(ConstMatrixView vector) const {
        const int workers = vector_workers(vector.rows);
        if (workers == 1)
            return Eigen::Map<const Eigen::VectorXd>(vector.data, vector.rows).norm();
        std::vector<double> partials(workers, 0.0);
        for_vector_rows(vector.rows, [&](auto first, auto count) {
            std::size_t worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            partials[worker] = Eigen::Map<const Eigen::VectorXd>(vector.data + first, count).norm();
        });
        double result = 0;
        for (const auto norm : partials)
            result = std::hypot(result, norm);
        return result;
    }
    void combine(MatrixView destination,
                 ConstMatrixView basis,
                 const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<Eigen::MatrixXd, 0, Stride> output(
            destination.data, destination.rows, destination.cols, Stride(destination.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        for_vector_rows(destination.rows, [&](auto first, auto count) {
            output.middleRows(first, count).noalias() =
                input.middleRows(first, count) * coefficients;
        });
    }
    Matrix project(ConstMatrixView basis, ConstMatrixView images) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input_basis(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input_images(
            images.data, images.rows, images.cols, Stride(images.ld));
        const int workers = vector_workers(basis.rows);
        if (workers == 1)
            return input_basis.transpose() * input_images;
        // Only the small Gram matrices are thread-local; the CI basis is borrowed.
        std::vector<Matrix> partials(workers, Matrix::Zero(basis.cols, images.cols));
        for_vector_rows(basis.rows, [&](auto first, auto count) {
            std::size_t worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            partials[worker].noalias() = input_basis.middleRows(first, count).transpose() *
                                         input_images.middleRows(first, count);
        });
        Matrix result = std::move(partials.front());
        for (int worker = 1; worker < workers; ++worker)
            result += partials[worker];
        return result;
    }
    void subtract_projection(MatrixView destination,
                             ConstMatrixView basis,
                             const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<Eigen::MatrixXd, 0, Stride> output(
            destination.data, destination.rows, destination.cols, Stride(destination.ld));
        Eigen::Map<const Eigen::MatrixXd, 0, Stride> input(
            basis.data, basis.rows, basis.cols, Stride(basis.ld));
        for_vector_rows(destination.rows, [&](auto first, auto count) {
            output.middleRows(first, count).noalias() -=
                input.middleRows(first, count) * coefficients;
        });
    }
    void precondition(MatrixView correction,
                      ConstMatrixView residual,
                      double energy,
                      const DavidsonPreconditioner& inverse) const {
        const Eigen::Map<const Eigen::VectorXd> r(residual.data, residual.rows);
        Eigen::Map<Eigen::VectorXd> p(correction.data, correction.rows);
        inverse(r, energy, p);
        if (!p.allFinite())
            throw std::runtime_error("Davidson residual preconditioner returned an invalid vector");
        scal(correction, -1.0);
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
        // The action borrows its input for this call, just as the basis view does.
        // Keep its column stride instead of copying a full CI block first.
        const Eigen::Map<const Matrix, 0, Eigen::OuterStride<>> input(
            vectors.data, vectors.rows, vectors.cols, Eigen::OuterStride<>(vectors.ld));
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

namespace {

template <class Diagonal>
Eigen::MatrixXd initial_vectors(Eigen::Index dimension, int n_vectors, const Diagonal& diagonal) {
    const int n_diagonal_vectors = n_vectors / 2;
    std::vector<Eigen::Index> diagonal_order(n_diagonal_vectors);
    std::ranges::partial_sort_copy(
        std::views::iota(Eigen::Index{0}, dimension),
        diagonal_order,
        [&](Eigen::Index left, Eigen::Index right) { return diagonal(left) < diagonal(right); });
    Eigen::MatrixXd vectors = Eigen::MatrixXd::Zero(dimension, n_vectors);
    for (int vector = 0; vector < n_diagonal_vectors; ++vector) {
        vectors(diagonal_order[vector], vector) = 1.0;
    }
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int vector = n_diagonal_vectors; vector < n_vectors; ++vector) {
        for (Eigen::Index row = 0; row < dimension; ++row) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vectors(row, vector) = (state & 1ULL) == 0ULL ? -1.0 : 1.0;
        }
    }
    return vectors;
}

} // namespace

Eigen::MatrixXd
make_davidson_initial_vectors(const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                              const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
                              int n_vectors) {
    return initial_vectors(hamiltonian_diagonal.size(), n_vectors, [&](Eigen::Index row) {
        return hamiltonian_diagonal[row] / overlap_diagonal[row];
    });
}

Eigen::MatrixXd make_davidson_initial_vectors(const Eigen::Ref<const Eigen::VectorXd>& diagonal,
                                              int n_vectors) {
    return initial_vectors(
        diagonal.size(), n_vectors, [&](Eigen::Index row) { return diagonal[row]; });
}

std::size_t davidson_workspace_bytes(Eigen::Index dimension, const DavidsonOptions& options) {
    const auto n = core::checked_index<std::size_t>(dimension);
    const auto m = core::checked_index<std::size_t>(options.max_subspace_dimension);
    const auto roots = core::checked_index<std::size_t>(options.n_roots);
    const std::size_t images = options.identity_overlap ? 2 : 3;
    // Live basis and restart copies, plus seeds, Ritz images and corrections.
    const auto vectors = core::checked_multiply(images, 2 * m + 6 * roots + 4);
    const auto projected = core::checked_multiply(12, core::checked_multiply(m, m));
    return core::checked_multiply(sizeof(double),
                                  core::checked_add(core::checked_multiply(n, vectors), projected));
}

DavidsonOptions make_davidson_options(Eigen::Index dimension,
                                      int n_roots,
                                      double energy_tolerance,
                                      double relative_residual_tolerance,
                                      double stationarity_tolerance,
                                      std::optional<std::size_t> workspace_bytes,
                                      bool identity_overlap) {
    if (dimension <= 0 || n_roots <= 0 || n_roots > dimension) {
        throw std::invalid_argument("invalid Davidson eigenproblem dimensions");
    }
    if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
        !std::isfinite(relative_residual_tolerance) || relative_residual_tolerance <= 0.0 ||
        !std::isfinite(stationarity_tolerance) || stationarity_tolerance <= 0.0) {
        throw std::invalid_argument("Davidson accuracy tolerances must be positive");
    }
    const auto bytes = workspace_bytes ? *workspace_bytes : workspace::available();
    const int minimum = core::checked_index<int>(std::min<Eigen::Index>(dimension, 2LL * n_roots));
    // Cap projected work by a restart dimension, independently of memory.
    // Keep at least four blocks of room beyond the retained Ritz subspace.
    const int preferred = core::checked_index<int>(std::min<Eigen::Index>(
        {dimension, std::max(32LL, 8LL * n_roots), std::numeric_limits<int>::max()}));
    DavidsonOptions options{n_roots,
                            core::checked_index<Eigen::Index>(core::checked_multiply(
                                2, core::checked_index<std::size_t>(dimension))),
                            minimum,
                            energy_tolerance,
                            relative_residual_tolerance,
                            stationarity_tolerance};
    options.identity_overlap = identity_overlap;
    if (davidson_workspace_bytes(dimension, options) > bytes)
        throw std::length_error("Davidson workspace cannot hold the selected-root expansion");
    int lo = minimum, hi = preferred;
    while (lo < hi) {
        options.max_subspace_dimension = lo + (hi - lo) / 2 + 1;
        if (davidson_workspace_bytes(dimension, options) <= bytes)
            lo = options.max_subspace_dimension;
        else
            hi = options.max_subspace_dimension - 1;
    }
    options.max_subspace_dimension = lo;
    return options;
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
