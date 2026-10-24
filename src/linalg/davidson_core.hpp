#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <lapacke.h>

#include "linalg/eigensolver.hpp"
#include "linalg/views.hpp"

namespace xmvb::linalg::detail {

struct ProjectedEigenpairs {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

inline ProjectedEigenpairs
solve_lowest_projected_eigenpairs(const Eigen::Ref<const Eigen::MatrixXd>& hamiltonian,
                                  int n_roots) {
    const int dimension = static_cast<int>(hamiltonian.rows());
    Eigen::MatrixXd matrix = hamiltonian;
    ProjectedEigenpairs result{Eigen::VectorXd(n_roots), Eigen::MatrixXd(dimension, n_roots)};
    std::vector<lapack_int> support(2 * n_roots);
    lapack_int n_converged = 0;
    const lapack_int info = LAPACKE_dsyevr(LAPACK_COL_MAJOR,
                                           'V',
                                           'I',
                                           'U',
                                           dimension,
                                           matrix.data(),
                                           dimension,
                                           0.0,
                                           0.0,
                                           1,
                                           n_roots,
                                           0.0,
                                           &n_converged,
                                           result.eigenvalues.data(),
                                           result.eigenvectors.data(),
                                           dimension,
                                           support.data());
    if (info != 0 || n_converged != n_roots) {
        throw std::runtime_error(
            "Davidson projected eigensolve failed (info=" + std::to_string(info) + ")");
    }
    return result;
}

inline void validate_davidson_options(int dimension, const DavidsonOptions& options) {
    if (dimension <= 0 || options.n_roots <= 0 || options.n_roots > dimension) {
        throw std::invalid_argument("invalid Davidson eigenproblem dimensions");
    }
    if (options.max_iterations <= 0 ||
        options.max_subspace_dimension < std::min(dimension, 2 * options.n_roots) ||
        options.max_subspace_dimension > dimension) {
        throw std::invalid_argument("invalid Davidson iteration or subspace budget");
    }
    if (!std::isfinite(options.residual_tolerance) || options.residual_tolerance <= 0.0) {
        throw std::invalid_argument("Davidson residual tolerance must be positive");
    }
    if (!std::isfinite(options.energy_tolerance) || options.energy_tolerance <= 0.0) {
        throw std::invalid_argument("Davidson energy tolerance must be positive");
    }
    if (!std::isfinite(options.stationarity_tolerance) || options.stationarity_tolerance <= 0.0) {
        throw std::invalid_argument("Davidson stationarity tolerance must be positive");
    }
}

inline void
validate_davidson_diagonals(const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
                            const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal) {
    if (overlap_diagonal.size() != hamiltonian_diagonal.size() ||
        !hamiltonian_diagonal.allFinite() || !overlap_diagonal.allFinite() ||
        (overlap_diagonal.array() <= 0.0).any()) {
        throw std::invalid_argument("invalid generalized eigenproblem diagonals");
    }
}

/** Existing orthogonalization scalars, retained only for failure reporting. */
struct DavidsonRankInfo {
    double initial_scale = 0.0, metric_norm_squared = 0.0, final_scale = 0.0, threshold = 0.0;
};

struct DavidsonProjectionInfo {
    double raw_norm = 0.0;
    double projected_norm = 0.0;
    double roundoff_bound = 0.0;
};

/** Standard floating-point accumulation bound gamma_n = nu/(1-nu). */
inline double rounding_bound(std::ptrdiff_t operations) {
    const double nu = operations * std::numeric_limits<double>::epsilon();
    return nu < 1.0 ? nu / (1.0 - nu) : std::numeric_limits<double>::infinity();
}

template <class Backend>
int append_s_orthonormal_block(Backend& backend,
                               ConstMatrixView candidates,
                               ConstMatrixView hamiltonian_candidates,
                               ConstMatrixView overlap_candidates,
                               MatrixView basis,
                               MatrixView hamiltonian_basis,
                               MatrixView overlap_basis,
                               int active_dimension,
                               bool identity_overlap,
                               Eigen::VectorXd& basis_norms,
                               DavidsonRankInfo* rank_info = nullptr) {
    const double dependence_floor = 64.0 * std::numeric_limits<double>::epsilon();
    auto vector_storage = backend.make(candidates.rows, 1);
    auto hamiltonian_storage = backend.make(candidates.rows, 1);
    auto overlap_storage = identity_overlap ? backend.make(0, 0) : backend.make(candidates.rows, 1);
    auto vector = backend.view(vector_storage);
    auto hamiltonian_image = backend.view(hamiltonian_storage);
    auto overlap_image = identity_overlap ? vector : backend.view(overlap_storage);

    for (int candidate = 0; candidate < candidates.cols && active_dimension < basis.cols;
         ++candidate) {
        backend.copy(vector, candidates.column(candidate));
        backend.copy(hamiltonian_image, hamiltonian_candidates.column(candidate));
        if (!identity_overlap) {
            backend.copy(overlap_image, overlap_candidates.column(candidate));
        }
        const double initial_metric_scale =
            std::max(std::abs(backend.dot(vector, overlap_image)),
                     backend.norm(vector) * backend.norm(overlap_image));

        for (int pass = 0; pass < 2; ++pass) {
            if (active_dimension == 0)
                break;
            const Eigen::MatrixXd coefficients =
                backend.project(basis.columns(0, active_dimension), overlap_image);
            backend.subtract_projection(vector, basis.columns(0, active_dimension), coefficients);
            backend.subtract_projection(
                hamiltonian_image, hamiltonian_basis.columns(0, active_dimension), coefficients);
            if (!identity_overlap) {
                backend.subtract_projection(
                    overlap_image, overlap_basis.columns(0, active_dimension), coefficients);
            }
        }
        const double metric_norm_squared = backend.dot(vector, overlap_image);
        const double vector_norm = backend.norm(vector);
        const double scale = vector_norm * backend.norm(overlap_image);
        const double threshold =
            dependence_floor * std::max(initial_metric_scale, std::numeric_limits<double>::min());
        if (rank_info)
            *rank_info = {initial_metric_scale, metric_norm_squared, scale, threshold};
        if (!std::isfinite(metric_norm_squared) || metric_norm_squared <= threshold ||
            metric_norm_squared <= 0.0 || !std::isfinite(scale)) {
            continue;
        }
        const double inverse_metric_norm = 1.0 / std::sqrt(metric_norm_squared);
        basis_norms[active_dimension] = vector_norm * inverse_metric_norm;
        backend.copy(basis.column(active_dimension), vector);
        backend.scal(basis.column(active_dimension), inverse_metric_norm);
        backend.copy(hamiltonian_basis.column(active_dimension), hamiltonian_image);
        backend.scal(hamiltonian_basis.column(active_dimension), inverse_metric_norm);
        if (!identity_overlap) {
            backend.copy(overlap_basis.column(active_dimension), overlap_image);
            backend.scal(overlap_basis.column(active_dimension), inverse_metric_norm);
        }
        ++active_dimension;
    }
    return active_dimension;
}

/** Remove redundant components BEFORE forming fresh H/M images.
 * By symmetry V^T M p = (MV)^T p, so this needs no new operator action.
 * Euclidean QR within the correction block preserves its joint span and
 * prevents cancellation against newly appended roots after the action too.
 */
template <class Backend>
DavidsonProjectionInfo project_davidson_corrections(Backend& backend,
                                                    MatrixView corrections,
                                                    ConstMatrixView basis,
                                                    ConstMatrixView overlap_basis,
                                                    const Eigen::VectorXd& basis_norms) {
    DavidsonProjectionInfo info;
    for (int column = 0; column < corrections.cols; ++column) {
        auto p = corrections.column(column);
        const double initial_norm = backend.norm(p);
        const Eigen::MatrixXd coefficients = backend.project(overlap_basis, p);
        const double terms = coefficients.col(0).cwiseAbs().dot(basis_norms.head(basis.cols));
        double error = rounding_bound(2 * basis.cols + 1) * (initial_norm + terms);
        backend.subtract_projection(p, basis, coefficients);
        for (int pass = 0; column > 0 && pass < 2; ++pass) {
            const auto previous = corrections.columns(0, column);
            const Eigen::MatrixXd weights = backend.project(previous, p);
            // Previous nonzero columns have Euclidean norm one. Zero columns
            // contribute no coefficient and need no separate packing or action.
            error += rounding_bound(2 * column + 1) * (backend.norm(p) + weights.cwiseAbs().sum());
            backend.subtract_projection(p, previous, weights);
        }
        const double projected_norm = backend.norm(p);
        info = {initial_norm, projected_norm, error};
        backend.scal(p, projected_norm <= error ? 0.0 : 1.0 / projected_norm);
    }
    return info;
}

template <class Backend>
void update_projected_hamiltonian(Backend& backend,
                                  ConstMatrixView basis,
                                  ConstMatrixView hamiltonian_basis,
                                  int first_new_column,
                                  int active_dimension,
                                  Eigen::MatrixXd& projected_hamiltonian) {
    const int n_new_columns = active_dimension - first_new_column;
    if (n_new_columns <= 0)
        return;
    const Eigen::MatrixXd new_columns =
        backend.project(basis.columns(0, active_dimension),
                        hamiltonian_basis.columns(first_new_column, n_new_columns));
    projected_hamiltonian.block(0, first_new_column, active_dimension, n_new_columns) = new_columns;
    if (first_new_column > 0) {
        projected_hamiltonian.block(first_new_column, 0, n_new_columns, first_new_column) =
            new_columns.topRows(first_new_column).transpose();
    }
    auto diagonal_block = projected_hamiltonian.block(
        first_new_column, first_new_column, n_new_columns, n_new_columns);
    diagonal_block = (0.5 * (diagonal_block + diagonal_block.transpose())).eval();
}

template <class Backend>
void validate_images(Backend& backend,
                     ConstMatrixView h,
                     ConstMatrixView s,
                     int dimension,
                     int width,
                     bool identity_overlap) {
    if (h.rows != dimension || h.cols != width ||
        (!identity_overlap && (s.rows != dimension || s.cols != width)) ||
        (identity_overlap && !s.empty())) {
        throw std::runtime_error("generalized eigenvalue action returned invalid block images");
    }
    for (int column = 0; column < width; ++column) {
        if (!std::isfinite(backend.norm(h.column(column))) ||
            (!identity_overlap && !std::isfinite(backend.norm(s.column(column))))) {
            throw std::runtime_error("generalized eigenvalue action returned invalid block images");
        }
    }
}

/** Shared iteration and stopping decisions for host and CUDA storage. */
template <class Result = DavidsonResult,
          class Backend,
          class InitialVectors,
          class HamiltonianDiagonal,
          class OverlapDiagonal>
Result solve_davidson(Backend& backend,
                      const HamiltonianDiagonal& hamiltonian_diagonal,
                      const OverlapDiagonal& overlap_diagonal,
                      const InitialVectors& initial_vectors,
                      const DavidsonOptions& options) {
    const int dimension = [&] {
        if constexpr (std::is_same_v<HamiltonianDiagonal, ConstMatrixView>) {
            backend.validate_diagonals(hamiltonian_diagonal, overlap_diagonal);
            return static_cast<int>(hamiltonian_diagonal.rows);
        } else {
            validate_davidson_diagonals(hamiltonian_diagonal, overlap_diagonal);
            return static_cast<int>(hamiltonian_diagonal.size());
        }
    }();
    validate_davidson_options(dimension, options);
    const auto initial_rows = [&] {
        if constexpr (std::is_same_v<InitialVectors, ConstMatrixView>) {
            return initial_vectors.rows;
        } else {
            return initial_vectors.rows();
        }
    }();
    const auto initial_cols = [&] {
        if constexpr (std::is_same_v<InitialVectors, ConstMatrixView>) {
            return initial_vectors.cols;
        } else {
            return initial_vectors.cols();
        }
    }();
    if (initial_rows != dimension || initial_cols < options.n_roots ||
        initial_cols > options.max_subspace_dimension) {
        throw std::invalid_argument("invalid Davidson initial vector block");
    }
    const bool initial_finite = [&] {
        if constexpr (std::is_same_v<InitialVectors, ConstMatrixView>) {
            return initial_vectors.data != nullptr && initial_vectors.ld >= initial_vectors.rows &&
                   backend.all_finite(initial_vectors);
        } else {
            return initial_vectors.allFinite();
        }
    }();
    if (!initial_finite) {
        throw std::invalid_argument("invalid Davidson initial vector block");
    }
    if constexpr (!Backend::supports_host_callbacks) {
        if (options.physical_residual_norm || options.residual_preconditioner) {
            throw std::invalid_argument("CUDA Davidson does not support host residual callbacks");
        }
    }

    const int max_subspace = options.max_subspace_dimension;
    Eigen::VectorXd basis_norms = Eigen::VectorXd::Zero(max_subspace);
    auto basis = backend.make(dimension, max_subspace);
    auto hamiltonian_basis = backend.make(dimension, max_subspace);
    auto overlap_basis = backend.make(dimension, options.identity_overlap ? 0 : max_subspace);
    Result result;
    int active_dimension = 0;
    {
        auto candidate = backend.upload(initial_vectors);
        auto hamiltonian_candidate = backend.make(dimension, initial_cols);
        auto overlap_candidate =
            backend.make(dimension, options.identity_overlap ? 0 : initial_cols);
        backend.apply(backend.cview(candidate),
                      backend.view(hamiltonian_candidate),
                      backend.view(overlap_candidate));
        validate_images(backend,
                        backend.cview(hamiltonian_candidate),
                        backend.cview(overlap_candidate),
                        dimension,
                        static_cast<int>(initial_cols),
                        options.identity_overlap);
        ++result.block_actions;
        active_dimension = append_s_orthonormal_block(backend,
                                                      backend.cview(candidate),
                                                      backend.cview(hamiltonian_candidate),
                                                      backend.cview(overlap_candidate),
                                                      backend.view(basis),
                                                      backend.view(hamiltonian_basis),
                                                      backend.view(overlap_basis),
                                                      0,
                                                      options.identity_overlap,
                                                      basis_norms);
    }
    if (active_dimension < options.n_roots) {
        throw std::runtime_error(
            "Davidson initial vectors are linearly dependent in the overlap metric");
    }
    Eigen::MatrixXd projected_hamiltonian = Eigen::MatrixXd::Zero(max_subspace, max_subspace);
    update_projected_hamiltonian(backend,
                                 backend.cview(basis),
                                 backend.cview(hamiltonian_basis),
                                 0,
                                 active_dimension,
                                 projected_hamiltonian);
    Eigen::VectorXd previous_eigenvalues;

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        result.iterations = iteration;
        result.peak_subspace_dimension = std::max(result.peak_subspace_dimension, active_dimension);
        const Eigen::MatrixXd active_projected_hamiltonian =
            projected_hamiltonian.topLeftCorner(active_dimension, active_dimension);
        const int n_projected_roots = active_dimension + options.n_roots > max_subspace
                                          ? active_dimension
                                          : std::min(active_dimension, options.n_roots + 1);
        const auto projected =
            solve_lowest_projected_eigenpairs(active_projected_hamiltonian, n_projected_roots);
        const auto& projected_eigenvalues = projected.eigenvalues;
        const auto& projected_eigenvectors = projected.eigenvectors;
        const Eigen::MatrixXd root_coefficients = projected_eigenvectors.leftCols(options.n_roots);
        auto roots = backend.make(dimension, options.n_roots);
        auto hamiltonian_roots = backend.make(dimension, options.n_roots);
        auto overlap_roots =
            backend.make(dimension, options.identity_overlap ? 0 : options.n_roots);
        backend.combine(backend.view(roots),
                        backend.cview(basis).columns(0, active_dimension),
                        root_coefficients);
        backend.combine(backend.view(hamiltonian_roots),
                        backend.cview(hamiltonian_basis).columns(0, active_dimension),
                        root_coefficients);
        if (!options.identity_overlap) {
            backend.combine(backend.view(overlap_roots),
                            backend.cview(overlap_basis).columns(0, active_dimension),
                            root_coefficients);
        }
        const auto s_roots =
            options.identity_overlap ? backend.cview(roots) : backend.cview(overlap_roots);
        const double selected_subspace_gap =
            projected_eigenvalues.size() > options.n_roots
                ? std::max(projected_eigenvalues[options.n_roots] -
                               projected_eigenvalues[options.n_roots - 1],
                           std::numeric_limits<double>::epsilon())
                : std::numeric_limits<double>::epsilon();

        result.relative_residual_norms.assign(options.n_roots, 0.0);
        result.stationarity_residual_norms.assign(options.n_roots, 0.0);
        bool converged = true;
        auto corrections = backend.make(dimension, options.n_roots);
        auto residual = backend.make(dimension, 1);
        int n_corrections = 0;
        for (int root = 0; root < options.n_roots; ++root) {
            const double eigenvalue = projected_eigenvalues[root];
            backend.copy(backend.view(residual), backend.cview(hamiltonian_roots).column(root));
            backend.axpy(backend.view(residual), -eigenvalue, s_roots.column(root));
            const double residual_norm = backend.norm(backend.cview(residual));
            const double residual_scale =
                std::max(1.0,
                         backend.norm(backend.cview(hamiltonian_roots).column(root)) +
                             std::abs(eigenvalue) * backend.norm(s_roots.column(root)));
            const double relative_residual = residual_norm / residual_scale;
            result.relative_residual_norms[root] = relative_residual;
            const double stationarity_residual =
                residual_norm / backend.norm(backend.cview(roots).column(root));
            result.stationarity_residual_norms[root] = stationarity_residual;
            const double energy_scale = std::max(1.0, std::abs(eigenvalue));
            const double first_iteration_residual_tolerance =
                options.energy_tolerance / energy_scale;
            const double spectral_energy_residual_tolerance = std::sqrt(
                (options.energy_tolerance / energy_scale) * (selected_subspace_gap / energy_scale));
            const bool ritz_energy_stable =
                previous_eigenvalues.size() == options.n_roots &&
                std::abs(eigenvalue - previous_eigenvalues[root]) <= options.energy_tolerance;
            const bool independently_certified =
                relative_residual <= first_iteration_residual_tolerance;
            const double required_residual =
                std::min(options.residual_tolerance, spectral_energy_residual_tolerance);
            bool root_converged = relative_residual <= required_residual &&
                                  stationarity_residual <= options.stationarity_tolerance &&
                                  (ritz_energy_stable || independently_certified);
            if (root_converged && options.physical_residual_norm) {
                if constexpr (Backend::supports_host_callbacks) {
                    const Eigen::VectorXd root_vector =
                        backend.download(backend.cview(roots).column(root));
                    const double physical = options.physical_residual_norm(root_vector, eigenvalue);
                    if (!std::isfinite(physical) || physical < 0.0) {
                        throw std::runtime_error("Davidson physical residual norm is invalid");
                    }
                    result.stationarity_residual_norms[root] = physical;
                    root_converged = physical <= options.stationarity_tolerance;
                }
            }
            if (root_converged)
                continue;
            converged = false;
            auto correction = backend.view(corrections).column(n_corrections);
            if (options.residual_preconditioner) {
                if constexpr (Backend::supports_host_callbacks) {
                    const Eigen::VectorXd host_residual = backend.download(backend.cview(residual));
                    Eigen::VectorXd host_correction =
                        options.residual_preconditioner(host_residual, eigenvalue);
                    if (host_correction.size() != dimension || !host_correction.allFinite()) {
                        throw std::runtime_error(
                            "Davidson residual preconditioner returned an invalid vector");
                    }
                    host_correction = -host_correction;
                    backend.upload_into(correction, host_correction);
                }
            } else {
                backend.precondition(correction,
                                     backend.cview(residual),
                                     hamiltonian_diagonal,
                                     overlap_diagonal,
                                     eigenvalue);
            }
            ++n_corrections;
        }

        if (options.observe_iteration) {
            options.observe_iteration(iteration,
                                      active_dimension,
                                      projected_eigenvalues[0],
                                      *std::max_element(result.relative_residual_norms.begin(),
                                                        result.relative_residual_norms.end()),
                                      *std::max_element(result.stationarity_residual_norms.begin(),
                                                        result.stationarity_residual_norms.end()));
        }
        if (converged) {
            result.eigenpairs.eigenvalues.assign(projected_eigenvalues.data(),
                                                 projected_eigenvalues.data() + options.n_roots);
            if constexpr (std::is_same_v<Result, DavidsonResult>) {
                const Eigen::MatrixXd host_roots = backend.download(backend.cview(roots));
                result.eigenpairs.eigenvector_matrix.assign(host_roots.data(),
                                                            host_roots.data() + host_roots.size());
                result.hamiltonian_eigenvectors =
                    backend.download(backend.cview(hamiltonian_roots));
                result.overlap_eigenvectors = backend.download(s_roots);
            } else {
                backend.finish_device_result(result,
                                             std::move(roots),
                                             std::move(hamiltonian_roots),
                                             std::move(overlap_roots),
                                             options.identity_overlap);
            }
            return result;
        }
        previous_eigenvalues = projected_eigenvalues.head(options.n_roots);

        if (active_dimension == dimension) {
            if (options.physical_residual_norm) {
                if constexpr (Backend::supports_host_callbacks) {
                    for (int root = 0; root < options.n_roots; ++root) {
                        const Eigen::VectorXd root_vector =
                            backend.download(backend.cview(roots).column(root));
                        const double physical = options.physical_residual_norm(
                            root_vector, projected_eigenvalues[root]);
                        if (!std::isfinite(physical) || physical < 0.0) {
                            throw std::runtime_error("Davidson physical residual norm is invalid");
                        }
                        result.stationarity_residual_norms[root] = physical;
                    }
                }
            }
            std::ostringstream message;
            message << "complete Davidson basis cannot meet the requested Ritz residual: "
                    << *std::max_element(result.relative_residual_norms.begin(),
                                         result.relative_residual_norms.end())
                    << (options.physical_residual_norm ? "; physical CI stationarity residual = "
                                                       : "; CI stationarity residual = ")
                    << *std::max_element(result.stationarity_residual_norms.begin(),
                                         result.stationarity_residual_norms.end())
                    << " (requested " << options.stationarity_tolerance << ')';
            double coefficient_norm = 0.0;
            double overlap_norm = 0.0;
            for (int root = 0; root < options.n_roots; ++root) {
                coefficient_norm =
                    std::max(coefficient_norm, backend.norm(backend.cview(roots).column(root)));
                overlap_norm = std::max(overlap_norm, backend.norm(s_roots.column(root)));
            }
            message << "; coefficient norm = " << coefficient_norm
                    << "; overlap-image norm = " << overlap_norm;
            throw std::runtime_error(message.str());
        }

        if (active_dimension + n_corrections > max_subspace) {
            const int n_keep =
                std::min(active_dimension, std::max(options.n_roots, max_subspace / 2));
            const Eigen::MatrixXd keep_coefficients = projected_eigenvectors.leftCols(n_keep);
            auto restarted_basis = backend.make(dimension, n_keep);
            auto restarted_hamiltonian_basis = backend.make(dimension, n_keep);
            auto restarted_overlap_basis =
                backend.make(dimension, options.identity_overlap ? 0 : n_keep);
            backend.combine(backend.view(restarted_basis),
                            backend.cview(basis).columns(0, active_dimension),
                            keep_coefficients);
            backend.combine(backend.view(restarted_hamiltonian_basis),
                            backend.cview(hamiltonian_basis).columns(0, active_dimension),
                            keep_coefficients);
            if (!options.identity_overlap) {
                backend.combine(backend.view(restarted_overlap_basis),
                                backend.cview(overlap_basis).columns(0, active_dimension),
                                keep_coefficients);
            }
            backend.copy(backend.view(basis).columns(0, n_keep), backend.cview(restarted_basis));
            backend.copy(backend.view(hamiltonian_basis).columns(0, n_keep),
                         backend.cview(restarted_hamiltonian_basis));
            if (!options.identity_overlap) {
                backend.copy(backend.view(overlap_basis).columns(0, n_keep),
                             backend.cview(restarted_overlap_basis));
            }
            projected_hamiltonian.topLeftCorner(n_keep, n_keep) =
                projected_eigenvalues.head(n_keep).asDiagonal();
            active_dimension = n_keep;
            // Restart forms new columns. Refresh only their scalar norm metadata;
            // ordinary extensions reuse norms already measured before normalization.
            for (int column = 0; column < n_keep; ++column) {
                basis_norms[column] = backend.norm(backend.cview(basis).column(column));
            }
        }

        const auto old_basis = backend.cview(basis).columns(0, active_dimension);
        const auto projection_info = project_davidson_corrections(
            backend,
            backend.view(corrections).columns(0, n_corrections),
            old_basis,
            options.identity_overlap ? old_basis
                                     : backend.cview(overlap_basis).columns(0, active_dimension),
            basis_norms);
        auto correction_images = backend.make(dimension, n_corrections);
        auto overlap_correction_images =
            backend.make(dimension, options.identity_overlap ? 0 : n_corrections);
        backend.apply(backend.cview(corrections).columns(0, n_corrections),
                      backend.view(correction_images),
                      backend.view(overlap_correction_images));
        validate_images(backend,
                        backend.cview(correction_images),
                        backend.cview(overlap_correction_images),
                        dimension,
                        n_corrections,
                        options.identity_overlap);
        ++result.block_actions;

        const int previous_dimension = active_dimension;
        DavidsonRankInfo rank_info;
        active_dimension =
            append_s_orthonormal_block(backend,
                                       backend.cview(corrections).columns(0, n_corrections),
                                       backend.cview(correction_images),
                                       backend.cview(overlap_correction_images),
                                       backend.view(basis),
                                       backend.view(hamiltonian_basis),
                                       backend.view(overlap_basis),
                                       active_dimension,
                                       options.identity_overlap,
                                       basis_norms,
                                       &rank_info);
        update_projected_hamiltonian(backend,
                                     backend.cview(basis),
                                     backend.cview(hamiltonian_basis),
                                     previous_dimension,
                                     active_dimension,
                                     projected_hamiltonian);
        if (active_dimension == previous_dimension) {
            std::ostringstream message;
            message.precision(17);
            message << "Davidson correction space became linearly dependent before convergence"
                    << "; iteration=" << iteration << "; subspace=" << active_dimension
                    << "; corrections=" << n_corrections << "; max relative residual="
                    << *std::max_element(result.relative_residual_norms.begin(),
                                         result.relative_residual_norms.end())
                    << " (requested " << options.residual_tolerance << ')'
                    << "; max stationarity residual="
                    << *std::max_element(result.stationarity_residual_norms.begin(),
                                         result.stationarity_residual_norms.end())
                    << " (requested " << options.stationarity_tolerance << ')'
                    << "; last correction initial scale=" << rank_info.initial_scale
                    << "; projected metric norm squared=" << rank_info.metric_norm_squared
                    << "; projected scale=" << rank_info.final_scale
                    << "; dependence threshold=" << rank_info.threshold
                    << "; raw correction norm=" << projection_info.raw_norm
                    << "; preprojected correction norm=" << projection_info.projected_norm
                    << "; preprojection roundoff bound=" << projection_info.roundoff_bound;
            throw std::runtime_error(message.str());
        }
    }
    std::ostringstream message;
    message << "Davidson did not converge in " << options.max_iterations
            << " iterations; largest relative residual = "
            << *std::max_element(result.relative_residual_norms.begin(),
                                 result.relative_residual_norms.end())
            << "; largest CI stationarity residual = "
            << *std::max_element(result.stationarity_residual_norms.begin(),
                                 result.stationarity_residual_norms.end())
            << " (requested " << options.stationarity_tolerance << ')';
    throw std::runtime_error(message.str());
}

} // namespace xmvb::linalg::detail
