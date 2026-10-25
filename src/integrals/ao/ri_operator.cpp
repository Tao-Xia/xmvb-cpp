#include "integrals/ao/ri_operator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "integrals/jk/contraction.hpp"

namespace xmvb::integrals {

namespace {

std::size_t packed_pair_count(int n_basis_functions) {
    const std::size_t n = n_basis_functions;
    return n * (n + 1) / 2;
}

struct SpectralFactorization {
    Eigen::MatrixXd scaled_eigenvectors;
    int n_positive_components = 0;
    int n_negative_components = 0;
    int effective_rank = 0;
    bool use_low_rank_path = false;
};

void unpack_packed_factor_row_lower_triangle(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    int n_basis_functions,
    Eigen::MatrixXd* factor_matrix) {
    if (factor_matrix == nullptr) {
        throw std::invalid_argument("RI factor-row unpack received null matrix output");
    }

    factor_matrix->resize(n_basis_functions, n_basis_functions);
    std::size_t packed_index = 0;
    for (int column = 0; column < n_basis_functions; ++column) {
        for (int row = 0; row <= column; ++row) {
            const double value =
                packed_factor_matrix(auxiliary_index, static_cast<Eigen::Index>(packed_index++));
            (*factor_matrix)(column, row) = value;
        }
    }
}

void add_scaled_packed_factor_row_to_lower_triangle(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    double scale,
    double* output_storage,
    int n_basis_functions) {
    if (output_storage == nullptr) {
        throw std::invalid_argument("packed-row lower-triangle accumulation received null storage");
    }
    std::size_t packed_index = 0;
    for (int column = 0; column < n_basis_functions; ++column) {
        const std::size_t column_offset = column * n_basis_functions;
        for (int row = 0; row <= column; ++row) {
            output_storage[column_offset + row] +=
                scale *
                packed_factor_matrix(auxiliary_index, static_cast<Eigen::Index>(packed_index++));
        }
    }
}

void subtract_lower_triangle_in_place(const Eigen::MatrixXd& exchange_matrix,
                                      double* output_storage,
                                      int n_basis_functions) {
    if (output_storage == nullptr) {
        throw std::invalid_argument("output_storage must not be null");
    }
    for (int column = 0; column < n_basis_functions; ++column) {
        const std::size_t column_offset = column * n_basis_functions;
        for (int row = 0; row <= column; ++row) {
            output_storage[column_offset + row] -= exchange_matrix(column, row);
        }
    }
}

void add_scaled_lower_triangle_in_place(const Eigen::MatrixXd& exchange_matrix,
                                        double scale,
                                        double* output_storage,
                                        int n_basis_functions) {
    for (int column = 0; column < n_basis_functions; ++column) {
        const std::size_t offset = static_cast<std::size_t>(column) * n_basis_functions;
        for (int row = 0; row <= column; ++row)
            output_storage[offset + row] += scale * exchange_matrix(column, row);
    }
}

double
packed_row_symmetric_matrix_dot(const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
                                int auxiliary_index,
                                const std::vector<double>& weighted_packed_matrix) {
    if (weighted_packed_matrix.empty()) {
        return 0.0;
    }

    double result = 0.0;
    for (std::size_t packed_index = 0; packed_index < weighted_packed_matrix.size();
         ++packed_index) {
        result += packed_factor_matrix(auxiliary_index, static_cast<Eigen::Index>(packed_index)) *
                  weighted_packed_matrix[packed_index];
    }
    return result;
}

void accumulate_dense_ri_factor_action(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    const std::vector<double>& weighted_packed_input,
    const Eigen::MatrixXd& factor_matrix,
    Eigen::MatrixXd* left_workspace,
    Eigen::MatrixXd* exchange_matrix,
    double* local_output,
    double coulomb_weight,
    double exchange_weight) {
    if (local_output == nullptr ||
        (exchange_weight != 0.0 && (left_workspace == nullptr || exchange_matrix == nullptr))) {
        throw std::invalid_argument("dense RI factor action received null workspace");
    }
    const int n_basis_functions = static_cast<int>(input_matrix.rows());
    if (coulomb_weight != 0.0) {
        const double coulomb_projection = packed_row_symmetric_matrix_dot(
            packed_factor_matrix, auxiliary_index, weighted_packed_input);
        add_scaled_packed_factor_row_to_lower_triangle(packed_factor_matrix,
                                                       auxiliary_index,
                                                       coulomb_weight * coulomb_projection,
                                                       local_output,
                                                       n_basis_functions);
    }
    if (exchange_weight == 0.0)
        return;

    cblas_dsymm(CblasColMajor,
                CblasLeft,
                CblasLower,
                n_basis_functions,
                n_basis_functions,
                1.0,
                factor_matrix.data(),
                n_basis_functions,
                input_matrix.data(),
                n_basis_functions,
                0.0,
                left_workspace->data(),
                n_basis_functions);
    cblas_dsymm(CblasColMajor,
                CblasRight,
                CblasLower,
                n_basis_functions,
                n_basis_functions,
                1.0,
                factor_matrix.data(),
                n_basis_functions,
                left_workspace->data(),
                n_basis_functions,
                0.0,
                exchange_matrix->data(),
                n_basis_functions);
    if (exchange_weight == -1.0)
        subtract_lower_triangle_in_place(*exchange_matrix, local_output, n_basis_functions);
    else
        add_scaled_lower_triangle_in_place(
            *exchange_matrix, exchange_weight, local_output, n_basis_functions);
}

void build_weighted_packed_symmetric_matrix(const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
                                            std::vector<double>* weighted_packed_matrix) {
    if (weighted_packed_matrix == nullptr) {
        throw std::invalid_argument("weighted packed matrix output must not be null");
    }
    const int n_basis_functions = static_cast<int>(input_matrix.rows());
    weighted_packed_matrix->resize(packed_pair_count(n_basis_functions));
    std::size_t packed_index = 0;
    for (int column = 0; column < n_basis_functions; ++column) {
        for (int row = 0; row <= column; ++row) {
            if (row == column) {
                (*weighted_packed_matrix)[packed_index++] = input_matrix(row, column);
            } else {
                (*weighted_packed_matrix)[packed_index++] =
                    input_matrix(row, column) + input_matrix(column, row);
            }
        }
    }
}

SpectralFactorization
build_spectral_factorization(const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
                             const AoEffectiveOneElectronRiOperatorOptions& options) {
    SpectralFactorization factorization;
    if (!options.attempt_spectral_factorization) {
        return factorization;
    }
    if (!(options.spectral_eigenvalue_cutoff >= 0.0)) {
        throw std::invalid_argument("spectral_eigenvalue_cutoff must be non-negative");
    }
    if (!(options.low_rank_fraction_cutoff > 0.0 && options.low_rank_fraction_cutoff <= 1.0)) {
        throw std::invalid_argument("low_rank_fraction_cutoff must lie in (0, 1]");
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(input_matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("failed to diagonalize RI AO-H1E input matrix");
    }

    const auto eigenvalues = solver.eigenvalues();
    const double max_abs_eigenvalue =
        eigenvalues.cwiseAbs().size() > 0 ? eigenvalues.cwiseAbs().maxCoeff() : 0.0;
    const double effective_cutoff =
        std::max(options.spectral_eigenvalue_cutoff,
                 std::numeric_limits<double>::epsilon() * static_cast<double>(input_matrix.rows()) *
                     max_abs_eigenvalue);

    int n_positive_components = 0;
    int n_negative_components = 0;
    for (int index = 0; index < eigenvalues.size(); ++index) {
        const double eigenvalue = eigenvalues(index);
        if (eigenvalue > effective_cutoff) {
            ++n_positive_components;
        } else if (eigenvalue < -effective_cutoff) {
            ++n_negative_components;
        }
    }
    const int effective_rank = n_positive_components + n_negative_components;
    factorization.n_positive_components = n_positive_components;
    factorization.n_negative_components = n_negative_components;
    factorization.effective_rank = effective_rank;
    if (effective_rank == 0) {
        factorization.use_low_rank_path = true;
        return factorization;
    }
    if (static_cast<double>(effective_rank) >
        options.low_rank_fraction_cutoff * static_cast<double>(input_matrix.rows())) {
        return factorization;
    }

    factorization.scaled_eigenvectors.resize(input_matrix.rows(), effective_rank);
    factorization.use_low_rank_path = true;

    int positive_column = 0;
    int negative_column = n_positive_components;
    for (int index = 0; index < eigenvalues.size(); ++index) {
        const double eigenvalue = eigenvalues(index);
        if (eigenvalue > effective_cutoff) {
            factorization.scaled_eigenvectors.col(positive_column++) =
                solver.eigenvectors().col(index) * std::sqrt(eigenvalue);
        } else if (eigenvalue < -effective_cutoff) {
            factorization.scaled_eigenvectors.col(negative_column++) =
                solver.eigenvectors().col(index) * std::sqrt(-eigenvalue);
        }
    }
    return factorization;
}

SpectralFactorization build_prefactorized_spectral_factorization(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors, int n_basis_functions) {
    SpectralFactorization factorization;
    if (low_rank_factors.scaled_factor_matrix.rows() != n_basis_functions) {
        throw std::invalid_argument("prefactorized RI operator row count mismatch");
    }
    if (low_rank_factors.n_positive_components < 0 || low_rank_factors.n_negative_components < 0) {
        throw std::invalid_argument(
            "prefactorized RI operator component counts must be non-negative");
    }

    const int effective_rank =
        low_rank_factors.n_positive_components + low_rank_factors.n_negative_components;
    if (low_rank_factors.scaled_factor_matrix.cols() != effective_rank) {
        throw std::invalid_argument("prefactorized RI operator factor column count mismatch");
    }

    factorization.n_positive_components = low_rank_factors.n_positive_components;
    factorization.n_negative_components = low_rank_factors.n_negative_components;
    factorization.effective_rank = effective_rank;
    factorization.use_low_rank_path = true;
    if (effective_rank == 0) {
        factorization.scaled_eigenvectors.resize(n_basis_functions, 0);
        return factorization;
    }

    factorization.scaled_eigenvectors = low_rank_factors.scaled_factor_matrix;
    return factorization;
}

int ri_auxiliary_thread_count(int n_auxiliary_functions, int n_bf, int matrices_per_worker) {
    // RI application partitions by auxiliary factor. The reduction buffers are
    // full AO matrices, so launching more workers than factors, or launching an
    // inner team under an existing exact_ctx parallel region, only increases
    // memory traffic without adding useful work. The aggregate worker budget
    // accounts for outputs AND scratch, not merely the reduction matrices.
    // One worker is indispensable even if its minimum storage exceeds budget.
    const std::size_t workspace_bytes = workspace::available();
    const std::size_t worker_elements = static_cast<std::size_t>(n_bf) * n_bf * matrices_per_worker;
    int n_threads = static_cast<int>(std::min<std::size_t>(
        xmvb::effective_openmp_thread_count(),
        std::max<std::size_t>(1, workspace_bytes / sizeof(double) / worker_elements)));
    if (n_auxiliary_functions <= 1) {
        return 1;
    }
    if (n_threads > n_auxiliary_functions) {
        n_threads = n_auxiliary_functions;
    }
    return std::max(1, n_threads);
}

void symmetrize_in_place(std::vector<double>* matrix_storage, int n_basis_functions) {
    if (matrix_storage == nullptr) {
        throw std::invalid_argument("matrix_storage must not be null");
    }
    Eigen::Map<Eigen::MatrixXd> matrix(
        matrix_storage->data(), n_basis_functions, n_basis_functions);
    for (int column = 0; column < n_basis_functions; ++column) {
        for (int row = 0; row < column; ++row) {
            // The RI sweep accumulates complementary contributions into opposite
            // triangles: Coulomb updates are written in packed-pair order while the
            // exchange kernels update a selfadjoint view. Finalization therefore
            // must merge both triangles, mirroring the exact AO path, rather than
            // averaging them away.
            const double symmetrized_sum = matrix(column, row) + matrix(row, column);
            matrix(column, row) = symmetrized_sum;
            matrix(row, column) = symmetrized_sum;
        }
    }
}

void symmetrize_in_place(Eigen::MatrixXd* matrix) {
    if (matrix == nullptr || matrix->rows() != matrix->cols()) {
        throw std::invalid_argument("invalid RI matrix symmetrization target");
    }
    for (int column = 0; column < matrix->cols(); ++column) {
        for (int row = 0; row < column; ++row) {
            const double symmetrized_sum = (*matrix)(column, row) + (*matrix)(row, column);
            (*matrix)(column, row) = symmetrized_sum;
            (*matrix)(row, column) = symmetrized_sum;
        }
    }
}

void validate_ri_factorization(const RiAoFactorization& ri_factorization, int n_basis_functions) {
    if (n_basis_functions <= 0) {
        throw std::invalid_argument("n_basis_functions must be positive");
    }
    if (ri_factorization.n_basis_functions != n_basis_functions) {
        throw std::invalid_argument("RI basis-function count mismatch");
    }
    if (ri_factorization.n_auxiliary_functions < 0) {
        throw std::invalid_argument("RI auxiliary-function count must be non-negative");
    }

    const std::size_t n_packed_pairs = packed_pair_count(n_basis_functions);
    if (ri_factorization.n_packed_ao_pairs != 0 &&
        ri_factorization.n_packed_ao_pairs != static_cast<int>(n_packed_pairs)) {
        throw std::invalid_argument("RI packed AO pair count mismatch");
    }
    if (ri_factorization.metric_whitened_ao_pair_factors.rows() !=
            ri_factorization.n_auxiliary_functions ||
        ri_factorization.metric_whitened_ao_pair_factors.cols() !=
            static_cast<Eigen::Index>(n_packed_pairs)) {
        throw std::invalid_argument("RI AO pair-factor matrix shape mismatch");
    }
}

void resize_fused_workspace(AoEffectiveOneElectronRiFusedWorkspace* workspace,
                            int n_threads,
                            int n_basis_functions) {
    if (workspace == nullptr) {
        throw std::invalid_argument("RI fused workspace must not be null");
    }
    workspace->partial_forward.resize(n_threads);
    workspace->partial_adjoint.resize(n_threads);
    workspace->factor_matrices.resize(n_threads);
    workspace->left_products.resize(n_threads);
    workspace->exchange_products.resize(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        workspace->partial_forward[thread].setZero(n_basis_functions, n_basis_functions);
        workspace->partial_adjoint[thread].setZero(n_basis_functions, n_basis_functions);
        workspace->factor_matrices[thread].resize(n_basis_functions, n_basis_functions);
        workspace->left_products[thread].resize(n_basis_functions, n_basis_functions);
        workspace->exchange_products[thread].resize(n_basis_functions, n_basis_functions);
    }
}

void apply_fused_dense_ri_operator(const Eigen::Ref<const Eigen::MatrixXd>& source,
                                   const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
                                   const RiAoFactorization& ri_factorization,
                                   AoEffectiveOneElectronRiFusedWorkspace* workspace,
                                   Eigen::MatrixXd* forward,
                                   Eigen::MatrixXd* transpose) {
    const int n_basis_functions = ri_factorization.n_basis_functions;
    const int n_threads =
        ri_auxiliary_thread_count(ri_factorization.n_auxiliary_functions, n_basis_functions, 5);
    resize_fused_workspace(workspace, n_threads, n_basis_functions);
    build_weighted_packed_symmetric_matrix(source, &workspace->weighted_packed_forward);
    build_weighted_packed_symmetric_matrix(adjoint, &workspace->weighted_packed_adjoint);

    const auto& packed_factor_matrix = ri_factorization.metric_whitened_ao_pair_factors;
#pragma omp parallel num_threads(n_threads)
    {
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        Eigen::MatrixXd& factor_matrix = workspace->factor_matrices[thread_index];
        Eigen::MatrixXd& left_product = workspace->left_products[thread_index];
        Eigen::MatrixXd& exchange_product = workspace->exchange_products[thread_index];
        Eigen::MatrixXd& local_forward = workspace->partial_forward[thread_index];
        Eigen::MatrixXd& local_adjoint = workspace->partial_adjoint[thread_index];

#pragma omp for schedule(static)
        for (std::ptrdiff_t auxiliary_offset = 0;
             auxiliary_offset < ri_factorization.n_auxiliary_functions;
             ++auxiliary_offset) {
            const int auxiliary_index = static_cast<int>(auxiliary_offset);
            unpack_packed_factor_row_lower_triangle(
                packed_factor_matrix, auxiliary_index, n_basis_functions, &factor_matrix);
            accumulate_dense_ri_factor_action(packed_factor_matrix,
                                              auxiliary_index,
                                              source,
                                              workspace->weighted_packed_forward,
                                              factor_matrix,
                                              &left_product,
                                              &exchange_product,
                                              local_forward.data(),
                                              2.0,
                                              -1.0);
            accumulate_dense_ri_factor_action(packed_factor_matrix,
                                              auxiliary_index,
                                              adjoint,
                                              workspace->weighted_packed_adjoint,
                                              factor_matrix,
                                              &left_product,
                                              &exchange_product,
                                              local_adjoint.data(),
                                              2.0,
                                              -1.0);
        }
    }

    forward->setZero(n_basis_functions, n_basis_functions);
    transpose->setZero(n_basis_functions, n_basis_functions);
    for (int thread = 0; thread < n_threads; ++thread) {
        *forward += workspace->partial_forward[thread];
        *transpose += workspace->partial_adjoint[thread];
    }
    symmetrize_in_place(forward);
    symmetrize_in_place(transpose);
}

std::vector<double> apply_low_rank_ri_operator(const RiAoFactorization& ri_factorization,
                                               int n_basis_functions,
                                               const SpectralFactorization& factorization,
                                               double coulomb_weight,
                                               double exchange_weight) {
    const std::size_t matrix_size = n_basis_functions * n_basis_functions;
    std::vector<double> output_storage(matrix_size, 0.0);
    if (factorization.scaled_eigenvectors.cols() == 0) {
        return output_storage;
    }

    const auto& packed_factor_matrix = ri_factorization.metric_whitened_ao_pair_factors;
    const int n_auxiliary = ri_factorization.n_auxiliary_functions;
    const int rank = factorization.effective_rank;

    // The auxiliary-major Z tile is shared. Workers own AO row slices and
    // disjoint output panels, never full exchange output replicas.
    const std::size_t workspace_bytes = workspace::available();
    const int n_threads = std::min(xmvb::effective_openmp_thread_count(), n_basis_functions);
    // At least one near-square lower-triangle block per worker; the number
    // follows p(p+1)/2 >= workers, not a molecule-specific block dimension.
    const int n_panels =
        std::min(n_basis_functions,
                 static_cast<int>(std::ceil((std::sqrt(1.0 + 8.0 * n_threads) - 1.0) / 2.0)));
    std::vector<std::pair<int, int>> output_blocks;
    for (int row = 0; row < n_panels; ++row) {
        for (int column = 0; column <= row; ++column) {
            output_blocks.emplace_back(row, column);
        }
    }
    const std::size_t per_auxiliary_elements =
        static_cast<std::size_t>(n_basis_functions) * (rank + n_threads) + n_threads;
    const int tile_width = static_cast<int>(std::min<std::size_t>(
        std::max(1, n_auxiliary),
        std::max<std::size_t>(1, workspace_bytes / sizeof(double) / per_auxiliary_elements)));
    Eigen::VectorXd auxiliary_projection;
    if (coulomb_weight != 0.0)
        auxiliary_projection = Eigen::VectorXd::Zero(n_auxiliary);
    Eigen::Map<Eigen::MatrixXd> output(output_storage.data(), n_basis_functions, n_basis_functions);
    for (int first_auxiliary = 0; first_auxiliary < n_auxiliary; first_auxiliary += tile_width) {
        const int count = std::min(tile_width, n_auxiliary - first_auxiliary);
        Eigen::MatrixXd transformed(static_cast<Eigen::Index>(count) * rank, n_basis_functions);
        std::vector<Eigen::VectorXd> partial_projections(n_threads);
        if (coulomb_weight != 0.0)
            for (auto& projection : partial_projections)
                projection.setZero(count);
#pragma omp parallel num_threads(n_threads)
        {
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            Eigen::MatrixXd ao_slice(count, n_basis_functions);
#pragma omp for schedule(static)
            for (int mu = 0; mu < n_basis_functions; ++mu) {
                for (int nu = 0; nu < n_basis_functions; ++nu) {
                    const int larger = std::max(mu, nu);
                    const int smaller = std::min(mu, nu);
                    const Eigen::Index pair =
                        static_cast<Eigen::Index>(larger) * (larger + 1) / 2 + smaller;
                    ao_slice.col(nu) =
                        packed_factor_matrix.col(pair).segment(first_auxiliary, count);
                }
                Eigen::Map<Eigen::MatrixXd> image(transformed.col(mu).data(), count, rank);
                image.noalias() = ao_slice * factorization.scaled_eigenvectors;
                // Consume each just-transformed AO slice before it leaves cache.
                if (coulomb_weight != 0.0)
                    for (int k = 0; k < rank; ++k) {
                        const double sign = k < factorization.n_positive_components ? 1.0 : -1.0;
                        partial_projections[thread].noalias() +=
                            (sign * factorization.scaled_eigenvectors(mu, k)) * image.col(k);
                    }
            }
#pragma omp single
            {
                if (coulomb_weight != 0.0)
                    for (const auto& projection : partial_projections)
                        auxiliary_projection.segment(first_auxiliary, count) += projection;
            }

            // Blocks own disjoint output rectangles. Near-square GEMMs avoid
            // extremely narrow full-height panels at high worker counts.
#pragma omp for schedule(dynamic, 1)
            for (std::size_t block = 0; block < (exchange_weight != 0.0 ? output_blocks.size() : 0);
                 ++block) {
                const auto [row_panel, column_panel] = output_blocks[block];
                const int row_begin = row_panel * n_basis_functions / n_panels;
                const int row_end = (row_panel + 1) * n_basis_functions / n_panels;
                const int begin = column_panel * n_basis_functions / n_panels;
                const int end = (column_panel + 1) * n_basis_functions / n_panels;
                const int width = end - begin;
                const int leading = static_cast<int>(transformed.rows());
                for (int sign_channel = 0; sign_channel < 2; ++sign_channel) {
                    const int components = sign_channel == 0 ? factorization.n_positive_components
                                                             : factorization.n_negative_components;
                    const int rows = count * components;
                    if (rows == 0)
                        continue;
                    const int offset =
                        sign_channel == 0 ? 0 : count * factorization.n_positive_components;
                    const double scale = sign_channel == 0 ? exchange_weight : -exchange_weight;
                    const double* panel =
                        transformed.data() + static_cast<Eigen::Index>(begin) * leading + offset;
                    if (row_panel == column_panel) {
                        // Diagonal panel: SYRK writes only its lower triangle.
                        cblas_dsyrk(CblasColMajor,
                                    CblasLower,
                                    CblasTrans,
                                    width,
                                    rows,
                                    scale,
                                    panel,
                                    leading,
                                    1.0,
                                    output.data() +
                                        static_cast<Eigen::Index>(begin) * n_basis_functions +
                                        begin,
                                    n_basis_functions);
                    } else {
                        // Strictly lower rectangle: every entry is owned by this block.
                        cblas_dgemm(CblasColMajor,
                                    CblasTrans,
                                    CblasNoTrans,
                                    row_end - row_begin,
                                    width,
                                    rows,
                                    scale,
                                    transformed.data() +
                                        static_cast<Eigen::Index>(row_begin) * leading + offset,
                                    leading,
                                    panel,
                                    leading,
                                    1.0,
                                    output.data() +
                                        static_cast<Eigen::Index>(begin) * n_basis_functions +
                                        row_begin,
                                    n_basis_functions);
                    }
                }
            }
        }
    }
    if (coulomb_weight != 0.0) {
        const Eigen::VectorXd packed_coulomb =
            coulomb_weight * packed_factor_matrix.transpose() * auxiliary_projection;
        Eigen::Index packed_index = 0;
        for (int column = 0; column < n_basis_functions; ++column) {
            for (int row = 0; row <= column; ++row) {
                output(row, column) += packed_coulomb(packed_index++);
            }
        }
    }
    symmetrize_in_place(&output_storage, n_basis_functions);
    return output_storage;
}

std::vector<double> apply_dense_ri_operator(const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
                                            const RiAoFactorization& ri_factorization,
                                            int n_basis_functions,
                                            double coulomb_weight,
                                            double exchange_weight) {
    const std::size_t matrix_size = n_basis_functions * n_basis_functions;
    std::vector<double> output_storage(matrix_size, 0.0);
    std::vector<double> weighted_packed_input;
    if (coulomb_weight != 0.0)
        build_weighted_packed_symmetric_matrix(input_matrix, &weighted_packed_input);
    const auto& packed_factor_matrix = ri_factorization.metric_whitened_ao_pair_factors;

    if (exchange_weight == 0.0) {
        if (coulomb_weight == 0.0)
            return output_storage;
        const Eigen::Map<const Eigen::VectorXd> weighted_input(
            weighted_packed_input.data(), static_cast<Eigen::Index>(weighted_packed_input.size()));
        const Eigen::VectorXd auxiliary_projection = packed_factor_matrix * weighted_input;
        const Eigen::VectorXd packed_coulomb =
            coulomb_weight * (packed_factor_matrix.transpose() * auxiliary_projection);
        Eigen::Map<Eigen::MatrixXd> output(
            output_storage.data(), n_basis_functions, n_basis_functions);
        Eigen::Index packed_index = 0;
        for (int column = 0; column < n_basis_functions; ++column)
            for (int row = 0; row <= column; ++row) {
                const double value = packed_coulomb(packed_index++);
                output(row, column) = value;
                output(column, row) = value;
            }
        return output_storage;
    }

    int n_threads = 1;
    n_threads = ri_auxiliary_thread_count(
        ri_factorization.n_auxiliary_functions, n_basis_functions, exchange_weight != 0.0 ? 4 : 1);
    std::vector<std::vector<double>> partial_outputs(n_threads,
                                                     std::vector<double>(matrix_size, 0.0));

    // The dense RI contraction has the same reduction shape as the low-rank path:
    // local AO matrices avoid write conflicts, while the capped team width keeps
    // the memory footprint proportional to useful auxiliary work.
#pragma omp parallel num_threads(n_threads)
    {
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        Eigen::Map<Eigen::MatrixXd> local_output(
            partial_outputs[thread_index].data(), n_basis_functions, n_basis_functions);
        Eigen::MatrixXd factor_matrix, left_workspace, exchange_matrix;
        if (exchange_weight != 0.0) {
            factor_matrix.resize(n_basis_functions, n_basis_functions);
            left_workspace.resize(n_basis_functions, n_basis_functions);
            exchange_matrix.resize(n_basis_functions, n_basis_functions);
        }

#pragma omp for schedule(static)
        for (std::ptrdiff_t auxiliary_offset = 0;
             auxiliary_offset < ri_factorization.n_auxiliary_functions;
             ++auxiliary_offset) {
            if (exchange_weight != 0.0)
                unpack_packed_factor_row_lower_triangle(packed_factor_matrix,
                                                        static_cast<int>(auxiliary_offset),
                                                        n_basis_functions,
                                                        &factor_matrix);

            // In the dense contraction we expose only the lower triangle of `L_A`
            // to BLAS symmetric kernels, which avoids treating the RI factor as a
            // generic dense matrix and matches the dense-lower cache layout.
            accumulate_dense_ri_factor_action(packed_factor_matrix,
                                              static_cast<int>(auxiliary_offset),
                                              input_matrix,
                                              weighted_packed_input,
                                              factor_matrix,
                                              &left_workspace,
                                              &exchange_matrix,
                                              local_output.data(),
                                              coulomb_weight,
                                              exchange_weight);
        }
    }

    for (const auto& partial_output : partial_outputs) {
        for (std::size_t index = 0; index < output_storage.size(); ++index) {
            output_storage[index] += partial_output[index];
        }
    }
    symmetrize_in_place(&output_storage, n_basis_functions);
    return output_storage;
}

} // namespace

std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const std::vector<double>& input_matrix,
    const RiAoFactorization& ri_factorization,
    int n_basis_functions,
    const AoEffectiveOneElectronRiOperatorOptions& options) {
    profiling::ScopedPhase phase(profiling::Phase::RiAoH1e);
    validate_ri_factorization(ri_factorization, n_basis_functions);

    const std::size_t matrix_size = n_basis_functions * n_basis_functions;
    if (input_matrix.size() != matrix_size) {
        throw std::invalid_argument("input_matrix size mismatch");
    }

    const Eigen::Map<const Eigen::MatrixXd> input(
        input_matrix.data(), n_basis_functions, n_basis_functions);
    const auto spectral_factorization = build_spectral_factorization(input, options);
    if (spectral_factorization.use_low_rank_path) {
        return apply_low_rank_ri_operator(
            ri_factorization, n_basis_functions, spectral_factorization, 2.0, -1.0);
    }
    return apply_dense_ri_operator(input, ri_factorization, n_basis_functions, 2.0, -1.0);
}

std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors,
    const RiAoFactorization& ri_factorization,
    int n_basis_functions) {
    profiling::ScopedPhase phase(profiling::Phase::RiAoH1e);
    validate_ri_factorization(ri_factorization, n_basis_functions);

    const auto spectral_factorization =
        build_prefactorized_spectral_factorization(low_rank_factors, n_basis_functions);
    return apply_low_rank_ri_operator(
        ri_factorization, n_basis_functions, spectral_factorization, 2.0, -1.0);
}

Eigen::MatrixXd contract_jk(const Eigen::Ref<const Eigen::MatrixXd>& density,
                            const RiAoFactorization& ri,
                            JkWeights weights) {
    const int n = ri.n_basis_functions;
    validate_ri_factorization(ri, n);
    if (density.rows() != n || density.cols() != n || !density.allFinite() ||
        !density.isApprox(density.transpose()) || !std::isfinite(weights.coulomb) ||
        !std::isfinite(weights.exchange))
        throw std::invalid_argument("invalid symmetric AO density or JK weights");
    if (weights.coulomb == 0.0 && weights.exchange == 0.0)
        return Eigen::MatrixXd::Zero(n, n);
    const bool hf_weights = weights.coulomb == 1.0 && weights.exchange == -0.5;
    if (hf_weights) {
        // Match the established RHF RI spectral admission and 2J-K action.
        const std::vector<double> input(density.data(),
                                        density.data() + static_cast<Eigen::Index>(n) * n);
        const auto storage = apply_ao_effective_one_electron_ri_operator(
            input, ri, n, {.attempt_spectral_factorization = true});
        Eigen::MatrixXd result = Eigen::Map<const Eigen::MatrixXd>(storage.data(), n, n);
        result *= 0.5;
        return result;
    }
    const auto storage = apply_dense_ri_operator(density, ri, n, weights.coulomb, weights.exchange);
    return Eigen::Map<const Eigen::MatrixXd>(storage.data(), n, n);
}

void apply_ao_effective_one_electron_ri_operator_fused(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
    const int n_basis_functions = ri_factorization.n_basis_functions;
    validate_ri_factorization(ri_factorization, n_basis_functions);
    if (source.rows() != n_basis_functions || source.cols() != n_basis_functions ||
        adjoint.rows() != n_basis_functions || adjoint.cols() != n_basis_functions) {
        throw std::invalid_argument("RI fused input matrix shape mismatch");
    }
    if (workspace == nullptr || forward == nullptr || transpose == nullptr) {
        throw std::invalid_argument("RI fused buffers must not be null");
    }
    if (forward == transpose || forward->data() == source.data() ||
        forward->data() == adjoint.data() || transpose->data() == source.data() ||
        transpose->data() == adjoint.data()) {
        throw std::invalid_argument("RI fused outputs must not alias inputs or each other");
    }

    apply_fused_dense_ri_operator(source, adjoint, ri_factorization, workspace, forward, transpose);
}

} // namespace xmvb::integrals
