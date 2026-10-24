#include "libcint/ri_provider.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include "lapacke.h"

#include "libcint/direct_shell.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::libcint {

namespace {

constexpr int kWhitenColumnBlockSize = 256;

struct ThreeCenterShellTask {
    int primary_left_shell = 0;
    int primary_right_shell = 0;
    int auxiliary_shell = 0;
};

int ao_pair_index(int first, int second) {
    if (first >= second) {
        return first * (first + 1) / 2 + second;
    }
    return second * (second + 1) / 2 + first;
}

std::vector<ThreeCenterShellTask> build_three_center_shell_tasks(int n_primary_shells,
                                                                 int n_auxiliary_shells) {
    std::vector<ThreeCenterShellTask> tasks;
    tasks.reserve(n_auxiliary_shells * n_primary_shells * (n_primary_shells + 1) / 2);
    for (int primary_left_shell = 0; primary_left_shell < n_primary_shells; ++primary_left_shell) {
        for (int primary_right_shell = 0; primary_right_shell <= primary_left_shell;
             ++primary_right_shell) {
            for (int auxiliary_shell = 0; auxiliary_shell < n_auxiliary_shells; ++auxiliary_shell) {
                ThreeCenterShellTask task;
                task.primary_left_shell = primary_left_shell;
                task.primary_right_shell = primary_right_shell;
                task.auxiliary_shell = auxiliary_shell;
                tasks.push_back(task);
            }
        }
    }
    return tasks;
}

void scatter_three_center_shell_block(const LibcintThreeCenterShellBlock& block,
                                      Eigen::MatrixXd* raw_ao_pair_factors) {
    if (raw_ao_pair_factors == nullptr) {
        throw std::invalid_argument("raw_ao_pair_factors must not be null");
    }
    for (int auxiliary_local = 0; auxiliary_local < block.auxiliary_ao_count; ++auxiliary_local) {
        const int auxiliary_index = block.auxiliary_ao_offset + auxiliary_local;
        for (int right_local = 0; right_local < block.primary_right_ao_count; ++right_local) {
            const int right_index = block.primary_right_ao_offset + right_local;
            for (int left_local = 0; left_local < block.primary_left_ao_count; ++left_local) {
                const int left_index = block.primary_left_ao_offset + left_local;
                if (right_index > left_index) {
                    continue;
                }
                const int packed_pair_index = ao_pair_index(left_index, right_index);
                const std::size_t local_index =
                    left_local + right_local * block.primary_left_ao_count +
                    auxiliary_local * block.primary_left_ao_count * block.primary_right_ao_count;
                (*raw_ao_pair_factors)(auxiliary_index, packed_pair_index) =
                    block.values[local_index];
            }
        }
    }
}

} // namespace

xmvb::integrals::RiAoFactorization
build_libcint_ri_factorization(const xmvb::integrals::LibcintInput& primary_input,
                               const xmvb::integrals::LibcintInput& auxiliary_input,
                               double metric_eigenvalue_cutoff) {
    if (!(metric_eigenvalue_cutoff >= 0.0)) {
        throw std::invalid_argument("metric_eigenvalue_cutoff must be non-negative");
    }

    xmvb::integrals::RiAoFactorization result;
    LibcintDirectShellEvaluator evaluator(primary_input, auxiliary_input);
    LibcintShellWorkspace workspace;

    result.n_basis_functions = evaluator.n_basis_functions();
    result.n_auxiliary_functions = evaluator.n_auxiliary_basis_functions();
    result.n_packed_ao_pairs = result.n_basis_functions * (result.n_basis_functions + 1) / 2;
    if (result.n_auxiliary_functions <= 0 || result.n_packed_ao_pairs <= 0) {
        throw std::runtime_error("invalid RI dimensions inferred from libcint inputs");
    }

    Eigen::MatrixXd auxiliary_metric(result.n_auxiliary_functions, result.n_auxiliary_functions);
    auxiliary_metric.setZero();
    for (int left_shell = 0; left_shell < auxiliary_input.n_shells; ++left_shell) {
        for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
            const auto& block =
                evaluator.evaluate_auxiliary_metric_shell_pair(left_shell, right_shell, workspace);
            for (int column = 0; column < block.right_ao_count; ++column) {
                const int global_column = block.right_ao_offset + column;
                for (int row = 0; row < block.left_ao_count; ++row) {
                    const int global_row = block.left_ao_offset + row;
                    const std::size_t local_index = row + column * block.left_ao_count;
                    const double value = block.values[local_index];
                    auxiliary_metric(global_row, global_column) = value;
                    auxiliary_metric(global_column, global_row) = value;
                }
            }
        }
    }
    Eigen::MatrixXd raw_ao_pair_factors(result.n_auxiliary_functions, result.n_packed_ao_pairs);
    raw_ao_pair_factors.setZero();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (n_threads <= 1) {
        for (int left_shell = 0; left_shell < primary_input.n_shells; ++left_shell) {
            for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
                for (int auxiliary_shell = 0; auxiliary_shell < auxiliary_input.n_shells;
                     ++auxiliary_shell) {
                    const auto& block = evaluator.evaluate_three_center_shell_block(
                        left_shell, right_shell, auxiliary_shell, workspace);
                    scatter_three_center_shell_block(block, &raw_ao_pair_factors);
                }
            }
        }
    } else {
        const auto three_center_tasks =
            build_three_center_shell_tasks(primary_input.n_shells, auxiliary_input.n_shells);

#pragma omp parallel
        {
            LibcintShellWorkspace local_workspace;
#pragma omp for schedule(dynamic)
            for (std::ptrdiff_t task_index = 0;
                 task_index < static_cast<std::ptrdiff_t>(three_center_tasks.size());
                 ++task_index) {
                const auto& task = three_center_tasks[task_index];
                const auto& block =
                    evaluator.evaluate_three_center_shell_block(task.primary_left_shell,
                                                                task.primary_right_shell,
                                                                task.auxiliary_shell,
                                                                local_workspace);
                scatter_three_center_shell_block(block, &raw_ao_pair_factors);
            }
        }
    }
    const int n_auxiliary = result.n_auxiliary_functions;
    Eigen::MatrixXd metric_factor = auxiliary_metric;
    const lapack_int cholesky_info =
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n_auxiliary, metric_factor.data(), n_auxiliary);
    if (cholesky_info < 0) {
        throw std::runtime_error("invalid RI auxiliary Cholesky argument");
    }
    Eigen::MatrixXd whitener;
    if (cholesky_info > 0) {
        // Rank-revealing spectral factorization for a non-positive-definite
        // auxiliary metric. Never form a dense symmetric inverse square root:
        // discarded directions must not survive through cancellation errors.
        metric_factor = auxiliary_metric;
        Eigen::VectorXd eigenvalues(n_auxiliary);
        const lapack_int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR,
                                               'V',
                                               'L',
                                               n_auxiliary,
                                               metric_factor.data(),
                                               n_auxiliary,
                                               eigenvalues.data());
        if (info != 0) {
            throw std::runtime_error("failed to diagonalize the RI auxiliary metric");
        }
        const int rank = (eigenvalues.array() > metric_eigenvalue_cutoff).count();
        if (rank == 0) {
            throw std::runtime_error("RI auxiliary metric has no retained directions");
        }
        whitener = metric_factor.rightCols(rank).transpose();
        for (int row = 0; row < rank; ++row) {
            whitener.row(row) /= std::sqrt(eigenvalues[n_auxiliary - rank + row]);
        }
        result.n_auxiliary_functions = rank;
    }
    result.metric_whitened_ao_pair_factors.resize(result.n_auxiliary_functions,
                                                  result.n_packed_ao_pairs);

#pragma omp parallel for schedule(static)
    for (int pair_column_offset = 0; pair_column_offset < result.n_packed_ao_pairs;
         pair_column_offset += kWhitenColumnBlockSize) {
        const int block_column_count =
            std::min(kWhitenColumnBlockSize, result.n_packed_ao_pairs - pair_column_offset);
        auto output = result.metric_whitened_ao_pair_factors.middleCols(pair_column_offset,
                                                                        block_column_count);
        const auto input = raw_ao_pair_factors.middleCols(pair_column_offset, block_column_count);
        if (cholesky_info == 0) {
            output = metric_factor.triangularView<Eigen::Lower>().solve(input);
        } else {
            output.noalias() = whitener * input;
        }
    }
    result.auxiliary_metric_matrix = std::move(auxiliary_metric);
    return result;
}

xmvb::integrals::RiAoFactorization
LibcintRiIntegralProvider::build(const xmvb::integrals::LibcintInput& primary_input,
                                 const xmvb::integrals::LibcintInput& auxiliary_input) const {
    return build(primary_input, auxiliary_input, LibcintRiIntegralProviderOptions{});
}

xmvb::integrals::RiAoFactorization
LibcintRiIntegralProvider::build(const xmvb::integrals::LibcintInput& primary_input,
                                 const xmvb::integrals::LibcintInput& auxiliary_input,
                                 const LibcintRiIntegralProviderOptions& options) const {
    return build_libcint_ri_factorization(
        primary_input, auxiliary_input, options.metric_eigenvalue_cutoff);
}

} // namespace xmvb::libcint
