#include "libcint/ri_provider.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include "lapacke.h"

#include "libcint/direct_shell.hpp"
#include "core/index.hpp"
#include "core/timing.hpp"
#include "integrals/ao/libcint.hpp"
#include "parallel/openmp.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::libcint {

namespace {

constexpr Eigen::Index kWhitenColumnBlockSize = 256;

using Clock = std::chrono::steady_clock;

struct ShellPair {
    int left;
    int right;
    Eigen::Index columns;
};

struct ShellPanel {
    std::size_t first;
    std::size_t end;
    Eigen::Index columns;
};

Eigen::Index ao_pair_index(int first, int second) {
    const Eigen::Index row = first;
    return row * (row + 1) / 2 + second;
}

/** Visit exactly the canonical AO columns of one primary shell pair. */
template<class Consumer>
void visit_ao_columns(int left_offset, int left_count, int right_offset, int right_count,
                      Consumer consume) {
    for (int j = 0; j < right_count; ++j)
        for (int i = std::max(0, right_offset + j - left_offset); i < left_count; ++i)
            consume(i, j, ao_pair_index(left_offset + i, right_offset + j));
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
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
    result.n_packed_ao_pairs = core::checked_index<Eigen::Index>(core::triangular_count(result.n_basis_functions));
    if (result.n_auxiliary_functions <= 0 || result.n_packed_ao_pairs <= 0) {
        throw std::runtime_error("invalid RI dimensions inferred from libcint inputs");
    }

    const bool timing = profiling::phase_totals().enabled;
    const auto metric_start = timing ? Clock::now() : Clock::time_point{};
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
    const double metric_seconds = timing ? elapsed(metric_start) : 0.0;
    const int workers = effective_openmp_thread_count();
    const auto target_columns = std::min<Eigen::Index>(
        kWhitenColumnBlockSize,
        1 + (result.n_packed_ao_pairs - 1) / (4LL * workers));

    // A shell pair stays whole so every Libcint shell triple is generated once.
    std::vector<ShellPair> pairs;
    std::vector<ShellPanel> panels;
    ShellPanel panel{};
    for (int i = 0; i < primary_input.n_shells; ++i) {
        const Eigen::Index n_i = integrals::shell_ao_count(primary_input, i);
        for (int j = 0; j <= i; ++j) {
            const Eigen::Index columns = i == j ? n_i * (n_i + 1) / 2
                                                : n_i * integrals::shell_ao_count(primary_input, j);
            if (panel.columns && panel.columns + columns > target_columns) {
                panels.push_back(panel);
                panel = {pairs.size(), pairs.size(), 0};
            }
            pairs.push_back({i, j, columns});
            panel.end = pairs.size();
            panel.columns += columns;
        }
    }
    if (panel.columns)
        panels.push_back(panel);

    const auto widest = std::max_element(panels.begin(), panels.end(), [](auto a, auto b) {
        return a.columns < b.columns;
    })->columns;
    result.metric_whitened_ao_pair_factors.resize(result.n_auxiliary_functions,
                                                 result.n_packed_ao_pairs);
    const auto build_start = timing ? Clock::now() : Clock::time_point{};
    double generation_worker_seconds = 0.0, whitening_worker_seconds = 0.0;
    int team_size = 1;
#pragma omp parallel num_threads(workers) reduction(+ : generation_worker_seconds, whitening_worker_seconds)
    {
#ifdef _OPENMP
#pragma omp single
        team_size = omp_get_num_threads();
#endif
        LibcintShellWorkspace local_workspace;
        Eigen::MatrixXd raw(n_auxiliary, target_columns);
        Eigen::MatrixXd whitened;
        if (cholesky_info != 0)
            whitened.resize(result.n_auxiliary_functions, target_columns);
        std::vector<Eigen::Index> output_columns(target_columns);
        std::vector<std::size_t> shell_offsets(target_columns);
#pragma omp for schedule(dynamic, 1)
        for (std::size_t task = 0; task < panels.size(); ++task) {
            const auto job = panels[task];
            const auto generation_start = timing ? Clock::now() : Clock::time_point{};
            if (job.columns > raw.cols()) {
                raw.resize(n_auxiliary, job.columns);
                output_columns.resize(job.columns);
                shell_offsets.resize(job.columns);
            }
            Eigen::Index first = 0;
            for (std::size_t item = job.first; item < job.end; ++item) {
                const auto pair = pairs[item];
                const int left_count = integrals::shell_ao_count(primary_input, pair.left);
                const int right_count = integrals::shell_ao_count(primary_input, pair.right);
                const auto stride = static_cast<std::size_t>(left_count) * right_count;
                Eigen::Index column = first;
                visit_ao_columns(integrals::shell_ao_offset(primary_input, pair.left), left_count,
                                 integrals::shell_ao_offset(primary_input, pair.right), right_count,
                                 [&](int i, int j, Eigen::Index packed) {
                    output_columns[column] = packed;
                    shell_offsets[column] = i + static_cast<std::size_t>(j) * left_count;
                    ++column;
                });
                for (int aux = 0; aux < auxiliary_input.n_shells; ++aux) {
                    const auto& block = evaluator.evaluate_three_center_shell_block(
                        pair.left, pair.right, aux, local_workspace);
                    for (Eigen::Index col = first; col < first + pair.columns; ++col)
                        for (int a = 0; a < block.auxiliary_ao_count; ++a)
                            raw(block.auxiliary_ao_offset + a, col) =
                                block.values[shell_offsets[col] +
                                             static_cast<std::size_t>(a) * stride];
                }
                first += pair.columns;
            }
            if (timing)
                generation_worker_seconds += elapsed(generation_start);
            const auto whiten_start = timing ? Clock::now() : Clock::time_point{};
            auto raw_panel = raw.leftCols(job.columns);
            if (cholesky_info == 0)
                metric_factor.triangularView<Eigen::Lower>().solveInPlace(raw_panel);
            else {
                if (job.columns > whitened.cols())
                    whitened.resize(result.n_auxiliary_functions, job.columns);
                whitened.leftCols(job.columns).noalias() = whitener * raw_panel;
            }
            const auto& source = cholesky_info == 0 ? raw : whitened;
            for (Eigen::Index col = 0; col < job.columns; ++col)
                result.metric_whitened_ao_pair_factors.col(output_columns[col]) = source.col(col);
            if (timing)
                whitening_worker_seconds += elapsed(whiten_start);
        }
    }
    if (timing)
        std::fprintf(stderr,
                     "XMVB_RI_BUILD,metric_seconds=%.9f,panels_seconds=%.9f,"
                     "generation_worker_seconds=%.9f,whitening_worker_seconds=%.9f,"
                     "threads=%d,panels=%zu,max_panel_columns=%ld,raw_aux=%d,rank=%d,ao_pairs=%ld\n",
                     metric_seconds, elapsed(build_start), generation_worker_seconds,
                     whitening_worker_seconds, team_size, panels.size(), static_cast<long>(widest),
                     n_auxiliary, result.n_auxiliary_functions,
                     static_cast<long>(result.n_packed_ao_pairs));
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
