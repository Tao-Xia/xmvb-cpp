#include "integrals/ao/exact.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"
#include "core/timing.hpp"

namespace xmvb::integrals {
namespace {
using Channels = AoH1eBatchWorkspace::Channels;

void validate_rows(const AoIntegralInput& ao, const Eigen::Ref<const Channels>& sources) {
    const Eigen::Index n = ao.n_basis_functions;
    const auto& graph = ao.pair_graph;
    if (n <= 0 || sources.rows() != n * n || !graph.has_row_graph() ||
        graph.row_offsets.size() != static_cast<std::size_t>(n * (n + 1) / 2 + 1))
        throw std::invalid_argument("density row contraction requires symmetric CSR integrals");
}

/** @brief One owned exchange row and, optionally, its ordered Coulomb row.
 * CSR values are raw ERIs: no unique-stream half weights belong here.
 * Right off-diagonal pairs supply two exchange destinations, not a factor 2.
 */
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
void contract_row(const AoIntegralInput& ao, int mu, const double* source,
    double* exchange, int width, Eigen::Index exchange_stride,
    double* coulomb, int core_width, Eigen::Index coulomb_stride) {
    const auto& graph = ao.pair_graph;
    const int n = ao.n_basis_functions;
    if (exchange)
        for (int nu = 0; nu < n; ++nu)
            std::fill_n(exchange + nu * exchange_stride, width, 0.0);
    for (int j = 0; j < n; ++j) {
        double* sum = core_width ? coulomb + j * coulomb_stride : nullptr;
        if (sum)
            std::fill_n(sum, core_width, 0.0);
        const int hi = std::max(mu, j), lo = std::min(mu, j);
        const int pair = hi * (hi + 1) / 2 + lo;
        for (int edge = graph.row_offsets[pair]; edge < graph.row_offsets[pair + 1]; ++edge) {
            const int right = graph.columns[edge];
            const int k = graph.pair_first[right], l = graph.pair_second[right];
            const double value = graph.values[edge];
            // D is symmetric: keep the ordered left partner j as the AO column.
            // All exchange RHS values then lie in one contiguous n-by-width stripe,
            // rather than touching n different pages for every integral row.
            if (exchange) {
                const double* djl = source + (static_cast<Eigen::Index>(j) * n + l) * width;
                double* target_k = exchange + k * exchange_stride;
#pragma omp simd
                for (int channel = 0; channel < width; ++channel)
                    target_k[channel] += value * djl[channel];
                if (k != l) {
                    const double* djk = source + (static_cast<Eigen::Index>(j) * n + k) * width;
                    double* target_l = exchange + l * exchange_stride;
#pragma omp simd
                    for (int channel = 0; channel < width; ++channel)
                        target_l[channel] += value * djk[channel];
                }
            }
            if (sum) {
                // Right packed pairs are ordered by k, then l. The symmetric AO
                // orientation follows that order in the shared density storage.
                const double* dkl = source + (static_cast<Eigen::Index>(k) * n + l) * width;
                const double scale = value * (k == l ? 1.0 : 2.0);
#pragma omp simd
                for (int channel = 0; channel < core_width; ++channel)
                    sum[channel] += scale * dkl[channel];
            }
        }
    }
}

void validate_weighted_graph(const AoIntegralInput& ao) {
    const int n = ao.n_basis_functions;
    if (n <= 0)
        throw std::invalid_argument("invalid exact AO JK basis size");
    const auto& graph = ao.pair_graph;
    const std::size_t n_pairs = static_cast<std::size_t>(n) * (n + 1) / 2;
    const bool csr = graph.has_row_graph() && graph.row_offsets.size() == n_pairs + 1 &&
                     graph.columns.size() == graph.values.size();
    const bool compact = graph.has_compact_integral_stream() && graph.valid_unique_storage();
    if ((!csr && !compact) || graph.pair_first.size() != n_pairs ||
        graph.pair_second.size() != n_pairs)
        throw std::invalid_argument("invalid exact AO JK integral representation");
}

/** Visit each distinct ordered permutation of one canonical quartet. */
void accumulate_unique_quartet(const Eigen::Ref<const Eigen::MatrixXd>& density,
                               int i,
                               int j,
                               int k,
                               int l,
                               double value,
                               double coulomb_weight,
                               double exchange_weight,
                               Eigen::MatrixXd* output) {
    for (int pair_swap = 0; pair_swap < 1 + (i != k || j != l); ++pair_swap) {
        const int first = pair_swap ? k : i, second = pair_swap ? l : j;
        const int third = pair_swap ? i : k, fourth = pair_swap ? j : l;
        for (int left_swap = 0; left_swap < 1 + (first != second); ++left_swap) {
            const int a = left_swap ? second : first;
            const int b = left_swap ? first : second;
            for (int right_swap = 0; right_swap < 1 + (third != fourth); ++right_swap) {
                const int c = right_swap ? fourth : third;
                const int d = right_swap ? third : fourth;
                if (coulomb_weight != 0.0)
                    (*output)(a, b) += coulomb_weight * value * density(c, d);
                if (exchange_weight != 0.0)
                    (*output)(a, c) += exchange_weight * value * density(b, d);
            }
        }
    }
}
} // namespace

void apply_symmetric_ao_h1e_rows(const AoIntegralInput& ao,
                                 int n_threads,
                                 const Eigen::Ref<const Channels>& sources,
                                 Channels* response) {
    validate_rows(ao, sources);
    if (!response)
        throw std::invalid_argument("null AO row response");
    const int n = ao.n_basis_functions, width = sources.cols();
    response->resize(sources.rows(), width);
    if (!width)
        return;
    n_threads = bounded_openmp_thread_count(n_threads, n);
#pragma omp parallel num_threads(n_threads)
    {
        Channels coulomb(n, width);
#pragma omp for schedule(dynamic, 1)
        for (int mu = 0; mu < n; ++mu) {
            double* target = response->data() + static_cast<Eigen::Index>(mu) * width;
            contract_row(ao,
                         mu,
                         sources.data(),
                         target,
                         width,
                         static_cast<Eigen::Index>(n) * width,
                         coulomb.data(),
                         width,
                         width);
            for (int nu = 0; nu < n; ++nu) {
#pragma omp simd
                for (int channel = 0; channel < width; ++channel)
                    (*response)(mu + nu * n, channel) =
                        2.0 * coulomb(nu, channel) - (*response)(mu + nu * n, channel);
            }
        }
    }
}

void apply_weighted_symmetric_ao_jk(const Eigen::Ref<const Eigen::MatrixXd>& density,
                                    const AoIntegralInput& ao,
                                    double coulomb_weight,
                                    double exchange_weight,
                                    int n_threads,
                                    Eigen::MatrixXd* response) {
    validate_weighted_graph(ao);
    const int n = ao.n_basis_functions;
    if (!response || density.rows() != n || density.cols() != n)
        throw std::invalid_argument("invalid exact AO JK density or response");
    response->setZero(n, n);
    if (coulomb_weight == 0.0 && exchange_weight == 0.0)
        return;
    const auto& graph = ao.pair_graph;
    if (!graph.has_row_graph()) {
        // The compact representation stores each integral once. Expand only its
        // distinct index permutations, so equal AO indices need no half weights.
        for (std::size_t index = 0; index < graph.integral_count(); ++index) {
            const AoPairIntegral eri = graph.integral(index);
            accumulate_unique_quartet(density,
                                      graph.pair_first[eri.left_pair],
                                      graph.pair_second[eri.left_pair],
                                      graph.pair_first[eri.right_pair],
                                      graph.pair_second[eri.right_pair],
                                      eri.value,
                                      coulomb_weight,
                                      exchange_weight,
                                      response);
        }
    } else {
        // Directed CSR values are raw ERIs. Every AO output row has one owner.
        n_threads = bounded_openmp_thread_count(n_threads, n);
#pragma omp parallel num_threads(n_threads)
        {
            std::vector<double> coulomb(coulomb_weight != 0.0 ? n : 0);
#pragma omp for schedule(dynamic, 1)
            for (int mu = 0; mu < n; ++mu) {
                contract_row(ao,
                             mu,
                             density.data(),
                             exchange_weight != 0.0 ? response->data() + mu : nullptr,
                             1,
                             n,
                             coulomb_weight != 0.0 ? coulomb.data() : nullptr,
                             coulomb_weight != 0.0 ? 1 : 0,
                             1);
                for (int nu = 0; nu < n; ++nu) {
                    const double j = coulomb_weight != 0.0 ? coulomb[nu] : 0.0;
                    const double k = exchange_weight != 0.0 ? (*response)(mu, nu) : 0.0;
                    (*response)(mu, nu) = coulomb_weight * j + exchange_weight * k;
                }
            }
        }
    }
    // Enforce one full symmetric AO matrix, including under differing sum order.
    for (int nu = 0; nu < n; ++nu)
        for (int mu = 0; mu < nu; ++mu)
            (*response)(mu, nu) = (*response)(nu, mu) =
                0.5 * ((*response)(mu, nu) + (*response)(nu, mu));
}

void apply_ao_density_diagonals(const AoIntegralInput& ao,
                                const Eigen::Ref<const Channels>& sources,
                                const Eigen::MatrixXd& frame,
                                int n_core_sources,
                                int n_threads,
                                AoDensityDiagonalWorkspace* workspace,
                                Eigen::MatrixXd* exchange_diagonal,
                                Eigen::MatrixXd* coulomb_diagonal) {
    validate_rows(ao, sources);
    const int n = ao.n_basis_functions, width = sources.cols(), n_mo = frame.cols();
    if (!workspace || !exchange_diagonal || !coulomb_diagonal || frame.rows() != n ||
        n_core_sources < 0 || n_core_sources > width)
        throw std::invalid_argument("invalid AO density diagonal projection");
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eFused);
    exchange_diagonal->resize(n_mo, width);
    coulomb_diagonal->setZero(n_mo, n_core_sources);
    if (!width)
        return;
    n_threads = bounded_openmp_thread_count(n_threads, n);
    workspace->rows.resize(n_threads);
    workspace->images.resize(n_threads);
    workspace->partial.resize(n_threads);
#pragma omp parallel num_threads(n_threads)
    {
        int worker = 0, team = 1;
#ifdef _OPENMP
        worker = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const int columns = width + n_core_sources;
        auto& rows = workspace->rows[worker];
        auto& images = workspace->images[worker];
        auto& partial = workspace->partial[worker];
        rows.resize(n, columns);
        images.resize(n_mo, columns);
        partial.setZero(n_mo, columns);
#pragma omp for schedule(dynamic, 1)
        for (int mu = 0; mu < n; ++mu) {
            contract_row(ao,
                         mu,
                         sources.data(),
                         rows.data(),
                         width,
                         columns,
                         rows.data() + width,
                         n_core_sources,
                         columns);
            // Both K and the required core J row are dead after this one GEMM.
            // Sum C(mu,p) [C^T row]_p directly, never forming the AO response.
            images.noalias() = frame.transpose() * rows;
            for (int p = 0; p < n_mo; ++p)
                partial.row(p).noalias() += frame(mu, p) * images.row(p);
        }
#pragma omp for schedule(static)
        for (int p = 0; p < n_mo; ++p)
            for (int channel = 0; channel < columns; ++channel) {
                double sum = 0.0;
                for (int other = 0; other < team; ++other)
                    sum += workspace->partial[other](p, channel);
                if (channel < width)
                    (*exchange_diagonal)(p, channel) = sum;
                else
                    (*coulomb_diagonal)(p, channel - width) = sum;
            }
    }
}
} // namespace xmvb::integrals
