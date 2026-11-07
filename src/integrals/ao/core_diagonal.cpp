#include "integrals/ao/exact.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cblas.h>

#include "core/index.hpp"
#include "core/pair_index.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::integrals {

bool apply_ao_core_diagonals(const AoIntegralInput& ao,
                             const Eigen::Ref<const Eigen::MatrixXd>& frame,
                             int n_core,
                             int n_threads,
                             Eigen::MatrixXd* exchange_diagonal,
                             Eigen::MatrixXd* coulomb_diagonal) {
    const int n = ao.n_basis_functions;
    const auto& graph = ao.pair_graph;
    const Eigen::Index pairs = core::triangular_count(n);
    if (!exchange_diagonal || !coulomb_diagonal || n <= 0 || n_core <= 0 || frame.rows() != n ||
        n_core > frame.cols() || !graph.has_rows() || graph.pair_first.size() != pairs ||
        graph.pair_second.size() != pairs)
        throw std::invalid_argument("invalid exact core diagonal input");

    const int n_mo = frame.cols();
    const int order = n + n % 2;
    const int ring = order - 1;
    const int workers = bounded_openmp_thread_count(n_threads, order / 2);
    // Reuse the previous per-worker AO slice allowance for half panels.
    // Each pair in the panel has n AO rows and n_core contiguous channels.
    const int panel_pairs = graph.tiles ? std::max(1, n / n_core) : 1;
    const auto square = static_cast<std::size_t>(n) * n;
    const auto panel_values = static_cast<std::size_t>(panel_pairs) * n * n_core;
    const auto scratch = graph.tiles ? panel_values : square + panel_values;
    const std::size_t bytes =
        sizeof(double) * ((square + pairs + n + 2 * n_mo) * n_core +
                          workers * (scratch + 2 * n * n_core + 4 * n_mo * n_core));
    workspace::Reservation reservation;
    if (!reservation.try_resize(bytes))
        return false;
    workspace::Temporary temporary(bytes);
    profiling::ScopedPhase phase(profiling::Phase::AoCoreDiagonal);

    using Panel = AoPairTiles::Matrix;
    const Panel core = frame.leftCols(n_core);
    Panel exchange(square, n_core);
    Eigen::MatrixXd coulomb(n_core, pairs);
    std::vector<std::mutex> exchange_locks(graph.tiles ? n : 0);
    std::vector<Eigen::MatrixXd> projected(workers, Eigen::MatrixXd::Zero(n_mo, 2 * n_core));
    const auto exchange_row = [&](int mu) {
        return exchange.middleRows(static_cast<Eigen::Index>(mu) * n, n);
    };
#pragma omp parallel num_threads(workers)
    {
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        const auto consume_half = [&](std::size_t pair, int mu, int nu, const auto& half) {
            // The same T(k,i) = sum_l (mu nu|k l) C(l,i) supplies both images.
            coulomb.col(pair) = (core.array() * half.array()).colwise().sum().transpose();
            const auto accumulate = [&](int target, int source) {
                std::unique_lock<std::mutex> lock;
                if (graph.tiles)
                    lock = std::unique_lock(exchange_locks[target]);
                exchange_row(target).array() += half.array().rowwise() * core.row(source).array();
            };
            accumulate(mu, nu);
            if (mu != nu)
                accumulate(nu, mu);
        };
#pragma omp for schedule(static)
        for (int mu = 0; mu < n; ++mu)
            exchange_row(mu).setZero();

        if (graph.tiles) {
            const auto& tiles = *graph.tiles;
            Panel half(static_cast<Eigen::Index>(panel_pairs) * n, n_core);
#pragma omp for schedule(dynamic, 1)
            for (std::size_t block = 0; block < tiles.row_blocks(); ++block) {
                for (int begin = 0; begin < tiles.row_count(block); begin += panel_pairs) {
                    const int count = std::min(panel_pairs, tiles.row_count(block) - begin);
                    half.topRows(static_cast<Eigen::Index>(count) * n).setZero();
                    tiles.visit_panel(block, begin, count, [&](std::size_t source, auto visit) {
                        const int k = graph.pair_first[source], l = graph.pair_second[source];
                        visit([&](int row, double value) {
                            const auto offset = static_cast<Eigen::Index>(row) * n;
                            half.row(offset + k).noalias() += value * core.row(l);
                            if (k != l)
                                half.row(offset + l).noalias() += value * core.row(k);
                        });
                    });
                    for (int row = 0; row < count; ++row) {
                        const auto pair = tiles.row_begin(block) + begin + row;
                        consume_half(pair, graph.pair_first[pair], graph.pair_second[pair],
                                     half.middleRows(static_cast<Eigen::Index>(row) * n, n));
                    }
                }
            }
        } else {
            Panel eri(n, n), half(n, n_core);
            const auto consume = [&](int mu, int nu) {
                const auto pair = core::PairIndex::packed_pair_index(mu, nu);
                eri.triangularView<Eigen::Lower>().setZero();
                graph.visit_row(pair, [&](std::size_t right, double value) {
                    eri(graph.pair_first[right], graph.pair_second[right]) = value;
                });
                cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, n, n_core, 1.0,
                            eri.data(), n, core.data(), n_core, 0.0, half.data(), n_core);
                consume_half(pair, mu, nu, half);
            };
#pragma omp for schedule(static)
            for (int mu = 0; mu < n; ++mu)
                consume(mu, mu);
            // Every unordered pair is visited once. The two output rows of each
            // pair are disjoint from the other workers' rows in the same round.
            for (int round = 0; round < ring; ++round) {
#pragma omp for schedule(static)
                for (int k = 0; k < order / 2; ++k) {
                    if (k != 0)
                        consume((round + k) % ring, (round + ring - k) % ring);
                    else if (ring < n)
                        consume(round, ring);
                }
            }
        }
        Eigen::MatrixXd rows(n, 2 * n_core), images(n_mo, 2 * n_core);
        auto& partial = projected[worker];
#pragma omp for schedule(static)
        for (int mu = 0; mu < n; ++mu) {
            rows.leftCols(n_core) = exchange_row(mu);
            for (int nu = 0; nu < n; ++nu)
                rows.row(nu).rightCols(n_core) =
                    coulomb.col(core::PairIndex::packed_pair_index(mu, nu)).transpose();
            images.noalias() = frame.transpose() * rows;
            for (int p = 0; p < n_mo; ++p)
                partial.row(p).noalias() += frame(mu, p) * images.row(p);
        }
    }
    exchange_diagonal->setZero(n_mo, n_core);
    coulomb_diagonal->setZero(n_mo, n_core);
    for (const auto& partial : projected) {
        *exchange_diagonal += partial.leftCols(n_core);
        *coulomb_diagonal += partial.rightCols(n_core);
    }
    return true;
}

} // namespace xmvb::integrals
