#include "integrals/ao/exact.hpp"

#include <algorithm>
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
    // One exchange image shared by the workers, packed J, and one AO-pair
    // slice plus half-transformed and projected row panels per worker.
    const auto square = static_cast<std::size_t>(n) * n;
    const std::size_t bytes =
        sizeof(double) * ((square + pairs + 2 * n_mo) * n_core +
                          workers * (square + 3 * n * n_core + 4 * n_mo * n_core));
    workspace::Reservation reservation;
    if (!reservation.try_resize(bytes))
        return false;
    workspace::Temporary temporary(bytes);
    profiling::ScopedPhase phase(profiling::Phase::AoCoreDiagonal);

    Eigen::MatrixXd exchange(n, static_cast<Eigen::Index>(n) * n_core);
    Eigen::MatrixXd coulomb(n_core, pairs);
    std::vector<Eigen::MatrixXd> projected(workers, Eigen::MatrixXd::Zero(n_mo, 2 * n_core));
    const auto exchange_row = [&](int mu) {
        return exchange.middleCols(static_cast<Eigen::Index>(mu) * n_core, n_core);
    };
#pragma omp parallel num_threads(workers)
    {
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        Eigen::MatrixXd eri(n, n), half(n, n_core);
        const auto consume = [&](int mu, int nu) {
            const auto pair = core::PairIndex::packed_pair_index(mu, nu);
            eri.triangularView<Eigen::Lower>().setZero();
            graph.visit_row(pair, [&](std::size_t right, double value) {
                eri(graph.pair_first[right], graph.pair_second[right]) = value;
            });
            // T(k,i) = sum_l (mu nu|k l) C(l,i). The same panel supplies
            // J_i(mu,nu) and the exchange updates to rows mu and nu.
            cblas_dsymm(CblasColMajor,
                        CblasLeft,
                        CblasLower,
                        n,
                        n_core,
                        1.0,
                        eri.data(),
                        n,
                        frame.data(),
                        frame.outerStride(),
                        0.0,
                        half.data(),
                        n);
            for (int i = 0; i < n_core; ++i) {
                coulomb(i, pair) = frame.col(i).dot(half.col(i));
                exchange_row(mu).col(i).noalias() += frame(nu, i) * half.col(i);
                if (mu != nu)
                    exchange_row(nu).col(i).noalias() += frame(mu, i) * half.col(i);
            }
        };
#pragma omp for schedule(static)
        for (int mu = 0; mu < n; ++mu) {
            exchange_row(mu).setZero();
            consume(mu, mu);
        }
        // Round-robin pair schedule. Every unordered pair is visited once
        // and no two workers update the same exchange row within a round.
        for (int round = 0; round < ring; ++round) {
#pragma omp for schedule(static)
            for (int k = 0; k < order / 2; ++k) {
                if (k != 0)
                    consume((round + k) % ring, (round + ring - k) % ring);
                else if (ring < n)
                    consume(round, ring);
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
