#pragma once

#include <memory>
#include <mutex>

#include "core/index.hpp"
#include "core/workspace.hpp"
#include "mcscf/ci/links.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf::detail {

/** Exact number-channel elimination. Only one-spin tables are retained. */
struct DiagonalChannels {
    struct Spin {
        Eigen::MatrixXd off, potential;
        Eigen::VectorXd self;
    };
    workspace::Temporary storage;
    Eigen::MatrixXd kernel;
    Spin alpha, other;
    bool distinct;

    DiagonalChannels(const Eigen::MatrixXd& k, const SpinConnections& a, const SpinConnections& b)
        : distinct(&a != &b) {
        const int m = a.orbitals, o = m * (m - 1) / 2;
        const auto strings =
            core::checked_add(static_cast<std::size_t>(a.n_strings()),
                              distinct ? static_cast<std::size_t>(b.n_strings()) : 0);
        storage.resize(core::checked_multiply(
            sizeof(double),
            core::checked_add(
                static_cast<std::size_t>(o) * o,
                core::checked_multiply(strings, static_cast<std::size_t>(o + m + 1)))));
        kernel.resize(o, o);
        workspace::Temporary scratch(static_cast<std::size_t>(m) * (o + m) * sizeof(double));
        Eigen::MatrixXd od(o, m), dd(m, m);
        for (int p = 0; p < m; ++p) {
            const int pp = p * (p + 1) / 2 + p;
            for (int q = 0; q < m; ++q)
                dd(p, q) = k(pp, q * (q + 1) / 2 + q);
            for (int q = 0; q < p; ++q) {
                const int pq = p * (p - 1) / 2 + q;
                const int packed = p * (p + 1) / 2 + q;
                for (int r = 0; r < m; ++r) {
                    od(pq, r) = k(packed, r * (r + 1) / 2 + r);
                    for (int s = 0; s < r; ++s)
                        kernel(pq, r * (r - 1) / 2 + s) = k(packed, r * (r + 1) / 2 + s);
                }
            }
        }
        const auto fill = [&](const SpinConnections& graph, Spin& result) {
            const auto count = graph.n_strings();
            result.off.resize(count, o);
            result.potential.resize(count, m);
            result.self.resize(count);
            const int workers = bounded_openmp_thread_count(effective_openmp_thread_count(), count);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
            for (StringIndex row = 0; row < count; ++row) {
                for (int p = 0; p < o; ++p) {
                    double value = 0;
                    for (int q : graph.occupied(row))
                        value += od(p, q);
                    result.off(row, p) = value;
                }
                for (int p = 0; p < m; ++p) {
                    double value = 0;
                    for (int q : graph.occupied(row))
                        value += dd(p, q);
                    result.potential(row, p) = value;
                }
                double value = 0;
                for (int p : graph.occupied(row))
                    value += result.potential(row, p);
                result.self[row] = value;
            }
        };
        fill(a, alpha);
        if (distinct)
            fill(b, other);
    }

    const Spin& beta() const {
        return distinct ? other : alpha;
    }
};

/** Built only for sigma-only consumers; fused response actions keep their own channels. */
class DiagonalChannelCache {
  public:
    const DiagonalChannels&
    get(const Eigen::MatrixXd& kernel, const SpinConnections& alpha, const SpinConnections& beta) {
        std::call_once(ready_,
                       [&] { data_ = std::make_unique<DiagonalChannels>(kernel, alpha, beta); });
        return *data_;
    }

  private:
    std::once_flag ready_;
    std::unique_ptr<DiagonalChannels> data_;
};

} // namespace xmvb::mcscf::detail
