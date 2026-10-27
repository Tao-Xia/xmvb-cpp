#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace conditional_audit {
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/** Shared fixed-point E^T K E on one spin sector, stored on its exact <=2 excitation graph.
 * No dense single-spin square or spin-product array. The builder uses the same
 * channels/signs as DirectCiSigmaAction, including absorbed one-electron terms.
 */
class SpinHamiltonian {
  public:
    std::vector<std::size_t> offsets;
    std::vector<int> columns;
    std::vector<double> values;
    SpinHamiltonian(const xmvb::mcscf::SpinConnections& graph, const Eigen::MatrixXd& kernel) {
        const std::size_t ne = graph.n_occupied(), nv = graph.orbitals - ne;
        const auto choose2 = [](std::size_t k) { return k < 2 ? 0 : k * (k - 1) / 2; };
        const std::size_t degree = 1 + ne * nv + choose2(ne) * choose2(nv);
        offsets.resize(graph.strings + 1);
        for (std::size_t i = 0; i < offsets.size(); ++i)
            offsets[i] = i * degree;
        columns.resize(offsets.back());
        values.resize(offsets.back());
#pragma omp parallel
        {
            std::vector<std::pair<int, double>> paths;
            paths.reserve((graph.degree + ne) * (graph.degree + ne) + 1);
            const auto visit = [&](int row, auto&& consume) {
                for (int p : graph.occupied(row))
                    consume(row, xmvb::core::PairIndex::packed_pair_index(p, p), 1.0);
                for (const auto& link : graph.singles(row))
                    consume(link.source, link.pair, link.sign);
            };
#pragma omp for schedule(static)
            for (int row = 0; row < graph.strings; ++row) {
                paths.clear();
                paths.emplace_back(row, 0.0);
                visit(row, [&](int mid, int q, double sq) {
                    visit(mid, [&](int col, int p, double sp) {
                        paths.emplace_back(col, sq * sp * kernel(q, p));
                    });
                });
                std::sort(paths.begin(), paths.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });
                auto out = offsets[row];
                for (std::size_t i = 0; i < paths.size();) {
                    const int col = paths[i].first;
                    double sum = 0;
                    do {
                        sum += paths[i++].second;
                    } while (i < paths.size() && paths[i].first == col);
                    columns[out] = col;
                    values[out++] = sum;
                }
                if (out != offsets[row + 1])
                    std::terminate(); // Exact graph identity, never a numerical cutoff.
            }
        }
    }
    std::size_t bytes() const {
        return offsets.capacity() * sizeof(std::size_t) + columns.capacity() * sizeof(int) +
               values.capacity() * sizeof(double);
    }
    void apply(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::Ref<Eigen::VectorXd> y) const {
#pragma omp parallel for schedule(static)
        for (int row = 0; row < x.size(); ++row) {
            double sum = 0;
            for (std::size_t j = offsets[row]; j < offsets[row + 1]; ++j)
                sum += values[j] * x[columns[j]];
            y[row] = sum;
        }
    }
};

struct Result {
    double norm = 0, numerator = 0, residual_norm = 0, row_residual_norm = 0;
};

/** One orthogonal beta output at a time. All large buffers are shared panels.
 * Arbitrary raw source pairs and unequal spin sectors use the same identities.
 */
class Columns {
    const xmvb::mcscf::DirectCiSigmaAction& sigma;
    const xmvb::vb::ExteriorOrbitalTransform &alpha, &beta;
    const xmvb::mcscf::RawCiPlan& source;
    const Eigen::VectorXd& source_values;
    SpinHamiltonian ha;
    std::unique_ptr<SpinHamiltonian> hb_distinct;
    const SpinHamiltonian& hb;
    int neighbours;
    Eigen::MatrixXd raw_beta, raw_alpha;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> alpha_rows, weights;
    Eigen::VectorXd h_alpha;

  public:
    double fill_seconds = 0, beta_seconds = 0, source_seconds = 0, alpha_seconds = 0,
           finish_seconds = 0;
    Columns(const xmvb::mcscf::DirectCiSigmaAction& s,
            const xmvb::vb::ExteriorOrbitalTransform& a,
            const xmvb::vb::ExteriorOrbitalTransform& b,
            const xmvb::mcscf::RawCiPlan& plan,
            const Eigen::VectorXd& values)
        : sigma(s), alpha(a), beta(b), source(plan), source_values(values),
          ha(s.alpha_connections(), s.channel_kernel()),
          hb_distinct(s.shared_spin_connections()
                          ? nullptr
                          : std::make_unique<SpinHamiltonian>(s.beta_connections_view(),
                                                              s.channel_kernel())),
          hb(hb_distinct ? *hb_distinct : ha), neighbours(1 + s.beta_connections_view().degree),
          raw_beta(b.dimension(), neighbours + 1), raw_alpha(a.dimension(), neighbours + 1),
          alpha_rows(a.dimension(), neighbours), weights(s.channel_kernel().rows(), neighbours),
          h_alpha(a.dimension()) {}

    std::size_t bytes() const {
        return sizeof(double) * (raw_alpha.size() + raw_beta.size() + alpha_rows.size() +
                                 weights.size() + h_alpha.size()) +
               ha.bytes() + (hb_distinct ? hb.bytes() : 0);
    }
    const Eigen::MatrixXd& alpha_panel() const {
        return raw_alpha;
    }
    const Eigen::VectorXd& image() const {
        return h_alpha;
    }
    Result evaluate(int b, double energy, bool timers = true) {
        auto t = Clock::now();
        raw_beta.setZero();
        weights.setZero();
        raw_beta(b, 0) = 1;
        const auto& graph = sigma.beta_connections_view();
        const auto& k = sigma.channel_kernel();
        // Combine beta diagonal channels on their common target before propagation.
        for (int p : graph.occupied(b))
            weights.col(0) += 2 * k.col(xmvb::core::PairIndex::packed_pair_index(p, p));
        int j = 1;
        for (const auto& link : graph.singles(b)) {
            raw_beta(link.source, j) = 1;
            weights.col(j++) = 2 * link.sign * k.col(link.pair);
        }
        // One linear combination replaces the entire beta double-excitation halo.
        for (std::size_t t = hb.offsets[b]; t < hb.offsets[b + 1]; ++t)
            raw_beta(hb.columns[t], neighbours) = hb.values[t];
        if (timers)
            fill_seconds += elapsed(t);
        t = Clock::now();
        beta.apply_adjoint_left_block(raw_beta);
        if (timers)
            beta_seconds += elapsed(t);
        t = Clock::now();
        source.multiply_source(source_values, raw_beta, raw_alpha);
        if (timers)
            source_seconds += elapsed(t);
        t = Clock::now();
        alpha.apply_left_block(raw_alpha);
        if (timers)
            alpha_seconds += elapsed(t);
        t = Clock::now();
        const auto y = raw_alpha.col(0);
        ha.apply(y, h_alpha);
        h_alpha += raw_alpha.col(neighbours);
        alpha_rows = raw_alpha.leftCols(neighbours);
        const auto& links = sigma.alpha_connections();
        // Contract the small beta-neighbour dimension directly on each alpha link.
        // Both dot operands are contiguous; no U_alpha*channel mixed panel exists.
#pragma omp parallel for schedule(static)
        for (int a = 0; a < alpha_rows.rows(); ++a) {
            double mixed = 0;
            for (int p : links.occupied(a))
                mixed += weights.row(xmvb::core::PairIndex::packed_pair_index(p, p))
                             .dot(alpha_rows.row(a));
            for (const auto& link : links.singles(a))
                mixed += link.sign * weights.row(link.pair).dot(alpha_rows.row(link.source));
            h_alpha[a] += mixed;
        }
        Result result;
        result.norm = y.squaredNorm();
        result.numerator = y.dot(h_alpha);
        for (int a = 0; a < y.size(); ++a) {
            const double r = h_alpha[a] - energy * y[a];
            result.residual_norm += r * r;
            if (y[a] != 0)
                result.row_residual_norm += r * r;
        }
        if (timers)
            finish_seconds += elapsed(t);
        return result;
    }
};
} // namespace conditional_audit
