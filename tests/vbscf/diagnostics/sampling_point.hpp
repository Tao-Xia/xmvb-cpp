#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace sampling_audit {
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/** Fixed-point column probe. All physical kernels are borrowed unchanged.
 * Only small structure-output contributions may be catalogued for statistics;
 * no complete spin-product coefficient array is allocated.
 */
class Point {
  public:
    xmvb::vb::StructureAction::OrthogonalView view;
    xmvb::mcscf::SimilarityHamiltonian op;
    xmvb::mcscf::RawCiPlan source;
    Eigen::VectorXd coefficients;
    Eigen::MatrixXd values;
    std::vector<int> targets;
    int structures, width;
    Eigen::MatrixXd left, right;
    std::size_t panel_bytes;

    Point(xmvb::vb::StructureAction::OrthogonalView v,
          const xmvb::mcscf::ActiveIntegrals& integrals,
          const Eigen::MatrixXd& orbital,
          const Eigen::VectorXd& c)
        : view(v), op(integrals, orbital),
          source(v.sigma.alpha_connections(), v.sigma.beta_connections_view(), v.products),
          coefficients(c), values(Eigen::MatrixXd::Zero(v.products.size(), 1)),
          targets(source.beta_columns().begin(), source.beta_columns().end()), structures(c.size()),
          width(std::min<std::size_t>(targets.size(), 8 * xmvb::cooperative_openmp_thread_count())),
          left(v.alpha.dimension(), 2 * width), right(v.beta.dimension(), width) {
#pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < view.products.size(); ++p)
            for (std::size_t t = view.offsets[p]; t < view.offsets[p + 1]; ++t)
                values(p, 0) += view.terms[t].coefficient * c[view.terms[t].structure];
        panel_bytes = sizeof(double) * (left.size() + right.size() + 3 * values.size());
    }

    /** Cheap features and full-support importance probabilities, no orbital sweeps. */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> proposal() {
        const int n = op.orbitals(), k = targets.size();
        Eigen::MatrixXd f = Eigen::MatrixXd::Zero(n + 2, k);
        Eigen::VectorXd norms(k);
        for (int first = 0; first < k; first += width) {
            const int count = std::min(width, k - first);
            const auto betas = std::span(targets).subspan(first, count);
            auto raw = left.leftCols(count);
            source.apply_columns(op, values, betas, raw);
            for (int j = 0; j < count; ++j) {
                const int b = betas[j];
                const auto bits = view.sigma.beta_connections_view().basis.masks()[b];
                f(0, first + j) = 1;
                for (int i = 0; i < n - 1; ++i)
                    f(i + 1, first + j) = (bits >> i) & 1;
                f(n, first + j) = raw.col(j).sum();
                f(n + 1, first + j) = norms[first + j] = raw.col(j).norm();
            }
        }
        for (int row = n; row < n + 2; ++row) {
            const double scale = f.row(row).norm();
            if (scale)
                f.row(row) /= scale;
        }
        const double mean2 = norms.squaredNorm() / k;
        // RMS completion gives positive support without a tuned probability floor.
        Eigen::VectorXd p = (norms.array().square() + mean2).sqrt();
        if (p.sum() == 0)
            p.setOnes();
        p /= p.sum();
        return {std::move(f), std::move(p)};
    }

    /** Generate each requested beta once; multiplicity weights enter only projection.
     * Catalogues contain (2*n_structures) x requested_columns, never U_alpha*U_beta.
     */
    Eigen::VectorXd apply(std::span<const int> ids,
                          const Eigen::VectorXd& weights,
                          Eigen::MatrixXd* catalogue = nullptr) {
        const int nb = view.beta.dimension();
        if (weights.size() != static_cast<Eigen::Index>(ids.size()))
            throw std::invalid_argument("sampling weight count differs");
        if (catalogue)
            catalogue->setZero(2 * structures, ids.size());
        Eigen::MatrixXd products = Eigen::MatrixXd::Zero(values.rows(), 2);
        std::vector<int> betas;
        betas.reserve(width);
        for (std::size_t first = 0; first < ids.size(); first += width) {
            const int count = std::min<std::size_t>(width, ids.size() - first);
            betas.clear();
            for (int j = 0; j < count; ++j)
                betas.push_back(targets.at(ids[first + j]));
            auto h = left.leftCols(count);
            auto m = left.middleCols(count, count);
            source.apply_columns(op, values, betas, h);
            source.source_columns(values, betas, m);
            auto l = left.leftCols(2 * count);
            auto r = right.leftCols(count);
            r.setZero();
            for (int j = 0; j < count; ++j)
                r(betas[j], j) = 1;
            view.alpha.apply_left_block(l);
            view.alpha.apply_adjoint_left_block(l);
            view.beta.apply_left_block(r);
            view.beta.apply_adjoint_left_block(r);
            if (catalogue) {
#pragma omp parallel for schedule(static)
                for (int structure = 0; structure < structures; ++structure)
                    for (std::size_t t = view.structure_offsets[structure];
                         t < view.structure_offsets[structure + 1];
                         ++t) {
                        const auto& term = view.terms[view.structure_terms[t]];
                        const int a = view.products[term.group] / nb,
                                  b = view.products[term.group] % nb;
                        for (int j = 0; j < count; ++j) {
                            const double z = term.coefficient * r(b, j);
                            (*catalogue)(structure, first + j) += z * h(a, j);
                            (*catalogue)(structures + structure, first + j) += z * m(a, j);
                        }
                    }
            }
#pragma omp parallel for schedule(static)
            for (std::size_t p = 0; p < view.products.size(); ++p) {
                const int a = view.products[p] / nb, b = view.products[p] % nb;
                double yh = 0, ym = 0;
                for (int j = 0; j < count; ++j) {
                    const double z = weights[first + j] * r(b, j);
                    yh += z * h(a, j);
                    ym += z * m(a, j);
                }
                products(p, 0) += yh;
                products(p, 1) += ym;
            }
        }
        Eigen::VectorXd result = Eigen::VectorXd::Zero(2 * structures);
#pragma omp parallel for schedule(static)
        for (int structure = 0; structure < structures; ++structure)
            for (std::size_t t = view.structure_offsets[structure];
                 t < view.structure_offsets[structure + 1];
                 ++t) {
                const auto& term = view.terms[view.structure_terms[t]];
                result[structure] += term.coefficient * products(term.group, 0);
                result[structures + structure] += term.coefficient * products(term.group, 1);
            }
        return result;
    }
};
} // namespace sampling_audit
