#include "core/index.hpp"
#include <cstdint>
#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {

OverlapTransportGradient
StructureAction::cross_overlap_transport_gradient(const Eigen::VectorXd& lambda,
                                                  const Eigen::VectorXd& base,
                                                  const Eigen::VectorXd& reference,
                                                  const Eigen::VectorXd& transported,
                                                  const ExteriorOrbitalTransform& ca,
                                                  const ExteriorOrbitalTransform& cb,
                                                  const ExteriorOrbitalTransform& oa,
                                                  const ExteriorOrbitalTransform& ob,
                                                  const Eigen::MatrixXd& direction,
                                                  std::size_t workspace_bytes) const {
    profiling::ScopedPhase phase(profiling::Phase::CiOverlapTransportGradient);
    const int n = ca.n_orbitals();
    if (lambda.size() != n_structures_ || base.size() != n_structures_ ||
        reference.size() != n_structures_ || transported.size() != n_structures_ ||
        ca.dimension() != n_unique_alpha_ || oa.dimension() != n_unique_alpha_ ||
        cb.dimension() != n_unique_beta_ || ob.dimension() != n_unique_beta_ ||
        direction.rows() != n || direction.cols() != n)
        throw std::invalid_argument("invalid finite transport gradient inputs");
    OverlapTransportGradient result{{Eigen::MatrixXd::Zero(n, n), Eigen::MatrixXd::Zero(n, n)},
                                    Eigen::VectorXd::Zero(n_structures_)};
    const auto da = oa.direction(direction), db = ob.direction(direction);
    const auto& sources = *sources_;
    // All consumers share one panel budget. Workers own columns, never panel copies.
    const std::size_t row_bytes =
        16 * sizeof(double) * (static_cast<std::size_t>(n_unique_alpha_) + n_unique_beta_);
    if (workspace_bytes < row_bytes)
        throw std::length_error("finite transport cannot hold one factor column");
    const int extent = std::max<std::size_t>(
        1,
        std::min<std::size_t>(sources.columns.size(),
                              std::min<std::size_t>(workspace_bytes / row_bytes,
                                                    8 * effective_openmp_thread_count())));
    const Eigen::VectorXd weights = expand_support(lambda);
    // All four structure vectors have the same certified spin-exchange parity.
    // With shared maps, exchange interchanges the two moment contributions and
    // the two parity signs cancel. The CI transpose image still uses one side.
    const bool shared_moment = spin_exchange_parity_ != 0 && &ca == &cb && &oa == &ob;
    const auto seed = [&](Eigen::MatrixXd panel, bool alpha) {
        const auto& from = alpha ? cb : ca;
        const auto& to = alpha ? ca : cb;
        from.apply_adjoint_left(&panel);
        Eigen::MatrixXd image = Eigen::MatrixXd::Zero(to.dimension(), panel.cols());
        const int workers =
            coarse_openmp_thread_count(cooperative_openmp_thread_count(), image.size());
        cooperative_openmp_for(panel.cols(), workers, [&](std::ptrdiff_t col) {
            const auto in = panel.col(col);
            auto out = image.col(col);
            if (alpha) {
                for (Eigen::Index ai = 0; ai < n_unique_alpha_; ++ai)
                    for (auto g = sources.alpha_offsets[ai]; g < sources.alpha_offsets[ai + 1]; ++g)
                        out[ai] += weights[g] * in[spin_products_[g] % n_unique_beta_];
            } else {
                for (Eigen::Index bi = 0; bi < n_unique_beta_; ++bi)
                    for (const auto g : sources.beta_groups[bi])
                        out[bi] += weights[g] * in[spin_products_[g] / n_unique_beta_];
            }
        });
        to.apply_left(&image);
        return image;
    };
    for (std::int64_t begin = 0; begin < core::checked_index<std::int64_t>(sources.columns.size());
         begin += extent) {
        const std::int64_t count = std::min<int>(extent, sources.columns.size() - begin);
        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n_unique_alpha_, count), du = u, ub = u, uc = u;
        Eigen::MatrixXd v = Eigen::MatrixXd::Zero(n_unique_beta_, count), dv = v, vc = v;
        for (int j = 0; j < count; ++j) {
            const std::int64_t b = sources.columns[begin + j];
            v(b, j) = vc(b, j) = 1;
            for (const auto g : sources.beta_groups[b]) {
                const std::int64_t a = spin_products_[g] / n_unique_beta_;
                for (auto t = spin_term_offsets_[g]; t < spin_term_offsets_[g + 1]; ++t) {
                    const auto& term = spin_terms_[t];
                    u(a, j) += term.coefficient * reference[term.structure];
                    ub(a, j) += term.coefficient * base[term.structure];
                    uc(a, j) += term.coefficient * transported[term.structure];
                }
            }
        }
        oa.apply_directional_left(da, &u, &du);
        ob.apply_directional_left(db, &v, &dv);
        oa.apply_left(&ub);
        ub += du;
        du.resize(0, 0);
        ca.apply_left(&uc);
        cb.apply_left(&vc);
        {
            // Target = (U_base+dU_ref) V_ref^T + U_ref dV_ref^T.
            // Each side consumes all three factor pairs in a single pass.
            Eigen::MatrixXd b(n_unique_beta_, 3 * count);
            b << v, dv, vc;
            auto left = seed(std::move(b), true);
            auto first_left = left.leftCols(count);
            const auto second_left = left.middleCols(count, count);
            const auto metric_left = left.rightCols(count);
            const ExteriorGeneratorTerm alpha_terms[]{{ub, first_left},
                                                      {u, second_left},
                                                      {uc, metric_left, -1.0},
                                                      {metric_left, uc, -1.0}};
            const double multiplicity = shared_moment ? 2.0 : 1.0;
            result.charts.left += multiplicity * ca.generator_pullback(alpha_terms);
            result.charts.right += multiplicity * oa.generator_pullback(first_left, u);
            if (!shared_moment) {
                Eigen::MatrixXd a(n_unique_alpha_, 3 * count);
                a << ub, u, uc;
                const auto right = seed(std::move(a), false);
                const auto first_right = right.leftCols(count);
                const auto second_right = right.middleCols(count, count);
                const auto metric_right = right.rightCols(count);
                const ExteriorGeneratorTerm beta_terms[]{{v, first_right},
                                                         {dv, second_right},
                                                         {vc, metric_right, -1.0},
                                                         {metric_right, vc, -1.0}};
                result.charts.left += cb.generator_pullback(beta_terms);
                result.charts.right += ob.generator_pullback(second_right, v);
            }
            oa.apply_adjoint_left_block(first_left);
            for (int j = 0; j < count; ++j)
                for (const auto g : sources.beta_groups[sources.columns[begin + j]]) {
                    const double value = first_left(spin_products_[g] / n_unique_beta_, j);
                    for (auto t = spin_term_offsets_[g]; t < spin_term_offsets_[g + 1]; ++t) {
                        const auto& term = spin_terms_[t];
                        result.image[term.structure] += term.coefficient * value;
                    }
                }
        }
    }
    return result;
}

} // namespace xmvb::vb
