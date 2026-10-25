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
    if (use_product_overlap(ca, cb, workspace_bytes, 5)) {
        // Broad support keeps the cheaper moment order, with five shared images
        // at peak. No new full-carrier residency requirement is introduced.
        auto left = expand_structure_block(lambda);
        ca.apply_left(&left);
        cb.apply_right(&left);
        auto current = expand_structure_block(transported);
        ca.apply_left(&current);
        cb.apply_right(&current);
        auto origin = expand_structure_block(reference);
        Eigen::MatrixXd tangent = Eigen::MatrixXd::Zero(origin.rows(), origin.cols());
        oa.apply_directional_left(da, &origin, &tangent);
        ob.apply_directional_right(db, &origin, &tangent);
        auto target = expand_structure_block(base);
        oa.apply_left(&target);
        ob.apply_right(&target);
        target += tangent - current;
        tangent.resize(0, 0);
        result.charts.left = ca.moment_pullback(carrier_generator_adjoint(target, left) -
                                                carrier_generator_adjoint(left, current));
        result.charts.right = oa.moment_pullback(carrier_generator_adjoint(left, origin));
        current.resize(0, 0);
        origin.resize(0, 0);
        target.resize(0, 0);
        ob.apply_adjoint_right(&left);
        oa.apply_adjoint_left(&left);
        result.image = contract_spin_product_block(left).col(0);
        return result;
    }

    const auto& sources = *sources_;
    // All consumers share one panel budget. Workers own rows, never panel copies.
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
    // Form F_current lambda contracted with an arbitrary factor U V^T.
    // Its two seed panels serve both chart pullbacks and the transpose image.
    const auto seeds = [&](Eigen::MatrixXd a, Eigen::MatrixXd b) {
        ca.apply_adjoint_left(&a);
        cb.apply_adjoint_left(&b);
        Eigen::MatrixXd za = Eigen::MatrixXd::Zero(n_unique_alpha_, b.cols());
        Eigen::MatrixXd zb = Eigen::MatrixXd::Zero(n_unique_beta_, a.cols());
#pragma omp parallel for schedule(static) if (n_unique_alpha_ > 32)
        for (int ai = 0; ai < n_unique_alpha_; ++ai)
            for (int g = sources.alpha_offsets[ai]; g < sources.alpha_offsets[ai + 1]; ++g)
                za.row(ai).noalias() += weights[g] * b.row(spin_products_[g] % n_unique_beta_);
#pragma omp parallel for schedule(static) if (n_unique_beta_ > 32)
        for (int bi = 0; bi < n_unique_beta_; ++bi)
            for (const int g : sources.beta_groups[bi])
                zb.row(bi).noalias() += weights[g] * a.row(spin_products_[g] / n_unique_beta_);
        ca.apply_left(&za);
        cb.apply_left(&zb);
        return UpperPair{std::move(za), std::move(zb)};
    };
    for (int begin = 0; begin < static_cast<int>(sources.columns.size()); begin += extent) {
        const int count = std::min<int>(extent, sources.columns.size() - begin);
        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n_unique_alpha_, count), du = u, ub = u, uc = u;
        Eigen::MatrixXd v = Eigen::MatrixXd::Zero(n_unique_beta_, count), dv = v, vc = v;
        for (int j = 0; j < count; ++j) {
            const int b = sources.columns[begin + j];
            v(b, j) = vc(b, j) = 1;
            for (const int g : sources.beta_groups[b]) {
                const int a = spin_products_[g] / n_unique_beta_;
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
            // Z_a for the first term also supplies the reference pullback/image;
            // Z_b for the second term supplies the reference beta pullback.
            auto first = seeds(ub, v);
            auto second = seeds(u, dv);
            const auto metric = seeds(uc, vc);
            // Linearity permits one link traversal per current-spin chart. Both
            // metric sides remain: they are different oriented bilinear moments.
            // All factor panels, including these six seeds, fit the existing 16
            // (U_alpha+U_beta)-column budget; no product carrier is introduced.
            const ExteriorGeneratorTerm alpha_terms[]{{ub, first.left},
                                                      {u, second.left},
                                                      {uc, metric.left, -1.0},
                                                      {metric.left, uc, -1.0}};
            const ExteriorGeneratorTerm beta_terms[]{{v, first.right},
                                                     {dv, second.right},
                                                     {vc, metric.right, -1.0},
                                                     {metric.right, vc, -1.0}};
            result.charts.left +=
                ca.generator_pullback(alpha_terms) + cb.generator_pullback(beta_terms);
            result.charts.right +=
                oa.generator_pullback(first.left, u) + ob.generator_pullback(second.right, v);
            oa.apply_adjoint_left(&first.left);
            for (int j = 0; j < count; ++j)
                for (const int g : sources.beta_groups[sources.columns[begin + j]]) {
                    const double value = first.left(spin_products_[g] / n_unique_beta_, j);
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
