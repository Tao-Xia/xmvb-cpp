#include "core/index.hpp"
#include <cstdint>
#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace xmvb::vb {
namespace {

/** @brief A bounded single-spin factor and, when requested, its tangent. */
struct Factor {
    Eigen::MatrixXd value, tangent;
    Factor(int rows, int columns, bool directional) : value(Eigen::MatrixXd::Zero(rows, columns)) {
        if (directional)
            tangent = value;
    }
};

void transform(const ExteriorOrbitalTransform& map,
               const ExteriorTransformDirection* direction,
               Factor& factor,
               bool adjoint = false) {
    if (direction) {
        if (adjoint)
            map.apply_directional_adjoint_left(*direction, &factor.value, &factor.tangent);
        else
            map.apply_directional_left(*direction, &factor.value, &factor.tangent);
    } else {
        if (adjoint)
            map.apply_adjoint_left(&factor.value);
        else
            map.apply_left(&factor.value);
    }
}

int panel_width(std::size_t bytes, int na, int nb, int columns, int copies) {
    const std::size_t row_bytes = copies * sizeof(double) * (static_cast<std::size_t>(na) + nb);
    if (bytes < row_bytes)
        throw std::length_error("cross overlap budget cannot hold one factor column");
    return static_cast<int>(std::max<std::size_t>(
        1,
        std::min<std::size_t>(
            columns,
            std::min(bytes / row_bytes,
                     static_cast<std::size_t>(effective_openmp_thread_count()) * 8))));
}

} // namespace

bool StructureAction::shared_overlap_sources() const noexcept {
    return sources_ &&
           std::all_of(sources_->columns.begin(), sources_->columns.end(), [&](Eigen::Index b) {
               return sources_->beta_groups[b].size() == 1;
           });
}

Eigen::MatrixXd StructureAction::cross_overlap(const Eigen::Ref<const Eigen::MatrixXd>& vectors,
                                               const ExteriorOrbitalTransform& la,
                                               const ExteriorOrbitalTransform& lb,
                                               const ExteriorOrbitalTransform& ra,
                                               const ExteriorOrbitalTransform& rb,
                                               const ExteriorTransformDirection* dla,
                                               const ExteriorTransformDirection* dlb,
                                               const ExteriorTransformDirection* dra,
                                               const ExteriorTransformDirection* drb,
                                               std::size_t workspace_bytes) const {
    const bool directional = dla || dlb || dra || drb;
    if (vectors.rows() != n_structures_ || vectors.cols() <= 0 ||
        la.dimension() != n_unique_alpha_ || ra.dimension() != n_unique_alpha_ ||
        lb.dimension() != n_unique_beta_ || rb.dimension() != n_unique_beta_ ||
        (directional && !(dla && dlb && dra && drb)))
        throw std::invalid_argument("invalid selected cross overlap");
    const auto& sources = *sources_;
    // Column-singleton support (including covalent structures) has one alpha string
    // per touched beta. Its scalar coefficient commutes with both exterior
    // maps. Generate each common basis image once, not once per CI RHS.
    const bool singleton = shared_overlap_sources();
    if (singleton && vectors.cols() > 1) {
        const std::size_t groups = topology_->products.size();
        const std::size_t factor_rows =
            (directional ? 4 : 2) * (static_cast<std::size_t>(n_unique_alpha_) + n_unique_beta_);
        // Source factors do not depend on the CI RHS. Project each factor panel
        // to structures once, then apply all RHS batches. Associativity removes
        // both groups-by-RHS arrays and repeated exterior maps across RHS batches.
        const std::size_t capacity = workspace_bytes / sizeof(double);
        if (capacity > factor_rows + static_cast<std::size_t>(n_structures_)) {
            const int width =
                std::min<std::size_t>(vectors.cols(), capacity - factor_rows - n_structures_);
            const int extent = std::min<std::size_t>(
                groups,
                std::min<std::size_t>(8 * effective_openmp_thread_count(),
                                      capacity / (factor_rows + n_structures_ + width)));
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_structures_, vectors.cols());
            for (std::int64_t begin = 0; begin < static_cast<std::int64_t>(groups);
                 begin += extent) {
                const std::int64_t count = std::min<int>(extent, groups - begin);
                Factor a(n_unique_alpha_, count, directional),
                    b(n_unique_beta_, count, directional);
                for (int j = 0; j < count; ++j) {
                    a.value(topology_->products[begin + j] / n_unique_beta_, j) = 1;
                    b.value(topology_->products[begin + j] % n_unique_beta_, j) = 1;
                }
                transform(ra, dra, a);
                transform(la, dla, a, true);
                transform(rb, drb, b);
                transform(lb, dlb, b, true);
                Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(n_structures_, count);
#pragma omp parallel for schedule(static) if (n_structures_ > 32)
                for (Eigen::Index s = 0; s < n_structures_; ++s)
                    for (auto t = topology_->structure_offsets[s];
                         t < topology_->structure_offsets[s + 1];
                         ++t) {
                        const auto& term = topology_->terms[topology_->structure_terms[t]];
                        const std::int64_t ai = topology_->products[term.group] / n_unique_beta_;
                        const std::int64_t bi = topology_->products[term.group] % n_unique_beta_;
                        if (directional)
                            projected.row(s).array() +=
                                term.coefficient *
                                (a.tangent.row(ai).array() * b.value.row(bi).array() +
                                 a.value.row(ai).array() * b.tangent.row(bi).array());
                        else
                            projected.row(s).array() += term.coefficient * a.value.row(ai).array() *
                                                        b.value.row(bi).array();
                    }
                for (std::int64_t first = 0; first < vectors.cols(); first += width) {
                    const int count_rhs = std::min<int>(width, vectors.cols() - first);
                    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(count, count_rhs);
#pragma omp parallel for schedule(static) if (count > 32)
                    for (int j = 0; j < count; ++j)
                        for (auto t = topology_->offsets[begin + j];
                             t < topology_->offsets[begin + j + 1];
                             ++t) {
                            const auto& term = topology_->terms[t];
                            coefficients.row(j).noalias() +=
                                term.coefficient *
                                vectors.row(term.structure).segment(first, count_rhs);
                        }
                    result.middleCols(first, count_rhs).noalias() += projected * coefficients;
                }
            }
            return result;
        }
    }
    const int extent = panel_width(workspace_bytes,
                                   n_unique_alpha_,
                                   n_unique_beta_,
                                   sources.columns.size(),
                                   directional ? 8 : 4);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_structures_, vectors.cols());
    for (std::int64_t column = 0; column < vectors.cols(); ++column) {
        Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(topology_->products.size());
        Eigen::VectorXd products = Eigen::VectorXd::Zero(topology_->products.size());
#pragma omp parallel for schedule(static) if (topology_->products.size() > 32)
        for (std::size_t group = 0; group < topology_->products.size(); ++group)
            for (std::size_t term = topology_->offsets[group]; term < topology_->offsets[group + 1];
                 ++term)
                coefficients[group] += topology_->terms[term].coefficient *
                                       vectors(topology_->terms[term].structure, column);

        for (std::int64_t begin = 0;
             begin < core::checked_index<std::int64_t>(sources.columns.size());
             begin += extent) {
            const std::int64_t count = std::min<int>(extent, sources.columns.size() - begin);
            Factor a(n_unique_alpha_, count, directional), b(n_unique_beta_, count, directional);
            for (int j = 0; j < count; ++j) {
                const std::int64_t beta = sources.columns[begin + j];
                b.value(beta, j) = 1.0;
                for (const auto group : sources.beta_groups[beta])
                    a.value(topology_->products[group] / n_unique_beta_, j) = coefficients[group];
            }
            // K_alpha X_J and K_beta E_J, with K_sigma = T_left^T T_right.
            // Associativity removes the transformed alpha-by-beta product entirely.
            transform(ra, dra, a);
            transform(la, dla, a, true);
            transform(rb, drb, b);
            transform(lb, dlb, b, true);
#pragma omp parallel for schedule(static) if (topology_->products.size() > 32)
            for (std::size_t group = 0; group < topology_->products.size(); ++group) {
                const std::int64_t alpha = topology_->products[group] / n_unique_beta_;
                const std::int64_t beta = topology_->products[group] % n_unique_beta_;
                products[group] += directional ? a.tangent.row(alpha).dot(b.value.row(beta)) +
                                                     a.value.row(alpha).dot(b.tangent.row(beta))
                                               : a.value.row(alpha).dot(b.value.row(beta));
            }
        }
#pragma omp parallel for schedule(static) if (n_structures_ > 32)
        for (Eigen::Index structure = 0; structure < n_structures_; ++structure)
            for (std::size_t offset = topology_->structure_offsets[structure];
                 offset < topology_->structure_offsets[structure + 1];
                 ++offset) {
                const auto& term = topology_->terms[topology_->structure_terms[offset]];
                result(structure, column) += term.coefficient * products[term.group];
            }
    }
    return result;
}

UpperPair StructureAction::cross_overlap_adjoint(const Eigen::VectorXd& left,
                                                 const Eigen::VectorXd& right,
                                                 const ExteriorOrbitalTransform& la,
                                                 const ExteriorOrbitalTransform& lb,
                                                 const ExteriorOrbitalTransform& ra,
                                                 const ExteriorOrbitalTransform& rb,
                                                 const Eigen::MatrixXd* delta_left,
                                                 const Eigen::MatrixXd* delta_right,
                                                 std::size_t workspace_bytes,
                                                 OverlapSide side) const {
    const bool directional = delta_left != nullptr;
    if (left.size() != n_structures_ || right.size() != n_structures_ ||
        directional != (delta_right != nullptr))
        throw std::invalid_argument("invalid cross overlap adjoint");
    // Scalar cross-Gram symmetry turns a left-only pullback into the same
    // right-only contraction, including both chart directions when present.
    if (side == OverlapSide::Left) {
        auto swapped = cross_overlap_adjoint(right,
                                             left,
                                             ra,
                                             rb,
                                             la,
                                             lb,
                                             delta_right,
                                             delta_left,
                                             workspace_bytes,
                                             OverlapSide::Right);
        return {std::move(swapped.right), {}};
    }
    const bool left_requested = side == OverlapSide::Both;
    const auto& sources = *sources_;
    const int extent = panel_width(workspace_bytes,
                                   n_unique_alpha_,
                                   n_unique_beta_,
                                   sources.columns.size(),
                                   directional ? 16 : 8);
    ExteriorTransformDirection dla, dlb, dra, drb;
    if (directional) {
        dla = la.direction(*delta_left);
        dlb = lb.direction(*delta_left);
        dra = ra.direction(*delta_right);
        drb = rb.direction(*delta_right);
    }
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(topology_->products.size(), 2);
#pragma omp parallel for schedule(static) if (topology_->products.size() > 32)
    for (std::size_t group = 0; group < topology_->products.size(); ++group)
        for (std::size_t t = topology_->offsets[group]; t < topology_->offsets[group + 1]; ++t) {
            const auto& term = topology_->terms[t];
            coefficients(group, 0) += term.coefficient * left[term.structure];
            coefficients(group, 1) += term.coefficient * right[term.structure];
        }
    UpperPair result{left_requested ? Eigen::MatrixXd::Zero(la.n_orbitals(), la.n_orbitals()).eval()
                                    : Eigen::MatrixXd{},
                     Eigen::MatrixXd::Zero(ra.n_orbitals(), ra.n_orbitals())};
    for (std::int64_t begin = 0; begin < core::checked_index<std::int64_t>(sources.columns.size());
         begin += extent) {
        const std::int64_t count = std::min<int>(extent, sources.columns.size() - begin);
        Factor u(n_unique_alpha_, count, directional), v(n_unique_beta_, count, directional);
        for (int j = 0; j < count; ++j) {
            const std::int64_t beta = sources.columns[begin + j];
            v.value(beta, j) = 1.0;
            for (const auto group : sources.beta_groups[beta])
                u.value(topology_->products[group] / n_unique_beta_, j) = coefficients(group, 1);
        }
        transform(ra, directional ? &dra : nullptr, u);
        transform(rb, directional ? &drb : nullptr, v);
        Factor a = u, b = v;
        transform(la, directional ? &dla : nullptr, a, true);
        transform(lb, directional ? &dlb : nullptr, b, true);
        Factor za(n_unique_alpha_, count, directional), zb(n_unique_beta_, count, directional);
        // Seeds X_left B and X_left^T A use only the original touched products.
#pragma omp parallel for schedule(static) if (n_unique_alpha_ > 32)
        for (std::int64_t alpha = 0; alpha < n_unique_alpha_; ++alpha)
            for (std::size_t group = sources.alpha_offsets[alpha];
                 group < sources.alpha_offsets[alpha + 1];
                 ++group) {
                const std::int64_t beta = topology_->products[group] % n_unique_beta_;
                za.value.row(alpha).noalias() += coefficients(group, 0) * b.value.row(beta);
                if (directional)
                    za.tangent.row(alpha).noalias() += coefficients(group, 0) * b.tangent.row(beta);
            }
#pragma omp parallel for schedule(static) if (n_unique_beta_ > 32)
        for (std::int64_t beta = 0; beta < n_unique_beta_; ++beta)
            for (const auto group : sources.beta_groups[beta]) {
                const std::int64_t alpha = topology_->products[group] / n_unique_beta_;
                zb.value.row(beta).noalias() += coefficients(group, 0) * a.value.row(alpha);
                if (directional)
                    zb.tangent.row(beta).noalias() += coefficients(group, 0) * a.tangent.row(alpha);
            }
        transform(la, directional ? &dla : nullptr, za);
        transform(lb, directional ? &dlb : nullptr, zb);
        const auto pullback = [&](const ExteriorOrbitalTransform& map,
                                  const Factor& seed,
                                  const Factor& image,
                                  const Eigen::MatrixXd* delta) {
            return map.generator_pullback(seed.value,
                                          image.value,
                                          directional ? &seed.tangent : nullptr,
                                          directional ? &image.tangent : nullptr,
                                          delta);
        };
        if (left_requested)
            result.left += pullback(la, u, za, delta_left) + pullback(lb, v, zb, delta_left);
        result.right += pullback(ra, za, u, delta_right) + pullback(rb, zb, v, delta_right);
    }
    return result;
}

} // namespace xmvb::vb
