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

void transform_right(const ExteriorOrbitalTransform& map,
                     const ExteriorTransformDirection* direction,
                     Factor& factor,
                     bool adjoint = false) {
    if (direction) {
        if (adjoint)
            map.apply_directional_adjoint_right(*direction, &factor.value, &factor.tangent);
        else
            map.apply_directional_right(*direction, &factor.value, &factor.tangent);
    } else {
        if (adjoint)
            map.apply_adjoint_right(&factor.value);
        else
            map.apply_right(&factor.value);
    }
}

Factor physical_product(const StructureAction& action,
                        const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                        const ExteriorOrbitalTransform& alpha,
                        const ExteriorOrbitalTransform& beta,
                        const ExteriorTransformDirection* da = nullptr,
                        const ExteriorTransformDirection* db = nullptr) {
    Factor result(0, 0, false);
    result.value = action.expand_structure_block(coefficients);
    if (da)
        result.tangent = Eigen::MatrixXd::Zero(result.value.rows(), result.value.cols());
    transform(alpha, da, result);
    transform_right(beta, db, result);
    return result;
}

bool same_chart(const ExteriorOrbitalTransform& a, const ExteriorOrbitalTransform& b) {
    return a.n_orbitals() == b.n_orbitals() &&
           (a.orbital_transform().array() == b.orbital_transform().array()).all();
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

bool StructureAction::use_product_overlap(const ExteriorOrbitalTransform& alpha,
                                          const ExteriorOrbitalTransform& beta,
                                          std::size_t workspace_bytes,
                                          int images) const {
    const std::size_t products = static_cast<std::size_t>(n_unique_alpha_) * n_unique_beta_;
    if (products > workspace_bytes / (images * sizeof(double)))
        return false;
    if (spin_products_.size() == products)
        return true;
    const double a = n_unique_alpha_, b = n_unique_beta_, r = sources_->columns.size();
    const double sa = alpha.shear_pair_count(), sb = beta.shear_pair_count();
    // Common W expansion/projection work cancels. Compare ordered exterior
    // maps with source maps plus a length-r dot at each touched product.
    const double product = 2 * (sa * b + sb * a) + 4 * a * b;
    const double factor = 2 * (sa + sb) * r + 2 * (a + b) * r + spin_products_.size() * r;
    return product <= factor;
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
    if (use_product_overlap(la, lb, workspace_bytes, directional ? 2 : 1)) {
        Eigen::MatrixXd result(n_structures_, vectors.cols());
        for (Eigen::Index column = 0; column < vectors.cols(); ++column) {
            auto image = physical_product(*this, vectors.col(column), ra, rb, dra, drb);
            transform_right(lb, dlb, image, true);
            transform(la, dla, image, true);
            result.col(column) =
                contract_spin_product_block(directional ? image.tangent : image.value);
        }
        return result;
    }
    // Column-singleton support (including covalent structures) has one alpha string
    // per touched beta. Its scalar coefficient commutes with both exterior
    // maps. Generate each common basis image once, not once per CI RHS.
    const bool singleton = std::all_of(sources.columns.begin(), sources.columns.end(), [&](int b) {
        return sources.beta_groups[b].size() == 1;
    });
    if (singleton && vectors.cols() > 1) {
        const std::size_t groups = spin_products_.size();
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
            for (int begin = 0; begin < static_cast<int>(groups); begin += extent) {
                const int count = std::min<int>(extent, groups - begin);
                Factor a(n_unique_alpha_, count, directional),
                    b(n_unique_beta_, count, directional);
                for (int j = 0; j < count; ++j) {
                    a.value(spin_products_[begin + j] / n_unique_beta_, j) = 1;
                    b.value(spin_products_[begin + j] % n_unique_beta_, j) = 1;
                }
                transform(ra, dra, a);
                transform(la, dla, a, true);
                transform(rb, drb, b);
                transform(lb, dlb, b, true);
                Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(n_structures_, count);
#pragma omp parallel for schedule(static) if (n_structures_ > 32)
                for (int s = 0; s < n_structures_; ++s)
                    for (auto t = structure_term_offsets_[s]; t < structure_term_offsets_[s + 1];
                         ++t) {
                        const auto& term = spin_terms_[structure_term_indices_[t]];
                        const int ai = spin_products_[term.group] / n_unique_beta_;
                        const int bi = spin_products_[term.group] % n_unique_beta_;
                        if (directional)
                            projected.row(s).array() +=
                                term.coefficient *
                                (a.tangent.row(ai).array() * b.value.row(bi).array() +
                                 a.value.row(ai).array() * b.tangent.row(bi).array());
                        else
                            projected.row(s).array() += term.coefficient * a.value.row(ai).array() *
                                                        b.value.row(bi).array();
                    }
                for (int first = 0; first < vectors.cols(); first += width) {
                    const int count_rhs = std::min<int>(width, vectors.cols() - first);
                    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(count, count_rhs);
#pragma omp parallel for schedule(static) if (count > 32)
                    for (int j = 0; j < count; ++j)
                        for (auto t = spin_term_offsets_[begin + j];
                             t < spin_term_offsets_[begin + j + 1];
                             ++t) {
                            const auto& term = spin_terms_[t];
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
    for (int column = 0; column < vectors.cols(); ++column) {
        Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(spin_products_.size());
        Eigen::VectorXd products = Eigen::VectorXd::Zero(spin_products_.size());
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
        for (int group = 0; group < static_cast<int>(spin_products_.size()); ++group)
            for (std::size_t term = spin_term_offsets_[group]; term < spin_term_offsets_[group + 1];
                 ++term)
                coefficients[group] +=
                    spin_terms_[term].coefficient * vectors(spin_terms_[term].structure, column);

        for (int begin = 0; begin < static_cast<int>(sources.columns.size()); begin += extent) {
            const int count = std::min<int>(extent, sources.columns.size() - begin);
            Factor a(n_unique_alpha_, count, directional), b(n_unique_beta_, count, directional);
            for (int j = 0; j < count; ++j) {
                const int beta = sources.columns[begin + j];
                b.value(beta, j) = 1.0;
                for (const int group : sources.beta_groups[beta])
                    a.value(spin_products_[group] / n_unique_beta_, j) = coefficients[group];
            }
            // K_alpha X_J and K_beta E_J, with K_sigma = T_left^T T_right.
            // Associativity removes the transformed alpha-by-beta product entirely.
            transform(ra, dra, a);
            transform(la, dla, a, true);
            transform(rb, drb, b);
            transform(lb, dlb, b, true);
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
            for (int group = 0; group < static_cast<int>(spin_products_.size()); ++group) {
                const int alpha = spin_products_[group] / n_unique_beta_;
                const int beta = spin_products_[group] % n_unique_beta_;
                products[group] += directional ? a.tangent.row(alpha).dot(b.value.row(beta)) +
                                                     a.value.row(alpha).dot(b.tangent.row(beta))
                                               : a.value.row(alpha).dot(b.value.row(beta));
            }
        }
#pragma omp parallel for schedule(static) if (n_structures_ > 32)
        for (int structure = 0; structure < n_structures_; ++structure)
            for (std::size_t offset = structure_term_offsets_[structure];
                 offset < structure_term_offsets_[structure + 1];
                 ++offset) {
                const auto& term = spin_terms_[structure_term_indices_[offset]];
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
    const bool dl = directional && !delta_left->isZero(0.0);
    const bool dr = directional && !delta_right->isZero(0.0);
    if (same_chart(la, lb) && same_chart(ra, rb) &&
        use_product_overlap(
            la, lb, workspace_bytes, 2 + static_cast<int>(dl) + static_cast<int>(dr))) {
        if (directional && !dl && !dr)
            return {left_requested ? Eigen::MatrixXd::Zero(la.n_orbitals(), la.n_orbitals()).eval()
                                   : Eigen::MatrixXd{},
                    Eigen::MatrixXd::Zero(ra.n_orbitals(), ra.n_orbitals())};
        ExteriorTransformDirection dla, dlb, dra, drb;
        if (dl) {
            dla = la.direction(*delta_left);
            dlb = lb.direction(*delta_left);
        }
        if (dr) {
            dra = ra.direction(*delta_right);
            drb = rb.direction(*delta_right);
        }
        const auto l =
            physical_product(*this, left, la, lb, dl ? &dla : nullptr, dl ? &dlb : nullptr);
        const auto r =
            physical_product(*this, right, ra, rb, dr ? &dra : nullptr, dr ? &drb : nullptr);
        // D(R,L)_{pq}=<R|E_pq|L>; the opposite orientation is D(R,L)^T.
        const Eigen::MatrixXd moment = carrier_generator_adjoint(r.value, l.value);
        const Eigen::MatrixXd transpose = moment.transpose();
        if (!directional)
            return {left_requested ? la.moment_pullback(moment) : Eigen::MatrixXd{},
                    ra.moment_pullback(transpose)};
        Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(moment.rows(), moment.cols());
        if (dl)
            delta += carrier_generator_adjoint(r.value, l.tangent);
        if (dr)
            delta += carrier_generator_adjoint(r.tangent, l.value);
        const Eigen::MatrixXd delta_transpose = delta.transpose();
        return {left_requested ? la.moment_pullback(moment, &delta, delta_left) : Eigen::MatrixXd{},
                ra.moment_pullback(transpose, &delta_transpose, delta_right)};
    }
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
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(spin_products_.size(), 2);
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
    for (int group = 0; group < static_cast<int>(spin_products_.size()); ++group)
        for (std::size_t t = spin_term_offsets_[group]; t < spin_term_offsets_[group + 1]; ++t) {
            const auto& term = spin_terms_[t];
            coefficients(group, 0) += term.coefficient * left[term.structure];
            coefficients(group, 1) += term.coefficient * right[term.structure];
        }
    UpperPair result{left_requested ? Eigen::MatrixXd::Zero(la.n_orbitals(), la.n_orbitals()).eval()
                                    : Eigen::MatrixXd{},
                     Eigen::MatrixXd::Zero(ra.n_orbitals(), ra.n_orbitals())};
    for (int begin = 0; begin < static_cast<int>(sources.columns.size()); begin += extent) {
        const int count = std::min<int>(extent, sources.columns.size() - begin);
        Factor u(n_unique_alpha_, count, directional), v(n_unique_beta_, count, directional);
        for (int j = 0; j < count; ++j) {
            const int beta = sources.columns[begin + j];
            v.value(beta, j) = 1.0;
            for (const int group : sources.beta_groups[beta])
                u.value(spin_products_[group] / n_unique_beta_, j) = coefficients(group, 1);
        }
        transform(ra, directional ? &dra : nullptr, u);
        transform(rb, directional ? &drb : nullptr, v);
        Factor a = u, b = v;
        transform(la, directional ? &dla : nullptr, a, true);
        transform(lb, directional ? &dlb : nullptr, b, true);
        Factor za(n_unique_alpha_, count, directional), zb(n_unique_beta_, count, directional);
        // Seeds X_left B and X_left^T A use only the original touched products.
#pragma omp parallel for schedule(static) if (n_unique_alpha_ > 32)
        for (int alpha = 0; alpha < n_unique_alpha_; ++alpha)
            for (int group = sources.alpha_offsets[alpha]; group < sources.alpha_offsets[alpha + 1];
                 ++group) {
                const int beta = spin_products_[group] % n_unique_beta_;
                za.value.row(alpha).noalias() += coefficients(group, 0) * b.value.row(beta);
                if (directional)
                    za.tangent.row(alpha).noalias() += coefficients(group, 0) * b.tangent.row(beta);
            }
#pragma omp parallel for schedule(static) if (n_unique_beta_ > 32)
        for (int beta = 0; beta < n_unique_beta_; ++beta)
            for (const int group : sources.beta_groups[beta]) {
                const int alpha = spin_products_[group] / n_unique_beta_;
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
