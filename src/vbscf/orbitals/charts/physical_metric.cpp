#include "vbscf/orbitals/charts/physical_metric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>

namespace xmvb::vb {

OrbitalPhysicalMetric::OrbitalPhysicalMetric(const OrbitalPreparationInput& input,
                                             bool active_subspace_gauge)
    : n_basis_functions_(input.n_basis_functions),
      n_inactive_((input.n_total_electrons - input.n_active_electrons) / 2),
      n_active_(input.n_active_orbitals), active_subspace_gauge_(active_subspace_gauge) {
    if (n_basis_functions_ <= 0 || n_inactive_ < 0 || n_active_ < 0 ||
        n_inactive_ + n_active_ > input.n_orbitals) {
        throw std::invalid_argument("invalid orbital partition in physical metric");
    }
    overlap_ = Eigen::Map<const Eigen::MatrixXd>(
        input.ao_overlap_matrix.data(), n_basis_functions_, n_basis_functions_);
    if (!overlap_.allFinite()) {
        throw std::invalid_argument("non-finite AO overlap in physical metric");
    }
    Eigen::LLT<Eigen::MatrixXd> ao_factor(overlap_);
    if (ao_factor.info() != Eigen::Success) {
        throw std::runtime_error("AO overlap is not positive definite");
    }
    ao_metric_upper_ = ao_factor.matrixU();

    inactive_ = Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
    active_ = Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
    orbital_supports_.resize(n_inactive_ + n_active_);
    for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
        const int count = stored_sparse_orbital_coefficient_count(input, orbital);
        orbital_supports_[orbital].resize(count);
        for (int slot = 0; slot < count; ++slot) {
            const int flat = orbital * n_basis_functions_ + slot;
            const int basis = input.orbital_basis_index_table[flat] - 1;
            if (basis < 0 || basis >= n_basis_functions_) {
                throw std::invalid_argument("invalid sparse AO index in physical metric");
            }
            orbital_supports_[orbital][slot] = basis;
            if (orbital < n_inactive_) {
                inactive_(basis, orbital) = input.orbital_value_table[flat];
            } else {
                active_(basis, orbital - n_inactive_) = input.orbital_value_table[flat];
            }
        }
    }

    complement_ = Eigen::MatrixXd::Identity(n_basis_functions_, n_basis_functions_);
    if (n_inactive_ > 0) {
        const Eigen::MatrixXd inactive_metric = inactive_.transpose() * overlap_ * inactive_;
        Eigen::LDLT<Eigen::MatrixXd> factor(inactive_metric);
        if (factor.info() != Eigen::Success || !factor.isPositive()) {
            throw std::runtime_error("singular inactive span in physical metric");
        }
        inactive_inverse_ = factor.solve(Eigen::MatrixXd::Identity(n_inactive_, n_inactive_));
        if (!inactive_inverse_.allFinite()) {
            throw std::runtime_error("non-finite inactive metric inverse");
        }
        complement_.noalias() -= inactive_ * inactive_inverse_ * inactive_.transpose() * overlap_;
        Eigen::LLT<Eigen::MatrixXd> inactive_factor(inactive_metric);
        if (inactive_factor.info() != Eigen::Success) {
            throw std::runtime_error("inactive overlap is not positive definite");
        }
        const Eigen::MatrixXd upper = inactive_factor.matrixU();
        inactive_inverse_upper_ = upper.triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_inactive_, n_inactive_));
    }
    projected_active_ = complement_ * active_;
    if (active_subspace_gauge_ && n_active_ > 0) {
        const Eigen::MatrixXd gram = projected_active_.transpose() * overlap_ * projected_active_;
        Eigen::LLT<Eigen::MatrixXd> factor(gram);
        if (factor.info() != Eigen::Success) {
            throw std::runtime_error("singular active subspace in physical metric");
        }
        active_inverse_ = factor.solve(Eigen::MatrixXd::Identity(n_active_, n_active_));
        const Eigen::MatrixXd upper = factor.matrixU();
        active_inverse_upper_ = upper.triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active_, n_active_));
        active_complement_ = Eigen::MatrixXd::Identity(n_basis_functions_, n_basis_functions_);
        active_complement_.noalias() -=
            projected_active_ * active_inverse_ * projected_active_.transpose() * overlap_;
    }
    projected_norm_squared_ = Eigen::VectorXd::Zero(n_active_);
    for (int active = 0; active < n_active_; ++active) {
        projected_norm_squared_[active] =
            projected_active_.col(active).dot(overlap_ * projected_active_.col(active));
        if (!(projected_norm_squared_[active] > 0.0) ||
            !std::isfinite(projected_norm_squared_[active])) {
            throw std::runtime_error("degenerate projected active ray in physical metric");
        }
    }
    if (n_active_ > 0) {
        overlap_active_ = overlap_ * active_;
        overlap_projected_active_ = overlap_ * projected_active_;
        transposed_overlap_projected_active_ = overlap_.transpose() * projected_active_;
    }
    if (n_inactive_ > 0 && n_active_ > 0) {
        const Eigen::MatrixXd inactive_active_overlap = inactive_.transpose() * overlap_ * active_;
        inactive_inverse_active_overlap_ = inactive_inverse_ * inactive_active_overlap;
        inactive_times_inverse_ = inactive_ * inactive_inverse_;
    }
}

Eigen::VectorXd OrbitalPhysicalMetric::physical_feature(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
    if (packed_direction.size() != layout.size() || !packed_direction.allFinite()) {
        throw std::invalid_argument("invalid packed direction in physical metric");
    }
    Eigen::MatrixXd delta_inactive = Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
    Eigen::MatrixXd delta_active = Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
    for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
        const int count = layout.orbital_coefficient_count(orbital);
        if (count > static_cast<int>(orbital_supports_[orbital].size())) {
            throw std::invalid_argument("metric layout exceeds orbital support");
        }
        for (int slot = 0; slot < count; ++slot) {
            const int packed = layout.packed_index(orbital, slot);
            const int basis = orbital_supports_[orbital][slot];
            if (orbital < n_inactive_) {
                delta_inactive(basis, orbital) = packed_direction[packed];
            } else {
                delta_active(basis, orbital - n_inactive_) = packed_direction[packed];
            }
        }
    }

    const Eigen::MatrixXd horizontal_inactive = complement_ * delta_inactive;
    Eigen::VectorXd feature = Eigen::VectorXd::Zero(n_basis_functions_ * (n_inactive_ + n_active_));
    if (n_inactive_ > 0) {
        Eigen::Map<Eigen::MatrixXd>(feature.data(), n_basis_functions_, n_inactive_) =
            ao_metric_upper_ * horizontal_inactive * inactive_inverse_upper_;
    }

    Eigen::MatrixXd delta_projected_active = complement_ * delta_active;
    if (n_inactive_ > 0 && n_active_ > 0) {
        const Eigen::MatrixXd horizontal_active_overlap =
            horizontal_inactive.transpose() * overlap_active_;
        delta_projected_active.noalias() -= horizontal_inactive * inactive_inverse_active_overlap_;
        delta_projected_active.noalias() -= inactive_times_inverse_ * horizontal_active_overlap;
    }
    if (active_subspace_gauge_ && n_active_ > 0) {
        // The complete CAS depends on the active span, not on individual rays.
        // Apply each AO operator to the narrow tangent, never form L*Q_a.
        const Eigen::MatrixXd horizontal_active = active_complement_ * delta_projected_active;
        Eigen::Map<Eigen::MatrixXd>(
            feature.data() + n_basis_functions_ * n_inactive_, n_basis_functions_, n_active_) =
            ao_metric_upper_ * horizontal_active * active_inverse_upper_;
        return feature;
    }
    for (int active = 0; active < n_active_; ++active) {
        const auto projected = projected_active_.col(active);
        Eigen::VectorXd horizontal = delta_projected_active.col(active);
        horizontal.noalias() -=
            projected * (transposed_overlap_projected_active_.col(active).dot(horizontal) /
                         projected_norm_squared_[active]);
        feature.segment(n_basis_functions_ * (n_inactive_ + active), n_basis_functions_) =
            ao_metric_upper_ * horizontal / std::sqrt(projected_norm_squared_[active]);
    }
    return feature;
}

Eigen::MatrixXd
OrbitalPhysicalMetric::orbital_block(int orbital,
                                     const std::vector<int>& support,
                                     const Eigen::Ref<const Eigen::MatrixXd>& lift) const {
    if (orbital < 0 || orbital >= n_inactive_ + n_active_ ||
        lift.rows() != static_cast<Eigen::Index>(support.size()) || !lift.allFinite())
        throw std::invalid_argument("invalid local physical metric lift");
    const Eigen::Index size = lift.rows();
    for (const int ao : support)
        if (ao < 0 || ao >= n_basis_functions_)
            throw std::invalid_argument("invalid local physical metric support");
    Eigen::MatrixXd local_overlap(size, size);
    Eigen::MatrixXd inactive_overlap(n_inactive_, size);
    Eigen::MatrixXd active_overlap(n_active_, size);
    for (Eigen::Index row = 0; row < size; ++row) {
        const int ao = support[row];
        for (Eigen::Index column = 0; column < size; ++column)
            local_overlap(row, column) = overlap_(ao, support[column]);
        for (int i = 0; i < n_inactive_; ++i)
            inactive_overlap(i, row) = inactive_.col(i).dot(overlap_.col(ao));
        for (int j = 0; j < n_active_; ++j)
            active_overlap(j, row) = overlap_projected_active_(ao, j);
    }
    // H = U^T S P U and W = V^T S U. Only support-local matrices
    // and occupied-by-local contractions are formed, never AO-by-AO copies.
    const Eigen::MatrixXd projection = inactive_overlap * lift;
    Eigen::MatrixXd horizontal = lift.transpose() * local_overlap * lift;
    if (n_inactive_ > 0)
        horizontal.noalias() -= projection.transpose() * inactive_inverse_ * projection;
    const Eigen::MatrixXd rays = active_overlap * lift;
    Eigen::MatrixXd block;
    if (active_subspace_gauge_ && n_active_ > 0) {
        const Eigen::MatrixXd ray_gram = rays.transpose() * active_inverse_ * rays;
        if (orbital < n_inactive_) {
            const double weight = inactive_inverse_(orbital, orbital);
            const auto coupling = inactive_inverse_active_overlap_.row(orbital);
            const double feedback = (coupling * active_inverse_ * coupling.transpose())(0, 0);
            block = (weight + feedback) * horizontal + (weight - feedback) * ray_gram;
        } else {
            block = active_inverse_(orbital - n_inactive_, orbital - n_inactive_) *
                    (horizontal - ray_gram);
        }
    } else if (orbital < n_inactive_) {
        const double weight = inactive_inverse_(orbital, orbital);
        double horizontal_weight = weight;
        Eigen::VectorXd ray_weights(n_active_);
        for (int j = 0; j < n_active_; ++j) {
            const double norm = projected_norm_squared_[j];
            const double coupling = inactive_inverse_active_overlap_(orbital, j);
            const double feedback = coupling * coupling / norm;
            horizontal_weight += feedback;
            ray_weights[j] = (weight - feedback) / norm;
        }
        block = horizontal_weight * horizontal;
        if (n_active_ > 0)
            block.noalias() += rays.transpose() * ray_weights.asDiagonal() * rays;
    } else {
        const int j = orbital - n_inactive_;
        const double norm = projected_norm_squared_[j];
        block = horizontal / norm - rays.row(j).transpose() * rays.row(j) / (norm * norm);
    }
    return (0.5 * (block + block.transpose())).eval();
}

bool OrbitalPhysicalMetric::has_full_support(const SparseParameterLayout& layout) const {
    for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
        if (layout.orbital_coefficient_count(orbital) != n_basis_functions_)
            return false;
        std::vector<bool> present(n_basis_functions_, false);
        for (const int basis : orbital_supports_[orbital])
            present[basis] = true;
        if (std::find(present.begin(), present.end(), false) != present.end())
            return false;
    }
    return true;
}

double OrbitalPhysicalMetric::subspace_dual_norm(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_covector) const {
    if (!active_subspace_gauge_ || !has_full_support(layout) ||
        packed_covector.size() != layout.size() || !packed_covector.allFinite()) {
        throw std::invalid_argument("flag dual norm requires a full-AO quotient covector");
    }
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_ + n_active_);
    for (int orbital = 0; orbital < gradient.cols(); ++orbital) {
        for (int slot = 0; slot < n_basis_functions_; ++slot) {
            gradient(orbital_supports_[orbital][slot], orbital) =
                packed_covector[layout.packed_index(orbital, slot)];
        }
    }
    // AO whitening is only a metric coordinate change, not an orbital update.
    const Eigen::MatrixXd white_gradient =
        ao_metric_upper_.transpose().triangularView<Eigen::Lower>().solve(gradient);
    Eigen::MatrixXd core = Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
    Eigen::MatrixXd active = Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
    Eigen::MatrixXd gi = white_gradient.leftCols(n_inactive_);
    Eigen::MatrixXd ga = white_gradient.rightCols(n_active_);
    if (n_inactive_ > 0) {
        core.noalias() = ao_metric_upper_ * inactive_ * inactive_inverse_upper_;
        if (n_active_ > 0) {
            gi.noalias() += ga * inactive_inverse_active_overlap_.transpose();
        }
        gi = (gi * inactive_inverse_upper_.triangularView<Eigen::Upper>()
                       .solve(Eigen::MatrixXd::Identity(n_inactive_, n_inactive_))
                       .transpose())
                 .eval();
    }
    if (n_active_ > 0) {
        active.noalias() = ao_metric_upper_ * projected_active_ * active_inverse_upper_;
        ga = (ga * active_inverse_upper_.triangularView<Eigen::Upper>()
                       .solve(Eigen::MatrixXd::Identity(n_active_, n_active_))
                       .transpose())
                 .eval();
    }
    // Core--active motion enters both span projectors, hence its factor 1/2
    // in the inverse metric. External motions enter one projector each.
    const Eigen::MatrixXd cross = active.transpose() * gi - ga.transpose() * core;
    const Eigen::MatrixXd external_i =
        gi - core * (core.transpose() * gi) - active * (active.transpose() * gi);
    const Eigen::MatrixXd external_a =
        ga - core * (core.transpose() * ga) - active * (active.transpose() * ga);
    const double squared =
        external_i.squaredNorm() + external_a.squaredNorm() + 0.5 * cross.squaredNorm();
    if (!std::isfinite(squared))
        throw std::runtime_error("non-finite flag dual norm");
    return std::sqrt(squared);
}

double OrbitalPhysicalMetric::squared_norm(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
    const double metric = physical_feature(layout, packed_direction).squaredNorm();
    if (!std::isfinite(metric) || metric < -1.0e-10) {
        throw std::runtime_error("non-finite or negative coupled orbital metric");
    }
    return std::max(0.0, metric);
}

Eigen::VectorXd
OrbitalPhysicalMetric::apply(const SparseParameterLayout& layout,
                             const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
    // Keep the metric and its norm on one computational graph: G = J^T J.
    // Expanded projector identities need not hold to machine precision when
    // inactive/active representatives are ill conditioned.
    return feature_adjoint(layout, physical_feature(layout, packed_direction));
}

Eigen::VectorXd
OrbitalPhysicalMetric::feature_adjoint(const SparseParameterLayout& layout,
                                       const Eigen::Ref<const Eigen::VectorXd>& feature) const {
    const Eigen::Map<const Eigen::MatrixXd> inactive_feature(
        feature.data(), n_basis_functions_, n_inactive_);
    const Eigen::Map<const Eigen::MatrixXd> active_feature(
        feature.data() + n_basis_functions_ * n_inactive_, n_basis_functions_, n_active_);
    Eigen::MatrixXd inactive_cotangent =
        ao_metric_upper_.transpose() * inactive_feature * inactive_inverse_upper_.transpose();
    Eigen::MatrixXd active_cotangent = Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
    if (active_subspace_gauge_ && n_active_ > 0) {
        active_cotangent.noalias() =
            ao_metric_upper_.transpose() * active_feature * active_inverse_upper_.transpose();
        active_cotangent = (active_complement_.transpose() * active_cotangent).eval();
    } else {
        active_cotangent.noalias() = ao_metric_upper_.transpose() * active_feature;
        for (int active = 0; active < n_active_; ++active) {
            const double norm_squared = projected_norm_squared_[active];
            active_cotangent.col(active) /= std::sqrt(norm_squared);
            active_cotangent.col(active).noalias() -=
                transposed_overlap_projected_active_.col(active) *
                (projected_active_.col(active).dot(active_cotangent.col(active)) / norm_squared);
        }
    }
    if (n_inactive_ > 0 && n_active_ > 0) {
        inactive_cotangent.noalias() -=
            active_cotangent * inactive_inverse_active_overlap_.transpose();
        inactive_cotangent.noalias() -=
            overlap_active_ * (inactive_times_inverse_.transpose() * active_cotangent).transpose();
    }
    inactive_cotangent = (complement_.transpose() * inactive_cotangent).eval();
    active_cotangent = (complement_.transpose() * active_cotangent).eval();
    Eigen::VectorXd packed_action = Eigen::VectorXd::Zero(layout.size());
    for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
        const int count = layout.orbital_coefficient_count(orbital);
        for (int slot = 0; slot < count; ++slot) {
            const int packed = layout.packed_index(orbital, slot);
            const int basis = orbital_supports_[orbital][slot];
            packed_action[packed] = orbital < n_inactive_
                                        ? inactive_cotangent(basis, orbital)
                                        : active_cotangent(basis, orbital - n_inactive_);
        }
    }
    if (!packed_action.allFinite()) {
        throw std::runtime_error("non-finite physical metric action");
    }
    return packed_action;
}

} // namespace xmvb::vb
