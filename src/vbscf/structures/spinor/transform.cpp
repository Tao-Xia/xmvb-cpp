#include "vbscf/structures/spinor/transform.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xmvb::vb::spinor {

Transform::Transform(std::shared_ptr<const Strings> strings,
                     const Eigen::Ref<const Eigen::MatrixXcd>& upper)
    : strings_(std::move(strings)) {
    if (!strings_ || upper.rows() != strings_->n_spinor() || upper.cols() != upper.rows() ||
        !upper.allFinite()) {
        throw std::invalid_argument("invalid spinor exterior transform");
    }
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * upper.rows() * std::max(1.0, upper.norm());
    if (upper.triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm() > tolerance ||
        (upper.rows() > 0 && upper.diagonal().imag().cwiseAbs().maxCoeff() > tolerance) ||
        (upper.diagonal().real().array() <= 0.0).any()) {
        throw std::invalid_argument("exterior R must be upper with positive real diagonal");
    }
    scales_.resize(strings_->dimension());
    for (int source = 0; source < strings_->dimension(); ++source) {
        std::complex<double> scale = 1.0;
        const auto mask = strings_->masks()[source];
        for (int p = 0; p < strings_->n_spinor(); ++p) {
            if ((mask & (std::uint64_t{1} << p)) != 0) {
                scale *= upper(p, p);
            }
        }
        scales_[source] = scale;
    }
    if (!scales_.allFinite() || (scales_.cwiseAbs().array() == 0.0).any()) {
        throw std::invalid_argument("exterior diagonal scales overflow or underflow");
    }

    // R = U D. Eliminate columns of unit-upper U on the right; inverse
    // elementary factors act on amplitudes in the recorded elimination order.
    Eigen::MatrixXcd unit = upper;
    for (int q = 0; q < unit.cols(); ++q) {
        unit.col(q) /= upper(q, q);
    }
    for (int q = 1; q < unit.cols(); ++q) {
        for (int p = q - 1; p >= 0; --p) {
            const auto coefficient = unit(p, q);
            if (coefficient != std::complex<double>{}) {
                shears_.push_back({p, q, coefficient});
                unit.col(q).noalias() -= coefficient * unit.col(p);
            }
        }
    }
}

void Transform::validate(const Eigen::Ref<const Eigen::MatrixXcd>& block) const {
    if (block.rows() != strings_->dimension() || block.cols() <= 0 || !block.allFinite()) {
        throw std::invalid_argument("invalid exterior carrier block");
    }
}

void Transform::apply(Eigen::Ref<Eigen::MatrixXcd> block) const {
    validate(block);
    block.array().colwise() *= scales_.array();
    for (const auto& shear : shears_) {
        for (int source = 0; source < strings_->dimension(); ++source) {
            int sign = 0;
            const int target = strings_->substitution(source, shear.created, shear.removed, &sign);
            if (target >= 0) {
                block.row(target) +=
                    shear.coefficient * static_cast<double>(sign) * block.row(source);
            }
        }
    }
}

void Transform::apply_adjoint(Eigen::Ref<Eigen::MatrixXcd> block) const {
    validate(block);
    for (auto shear = shears_.rbegin(); shear != shears_.rend(); ++shear) {
        for (int source = 0; source < strings_->dimension(); ++source) {
            int sign = 0;
            const int target =
                strings_->substitution(source, shear->created, shear->removed, &sign);
            if (target >= 0) {
                block.row(source) +=
                    std::conj(shear->coefficient) * static_cast<double>(sign) * block.row(target);
            }
        }
    }
    block.array().colwise() *= scales_.conjugate().array();
}

} // namespace xmvb::vb::spinor
