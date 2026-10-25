#include "vbscf/structures/spinor/basis.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::vb::spinor {

Basis::Basis(std::shared_ptr<const Strings> strings,
             Eigen::SparseMatrix<std::complex<double>> mapping)
    : strings_(std::move(strings)), mapping_(std::move(mapping)) {
    if (!strings_ || mapping_.rows() != strings_->dimension() || mapping_.cols() <= 0 ||
        mapping_.cols() > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid spinor VB structure map dimensions");
    }
    mapping_.makeCompressed();
    for (int col = 0; col < mapping_.outerSize(); ++col) {
        bool nonzero = false;
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator entry(mapping_, col); entry;
             ++entry) {
            if (!std::isfinite(entry.value().real()) || !std::isfinite(entry.value().imag())) {
                throw std::invalid_argument("nonfinite spinor VB structure coefficient");
            }
            nonzero = nonzero || entry.value() != std::complex<double>{};
        }
        if (!nonzero) {
            throw std::invalid_argument("zero spinor VB parent structure");
        }
    }
}

Eigen::MatrixXcd Basis::expand(const Eigen::Ref<const Eigen::MatrixXcd>& structures) const {
    if (structures.rows() != size() || structures.cols() <= 0 || !structures.allFinite()) {
        throw std::invalid_argument("invalid spinor VB input block");
    }
    return mapping_ * structures;
}

Eigen::MatrixXcd Basis::contract(const Eigen::Ref<const Eigen::MatrixXcd>& carrier) const {
    if (carrier.rows() != strings_->dimension() || carrier.cols() <= 0 || !carrier.allFinite()) {
        throw std::invalid_argument("invalid spinor carrier contraction block");
    }
    return mapping_.adjoint() * carrier;
}

} // namespace xmvb::vb::spinor
