#include "scf/optimization/orbitals.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "linalg/exponential.hpp"

namespace xmvb::scf {

OrbitalModel::OrbitalModel(const SpinMatrices& orbitals,
                           const std::array<int, 2>& occupied,
                           int channels,
                           double occupation,
                           const SpinMatrices& density,
                           const SpinMatrices& fock,
                           const ApplyUnrestrictedResponse& response)
    : orbitals_(orbitals), density_(density), response_(response), occupied_(occupied),
      channels_(channels), occupation_(occupation) {
    int size = 0;
    for (int spin = 0; spin < channels_; ++spin) {
        const int n = orbitals_[spin].cols(), no = occupied_[spin];
        if (no < 0 || no > n)
            throw std::invalid_argument("invalid orbital occupations");
        offsets_[spin] = size;
        sizes_[spin] = no * (n - no);
        size += sizes_[spin];
    }
    gradient_.resize(size);
    diagonal_.resize(size);
    for (int spin = 0; spin < channels_; ++spin) {
        const int n = orbitals_[spin].cols(), no = occupied_[spin], nv = n - no;
        const Eigen::MatrixXd f = orbitals_[spin].transpose() * fock[spin] * orbitals_[spin];
        foo_[spin] = f.topLeftCorner(no, no);
        fvv_[spin] = f.bottomRightCorner(nv, nv);
        Eigen::Map<Eigen::MatrixXd> g(gradient_.data() + offsets_[spin], nv, no);
        g = 2.0 * occupation_ * f.bottomLeftCorner(nv, no);
        Eigen::Map<Eigen::MatrixXd> diagonal(diagonal_.data() + offsets_[spin], nv, no);
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                diagonal(a, i) = 2.0 * occupation_ * (fvv_[spin](a, a) - foo_[spin](i, i));
    }
}

Eigen::MatrixXd OrbitalModel::generator(const Eigen::VectorXd& p, int spin) const {
    if (p.size() != gradient_.size())
        throw std::invalid_argument("orbital direction size mismatch");
    const int n = orbitals_[spin].cols(), no = occupied_[spin], nv = n - no;
    const Eigen::Map<const Eigen::MatrixXd> x(p.data() + offsets_[spin], nv, no);
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(n, n);
    k.bottomLeftCorner(nv, no) = x;
    k.topRightCorner(no, nv) = -x.transpose();
    return k;
}

SpinMatrices OrbitalModel::rotate(const Eigen::VectorXd& p) const {
    SpinMatrices result;
    for (int spin = 0; spin < channels_; ++spin)
        result[spin] = orbitals_[spin] * linalg::exponential(generator(p, spin));
    return result;
}

SpinMatrices OrbitalModel::density(const SpinMatrices& orbitals) const {
    SpinMatrices result;
    for (int spin = 0; spin < channels_; ++spin) {
        const auto occupied = orbitals[spin].leftCols(occupied_[spin]);
        result[spin] = occupation_ * (occupied * occupied.transpose()).eval();
    }
    return result;
}

Eigen::VectorXd OrbitalModel::apply(const Eigen::VectorXd& p) const {
    if (p.size() != gradient_.size())
        throw std::invalid_argument("orbital direction size mismatch");
    SpinMatrices delta;
    for (int spin = 0; spin < channels_; ++spin) {
        const int no = occupied_[spin], nv = orbitals_[spin].cols() - no;
        const Eigen::Map<const Eigen::MatrixXd> x(p.data() + offsets_[spin], nv, no);
        const Eigen::MatrixXd d = occupation_ * orbitals_[spin].rightCols(nv) * x *
                                  orbitals_[spin].leftCols(no).transpose();
        delta[spin] = d + d.transpose();
    }
    const auto df = response_(density_, delta);
    Eigen::VectorXd result(p.size());
    for (int spin = 0; spin < channels_; ++spin) {
        const int no = occupied_[spin], nv = orbitals_[spin].cols() - no;
        const Eigen::Map<const Eigen::MatrixXd> x(p.data() + offsets_[spin], nv, no);
        Eigen::Map<Eigen::MatrixXd> image(result.data() + offsets_[spin], nv, no);
        image =
            2.0 * occupation_ *
            (fvv_[spin] * x - x * foo_[spin] +
             orbitals_[spin].rightCols(nv).transpose() * df[spin] * orbitals_[spin].leftCols(no));
    }
    return result;
}

Eigen::VectorXd OrbitalModel::precondition(const Eigen::VectorXd& r, double shift) const {
    // PySCF SOSCF's gap preconditioner, not a modification of the physical HVP.
    Eigen::VectorXd result(r.size());
    for (Eigen::Index i = 0; i < r.size(); ++i) {
        const double d = diagonal_[i] - shift;
        result[i] = r[i] / (std::abs(d) < 1e-8 ? 1e-8 : d);
    }
    return result;
}

Eigen::VectorXd OrbitalModel::positive_precondition(const Eigen::VectorXd& r, double shift) const {
    return (r.array() / (diagonal_.array() - shift).abs().max(1e-8)).matrix();
}

Eigen::VectorXd OrbitalModel::pullback(const SpinMatrices& fock,
                                       const Eigen::VectorXd& p,
                                       const SpinMatrices& rotated) const {
    Eigen::VectorXd result(p.size());
    for (int spin = 0; spin < channels_; ++spin) {
        const int n = orbitals_[spin].cols(), no = occupied_[spin], nv = n - no;
        Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(n, n);
        raw.leftCols(no) = 2.0 * occupation_ * orbitals_[spin].transpose() * fock[spin] *
                           rotated[spin].leftCols(no);
        const auto b = linalg::exponential_pullback(generator(p, spin), raw);
        Eigen::Map<Eigen::MatrixXd> g(result.data() + offsets_[spin], nv, no);
        g = b.bottomLeftCorner(nv, no) - b.topRightCorner(no, nv).transpose();
    }
    return result;
}

Eigen::VectorXd OrbitalModel::displacement(const SpinMatrices& change,
                                           const Eigen::Ref<const Eigen::MatrixXd>& overlap) const {
    Eigen::VectorXd p(gradient_.size());
    for (int spin = 0; spin < channels_; ++spin) {
        const int no = occupied_[spin], nv = orbitals_[spin].cols() - no;
        Eigen::Map<Eigen::MatrixXd> x(p.data() + offsets_[spin], nv, no);
        x = (overlap * orbitals_[spin].rightCols(nv)).transpose() * change[spin] *
            (overlap * orbitals_[spin].leftCols(no)) / occupation_;
    }
    return p;
}

} // namespace xmvb::scf
