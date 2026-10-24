#include "casscf/orbitals/rotation.hpp"

#include <stdexcept>
#include "linalg/exponential.hpp"

namespace xmvb::casscf {

RotationChart::RotationChart(int nmo, int ncore, int nactive) : nmo_(nmo) {
    if (nmo < 1 || ncore < 0 || nactive < 1 || ncore + nactive > nmo)
        throw std::invalid_argument("invalid complete-CAS orbital partition");
    const auto sector = [&](int i) { return i < ncore ? 0 : (i < ncore + nactive ? 1 : 2); };
    for (int i = 0; i < nmo; ++i)
        for (int j = i + 1; j < nmo; ++j)
            if (sector(i) != sector(j))
                pairs_.emplace_back(i, j);
}

void RotationChart::check_matrix(const Eigen::MatrixXd& matrix) const {
    if (matrix.rows() != nmo_ || matrix.cols() != nmo_ || !matrix.allFinite())
        throw std::invalid_argument("rotation chart matrix shape mismatch");
}

Eigen::MatrixXd RotationChart::generator(const Eigen::VectorXd& parameters) const {
    if (parameters.size() != size() || !parameters.allFinite())
        throw std::invalid_argument("rotation parameter shape mismatch");
    Eigen::MatrixXd skew = Eigen::MatrixXd::Zero(nmo_, nmo_);
    for (int k = 0; k < size(); ++k) {
        const auto [i, j] = pairs_[k];
        skew(j, i) = parameters[k];
        skew(i, j) = -parameters[k];
    }
    return skew;
}

Eigen::MatrixXd RotationChart::retract(const Eigen::MatrixXd& orbitals,
                                       const Eigen::VectorXd& parameters) const {
    check_matrix(orbitals);
    const Eigen::MatrixXd skew = generator(parameters);
    const Eigen::MatrixXd exponential = linalg::exponential(skew);
    return orbitals * exponential;
}

Eigen::VectorXd RotationChart::pullback(const Eigen::MatrixXd& orbitals,
                                        const Eigen::MatrixXd& raw_gradient,
                                        const Eigen::VectorXd& parameters) const {
    check_matrix(orbitals);
    check_matrix(raw_gradient);
    const Eigen::MatrixXd skew = generator(parameters);
    Eigen::VectorXd gradient(size());
    if (parameters.squaredNorm() == 0.0) {
        const Eigen::MatrixXd frame_gradient = orbitals.transpose() * raw_gradient;
        for (int k = 0; k < size(); ++k) {
            const auto [i, j] = pairs_[k];
            gradient[k] = frame_gradient(j, i) - frame_gradient(i, j);
        }
        return gradient;
    }
    const Eigen::MatrixXd frame_gradient = orbitals.transpose() * raw_gradient;
    // <B, L_exp(K,E)> = <L_exp(K^T,B), E>. One adjoint Frechet
    // derivative supplies every rotation component, not one exponential per E.
    const auto cotangent = linalg::exponential_pullback(skew, frame_gradient);
    for (int k = 0; k < size(); ++k) {
        const auto [i, j] = pairs_[k];
        gradient[k] = cotangent(j, i) - cotangent(i, j);
    }
    return gradient;
}

Eigen::VectorXd RotationChart::origin_hessian_vector(const Eigen::MatrixXd& orbitals,
                                                     const Eigen::MatrixXd& raw_gradient,
                                                     const Eigen::MatrixXd& raw_gradient_direction,
                                                     const Eigen::VectorXd& direction) const {
    check_matrix(orbitals);
    check_matrix(raw_gradient);
    check_matrix(raw_gradient_direction);
    const Eigen::MatrixXd skew = generator(direction);
    const Eigen::MatrixXd frame_gradient = orbitals.transpose() * raw_gradient;
    const Eigen::MatrixXd frame_direction =
        orbitals.transpose() * raw_gradient_direction +
        0.5 * (skew.transpose() * frame_gradient + frame_gradient * skew.transpose());
    Eigen::VectorXd result(size());
    for (int k = 0; k < size(); ++k) {
        const auto [i, j] = pairs_[k];
        result[k] = frame_direction(j, i) - frame_direction(i, j);
    }
    return result;
}

} // namespace xmvb::casscf
