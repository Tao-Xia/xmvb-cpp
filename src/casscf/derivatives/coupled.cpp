#include "casscf/derivatives/coupled.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "casscf/derivatives/diagonal.hpp"

namespace xmvb::casscf {
namespace {
constexpr double sqrt_two = 1.4142135623730950488;
}

CoupledModel::CoupledModel(const FixedCiPoint& point,
                           const integrals::AoIntegralInput& ao,
                           int ncore,
                           int nactive)
    : point_(point), chart_(static_cast<int>(point.orbitals().cols()), ncore, nactive),
      ci_size_(point.coefficients().size()), raw_gradient_(point.orbital_gradient_raw()) {
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(chart_.size());
    gradient_ = pack(chart_.pullback(point.orbitals(), raw_gradient_, zero),
                     sqrt_two * point.ci_residual());
    diagonal_.resize(size());
    diagonal_.head(orbital_size()) = orbital_hessian_diagonal(point, chart_, ao, ncore, nactive);
    const Eigen::MatrixXd ci_diagonal =
        point.sigma().diagonal(point.integrals().active.one_electron);
    diagonal_.tail(ci_size_) = Eigen::Map<const Eigen::VectorXd>(ci_diagonal.data(), ci_size_);
    diagonal_.tail(ci_size_).array() -= point.active_energy();
}

Eigen::MatrixXd CoupledModel::ci_direction(const Eigen::VectorXd& vector) const {
    if (vector.size() != size() || !vector.allFinite())
        throw std::invalid_argument("invalid coupled orbital/CI direction");
    const auto& c = point_.coefficients();
    const Eigen::Map<const Eigen::MatrixXd> raw(vector.data() + orbital_size(), c.rows(), c.cols());
    Eigen::MatrixXd q = point_.project_ci(raw);
    q.noalias() -= c * (c.array() * q.array()).sum();
    return q;
}

Eigen::VectorXd CoupledModel::pack(const Eigen::VectorXd& orbital,
                                   const Eigen::MatrixXd& ci) const {
    Eigen::VectorXd result(size());
    result.head(orbital_size()) = orbital;
    const auto& c = point_.coefficients();
    Eigen::MatrixXd tangent = point_.project_ci(ci);
    tangent.noalias() -= c * (c.array() * tangent.array()).sum();
    result.tail(ci_size_) = Eigen::Map<const Eigen::VectorXd>(tangent.data(), ci_size_);
    return result;
}

CoupledImages CoupledModel::apply_split(const Eigen::VectorXd& direction) const {
    const auto& u = point_.orbitals();
    const auto& c = point_.coefficients();
    const Eigen::MatrixXd q = ci_direction(direction);
    const Eigen::VectorXd p = direction.head(orbital_size());
    const Eigen::MatrixXd du = u * chart_.generator(p);
    const auto orbital_response = point_.direction(du);
    const mcscf::DirectCiSigmaAction delta_sigma(point_.sigma(), orbital_response.integrals);
    // One response sweep supplies Hq and the integral adjoint for B^Tq.
    const auto response = point_.sigma().apply_with_adjoint(q, c);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(orbital_size());
    const Eigen::VectorXd orbital_image =
        chart_.origin_hessian_vector(u, raw_gradient_, orbital_response.orbital_hessian, p);
    const Eigen::VectorXd ci_orbital_image =
        sqrt_two *
        chart_.pullback(u, point_.orbital_gradient_from_adjoint(response.integrals), zero);
    return {pack(orbital_image, sqrt_two * delta_sigma.apply(c)),
            pack(ci_orbital_image, response.sigma - point_.active_energy() * q)};
}

Eigen::VectorXd CoupledModel::apply(const Eigen::VectorXd& direction) const {
    const auto images = apply_split(direction);
    return images.orbital + images.ci;
}

Eigen::VectorXd CoupledModel::project_direction(const Eigen::VectorXd& direction) const {
    const Eigen::MatrixXd ci = ci_direction(direction);
    return pack(direction.head(orbital_size()), ci);
}

Eigen::VectorXd CoupledModel::precondition(const Eigen::VectorXd& residual, double shift) const {
    if (residual.size() != size() || !residual.allFinite() || !std::isfinite(shift))
        throw std::invalid_argument("invalid coupled preconditioner residual");
    Eigen::VectorXd result(residual.size());
    for (Eigen::Index i = 0; i < size(); ++i) {
        double denominator = diagonal_[i] - shift;
        // Same signed AH denominator protection as the shared VB diagonal model.
        if (std::abs(denominator) < 1e-8)
            denominator = 1e-8;
        result[i] = residual[i] / denominator;
    }
    return pack(result.head(orbital_size()), ci_direction(result));
}

Eigen::MatrixXd CoupledModel::orbitals(const Eigen::VectorXd& displacement) const {
    return chart_.retract(point_.orbitals(), displacement.head(orbital_size()));
}

Eigen::MatrixXd CoupledModel::coefficients(const Eigen::VectorXd& displacement) const {
    Eigen::MatrixXd result = point_.coefficients() + ci_direction(displacement) / sqrt_two;
    result /= result.norm();
    return result;
}

mcscf::CiahKeyframe CoupledModel::keyframe(const FixedCiPoint& trial,
                                           const Eigen::VectorXd& displacement) const {
    const Eigen::MatrixXd q = ci_direction(displacement);
    const double norm = std::sqrt(1 + 0.5 * q.squaredNorm());
    const Eigen::MatrixXd ci = sqrt_two / norm * trial.ci_residual();
    return {pack(chart_.pullback(point_.orbitals(),
                                 trial.orbital_gradient_raw(),
                                 displacement.head(orbital_size())),
                 ci),
            trial.energy(),
            trial.energy()};
}

double CoupledModel::maximum_component(const Eigen::VectorXd& direction) const {
    const double orbital =
        orbital_size() ? direction.head(orbital_size()).cwiseAbs().maxCoeff() : 0;
    const double ci = ci_direction(direction).cwiseAbs().maxCoeff() / sqrt_two;
    return std::max(orbital, ci);
}

} // namespace xmvb::casscf
