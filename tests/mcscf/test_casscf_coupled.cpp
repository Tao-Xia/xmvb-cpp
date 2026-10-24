#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "casscf/derivatives/coupled.hpp"
#include "casscf/derivatives/point.hpp"
#include "casscf/driver/casscf.hpp"
#include "integrals/ao/graph.hpp"
#include "mcscf/ci/spin.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

// A positive-semidefinite AO pair kernel gives a physical Coulomb metric;
// deterministic irrational-looking entries prevent accidental symmetries.
xmvb::integrals::AoIntegralInput make_ao(int n) {
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd factors(5, pairs);
    for (int k = 0; k < factors.rows(); ++k)
        for (int p = 0; p < pairs; ++p)
            factors(k, p) = 0.14 * std::sin(0.31 * (k + 1) * (p + 2)) +
                            0.07 * std::cos(0.53 * (k + 2) * (p + 1));
    const Eigen::MatrixXd kernel = factors.transpose() * factors;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            values.push_back(kernel(p, q));
        }
    xmvb::integrals::AoIntegralInput ao;
    ao.n_basis_functions = n;
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, n);
    ao.ao_core_hamiltonian_matrix.resize(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            ao.ao_core_hamiltonian_matrix(i, j) =
                i == j ? -1.2 + 0.33 * i : 0.11 * std::sin((i + 1) * (j + 1));
    return ao;
}

Eigen::MatrixXd make_frame(int n) {
    Eigen::MatrixXd seed(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            seed(i, j) = std::sin(0.37 * (i + 1) * (j + 2)) + (i == j ? 1.0 : 0.0);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(seed);
    return qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
}

void check_case(int nactive, int electrons, int spin) {
    using xmvb::casscf::CoupledModel;
    using xmvb::casscf::FixedCiPoint;
    constexpr int ncore = 1, nvirtual = 1;
    constexpr double nuclear = 0.37;
    const int n = ncore + nactive + nvirtual;
    const auto ao = make_ao(n);
    const Eigen::MatrixXd u = make_frame(n);
    const int na = (electrons + spin) / 2, nb = electrons - na;
    xmvb::mcscf::SpinSpace space(nactive, na, nb);
    Eigen::MatrixXd raw(space.alpha().size(), space.beta().size());
    for (int i = 0; i < raw.rows(); ++i)
        for (int j = 0; j < raw.cols(); ++j)
            raw(i, j) =
                std::sin(0.7 * (i + 1) + 0.4 * (j + 2)) + 0.3 * std::cos(0.29 * (i + 2) * (j + 1));
    const Eigen::MatrixXd pure = space.project(raw, spin);
    require(pure.norm() > 1e-8, "pure-spin CI fixture collapsed");
    const Eigen::MatrixXd c = pure / pure.norm();
    FixedCiPoint point(ao, u, ncore, nactive, nuclear, electrons, spin, c);
    CoupledModel model(point, ao, ncore, nactive);
    const Eigen::Index no = model.orbital_size(), nc = c.size();
    require(model.size() == no + nc && model.size() > 0,
            "coupled model has an inconsistent coordinate size");

    const auto energy_at = [&](const Eigen::VectorXd& p) {
        FixedCiPoint trial(
            ao, model.orbitals(p), ncore, nactive, nuclear, electrons, spin, model.coefficients(p));
        return trial.energy();
    };
    const auto keyframe_at = [&](const Eigen::VectorXd& p) {
        FixedCiPoint trial(
            ao, model.orbitals(p), ncore, nactive, nuclear, electrons, spin, model.coefficients(p));
        return model.keyframe(trial, p);
    };
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(model.size());
    const auto origin = model.keyframe(point, zero);
    require(origin.gradient.allFinite() && std::isfinite(origin.energy),
            "origin coupled keyframe is nonfinite");
    require((origin.gradient - model.gradient()).norm() < 2e-10,
            "origin coupled keyframe differs from model gradient");
    require(std::abs(origin.energy - point.energy()) < 2e-11,
            "origin coupled keyframe changed the source energy");

    Eigen::VectorXd v(model.size()), w(model.size());
    for (Eigen::Index k = 0; k < model.size(); ++k) {
        v[k] = std::sin(0.57 * (k + 1)) + 0.2 * std::cos(0.91 * (k + 2));
        w[k] = std::cos(0.43 * (k + 2)) - 0.3 * std::sin(0.71 * (k + 1));
    }
    v.normalize();
    w.normalize();
    const Eigen::VectorXd seed = model.project_direction(v);
    require((model.project_direction(seed) - seed).norm() < 2e-10 &&
                (seed.head(no) - v.head(no)).norm() < 2e-12,
            "recycled direction projection changed orbital labels or is not idempotent");
    const Eigen::VectorXd p = 0.08 * v;
    const auto displaced = keyframe_at(p);
    require(displaced.gradient.allFinite() && std::isfinite(displaced.energy),
            "displaced coupled keyframe is nonfinite");
    const Eigen::MatrixXd displaced_ci = model.coefficients(p);
    require((space.project(displaced_ci, spin) - displaced_ci).norm() < 2e-10 &&
                std::abs(displaced_ci.norm() - 1.0) < 2e-10,
            "displaced CI retraction left the normalized target-spin space");
    const Eigen::Map<const Eigen::MatrixXd> displaced_ci_gradient(
        displaced.gradient.data() + no, c.rows(), c.cols());
    require((space.project(displaced_ci_gradient, spin) - displaced_ci_gradient).norm() < 2e-9,
            "displaced keyframe gradient left the target-spin space");
    const double h = 2e-5;
    // Check every active chart coordinate, including redundant CI coordinates.
    for (Eigen::Index k = 0; k < model.size(); ++k) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(model.size());
        unit[k] = 1.0;
        const double origin_fd = (energy_at(h * unit) - energy_at(-h * unit)) / (2 * h);
        const double displaced_fd = (energy_at(p + h * unit) - energy_at(p - h * unit)) / (2 * h);
        require(std::abs(origin_fd - origin.gradient[k]) < 5e-7,
                "origin joint energy derivative disagrees with coupled gradient");
        require(std::abs(displaced_fd - displaced.gradient[k]) < 8e-7,
                "displaced joint energy derivative disagrees with keyframe gradient");
    }

    const Eigen::VectorXd hv = model.apply(v);
    const Eigen::VectorXd hw = model.apply(w);
    require(hv.allFinite() && hw.allFinite(), "coupled HVP is nonfinite");
    for (const Eigen::VectorXd* direction : {&v, &w}) {
        const auto images = model.apply_split(*direction);
        Eigen::VectorXd orbital_direction = Eigen::VectorXd::Zero(model.size());
        orbital_direction.head(no) = direction->head(no);
        Eigen::VectorXd ci_direction = Eigen::VectorXd::Zero(model.size());
        ci_direction.tail(nc) = direction->tail(nc);
        const Eigen::VectorXd full_image = model.apply(*direction);
        const double scale = std::max(1.0, full_image.norm());
        require(images.orbital.size() == model.size() && images.ci.size() == model.size() &&
                    images.orbital.allFinite() && images.ci.allFinite(),
                "split coupled images have invalid coordinates");
        require((images.orbital + images.ci - full_image).norm() < 2e-10 * scale,
                "split coupled images do not sum to the full action");
        require((images.orbital - model.apply(orbital_direction)).norm() < 2e-10 * scale,
                "orbital split image differs from the orbital-only action");
        require((images.ci - model.apply(ci_direction)).norm() < 2e-10 * scale,
                "CI split image differs from the CI-only action");
        for (const Eigen::VectorXd* image : {&images.orbital, &images.ci}) {
            const Eigen::Map<const Eigen::MatrixXd> tail(image->data() + no, c.rows(), c.cols());
            require((space.project(tail, spin) - tail).norm() < 2e-9 &&
                        std::abs((tail.array() * c.array()).sum()) < 2e-9,
                    "split coupled CI image left the physical target-spin tangent");
        }
    }
    const Eigen::VectorXd fd_hv =
        (keyframe_at(h * v).gradient - keyframe_at(-h * v).gradient) / (2 * h);
    const Eigen::VectorXd fd_hw =
        (keyframe_at(h * w).gradient - keyframe_at(-h * w).gradient) / (2 * h);
    require((hv - fd_hv).norm() < 5e-6 * std::max(1.0, hv.norm()),
            "coupled HVP differs from exact keyframe finite difference");
    require((hw - fd_hw).norm() < 5e-6 * std::max(1.0, hw.norm()),
            "second coupled HVP differs from exact keyframe finite difference");
    require(std::abs(v.dot(hw) - w.dot(hv)) < 5e-8 * std::max({1.0, hv.norm(), hw.norm()}),
            "coupled Hessian is not bilinear symmetric");

    Eigen::VectorXd orbital = Eigen::VectorXd::Zero(model.size());
    orbital.head(no) = v.head(no).normalized();
    Eigen::VectorXd ci = Eigen::VectorXd::Zero(model.size());
    ci.tail(nc) = model.gradient().tail(nc);
    if (space.dimension(spin) > 1) {
        require(ci.tail(nc).norm() > 1e-7, "the coupled CI fixture is accidentally stationary");
        ci.normalize();
        const double forward = ci.dot(model.apply(orbital));
        const double reverse = orbital.dot(model.apply(ci));
        require(std::abs(forward - reverse) <
                    5e-8 * std::max({1.0, std::abs(forward), std::abs(reverse)}),
                "orbital-CI cross blocks violate reciprocity");
    }

    // The carrier retains c, but the physical CI tangent excludes it.
    Eigen::VectorXd null_ci = Eigen::VectorXd::Zero(model.size());
    null_ci.tail(nc) = Eigen::Map<const Eigen::VectorXd>(c.data(), nc);
    require(model.project_direction(null_ci).norm() < 2e-10,
            "recycled initial guess retained the CI scale direction");
    require(model.apply(null_ci).norm() < 2e-9,
            "coefficient normalization direction survived tangent projection");
    require((model.coefficients(null_ci) - c).norm() < 2e-10,
            "normalization direction changed the retracted CI state");
    if (space.dimension(spin) == 1) {
        require(model.gradient().tail(nc).norm() < 2e-9,
                "one-dimensional high-spin carrier has a CI gradient");
        require(hv.tail(nc).norm() < 2e-9,
                "one-dimensional high-spin carrier has a CI Hessian image");
    }
    const Eigen::Map<const Eigen::MatrixXd> ci_image(hv.data() + no, c.rows(), c.cols());
    require((space.project(ci_image, spin) - ci_image).norm() < 2e-9,
            "coupled HVP left the requested spin sector");
    require(std::abs((ci_image.array() * c.array()).sum()) < 2e-9,
            "coupled HVP contains a coefficient normalization component");
}

} // namespace

int main() {
    try {
        check_case(2, 2, 0); // Singlet: active CI and orbital response.
        check_case(3, 3, 1); // Unbalanced alpha/beta doublet carrier.
        check_case(4, 4, 4); // Quintet: no physical CI tangent.
        const auto ao = make_ao(2);
        const Eigen::MatrixXd frame = make_frame(2);
        const auto full_ci = xmvb::casscf::solve_casci(
            xmvb::casscf::make_active_integrals(ao, frame, 0, 2, 0.37), 2, 0);
        const auto full_cas = xmvb::casscf::solve_casscf(ao, frame, 0, 2, 2, 0, 0.37);
        require(full_cas.converged && full_cas.hvp_actions == 0 && full_cas.iterations.empty() &&
                    std::abs(full_cas.energy - full_ci.energy()) < 2e-10,
                "complete all-orbital CAS did not reduce directly to CASCI");
        std::cout << "coupled CASSCF joint derivative checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
