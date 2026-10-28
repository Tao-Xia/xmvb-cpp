#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "casscf/derivatives/point.hpp"
#include "casscf/derivatives/coupled.hpp"
#include "casscf/derivatives/diagonal.hpp"
#include "casscf/orbitals/frame.hpp"
#include "casscf/orbitals/rotation.hpp"
#include "integrals/ao/graph.hpp"
#include "mcscf/ci/spin.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_frame_completion(int ncore, int nactive, int nvirtual) {
    const int n = ncore + nactive + nvirtual;
    Eigen::MatrixXd lower = Eigen::MatrixXd::Identity(n, n);
    for (int row = 1; row < n; ++row)
        for (int col = 0; col < row; ++col)
            lower(row, col) = 0.09 * std::sin((row + 1) * (col + 2));
    const Eigen::MatrixXd overlap = lower * lower.transpose();
    Eigen::MatrixXd seed(n, ncore + nactive);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < seed.cols(); ++col)
            seed(row, col) = (row == col ? 1.0 : 0.0) + 0.07 * std::cos((row + 2) * (col + 1));
    const Eigen::MatrixXd frame =
        xmvb::casscf::complete_metric_orthonormal_frame(seed, overlap, ncore, nactive);
    require((frame.transpose() * overlap * frame - Eigen::MatrixXd::Identity(n, n)).norm() < 2e-13,
            "completed MO frame is not AO-metric orthonormal");
    if (ncore) {
        const Eigen::MatrixXd core = frame.leftCols(ncore);
        require((seed.leftCols(ncore) - core * (core.transpose() * overlap * seed.leftCols(ncore)))
                        .norm() < 2e-13,
                "completed MO frame changed the core span");
    }
    const Eigen::MatrixXd active = frame.middleCols(ncore, nactive);
    Eigen::MatrixXd active_seed = seed.rightCols(nactive);
    if (ncore) {
        const Eigen::MatrixXd core = frame.leftCols(ncore);
        const Eigen::MatrixXd projection = core.transpose() * overlap * active_seed;
        active_seed.noalias() -= core * projection;
    }
    require((active_seed - active * (active.transpose() * overlap * active_seed)).norm() < 2e-13,
            "completed MO frame changed the active span");
    if (ncore) {
        Eigen::MatrixXd dependent = seed;
        dependent.col(ncore) = dependent.col(0);
        bool rejected = false;
        try {
            (void)xmvb::casscf::complete_metric_orthonormal_frame(
                dependent, overlap, ncore, nactive);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "linearly dependent core/active seeds were accepted");
    }
}

xmvb::integrals::AoIntegralInput make_ao(int n) {
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd factors(5, pairs);
    for (int k = 0; k < factors.rows(); ++k)
        for (int p = 0; p < pairs; ++p)
            factors(k, p) = 0.14 * std::sin(0.31 * (k + 1) * (p + 2)) +
                            0.07 * std::cos(0.53 * (k + 2) * (p + 1));
    const Eigen::MatrixXd kernel = factors.transpose() * factors;
    std::vector<std::int64_t> left, right;
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

void check_case(int ncore, int nactive, int nvirtual) {
    using xmvb::casscf::FixedCiPoint;
    using xmvb::casscf::RotationChart;
    const int n = ncore + nactive + nvirtual;
    const auto ao = make_ao(n);
    const Eigen::MatrixXd u = make_frame(n);
    RotationChart chart(n, ncore, nactive);
    const int electrons = 2, spin = 0;
    xmvb::mcscf::SpinSpace space(nactive, 1, 1);
    Eigen::MatrixXd trial(space.alpha().size(), space.beta().size());
    for (int i = 0; i < trial.rows(); ++i)
        for (int j = 0; j < trial.cols(); ++j)
            trial(i, j) = std::sin(0.7 * (i + 1) + 0.4 * (j + 2));
    const Eigen::MatrixXd projected = space.project(trial, spin);
    const Eigen::MatrixXd c = projected / projected.norm();
    constexpr double nuclear = 0.37;
    FixedCiPoint point(ao, u, ncore, nactive, nuclear, electrons, spin, c);
    const auto energy_at = [&](const Eigen::VectorXd& p) {
        FixedCiPoint at(ao, chart.retract(u, p), ncore, nactive, nuclear, electrons, spin, c);
        return at.energy();
    };
    const auto gradient_at = [&](const Eigen::VectorXd& p) {
        FixedCiPoint at(ao, chart.retract(u, p), ncore, nactive, nuclear, electrons, spin, c);
        return chart.pullback(u, at.orbital_gradient_raw(), p);
    };
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(chart.size());
    const Eigen::VectorXd gradient = chart.pullback(u, point.orbital_gradient_raw(), zero);
    const double step = 2e-5;
    for (int k = 0; k < chart.size(); ++k) {
        Eigen::VectorXd direction = Eigen::VectorXd::Zero(chart.size());
        direction[k] = 1.0;
        const double finite =
            (energy_at(step * direction) - energy_at(-step * direction)) / (2 * step);
        require(std::abs(finite - gradient[k]) < 2e-7, "origin orbital gradient mismatch");
    }
    if (chart.size() == 0)
        return;
    Eigen::VectorXd p(chart.size()), v(chart.size()), w(chart.size());
    for (int k = 0; k < chart.size(); ++k) {
        p[k] = 0.12 * std::sin(k + 1.0);
        v[k] = std::cos(0.7 * (k + 1));
        w[k] = std::sin(0.9 * (k + 2));
    }
    p /= std::max(1.0, p.norm());
    v.normalize();
    w.normalize();
    const double finite_keyframe = (energy_at(p + step * v) - energy_at(p - step * v)) / (2 * step);
    require(std::abs(finite_keyframe - gradient_at(p).dot(v)) < 3e-7,
            "finite keyframe Frechet gradient mismatch");
    const Eigen::MatrixXd du = u * chart.generator(v);
    const auto response_v = point.direction(du);
    const Eigen::VectorXd hv =
        chart.origin_hessian_vector(u, point.orbital_gradient_raw(), response_v.orbital_hessian, v);
    const Eigen::VectorXd fd_hv = (gradient_at(step * v) - gradient_at(-step * v)) / (2 * step);
    require((hv - fd_hv).norm() < 2e-6, "analytic orbital Hessian action mismatch");
    const Eigen::VectorXd hw =
        chart.origin_hessian_vector(u,
                                    point.orbital_gradient_raw(),
                                    point.direction(u * chart.generator(w)).orbital_hessian,
                                    w);
    require(std::abs(v.dot(hw) - w.dot(hv)) < 2e-8, "orbital Hessian symmetry mismatch");
    const Eigen::VectorXd analytic_diagonal =
        xmvb::casscf::orbital_hessian_diagonal(point, chart, ao, ncore, nactive);
    for (int k = 0; k < chart.size(); ++k) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(chart.size());
        unit[k] = 1.0;
        const Eigen::VectorXd column =
            chart.origin_hessian_vector(u,
                                        point.orbital_gradient_raw(),
                                        point.direction(u * chart.generator(unit)).orbital_hessian,
                                        unit);
        require(std::abs(analytic_diagonal[k] - column[k]) < 2e-7,
                "analytic orbital diagonal differs from exact HVP");
    }

    const auto& integral_direction = response_v.integrals;
    FixedCiPoint plus(ao, chart.retract(u, step * v), ncore, nactive, nuclear, electrons, spin, c);
    FixedCiPoint minus(
        ao, chart.retract(u, -step * v), ncore, nactive, nuclear, electrons, spin, c);
    const auto& plus_int = plus.integrals().active;
    const auto& minus_int = minus.integrals().active;
    require(((plus_int.one_electron - minus_int.one_electron) / (2 * step) -
             integral_direction.one_electron)
                    .norm() < 2e-7,
            "active one-electron integral direction mismatch");
    require(((plus_int.pair_kernel - minus_int.pair_kernel) / (2 * step) -
             integral_direction.pair_kernel)
                    .norm() < 2e-7,
            "active two-electron integral direction mismatch");

    Eigen::MatrixXd other = Eigen::MatrixXd::Zero(c.rows(), c.cols());
    other(0, 0) = 1.0;
    other = space.project(other, spin);
    const auto transition = [&](const FixedCiPoint& at) {
        return (other.array() * at.sigma().apply(c).array()).sum();
    };
    const double finite_transition = (transition(plus) - transition(minus)) / (2 * step);
    const Eigen::MatrixXd raw_transition = point.transition_orbital_gradient(other, c);
    const double predicted_transition = chart.pullback(u, raw_transition, zero).dot(v);
    require(std::abs(finite_transition - predicted_transition) < 2e-7,
            "transition integral pullback mismatch");

    other.normalize();
    FixedCiPoint fresh(ao, u, ncore, nactive, nuclear, electrons, spin, other);
    point.set_coefficients(other);
    require(std::abs(point.energy() - fresh.energy()) < 1e-11 &&
                (point.orbital_gradient_raw() - fresh.orbital_gradient_raw()).norm() < 1e-10,
            "CI state update reused stale integral adjoints");
}

} // namespace

int main() {
    try {
        check_frame_completion(1, 2, 2);
        check_frame_completion(0, 2, 1);
        check_frame_completion(1, 2, 0);
        check_case(1, 2, 1);
        check_case(0, 2, 1);
        check_case(1, 2, 0);
        std::cout << "complete-CAS analytic derivative checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
