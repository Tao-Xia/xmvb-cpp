#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Projection {
    Eigen::MatrixXd f;
    Eigen::MatrixXd metric;
    Eigen::VectorXd target;
    Eigen::VectorXd coefficients;
    Eigen::VectorXd residual;
};

Projection project(const Eigen::MatrixXd& f, const Eigen::VectorXd& target) {
    Projection result;
    result.f = f;
    result.metric = f.transpose() * f;
    result.target = target;
    Eigen::LLT<Eigen::MatrixXd> factor(result.metric);
    require(factor.info() == Eigen::Success, "selected metric is not positive definite");
    result.coefficients = factor.solve(f.transpose() * target);
    result.residual = target - f * result.coefficients;
    return result;
}

Eigen::VectorXd solve_metric(const Projection& point, const Eigen::VectorXd& rhs) {
    Eigen::LLT<Eigen::MatrixXd> factor(point.metric);
    require(factor.info() == Eigen::Success, "selected metric solve failed");
    return factor.solve(rhs);
}

double rayleigh(const Eigen::MatrixXd& h, const Eigen::VectorXd& carrier) {
    return carrier.dot(h * carrier) / carrier.squaredNorm();
}

Eigen::VectorXd coefficient_gradient(const Projection& point, const Eigen::MatrixXd& h) {
    const Eigen::VectorXd carrier = point.f * point.coefficients;
    const double energy = rayleigh(h, carrier);
    return (2.0 / carrier.squaredNorm()) * point.f.transpose() * (h * carrier - energy * carrier);
}

double relative_error(double actual, double expected) {
    return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

struct Synthetic {
    Eigen::MatrixXd f0;
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd h;
    Eigen::VectorXd c0;
    Eigen::VectorXd l;

    Synthetic() : f0(5, 3), a(5, 3), b(5, 3), h(5, 5), c0(3), l(3) {
        f0 << 1.0, 0.1, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 1.0, 0.25, -0.15, 0.05, -0.1, 0.2, 0.3;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 3; ++j) {
                a(i, j) = 0.23 * std::sin(0.71 * (i + 1) * (j + 2));
                b(i, j) = 0.17 * std::cos(0.53 * (i + 2) * (j + 1));
            }
        }
        h.setZero();
        h.diagonal() << 0.2, 1.1, 2.0, -0.8, 0.6;
        for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 5; ++j) {
                h(i, j) = h(j, i) = 0.07 * std::sin(0.37 * (i + 1) * (j + 2));
            }
        }
        c0 << 0.6, -0.3, 0.5;
        const Eigen::MatrixXd m0 = f0.transpose() * f0;
        c0 /= std::sqrt(c0.dot(m0 * c0));
        l << 0.3, 0.4, -0.2;
        l -= c0 * c0.dot(m0 * l);
    }

    Eigen::MatrixXd f(double p) const {
        return f0 + p * a + 0.5 * p * p * b;
    }

    Projection transported(double p, double q) const {
        return project(f(p), f0 * (c0 + q * l) + p * a * c0);
    }

    double energy(double p, double q, bool transport) const {
        const Eigen::VectorXd c =
            transport ? transported(p, q).coefficients : Eigen::VectorXd(c0 + q * l);
        return rayleigh(h, f(p) * c);
    }
};

struct Results {
    double maximum_error = 0.0;
    double nonclosure = 0.0;

    void check(double error, double tolerance, const std::string& label) {
        require(std::isfinite(error) && error < tolerance, label);
        maximum_error = std::max(maximum_error, error);
    }
};

void check_synthetic_transport(Results* results) {
    const Synthetic model;
    const Projection origin = model.transported(0.0, 0.0);
    results->check((origin.coefficients - model.c0).norm(),
                   1.0e-13,
                   "transport changed the origin coefficients");
    const double p = 0.29, q = -0.18;
    const Projection point = model.transported(p, q);
    const double residual = point.residual.norm();
    require(residual > 1.0e-5, "synthetic selected space unexpectedly closed");
    results->nonclosure = std::max(results->nonclosure, residual);
    results->check(
        (point.f.transpose() * point.residual).norm(), 1.0e-13, "synthetic normal equation failed");

    const double u = 0.8, w = -0.35;
    const double h = 1.0e-4;
    const Eigen::VectorXd plus = model.transported(h * u, h * w).coefficients;
    const Eigen::VectorXd minus = model.transported(-h * u, -h * w).coefficients;
    const Eigen::VectorXd first = (plus - minus) / (2.0 * h);
    results->check((first - w * model.l).norm(), 1.0e-8, "transport changed the origin first jet");
    const Projection expected_origin = model.transported(0.0, 0.0);
    const Eigen::VectorXd second_expected = -solve_metric(
        expected_origin,
        model.f0.transpose() * (u * u * model.b * model.c0 + 2.0 * u * w * model.a * model.l));
    const Eigen::VectorXd second_finite = (plus - 2.0 * model.c0 + minus) / (h * h);
    results->check((second_finite - second_expected).norm(),
                   2.0e-7,
                   "transport second jet disagrees with projection formula");

    const Eigen::VectorXd g_c = coefficient_gradient(point, model.h);
    const Eigen::VectorXd carrier = point.f * point.coefficients;
    const double energy = rayleigh(model.h, carrier);
    const Eigen::VectorXd h_residual = model.h * carrier - energy * carrier;
    const Eigen::MatrixXd d_f = model.a + p * model.b;
    const Eigen::VectorXd d_c_p =
        solve_metric(point,
                     d_f.transpose() * point.residual +
                         point.f.transpose() * (model.a * model.c0 - d_f * point.coefficients));
    const Eigen::VectorXd d_c_p_without_residual =
        solve_metric(point, point.f.transpose() * (model.a * model.c0 - d_f * point.coefficients));
    const Eigen::VectorXd d_c_q = solve_metric(point, point.f.transpose() * model.f0 * model.l);
    const double g_p =
        (2.0 / carrier.squaredNorm()) * (d_f * point.coefficients).dot(h_residual) + g_c.dot(d_c_p);
    const double g_q = g_c.dot(d_c_q);
    const double d = 1.0e-6;
    const double fd_p = (model.energy(p + d, q, true) - model.energy(p - d, q, true)) / (2.0 * d);
    const double fd_q = (model.energy(p, q + d, true) - model.energy(p, q - d, true)) / (2.0 * d);
    results->check(relative_error(g_p, fd_p),
                   2.0e-9,
                   "orbital gradient omitted transport chain or residual term");
    require(std::abs(g_c.dot(d_c_p - d_c_p_without_residual)) > 1.0e-6,
            "synthetic residual term is too small to test its omission");
    results->check(relative_error(g_q, fd_q), 2.0e-9, "CI gradient omitted transport chain");
    results->check((d_c_p - (model.transported(p + d, q).coefficients -
                             model.transported(p - d, q).coefficients) /
                                (2.0 * d))
                       .norm(),
                   2.0e-9,
                   "implicit orbital coefficient derivative is wrong");

    const Eigen::VectorXd g_c0 = coefficient_gradient(origin, model.h);
    require(g_c0.norm() > 1.0e-3, "synthetic CI was accidentally stationary");
    const double curvature_expected = g_c0.dot(second_expected);
    require(std::abs(curvature_expected) > 1.0e-4,
            "nonstationary curvature correction is accidentally negligible");
    const double curvature_finite =
        ((model.energy(h * u, h * w, true) - model.energy(h * u, h * w, false)) +
         (model.energy(-h * u, -h * w, true) - model.energy(-h * u, -h * w, false))) /
        (h * h);
    results->check(relative_error(curvature_finite, curvature_expected),
                   5.0e-7,
                   "nonstationary Hessian correction was lost");
}

Eigen::MatrixXd exterior_selected_map(const Eigen::Matrix3d& upper,
                                      const Eigen::Matrix<double, 9, 3>& selected) {
    const std::vector<std::vector<int>> occupations{{0, 1}, {0, 2}, {1, 2}};
    const xmvb::vb::ExteriorOrbitalTransform transform(occupations, upper);
    Eigen::MatrixXd map(9, 3);
    for (int structure = 0; structure < 3; ++structure) {
        Eigen::MatrixXd spin_product(3, 3);
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int beta = 0; beta < 3; ++beta)
                spin_product(alpha, beta) = selected(3 * alpha + beta, structure);
        transform.apply_left(&spin_product);
        transform.apply_right(&spin_product);
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int beta = 0; beta < 3; ++beta)
                map(3 * alpha + beta, structure) = spin_product(alpha, beta);
    }
    return map;
}

void check_exterior_selected_space(Results* results) {
    Eigen::Matrix<double, 9, 3> selected = Eigen::Matrix<double, 9, 3>::Zero();
    selected(0, 0) = selected(4, 1) = selected(8, 2) = 1.0;
    Eigen::Matrix3d upper_direction = Eigen::Matrix3d::Zero();
    upper_direction(0, 0) = 0.09;
    upper_direction(1, 1) = -0.06;
    upper_direction(2, 2) = 0.04;
    upper_direction(0, 1) = 0.27;
    upper_direction(0, 2) = -0.19;
    upper_direction(1, 2) = 0.23;
    const auto f = [&](double p) {
        const Eigen::Matrix3d upper = Eigen::Matrix3d::Identity() + p * upper_direction;
        return exterior_selected_map(upper, selected);
    };
    const Eigen::MatrixXd f0 = f(0.0);
    const double derivative_step = 1.0e-5;
    // Two-electron exterior powers are quadratic in this affine orbital path.
    const Eigen::MatrixXd a = (f(derivative_step) - f(-derivative_step)) / (2.0 * derivative_step);
    Eigen::Vector3d c0(0.7, -0.2, 0.6);
    c0.normalize();
    Eigen::Vector3d l(0.2, 0.5, -0.1);
    l -= c0 * c0.dot(l);
    const auto transported = [&](double p, double q) {
        return project(f(p), f0 * (c0 + q * l) + p * a * c0);
    };
    const double p = 0.34, q = 0.16;
    const Projection point = transported(p, q);
    require(point.residual.norm() > 1.0e-5,
            "three selected spin products unexpectedly form a closed space");
    results->nonclosure = std::max(results->nonclosure, point.residual.norm());
    results->check((point.f.transpose() * point.residual).norm(),
                   1.0e-12,
                   "exterior selected-space normal equation failed");
    require((point.coefficients - (c0 + q * l)).norm() > 1.0e-7,
            "exterior transport did not change selected coefficients");

    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(9, 9);
    for (int i = 0; i < 9; ++i)
        h(i, i) = 0.13 * (i + 1);
    for (int i = 0; i < 9; ++i)
        for (int j = i + 1; j < 9; ++j)
            h(i, j) = h(j, i) = 0.025 * std::sin(0.23 * (i + 1) * (j + 2));
    const Eigen::VectorXd g_c = coefficient_gradient(point, h);
    const Eigen::VectorXd d_c_q = solve_metric(point, point.f.transpose() * f0 * l);
    const double chain_gradient = g_c.dot(d_c_q);
    const auto energy = [&](double q_value) {
        const Projection trial = transported(p, q_value);
        return rayleigh(h, trial.f * trial.coefficients);
    };
    const double step = 1.0e-6;
    const double finite_gradient = (energy(q + step) - energy(q - step)) / (2.0 * step);
    results->check(relative_error(chain_gradient, finite_gradient),
                   3.0e-9,
                   "exterior selected-space CI gradient is inconsistent");
}

} // namespace

int main() {
    try {
        Results results;
        check_synthetic_transport(&results);
        check_exterior_selected_space(&results);
        std::cout << std::setprecision(12) << "ci_transport max_error=" << results.maximum_error
                  << " nonclosure_residual=" << results.nonclosure << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ci_transport: " << error.what() << '\n';
        return 1;
    }
}
