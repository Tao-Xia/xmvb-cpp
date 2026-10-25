#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "integrals/active/transform.hpp"

namespace {

int pair(int a, int b) {
    if (a < b)
        std::swap(a, b);
    return a * (a + 1) / 2 + b;
}

void require_close(const Eigen::MatrixXd& actual,
                   const Eigen::MatrixXd& expected,
                   const char* message,
                   double tolerance = 2e-11) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
        (actual - expected).norm() > tolerance * std::max(1.0, expected.norm())) {
        throw std::runtime_error(message);
    }
}

Eigen::MatrixXd fixture(int rows, int columns, double shift) {
    Eigen::MatrixXd result(rows, columns);
    for (int j = 0; j < columns; ++j) {
        for (int i = 0; i < rows; ++i) {
            result(i, j) = std::sin(shift + 0.31 * (i + 1) * (j + 2)) +
                           0.27 * std::cos(shift + 0.19 * (i + 3) * (j + 1));
        }
    }
    return result;
}

// Independent dense oracle: explicitly construct the packed pair map.
Eigen::MatrixXd pair_map(const Eigen::MatrixXd& q) {
    const int n = static_cast<int>(q.rows());
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd result(pairs, pairs);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            for (int b = 0; b < n; ++b) {
                for (int a = b; a < n; ++a) {
                    result(pair(a, b), pair(i, j)) =
                        q(a, i) * q(b, j) + (a == b ? 0.0 : q(b, i) * q(a, j));
                }
            }
        }
    }
    return result;
}

Eigen::MatrixXd pair_map_direction(const Eigen::MatrixXd& q, const Eigen::MatrixXd& dq) {
    const int n = static_cast<int>(q.rows());
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd result(pairs, pairs);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            for (int b = 0; b < n; ++b) {
                for (int a = b; a < n; ++a) {
                    result(pair(a, b), pair(i, j)) =
                        dq(a, i) * q(b, j) + q(a, i) * dq(b, j) +
                        (a == b ? 0.0 : dq(b, i) * q(a, j) + q(b, i) * dq(a, j));
                }
            }
        }
    }
    return result;
}

void check_pullback_generator_direction(const Eigen::MatrixXd& kernel,
                                        const Eigen::MatrixXd& gradient,
                                        const Eigen::MatrixXd& direction) {
    const int n = static_cast<int>(direction.rows());
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd l = pair_map_direction(identity, direction);
    const Eigen::MatrixXd expected = l * gradient + gradient * l.transpose();
    const Eigen::MatrixXd actual = xmvb::integrals::pullback_pair_generator(gradient, direction);
    require_close(actual, expected, "pair generator pullback differs from explicit pair map");

    constexpr double step = 1e-6;
    const Eigen::MatrixXd plus = pair_map(identity + step * direction);
    const Eigen::MatrixXd minus = pair_map(identity - step * direction);
    require_close(actual,
                  (plus * gradient * plus.transpose() - minus * gradient * minus.transpose()) /
                      (2.0 * step),
                  "pair generator pullback fails independent finite differences",
                  3e-8);

    const double left = (kernel.array() * actual.array()).sum();
    const double right =
        (xmvb::integrals::apply_pair_generator(kernel, direction).array() * gradient.array()).sum();
    if (std::abs(left - right) > 2e-11 * std::max({1.0, std::abs(left), std::abs(right)})) {
        throw std::runtime_error("pair generator pullback violates Frobenius duality");
    }
}

void check_size(int n) {
    const int pairs = n * (n + 1) / 2;
    const Eigen::MatrixXd k = fixture(pairs, pairs, 0.13);
    const Eigen::MatrixXd dk = fixture(pairs, pairs, -0.41);
    const Eigen::MatrixXd g = fixture(pairs, pairs, 0.73);
    const Eigen::MatrixXd dg = fixture(pairs, pairs, -0.62);
    Eigen::MatrixXd q = fixture(n, n, 0.27);
    q += Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd dq = fixture(n, n, -0.18);
    const Eigen::MatrixXd t = pair_map(q);
    const Eigen::MatrixXd dt = pair_map_direction(q, dq);

    const Eigen::MatrixXd forward = t.transpose() * k * t;
    const Eigen::MatrixXd forward_direction =
        dt.transpose() * k * t + t.transpose() * dk * t + t.transpose() * k * dt;
    const Eigen::MatrixXd backward = t * g * t.transpose();
    const Eigen::MatrixXd backward_direction =
        dt * g * t.transpose() + t * dg * t.transpose() + t * g * dt.transpose();
    require_close(xmvb::integrals::transform_pair_kernel(k, q),
                  forward,
                  "four-index transform differs from explicit pair map");
    require_close(xmvb::integrals::transform_pair_kernel_direction(k, dk, q, dq),
                  forward_direction,
                  "four-index transform direction is wrong");
    require_close(xmvb::integrals::pullback_pair_kernel(g, q),
                  backward,
                  "pair-kernel pullback misses packed multiplicities");
    require_close(xmvb::integrals::pullback_pair_kernel_direction(g, dg, q, dq),
                  backward_direction,
                  "pair-kernel pullback direction is wrong");

    const double duality_left = (g.array() * forward.array()).sum();
    const double duality_right = (backward.array() * k.array()).sum();
    if (std::abs(duality_left - duality_right) >
        2e-11 * std::max({1.0, std::abs(duality_left), std::abs(duality_right)})) {
        throw std::runtime_error("packed pair transform and pullback are not dual");
    }

    constexpr double step = 1e-6;
    const Eigen::MatrixXd plus_q = q + step * dq;
    const Eigen::MatrixXd minus_q = q - step * dq;
    require_close(xmvb::integrals::transform_pair_kernel_direction(k, dk, q, dq),
                  (xmvb::integrals::transform_pair_kernel(k + step * dk, plus_q) -
                   xmvb::integrals::transform_pair_kernel(k - step * dk, minus_q)) /
                      (2.0 * step),
                  "four-index transform fails directional finite differences",
                  3e-8);
    require_close(xmvb::integrals::pullback_pair_kernel_direction(g, dg, q, dq),
                  (xmvb::integrals::pullback_pair_kernel(g + step * dg, plus_q) -
                   xmvb::integrals::pullback_pair_kernel(g - step * dg, minus_q)) /
                      (2.0 * step),
                  "pair-kernel pullback fails directional finite differences",
                  3e-8);

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd nonsymmetric_g = g;
    if (pairs > 1)
        nonsymmetric_g(0, pairs - 1) += 0.41;
    Eigen::MatrixXd diagonal_direction = Eigen::MatrixXd::Zero(n, n);
    diagonal_direction.diagonal() = dq.diagonal();
    Eigen::MatrixXd off_diagonal_direction = Eigen::MatrixXd::Zero(n, n);
    if (n > 1) {
        off_diagonal_direction(0, n - 1) = 0.37;
        off_diagonal_direction(n - 1, 0) = -0.29;
    }
    const Eigen::MatrixXd zero_direction = Eigen::MatrixXd::Zero(n, n);
    const std::array<Eigen::MatrixXd, 4> pullback_directions{
        dq, diagonal_direction, off_diagonal_direction, zero_direction};
    for (const auto& direction : pullback_directions) {
        check_pullback_generator_direction(k, nonsymmetric_g, direction);
    }
    require_close(xmvb::integrals::pullback_pair_generator(nonsymmetric_g, zero_direction),
                  Eigen::MatrixXd::Zero(pairs, pairs),
                  "zero orbital direction must have zero pair-generator pullback",
                  0.0);
    const Eigen::MatrixXd generator = xmvb::integrals::orbital_generator_adjoint(k, g);
    const Eigen::MatrixXd identity_map = pair_map(identity);
    const Eigen::MatrixXd identity_direction = pair_map_direction(identity, dq);
    const Eigen::MatrixXd explicit_gradient = identity_direction.transpose() * k * identity_map +
                                              identity_map.transpose() * k * identity_direction;
    const Eigen::MatrixXd generator_apply = xmvb::integrals::apply_pair_generator(k, dq);
    require_close(generator_apply,
                  explicit_gradient,
                  "pair generator differs from explicit pair-map direction");
    require_close(generator_apply,
                  xmvb::integrals::transform_pair_kernel_direction(
                      k, Eigen::MatrixXd::Zero(pairs, pairs), identity, dq),
                  "pair generator differs from general four-index direction");
    require_close(generator_apply,
                  (xmvb::integrals::transform_pair_kernel(k, identity + step * dq) -
                   xmvb::integrals::transform_pair_kernel(k, identity - step * dq)) /
                      (2.0 * step),
                  "pair generator fails finite differences",
                  3e-8);
    const double predicted = (generator.array() * dq.array()).sum();
    const double expected = (g.array() * explicit_gradient.array()).sum();
    const double dual_apply = (g.array() * generator_apply.array()).sum();
    if (std::abs(predicted - dual_apply) >
        2e-11 * std::max({1.0, std::abs(predicted), std::abs(dual_apply)})) {
        throw std::runtime_error("pair generator and its orbital adjoint are not dual");
    }
    if (std::abs(predicted - expected) >
        2e-11 * std::max({1.0, std::abs(predicted), std::abs(expected)})) {
        throw std::runtime_error("orbital generator adjoint differs from explicit pair map");
    }
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(n, n);
            basis(row, column) = 1.0;
            const Eigen::MatrixXd basis_map = pair_map_direction(identity, basis);
            const Eigen::MatrixXd basis_output = basis_map.transpose() * k + k * basis_map;
            const double entry = (g.array() * basis_output.array()).sum();
            if (std::abs(generator(row, column) - entry) > 2e-11 * std::max(1.0, std::abs(entry))) {
                throw std::runtime_error("orbital generator adjoint has a wrong matrix entry");
            }
        }
    }
    const double finite_difference =
        (g.array() * (xmvb::integrals::transform_pair_kernel(k, identity + step * dq) -
                      xmvb::integrals::transform_pair_kernel(k, identity - step * dq))
                         .array())
            .sum() /
        (2.0 * step);
    if (std::abs(predicted - finite_difference) >
        3e-8 * std::max(1.0, std::abs(finite_difference))) {
        throw std::runtime_error("orbital generator adjoint fails finite differences");
    }
}

} // namespace

int main() {
    for (int n : {1, 2, 3, 4}) {
        check_size(n);
    }
}
