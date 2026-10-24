#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/diagnostics/orbitals/step_geometry.hpp"

namespace {
void close(double actual, double reference) {
    if (std::abs(actual - reference) > 2e-12)
        throw std::runtime_error("orbital step geometry disagrees with analytic reference");
}
} // namespace

int main() {
    using xmvb::vb::diagnostics::measure_orbital_step;
    Eigen::Matrix3d s;
    s << 2.0, 0.2, -0.1, 0.2, 1.4, 0.15, -0.1, 0.15, 1.1;
    const Eigen::Vector3d b(1.3, -0.2, 0.4);
    const double norm = std::sqrt(b.dot(s * b));
    const Eigen::Vector3d x = b / norm;
    Eigen::Vector3d tangent(0.3, 0.7, -0.1);
    tangent -= x * x.dot(s * tangent);
    tangent /= std::sqrt(tangent.dot(s * tangent));
    for (const double alpha : {0.0, 0.4, -0.3, -2.0}) {
        for (const double theta : {0.0, 0.03, 0.3, 1.0}) {
            const Eigen::Vector3d delta = norm * (alpha * x + theta * tangent);
            const auto g = measure_orbital_step(b, delta, s);
            close(g.alpha, alpha);
            close(g.theta, theta);
            close(g.step_norm, std::hypot(alpha, theta));
            close(g.norm_ratio, std::hypot(1.0 + alpha, theta));
            close(g.actual_angle, std::atan2(theta, 1.0 + alpha));
            const Eigen::Vector3d updated =
                (b + delta) / std::sqrt((b + delta).dot(s * (b + delta)));
            close(x.dot(s * updated), std::cos(g.actual_angle));
            const auto scaled = measure_orbital_step(7.0 * b, 7.0 * delta, s);
            close(scaled.alpha, g.alpha);
            close(scaled.theta, g.theta);
            close(scaled.actual_angle, g.actual_angle);
        }
    }
    // Tangent norm must not be obtained by subtracting two nearly equal norms.
    const auto almost_radial = measure_orbital_step(b, norm * (x + 1e-8 * tangent), s);
    close(almost_radial.theta, 1e-8);
    std::cout << "AO-metric radial/tangential decomposition and finite angles: passed\n";
}
