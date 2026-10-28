#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "mcscf/optimization/ciah/bounded.hpp"

namespace xmvb::test {

/** @brief Bounded sampled quadratic with a direction chosen before clipping.
 * Reference experiment only. The shifted Newton solution is chosen with
 * H+mu*M positive semidefinite and the same physical component bound as CIAH.
 * Bisection needs a continuous feasible/infeasible bracket, not an assumption
 * that the component infinity norm is monotone in mu.
 */
inline Eigen::VectorXd
restricted_quadratic_reference(const Eigen::MatrixXd& h,
                               const Eigen::MatrixXd& mass,
                               const Eigen::VectorXd& g,
                               const std::function<double(const Eigen::VectorXd&)>& maximum,
                               double cap) {
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(h, mass);
    if (spectrum.info() != Eigen::Success)
        throw std::runtime_error("restricted reference spectrum failed");
    return mcscf::bound_ciah_step(spectrum.eigenvalues(), spectrum.eigenvectors(), g, maximum, cap)
        .coefficients;
}

/** @brief Production-style fixed-scale AH ray followed by its existing cap. */
inline Eigen::VectorXd
clipped_ah_reference(const Eigen::MatrixXd& h,
                     const Eigen::MatrixXd& mass,
                     const Eigen::VectorXd& g,
                     const std::function<double(const Eigen::VectorXd&)>& maximum,
                     double cap) {
    const Eigen::Index n = g.size();
    Eigen::MatrixXd ah = Eigen::MatrixXd::Zero(n + 1, n + 1), am = ah;
    ah.bottomRightCorner(n, n) = h;
    ah.block(1, 0, n, 1) = g;
    ah.block(0, 1, 1, n) = g.transpose();
    am(0, 0) = 1;
    am.bottomRightCorner(n, n) = mass;
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigen(ah, am);
    if (eigen.info() != Eigen::Success)
        throw std::runtime_error("clipped AH reference failed");
    Eigen::Index root = 0;
    const double first = eigen.eigenvectors()(0, 0);
    const Eigen::VectorXd tail = eigen.eigenvectors().col(0).tail(n);
    const bool curvature = std::abs(first) <= .1 && tail.dot(h * tail) < 0;
    while (!curvature && root <= n && std::abs(eigen.eigenvectors()(0, root)) <= .1)
        ++root;
    if (root > n)
        throw std::runtime_error("clipped reference has no connected root");
    Eigen::VectorXd p = eigen.eigenvectors().col(root).tail(n);
    if (!curvature)
        p /= eigen.eigenvectors()(0, root);
    else if (g.dot(p) > 0)
        p = -p;
    const double largest = maximum(p);
    if (curvature || largest > cap)
        p *= cap / largest;
    return p;
}
} // namespace xmvb::test
