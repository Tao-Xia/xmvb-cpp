#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace xmvb::test {

/** @brief Bounded response-space oracle, using supplied diagnostic A/B images.
 * For SPD A, C-B V(V^T A V)^-1 V^T B^T decreases towards the exact Schur
 * complement as V grows. This does not change a production solver or its budget.
 */
template <class Inverse>
void audit_coupling_response(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b,
                             const Eigen::MatrixXd& c,
                             const Eigen::MatrixXd& metric,
                             Eigen::Index capacity,
                             const Inverse& inverse) {
    const Eigen::Index n = a.rows(), rhs_count = b.rows();
    const Eigen::VectorXd scale = metric.diagonal().array().sqrt().inverse();
    const Eigen::MatrixXd rhs = b.transpose() * scale.asDiagonal();
    const Eigen::MatrixXd scaled_c = scale.asDiagonal() * c * scale.asDiagonal();
    const Eigen::MatrixXd scaled_m = scale.asDiagonal() * metric * scale.asDiagonal();
    capacity = std::min(n, capacity);
    Eigen::MatrixXd basis(n, capacity), images(n, capacity);
    Eigen::MatrixXd residual = rhs;
    Eigen::Index size = 0;
    while (size < capacity) {
        const Eigen::Index before = size;
        for (Eigen::Index j = 0; j < rhs_count && size < capacity; ++j) {
            Eigen::VectorXd v = inverse(residual.col(j));
            const double initial = v.stableNorm();
            for (int pass = 0; pass < 2; ++pass)
                v.noalias() -= basis.leftCols(size) * (basis.leftCols(size).transpose() * v).eval();
            const double norm = v.stableNorm();
            if (!(norm > std::sqrt(std::numeric_limits<double>::epsilon()) * initial))
                continue;
            basis.col(size) = v / norm;
            images.col(size).noalias() = a * basis.col(size);
            ++size;
        }
        if (size == before)
            break;
        const auto v = basis.leftCols(size);
        const auto av = images.leftCols(size);
        const Eigen::MatrixXd raw = v.transpose() * av;
        const Eigen::MatrixXd projected = .5 * (raw + raw.transpose());
        const Eigen::VectorXd equilibrate = projected.diagonal().array().sqrt().inverse();
        const Eigen::MatrixXd normalized =
            equilibrate.asDiagonal() * projected * equilibrate.asDiagonal();
        const Eigen::LLT<Eigen::MatrixXd> factor(normalized);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("diagnostic response A projection is not positive");
        const Eigen::MatrixXd projected_rhs = v.transpose() * rhs;
        const Eigen::MatrixXd coefficients =
            equilibrate.asDiagonal() * factor.solve(equilibrate.asDiagonal() * projected_rhs);
        residual = rhs - av * coefficients;
        const Eigen::MatrixXd schur = scaled_c - projected_rhs.transpose() * coefficients;
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
            (.5 * (schur + schur.transpose())).eval(), scaled_m, Eigen::EigenvaluesOnly);
        if (spectrum.info() != Eigen::Success)
            throw std::runtime_error("diagnostic response Schur spectrum failed");
        std::cout << "RESPONSE_SPACE,orbital_actions=" << size
                  << ",minimum=" << spectrum.eigenvalues()[0]
                  << ",maximum=" << spectrum.eigenvalues().tail(1)[0]
                  << ",rhs_relative_residual=" << residual.norm() / rhs.norm() << std::endl;
        if (spectrum.eigenvalues()[0] < 0)
            break;
    }
}

} // namespace xmvb::test
