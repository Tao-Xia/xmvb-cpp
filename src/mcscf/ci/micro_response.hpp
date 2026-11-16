#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Eigenvalues>

#include "core/index.hpp"
#include "core/workspace.hpp"
#include "linalg/eigensolver.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf {

struct CiMicroResponse {
    Eigen::VectorXd coefficients;
    Eigen::VectorXd hamiltonian;
    double energy = 0.0;
    double residual_norm = 0.0;
    int actions = 0;
};

/** A short variational CI update, not a converged eigensolve.
 * The four-vector capacity follows PySCF's default CI response space;
 * an optional existing CI preconditioner supplies the expansion directions.
 * Orthogonalize explicitly instead of solving an ill-conditioned small Gram.
 * The caller supplies isometric coordinates and the production residual target.
 */
template <class Action>
CiMicroResponse ci_micro_response(const Eigen::VectorXd& initial,
                                  Action&& action,
                                  double residual_tolerance,
                                  const linalg::DavidsonPreconditioner& precondition = {}) {
    const Eigen::Index n = initial.size();
    const Eigen::Index capacity = std::min<Eigen::Index>(4, n);
    const double initial_norm = initial.norm();
    if (n == 0 || !(initial_norm > 0.0) || !(residual_tolerance > 0.0))
        throw std::invalid_argument("invalid CI microresponse seed or tolerance");
    const workspace::Temporary storage(core::checked_multiply(
        sizeof(double) * static_cast<std::size_t>(2 * capacity + 5), static_cast<std::size_t>(n)));
    Eigen::MatrixXd basis(n, capacity), images(n, capacity);
    Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(capacity, capacity);
    basis.col(0) = initial / initial_norm;
    CiMicroResponse result;
    Eigen::VectorXd weights;
    Eigen::VectorXd residual(n);
    // Intermediate Ritz vectors are consumed only through their residual.
    // Keep their row panels on the stack, not in two structure-sized arrays.
    constexpr Eigen::Index panel_rows = 256;
    const Eigen::Index panels = n / panel_rows + (n % panel_rows != 0);
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(panels));
    for (Eigen::Index j = 0; j < capacity; ++j) {
        images.col(j) = action(basis.col(j));
        ++result.actions;
        for (Eigen::Index i = 0; i <= j; ++i)
            projected(i, j) = projected(j, i) = basis.col(i).dot(images.col(j));
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ritz(
            projected.topLeftCorner(j + 1, j + 1));
        if (ritz.info() != Eigen::Success)
            throw std::runtime_error("CI microresponse Ritz solve failed");
        weights = ritz.eigenvectors().col(0);
        result.energy = ritz.eigenvalues()[0];
        const bool final_panel = j + 1 == capacity;
        if (final_panel) {
            result.coefficients.resize(n);
            result.hamiltonian.resize(n);
        }
#pragma omp parallel for schedule(static) num_threads(workers) if (workers > 1)
        for (Eigen::Index panel = 0; panel < panels; ++panel) {
            const Eigen::Index first = panel * panel_rows;
            const Eigen::Index count = std::min(panel_rows, n - first);
            Eigen::Matrix<double, panel_rows, 1> c, hc;
            // At most four columns: packet-wise sums avoid thousands of tiny
            // BLAS GEMV calls when Eigen is built with EIGEN_USE_BLAS.
            c.head(count) = weights[0] * basis.col(0).segment(first, count);
            hc.head(count) = weights[0] * images.col(0).segment(first, count);
            for (Eigen::Index k = 1; k <= j; ++k) {
                c.head(count) += weights[k] * basis.col(k).segment(first, count);
                hc.head(count) += weights[k] * images.col(k).segment(first, count);
            }
            residual.segment(first, count) = hc.head(count) - result.energy * c.head(count);
            if (final_panel) {
                result.coefficients.segment(first, count) = c.head(count);
                result.hamiltonian.segment(first, count) = hc.head(count);
            }
        }
        result.residual_norm = residual.stableNorm();
        if (result.residual_norm <= residual_tolerance || j + 1 == capacity)
            break;
        if (precondition)
            precondition(residual, result.energy, residual);
        const double direction_norm = residual.stableNorm();
        for (int pass = 0; pass < 2; ++pass)
            for (Eigen::Index i = 0; i <= j; ++i)
                residual -= basis.col(i).dot(residual) * basis.col(i);
        const double norm = residual.stableNorm();
        if (!(norm > 32 * std::numeric_limits<double>::epsilon() * direction_norm))
            break;
        basis.col(j + 1) = residual / norm;
    }
    // Materialize both outputs exactly once, including a dependent-direction exit.
    if (result.coefficients.size() == 0) {
        result.coefficients.noalias() = basis.leftCols(weights.size()) * weights;
        result.hamiltonian.noalias() = images.leftCols(weights.size()) * weights;
    }
    if (result.coefficients.dot(initial) < 0) {
        result.coefficients *= -1;
        result.hamiltonian *= -1;
    }
    return result;
}

} // namespace xmvb::mcscf
