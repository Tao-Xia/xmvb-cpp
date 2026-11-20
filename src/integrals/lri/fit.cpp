#include "integrals/lri/fit.hpp"
#include <Eigen/Eigenvalues>
#include <stdexcept>

namespace xmvb::integrals::lri {
FitResult fit_pair(const Eigen::MatrixXd& v, const Eigen::MatrixXd& b, double cutoff) {
    if (v.rows() == 0 || v.rows() != v.cols() || b.rows() != v.rows() || !v.allFinite() ||
        !b.allFinite() || !v.isApprox(v.transpose(), 1e-12) || !(cutoff > 0 && cutoff < 1))
        throw std::invalid_argument("invalid local Coulomb fit");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(v);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("local Coulomb eigensolve failed");
    const auto& eigenvalues = solver.eigenvalues();
    const double largest = eigenvalues.maxCoeff();
    if (!(largest > 0) || eigenvalues.minCoeff() < -cutoff * largest)
        throw std::runtime_error("local Coulomb metric is not positive semidefinite");
    Eigen::MatrixXd transformed = solver.eigenvectors().transpose() * b;
    std::size_t rank = 0;
    for (Eigen::Index i = 0; i < v.rows(); ++i) {
        if (eigenvalues[i] > cutoff * largest) {
            transformed.row(i) /= eigenvalues[i];
            ++rank;
        } else
            transformed.row(i).setZero();
    }
    Eigen::MatrixXd coefficients = solver.eigenvectors() * transformed;
    const double residual = (v * coefficients - b).norm() / std::max(1.0, b.norm());
    return {std::move(coefficients), rank, residual};
}
} // namespace xmvb::integrals::lri
