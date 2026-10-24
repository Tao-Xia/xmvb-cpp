#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/cholesky.hpp"
#include "vbscf/orbitals/preparation/overlap_pullback.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {

using namespace xmvb::vb;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Actual, typename Reference>
double relative_error(const Eigen::MatrixBase<Actual>& actual,
                      const Eigen::MatrixBase<Reference>& reference) {
    return (actual - reference).norm() / std::max(1.0, reference.norm());
}

OrbitalPreparationInput sparse_fixture() {
    OrbitalPreparationInput input;
    input.n_basis_functions = 5;
    input.n_orbitals = 3;
    input.n_active_orbitals = 2;
    input.n_active_electrons = 2;
    input.n_total_electrons = 4;
    input.orbital_type = 1;
    input.ao_overlap_matrix = Eigen::MatrixXd::Identity(5, 5);
    input.ao_overlap_matrix.array() += 0.07;
    input.ao_overlap_matrix(0, 2) += 0.04;
    input.ao_overlap_matrix(2, 0) += 0.04;
    input.ao_overlap_matrix(1, 4) -= 0.03;
    input.ao_overlap_matrix(4, 1) -= 0.03;
    input.orbital_basis_counts = {3, 4, 3};
    input.orbital_basis_index_table = {1, 2, 4, 0, 0, 1, 2, 3, 5, 0, 2, 4, 5, 0, 0};
    input.orbital_value_table = {
        1.4, 0.2, 0.3, 0.0, 0.0, 0.1, 0.8, 0.3, 0.1, 0.0, 0.1, 1.3, 0.4, 0.0, 0.0};
    return input;
}

Eigen::MatrixXd overlap(const OrbitalPreparationInput& input) {
    const auto result = ActiveSpaceOrbitalPreparer().prepare(input);
    const Eigen::Index n = static_cast<Eigen::Index>(input.n_active_orbitals);
    require(result.active_orbital_overlap_matrix.size() == static_cast<std::size_t>(n * n),
            "unexpected active overlap size");
    return Eigen::Map<const Eigen::MatrixXd>(result.active_orbital_overlap_matrix.data(), n, n);
}

OrbitalPreparationInput displaced(const OrbitalPreparationInput& input,
                                  const SparseParameterLayout& layout,
                                  const Eigen::VectorXd& direction,
                                  double step) {
    auto result = input;
    layout.unpack(layout.pack(input) + step * direction, &result);
    return result;
}

Eigen::VectorXd gradient_at(const OrbitalPreparationInput& input, const Eigen::MatrixXd& upstream) {
    const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
    const SparseParameterLayout layout(input);
    return OrbitalOverlapPullback(input, prepared, layout).gradient(upstream);
}

Eigen::MatrixXd independent_upper(const Eigen::MatrixXd& overlap_matrix) {
    Eigen::LLT<Eigen::MatrixXd> factor(overlap_matrix);
    require(factor.info() == Eigen::Success, "Cholesky test fixture lost positive definiteness");
    return factor.matrixU();
}

void check_cholesky_pullback() {
    Eigen::Matrix3d overlap_matrix;
    overlap_matrix << 1.34, 0.21, -0.16, 0.21, 1.67, 0.29, -0.16, 0.29, 1.22;
    Eigen::Matrix3d direction;
    direction << 0.31, -0.24, 0.13, -0.24, -0.18, 0.27, 0.13, 0.27, 0.22;
    Eigen::Matrix3d upper_gradient = Eigen::Matrix3d::Zero();
    upper_gradient(0, 0) = 0.62;
    upper_gradient(0, 1) = -0.37;
    upper_gradient(0, 2) = 0.48;
    upper_gradient(1, 1) = -0.54;
    upper_gradient(1, 2) = 0.23;
    upper_gradient(2, 2) = 0.71;
    Eigen::Matrix3d upper_gradient_direction = Eigen::Matrix3d::Zero();
    upper_gradient_direction(0, 0) = -0.16;
    upper_gradient_direction(0, 1) = 0.39;
    upper_gradient_direction(0, 2) = -0.22;
    upper_gradient_direction(1, 1) = 0.44;
    upper_gradient_direction(1, 2) = -0.31;
    upper_gradient_direction(2, 2) = 0.25;
    require((upper_gradient - upper_gradient.transpose()).norm() > 0.1,
            "Cholesky upstream fixture is accidentally symmetric");

    const Eigen::MatrixXd r = overlap_upper(overlap_matrix);
    require(relative_error(r, independent_upper(overlap_matrix)) < 1e-14,
            "Cholesky upper factor differs from independent LLT");
    require(relative_error(r.transpose() * r, overlap_matrix) < 1e-14,
            "Cholesky upper factor does not reconstruct the active overlap");

    constexpr double h = 1e-6;
    const Eigen::MatrixXd dr = overlap_upper_direction(r, direction);
    const Eigen::MatrixXd finite_dr = (independent_upper(overlap_matrix + h * direction) -
                                       independent_upper(overlap_matrix - h * direction)) /
                                      (2.0 * h);
    require(relative_error(dr, finite_dr) < 2e-9,
            "Cholesky upper direction differs from independent LLT differences");
    require(dr.triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm() < 1e-14,
            "Cholesky upper direction has lower-triangular entries");

    const Eigen::MatrixXd gradient = overlap_upper_gradient(r, upper_gradient);
    const double dual = (gradient.array() * direction.array()).sum();
    const double tangent = (upper_gradient.array() * dr.array()).sum();
    require(std::abs(dual - tangent) < 1e-12, "Cholesky upper VJP violates Frobenius duality");
    const double finite_scalar =
        ((upper_gradient.array() * independent_upper(overlap_matrix + h * direction).array())
             .sum() -
         (upper_gradient.array() * independent_upper(overlap_matrix - h * direction).array())
             .sum()) /
        (2.0 * h);
    require(std::abs(dual - finite_scalar) < 2e-9,
            "Cholesky upper VJP differs from scalar LLT differences");
    require(relative_error(gradient, gradient.transpose()) < 1e-14,
            "Cholesky overlap adjoint is not symmetric");

    const Eigen::MatrixXd analytic_gradient_direction =
        overlap_upper_gradient_direction(r, dr, upper_gradient, upper_gradient_direction);
    const Eigen::MatrixXd finite_gradient_direction =
        (overlap_upper_gradient(independent_upper(overlap_matrix + h * direction),
                                upper_gradient + h * upper_gradient_direction) -
         overlap_upper_gradient(independent_upper(overlap_matrix - h * direction),
                                upper_gradient - h * upper_gradient_direction)) /
        (2.0 * h);
    require(relative_error(analytic_gradient_direction, finite_gradient_direction) < 2e-8,
            "Cholesky VJP direction differs from LLT finite differences");
    require(
        (analytic_gradient_direction - overlap_upper_gradient(r, upper_gradient_direction)).norm() >
            1e-3,
        "Cholesky fixture does not expose factor-dependent VJP curvature");
}

void check_sparse_overlap_pullback() {
    const auto input = sparse_fixture();
    const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
    const SparseParameterLayout layout(input);
    const OrbitalOverlapPullback pullback(input, prepared, layout);
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(layout.size(), -0.72, 0.61);
    u.normalize();
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(layout.size(), 0.39, -0.83);
    v.normalize();
    Eigen::Matrix2d g;
    g << 0.7, -0.4, 0.2, 1.1;
    Eigen::Matrix2d dg;
    dg << -0.3, 0.6, 0.1, 0.5;
    const Eigen::Matrix2d skew = g - g.transpose();
    require(skew.norm() > 0.1, "upstream fixture is accidentally symmetric");
    require((overlap(input) - overlap(input).transpose()).norm() < 1e-13,
            "preparer did not return a symmetric active overlap");

    constexpr double h = 1e-5;
    const Eigen::MatrixXd finite_direction =
        (overlap(displaced(input, layout, u, h)) - overlap(displaced(input, layout, u, -h))) /
        (2.0 * h);
    const Eigen::MatrixXd analytic_direction = pullback.overlap_direction(u);
    require(relative_error(analytic_direction, finite_direction) < 2e-8,
            "active overlap direction differs from preparation finite difference");
    require(relative_error(analytic_direction, analytic_direction.transpose()) < 1e-12,
            "active overlap direction is not symmetric");

    const Eigen::VectorXd analytic_gradient = pullback.gradient(g);
    Eigen::VectorXd finite_gradient(layout.size());
    for (Eigen::Index j = 0; j < finite_gradient.size(); ++j) {
        const Eigen::VectorXd basis = Eigen::VectorXd::Unit(layout.size(), j);
        const double plus = (g.array() * overlap(displaced(input, layout, basis, h)).array()).sum();
        const double minus =
            (g.array() * overlap(displaced(input, layout, basis, -h)).array()).sum();
        finite_gradient[j] = (plus - minus) / (2.0 * h);
    }
    require(relative_error(analytic_gradient, finite_gradient) < 2e-8,
            "active overlap VJP differs from independent scalar differences");
    require(std::abs(analytic_gradient.dot(v) -
                     (g.array() * pullback.overlap_direction(v).array()).sum()) < 1e-11,
            "active overlap VJP violates directional duality");
    require(pullback.gradient(g - g.transpose()).norm() < 1e-11,
            "antisymmetric active-overlap adjoint did not cancel");

    const Eigen::VectorXd analytic_gradient_direction = pullback.gradient_direction(g, dg, u);
    const Eigen::VectorXd finite_gradient_direction =
        (gradient_at(displaced(input, layout, u, h), g + h * dg) -
         gradient_at(displaced(input, layout, u, -h), g - h * dg)) /
        (2.0 * h);
    require(relative_error(analytic_gradient_direction, finite_gradient_direction) < 3e-7,
            "active overlap gradient direction differs from finite differences");
    require((analytic_gradient_direction - pullback.gradient(dg)).norm() > 1e-4,
            "fixture does not expose orbital-dependent pullback curvature");
}

} // namespace

int main() {
    try {
        check_cholesky_pullback();
        check_sparse_overlap_pullback();
        std::cout << "Cholesky and sparse active-overlap pullbacks: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
