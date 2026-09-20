#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/krylov/minres.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& reference,
    double tolerance,
    const char* message) {
  const double scale = std::max(1.0, reference.stableNorm());
  require((actual - reference).stableNorm() <= tolerance * scale, message);
}

}  // namespace

int main() {
  try {
    using xmvb::vb::MinresOptions;
    using xmvb::vb::MinresStopReason;
    using xmvb::vb::solve_symmetric_minres;

    // A symmetric-indefinite system exercises both signs of curvature.
    Eigen::MatrixXd h(5, 5);
    h << 4.0, 1.0, 0.0, 0.0, 0.2,
         1.0, -3.0, 0.5, 0.0, 0.0,
         0.0, 0.5, 2.0, 0.3, 0.0,
         0.0, 0.0, 0.3, -1.5, 0.4,
         0.2, 0.0, 0.0, 0.4, 1.0;
    const Eigen::VectorXd rhs =
        (Eigen::VectorXd(5) << 1.0, -2.0, 0.5, 3.0, -1.0).finished();
    const Eigen::VectorXd reference = h.fullPivLu().solve(rhs);
    MinresOptions strict;
    strict.relative_residual_tolerance = 1.0e-12;
    const auto indefinite = solve_symmetric_minres(
        [&](const Eigen::VectorXd& vector) { return h * vector; }, rhs, strict);
    require(indefinite.converged(), "indefinite MINRES did not converge");
    require_close(indefinite.solution, reference, 1.0e-11,
                  "indefinite MINRES solution is inaccurate");
    require_close(indefinite.residual, rhs - h * indefinite.solution, 1.0e-15,
                  "MINRES did not return its explicit residual");
    require_close(indefinite.operator_image, h * indefinite.solution, 1.0e-15,
                  "MINRES did not return its final operator image");
    require(indefinite.residual_norm <= indefinite.residual_target,
            "MINRES convergence was not explicitly certified");

    // A symmetric block action may shift only its leading block.
    Eigen::Matrix2d a;
    a << 2.0, 0.3, 0.3, 1.4;
    Eigen::Matrix2d metric;
    metric << 1.5, 0.1, 0.1, 0.9;
    Eigen::Matrix<double, 3, 2> b;
    b << 0.5, -0.2,
         0.1, 0.4,
         -0.3, 0.2;
    Eigen::Matrix3d c;
    c << -1.2, 0.1, 0.0,
         0.1, 0.8, -0.2,
         0.0, -0.2, -0.6;
    constexpr double lambda = 0.7;
    Eigen::MatrixXd block_matrix(5, 5);
    block_matrix.topLeftCorner<2, 2>() = a + lambda * metric;
    block_matrix.topRightCorner<2, 3>() = b.transpose();
    block_matrix.bottomLeftCorner<3, 2>() = b;
    block_matrix.bottomRightCorner<3, 3>() = c;
    const Eigen::VectorXd block_rhs =
        (Eigen::VectorXd(5) << -1.0, 0.7, 0.0, 0.0, 0.0).finished();
    const auto block_result = solve_symmetric_minres(
        [&](const Eigen::VectorXd& vector) {
          Eigen::VectorXd image(5);
          const Eigen::Vector2d p = vector.head<2>();
          const Eigen::Vector3d z = vector.tail<3>();
          image.head<2>() = a * p + lambda * metric * p + b.transpose() * z;
          image.tail<3>() = b * p + c * z;
          return image;
        },
        block_rhs,
        strict);
    require(block_result.converged(),
            "symmetric block MINRES did not converge");
    require_close(
        block_result.solution,
        block_matrix.fullPivLu().solve(block_rhs),
        1.0e-11,
        "symmetric block MINRES solution is inaccurate");

    // A positive inverse preconditioner is valid even though A is indefinite.
    const Eigen::VectorXd inverse_diagonal =
        h.diagonal().cwiseAbs().cwiseMax(0.5).cwiseInverse();
    const auto preconditioned = solve_symmetric_minres(
        [&](const Eigen::VectorXd& vector) { return h * vector; },
        rhs,
        strict,
        [&](const Eigen::VectorXd& vector) {
          return inverse_diagonal.array() * vector.array();
        });
    require(preconditioned.converged(),
            "preconditioned indefinite MINRES did not converge");
    require_close(preconditioned.solution, reference, 1.0e-11,
                  "preconditioned MINRES solution is inaccurate");
    require(preconditioned.preconditioner_actions > 0,
            "inverse preconditioner was not applied");

    MinresOptions limited = strict;
    limited.maximum_iterations = 1;
    const auto incomplete = solve_symmetric_minres(
        [&](const Eigen::VectorXd& vector) { return h * vector; }, rhs, limited);
    require(!incomplete.converged() &&
                incomplete.stop_reason == MinresStopReason::IterationLimit,
            "work-limited MINRES reported false convergence");
    require_close(incomplete.residual, rhs - h * incomplete.solution, 1.0e-15,
                  "work-limited MINRES residual is not explicit");

    std::cout << "symmetric MINRES: passed (indefinite, block, preconditioned)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
