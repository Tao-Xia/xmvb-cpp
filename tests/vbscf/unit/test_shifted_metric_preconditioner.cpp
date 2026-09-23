#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "vbscf/optimization/preconditioners/shifted_metric.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_scaled_inner_budget() {
  Eigen::Matrix3d model;
  model << 3.0, 0.0, 0.0,
           0.0, 1.8, 0.0,
           0.0, 0.0, 0.9;
  Eigen::Matrix3d metric;
  metric << 1.4, 0.35, -0.1,
            0.35, 1.1, 0.2,
           -0.1, 0.2, 0.8;
  const Eigen::Vector3d rhs(0.7, -1.2, 0.4);
  constexpr double shift = 0.65;
  constexpr double target = 1.0e-4;
  const Eigen::Vector3d result =
      xmvb::vb::apply_shifted_metric_preconditioner(
          rhs,
          shift,
          target,
          [model](const Eigen::VectorXd& vector) {
            return (model * vector).eval();
          },
          [metric](const Eigen::VectorXd& vector) {
            return (metric * vector).eval();
          },
          [model, shift](const Eigen::VectorXd& vector) {
            return (model + shift * Eigen::Matrix3d::Identity())
                .ldlt()
                .solve(vector);
          });
  const Eigen::Vector3d residual =
      rhs - (model + shift * metric) * result;
  require(residual.norm() <= std::sqrt(target * rhs.norm()),
          "preconditioner exceeded its scaled inner residual budget");
}

void check_zero_shift_uses_block_inverse() {
  const Eigen::Vector2d diagonal(2.0, 5.0);
  const Eigen::Vector2d rhs(1.0, -3.0);
  int model_calls = 0;
  const Eigen::Vector2d result =
      xmvb::vb::apply_shifted_metric_preconditioner(
          rhs,
          0.0,
          0.1,
          [diagonal, &model_calls](const Eigen::VectorXd& vector) {
            ++model_calls;
            return (diagonal.asDiagonal() * vector).eval();
          },
          [](const Eigen::VectorXd& vector) { return vector; },
          [diagonal](const Eigen::VectorXd& vector) {
            return (vector.array() / diagonal.array()).matrix();
          });
  require(model_calls == 0, "zero shift performed an iterative solve");
  require(
      (result - Eigen::Vector2d(0.5, -0.6)).norm() < 1.0e-15,
      "zero-shift block inverse is incorrect");
}

}  // namespace

int main() {
  try {
    check_scaled_inner_budget();
    check_zero_shift_uses_block_inverse();
    std::cout << "shifted-metric preconditioner tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "shifted-metric preconditioner test failed: "
              << error.what() << '\n';
    return 1;
  }
}
