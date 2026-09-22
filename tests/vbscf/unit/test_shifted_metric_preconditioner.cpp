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

void check_nonidentity_metric_shift() {
  Eigen::Matrix3d model;
  model << 3.0, 0.0, 0.0,
           0.0, 1.8, 0.0,
           0.0, 0.0, 0.9;
  Eigen::Matrix3d metric;
  metric << 1.4, 0.35, -0.1,
            0.35, 1.1, 0.2,
           -0.1, 0.2, 0.8;
  const Eigen::Vector3d rhs(0.7, -1.2, 0.4);
  const double shift = 0.65;
  const Eigen::Vector3d result =
      xmvb::vb::apply_inverse_shifted_metric_model(
          rhs,
          shift,
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
  const Eigen::Vector3d reference =
      (model + shift * metric).ldlt().solve(rhs);
  require(
      (result - reference).norm() <= 1.0e-12 * reference.norm(),
      "shifted preconditioner replaced the physical metric by identity");
}

void check_zero_shift_uses_local_inverse() {
  const Eigen::Vector2d diagonal(2.0, 5.0);
  const Eigen::Vector2d rhs(1.0, -3.0);
  int model_calls = 0;
  const Eigen::Vector2d result =
      xmvb::vb::apply_inverse_shifted_metric_model(
          rhs,
          0.0,
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
      "zero-shift local inverse is incorrect");
}

}  // namespace

int main() {
  try {
    check_nonidentity_metric_shift();
    check_zero_shift_uses_local_inverse();
    std::cout << "shifted-metric preconditioner tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "shifted-metric preconditioner test failed: "
              << error.what() << '\n';
    return 1;
  }
}
