#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/optimization/objective/secant_hvp.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  xmvb::vb::ExactHvpOperator::Diagnostics response_cost;
  response_cost.estimated_core_pair_contraction_work = 100.0;
  response_cost.estimated_outer_string_contraction_work = 80.0;
  require(
      xmvb::vb::outer_response_scale_is_affordable(response_cost),
      "affordable response work was rejected");
  response_cost.estimated_outer_string_contraction_work = 120.0;
  require(
      !xmvb::vb::outer_response_scale_is_affordable(response_cost),
      "response work larger than core work was admitted");

  xmvb::vb::SymmetricSecantCorrection correction;
  Eigen::Vector2d step;
  step << 1.0, 0.0;
  Eigen::Vector2d target;
  target << 2.0, 3.0;
  require(correction.add_pair(step, target), "secant pair was rejected");
  require(
      (correction.apply(step) - target).norm() <= 1.0e-14,
      "symmetric correction does not satisfy its secant");

  Eigen::Vector2d left;
  left << -0.4, 0.7;
  Eigen::Vector2d right;
  right << 0.2, -0.9;
  const double symmetry_error = std::abs(
      left.dot(correction.apply(right)) -
      right.dot(correction.apply(left)));
  require(symmetry_error <= 1.0e-14, "secant correction is not symmetric");

  Eigen::Matrix<double, 2, 2> vectors;
  vectors << 0.3, -0.8,
             0.6,  0.1;
  Eigen::Matrix<double, 2, 2> scalar_images;
  scalar_images.col(0) = correction.apply(vectors.col(0));
  scalar_images.col(1) = correction.apply(vectors.col(1));
  require(
      (correction.apply_batch(vectors) - scalar_images).norm() <= 1.0e-14,
      "batched secant correction disagrees with scalar actions");
  return 0;
}
