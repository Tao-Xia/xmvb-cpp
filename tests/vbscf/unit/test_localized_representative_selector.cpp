#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/orbitals/gauge/localized.hpp"

namespace {

void require_close(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& expected,
    double tolerance,
    const char* label) {
  if (actual.rows() != expected.rows() ||
      actual.cols() != expected.cols() ||
      (actual - expected).cwiseAbs().maxCoeff() > tolerance) {
    throw std::runtime_error(label);
  }
}

}  // namespace

int main() {
  Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(4, 4);
  overlap.diagonal() << 1.0, 1.5, 0.8, 2.0;

  Eigen::MatrixXd inactive_frame(4, 2);
  inactive_frame <<
      1.0, 0.2,
      0.1, 0.9,
      0.3, -0.1,
      0.0, 0.25;
  Eigen::MatrixXd inactive_transform(2, 2);
  inactive_transform <<
      1.2, -0.3,
      0.4, 0.8;
  Eigen::MatrixXd active_coefficients(2, 1);
  active_coefficients << 0.35, -0.2;
  Eigen::MatrixXd active_auxiliary(4, 1);
  active_auxiliary << 0.15, -0.4, 0.8, 0.25;

  const Eigen::MatrixXd inactive_physical =
      inactive_frame * inactive_transform;
  const Eigen::MatrixXd active_physical =
      active_auxiliary + inactive_frame * active_coefficients;
  const Eigen::MatrixXd metric =
      inactive_frame.transpose() * overlap * inactive_frame;
  if ((metric - Eigen::MatrixXd::Identity(2, 2))
          .cwiseAbs()
          .maxCoeff() < 1.0e-2) {
    throw std::runtime_error("test frame is accidentally metric orthonormal");
  }

  const auto selector =
      xmvb::vb::build_localized_representative_selector(
          inactive_physical,
          inactive_frame,
          active_physical,
          active_auxiliary,
          overlap);
  require_close(
      selector.inactive_right_transform,
      inactive_transform,
      1.0e-12,
      "inactive selector did not solve through the metric Gram matrix");
  require_close(
      selector.active_inactive_coefficients,
      active_coefficients,
      1.0e-12,
      "active selector did not solve through the metric Gram matrix");
  require_close(
      selector.inactive_inverse_transpose_right_transform,
      inactive_transform.inverse().transpose(),
      1.0e-12,
      "inactive inverse-transpose selector is inconsistent");
  return 0;
}
