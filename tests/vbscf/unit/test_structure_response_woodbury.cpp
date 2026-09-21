#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/preconditioners/structure_response_woodbury.hpp"

namespace {

using xmvb::vb::StructureResponseWoodburyPreconditioner;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::MatrixXd& actual,
    const Eigen::MatrixXd& reference,
    double tolerance,
    const std::string& message) {
  require(
      (actual - reference).stableNorm() <=
          tolerance * std::max(1.0, reference.stableNorm()),
      message);
}

void check_dense_schur_inverse() {
  Eigen::Matrix4d base = Eigen::Matrix4d::Zero();
  base.diagonal() << 0.8, 0.5, 1.1, 0.6;
  Eigen::Matrix<double, 4, 2> coupling;
  coupling << 0.4, -0.1,
              0.2,  0.3,
             -0.2,  0.5,
              0.1,  0.2;
  Eigen::Matrix2d structure;
  structure << 2.4, 0.2,
               0.2, 1.7;
  const StructureResponseWoodburyPreconditioner preconditioner(
      coupling,
      structure,
      [base](const Eigen::MatrixXd& rhs) { return base * rhs; });
  require(preconditioner.available(),
          "regular Woodbury preconditioner is unavailable");
  require(preconditioner.retained_rank() == 2,
          "regular Woodbury system lost rank");
  require(preconditioner.pseudoinverse_residual() <= 1.0e-12,
          "regular Woodbury pseudoinverse residual is too large");

  Eigen::Matrix<double, 4, 3> rhs;
  rhs << 0.2, -0.4, 0.5,
         0.7,  0.1, 0.3,
        -0.3,  0.6, 0.4,
         0.8, -0.2, 0.1;
  const Eigen::MatrixXd small =
      structure - coupling.transpose() * base * coupling;
  const Eigen::MatrixXd explicit_update = base +
      base * coupling * small.inverse() * coupling.transpose() * base;
  require_close(
      preconditioner.apply_block(rhs), explicit_update * rhs, 2.0e-12,
      "Woodbury block action differs from its dense formula");
  require_close(
      preconditioner.apply(rhs.col(1)), explicit_update * rhs.col(1),
      2.0e-12,
      "Woodbury vector action differs from its dense formula");

  const Eigen::MatrixXd explicit_schur =
      base.inverse() -
      coupling * structure.inverse() * coupling.transpose();
  require_close(
      explicit_update, explicit_schur.inverse(), 2.0e-12,
      "Woodbury update is not the inverse Schur complement");
}

void check_symmetric_pseudoinverse_cutoff() {
  const Eigen::Matrix3d base = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 2> coupling =
      Eigen::Matrix<double, 3, 2>::Zero();
  coupling(0, 0) = 1.0;
  coupling(1, 1) = 1.0;
  Eigen::Matrix2d structure = Eigen::Matrix2d::Zero();
  structure(0, 0) = 2.0;
  structure(1, 1) = 1.0;
  const StructureResponseWoodburyPreconditioner preconditioner(
      coupling,
      structure,
      [base](const Eigen::MatrixXd& rhs) { return base * rhs; });
  require(preconditioner.available(),
          "singular symmetric Woodbury system is unavailable");
  require(preconditioner.retained_rank() == 1,
          "symmetric spectral cutoff retained a null mode");
  require(preconditioner.spectral_cutoff() > 0.0,
          "symmetric spectral cutoff was not recorded");

  const Eigen::Vector3d rhs(0.4, 0.7, -0.2);
  Eigen::Matrix3d reference = Eigen::Matrix3d::Identity();
  reference(0, 0) = 2.0;
  require_close(
      preconditioner.apply(rhs), reference * rhs, 2.0e-12,
      "singular Woodbury action differs from the spectral pseudoinverse");
}

void check_invalid_symmetry_is_explicitly_unavailable() {
  Eigen::Matrix2d nonsymmetric_base;
  nonsymmetric_base << 1.0, 1.0,
                       0.0, 1.0;
  const Eigen::Matrix2d two_direction_coupling =
      Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d two_direction_structure =
      3.0 * Eigen::Matrix2d::Identity();
  const StructureResponseWoodburyPreconditioner preconditioner(
      two_direction_coupling,
      two_direction_structure,
      [nonsymmetric_base](const Eigen::MatrixXd& rhs) {
        return nonsymmetric_base * rhs;
      });
  require(!preconditioner.available(),
          "nonsymmetric base inverse was accepted");
  require(!preconditioner.unavailability_reason().empty(),
          "unavailable Woodbury object has no diagnostic");
  bool rejected = false;
  try {
    static_cast<void>(preconditioner.apply(Eigen::Vector2d::Ones()));
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("unavailable") !=
        std::string::npos;
  }
  require(rejected,
          "unavailable Woodbury object silently used the base inverse");
}

void check_nonsymmetric_structure_is_unavailable() {
  const Eigen::Matrix2d coupling = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d structure;
  structure << 2.0, 0.4,
               0.0, 1.5;
  const StructureResponseWoodburyPreconditioner preconditioner(
      coupling,
      structure,
      [](const Eigen::MatrixXd& rhs) { return rhs; });
  require(!preconditioner.available(),
          "nonsymmetric structure block was accepted");
  require(preconditioner.unavailability_reason().find("symmetric") !=
              std::string::npos,
          "nonsymmetric structure rejection lacks a diagnostic");
}

void check_failed_application_never_falls_back() {
  int calls = 0;
  const StructureResponseWoodburyPreconditioner preconditioner(
      Eigen::Matrix2d::Identity(),
      3.0 * Eigen::Matrix2d::Identity(),
      [&calls](const Eigen::MatrixXd& rhs) {
        ++calls;
        return calls == 1
            ? rhs
            : Eigen::MatrixXd::Zero(rhs.rows() + 1, rhs.cols());
      });
  require(preconditioner.available(),
          "valid initial base inverse was rejected");
  bool rejected = false;
  try {
    static_cast<void>(preconditioner.apply(Eigen::Vector2d::Ones()));
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("unavailable") !=
        std::string::npos;
  }
  require(rejected,
          "failed Woodbury application silently returned a base step");
}

}  // namespace

int main() {
  try {
    check_dense_schur_inverse();
    check_symmetric_pseudoinverse_cutoff();
    check_invalid_symmetry_is_explicitly_unavailable();
    check_nonsymmetric_structure_is_unavailable();
    check_failed_application_never_falls_back();
    std::cout << "structure-response Woodbury tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "structure-response Woodbury test failed: "
              << error.what() << '\n';
    return 1;
  }
}
