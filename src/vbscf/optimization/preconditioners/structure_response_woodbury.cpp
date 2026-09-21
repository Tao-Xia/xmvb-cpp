#include "vbscf/optimization/preconditioners/structure_response_woodbury.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Eigenvalues>

namespace xmvb::vb {
namespace {

double symmetry_tolerance(const Eigen::MatrixXd& matrix) {
  return 256.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, matrix.rows())) *
      std::max(std::numeric_limits<double>::min(), matrix.stableNorm());
}

}  // namespace

StructureResponseWoodburyPreconditioner::
    StructureResponseWoodburyPreconditioner(
    Eigen::MatrixXd coupling,
    Eigen::MatrixXd structure_block,
    BlockInverseAction base_inverse)
    : coupling_(std::move(coupling)),
      base_inverse_(std::move(base_inverse)) {
  if (coupling_.rows() <= 0 || coupling_.cols() <= 0 ||
      structure_block.rows() != coupling_.cols() ||
      structure_block.cols() != coupling_.cols() ||
      !coupling_.allFinite() || !structure_block.allFinite() ||
      !base_inverse_) {
    throw std::invalid_argument(
        "invalid structure-response Woodbury data");
  }

  const double structure_skew =
      (structure_block - structure_block.transpose()).stableNorm();
  if (structure_skew > symmetry_tolerance(structure_block)) {
    make_unavailable(
        "structure block is not numerically symmetric");
    return;
  }
  structure_block =
      0.5 * (structure_block + structure_block.transpose()).eval();

  try {
    base_coupling_ = checked_base_inverse(coupling_);
  } catch (const std::exception& error) {
    make_unavailable(error.what());
    return;
  }
  const Eigen::MatrixXd projected_base =
      coupling_.transpose() * base_coupling_;
  const double projected_skew =
      (projected_base - projected_base.transpose()).stableNorm();
  if (projected_skew > symmetry_tolerance(projected_base)) {
    make_unavailable(
        "base inverse is not symmetric on the response coupling space");
    return;
  }

  small_matrix_ = structure_block -
      0.5 * (projected_base + projected_base.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(small_matrix_);
  if (spectrum.info() != Eigen::Success ||
      !spectrum.eigenvalues().allFinite()) {
    make_unavailable("small symmetric eigensystem failed");
    return;
  }
  const double spectral_scale = std::max(
      std::numeric_limits<double>::min(),
      spectrum.eigenvalues().cwiseAbs().maxCoeff());
  spectral_cutoff_ =
      256.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, small_matrix_.rows())) *
      spectral_scale;

  Eigen::VectorXd inverse = Eigen::VectorXd::Zero(
      spectrum.eigenvalues().size());
  Eigen::VectorXd retained = Eigen::VectorXd::Zero(inverse.size());
  for (Eigen::Index index = 0; index < inverse.size(); ++index) {
    const double eigenvalue = spectrum.eigenvalues()[index];
    if (std::abs(eigenvalue) > spectral_cutoff_) {
      inverse[index] = 1.0 / eigenvalue;
      retained[index] = 1.0;
      ++retained_rank_;
    }
  }
  small_pseudoinverse_ = spectrum.eigenvectors() * inverse.asDiagonal() *
      spectrum.eigenvectors().transpose();
  small_range_projector_ = spectrum.eigenvectors() * retained.asDiagonal() *
      spectrum.eigenvectors().transpose();

  const Eigen::MatrixXd reconstruction =
      small_matrix_ * small_pseudoinverse_ * small_matrix_;
  pseudoinverse_residual_ =
      (reconstruction - small_matrix_).stableNorm();
  const double residual_target =
      2048.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, small_matrix_.rows())) *
      std::max(
          std::numeric_limits<double>::min(),
          small_matrix_.stableNorm());
  if (!small_pseudoinverse_.allFinite() ||
      !small_range_projector_.allFinite() ||
      !std::isfinite(pseudoinverse_residual_) ||
      pseudoinverse_residual_ > residual_target) {
    make_unavailable(
        "small-system spectral pseudoinverse failed its residual certificate");
    return;
  }
  available_ = true;
}

int StructureResponseWoodburyPreconditioner::dimension() const noexcept {
  return static_cast<int>(coupling_.rows());
}

int StructureResponseWoodburyPreconditioner::
response_dimension() const noexcept {
  return static_cast<int>(coupling_.cols());
}

int StructureResponseWoodburyPreconditioner::retained_rank() const noexcept {
  return retained_rank_;
}

double StructureResponseWoodburyPreconditioner::
spectral_cutoff() const noexcept {
  return spectral_cutoff_;
}

double StructureResponseWoodburyPreconditioner::
    pseudoinverse_residual() const noexcept {
  return pseudoinverse_residual_;
}

bool StructureResponseWoodburyPreconditioner::available() const noexcept {
  return available_;
}

const std::string& StructureResponseWoodburyPreconditioner::
    unavailability_reason() const noexcept {
  return unavailability_reason_;
}

Eigen::VectorXd StructureResponseWoodburyPreconditioner::apply(
    const Eigen::VectorXd& covector) const {
  if (covector.size() != coupling_.rows() || !covector.allFinite()) {
    throw std::invalid_argument(
        "invalid structure-response Woodbury covector");
  }
  Eigen::MatrixXd block = covector;
  return apply_block(block).col(0);
}

Eigen::MatrixXd StructureResponseWoodburyPreconditioner::apply_block(
    const Eigen::MatrixXd& covectors) const {
  require_available();
  if (covectors.rows() != coupling_.rows() || !covectors.allFinite()) {
    throw std::invalid_argument(
        "invalid structure-response Woodbury covector block");
  }
  if (covectors.cols() == 0) {
    return Eigen::MatrixXd::Zero(coupling_.rows(), 0);
  }
  Eigen::MatrixXd base;
  try {
    base = checked_base_inverse(covectors);
  } catch (const std::exception& error) {
    throw std::runtime_error(
        std::string(
            "structure-response Woodbury preconditioner is unavailable: ") +
        error.what());
  }
  const Eigen::MatrixXd right_hand_side = coupling_.transpose() * base;
  const Eigen::MatrixXd coordinates =
      small_pseudoinverse_ * right_hand_side;
  const Eigen::MatrixXd projected_right_hand_side =
      small_range_projector_ * right_hand_side;
  const double residual =
      (small_matrix_ * coordinates - projected_right_hand_side).stableNorm();
  const double residual_target =
      2048.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, small_matrix_.rows())) *
      std::max({
          std::numeric_limits<double>::min(),
          projected_right_hand_side.stableNorm(),
          (small_matrix_ * coordinates).stableNorm()});
  if (!coordinates.allFinite() || !std::isfinite(residual) ||
      residual > residual_target) {
    throw std::runtime_error(
        "structure-response Woodbury preconditioner is unavailable: "
        "small-system solve failed its residual certificate");
  }
  Eigen::MatrixXd result = base + base_coupling_ * coordinates;
  if (!result.allFinite()) {
    throw std::runtime_error(
        "structure-response Woodbury preconditioner is unavailable: "
        "updated inverse action is non-finite");
  }
  return result;
}

void StructureResponseWoodburyPreconditioner::make_unavailable(
    std::string reason) {
  available_ = false;
  unavailability_reason_ = std::move(reason);
}

Eigen::MatrixXd StructureResponseWoodburyPreconditioner::checked_base_inverse(
    const Eigen::MatrixXd& covectors) const {
  Eigen::MatrixXd result = base_inverse_(covectors);
  if (result.rows() != coupling_.rows() ||
      result.cols() != covectors.cols() || !result.allFinite()) {
    throw std::runtime_error(
        "base shifted-orbital inverse returned an invalid block");
  }
  return result;
}

void StructureResponseWoodburyPreconditioner::require_available() const {
  if (available_) return;
  throw std::runtime_error(
      "structure-response Woodbury preconditioner is unavailable: " +
      unavailability_reason_);
}

}  // namespace xmvb::vb
