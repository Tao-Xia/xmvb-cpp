#include "vbscf/optimization/objective/secant_hvp.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"

namespace xmvb::vb {

Eigen::VectorXd SymmetricSecantCorrection::apply(
    const Eigen::VectorXd& vector) const {
  Eigen::VectorXd image = Eigen::VectorXd::Zero(vector.size());
  for (const auto& update : updates_) {
    const double step_projection = update.step.dot(vector);
    const double residual_projection = update.residual.dot(vector);
    const double scaled_step_projection =
        update.inverse_step_norm_squared * step_projection;
    image.noalias() += update.residual * scaled_step_projection;
    image.noalias() += update.step *
        (update.inverse_step_norm_squared * residual_projection -
         update.residual_step_inner_product *
             update.inverse_step_norm_squared * scaled_step_projection);
  }
  return image;
}

Eigen::MatrixXd SymmetricSecantCorrection::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  Eigen::MatrixXd images = Eigen::MatrixXd::Zero(
      vectors.rows(), vectors.cols());
  for (const auto& update : updates_) {
    const Eigen::RowVectorXd step_projection =
        update.step.transpose() * vectors;
    const Eigen::RowVectorXd residual_projection =
        update.residual.transpose() * vectors;
    images.noalias() += update.residual *
        (update.inverse_step_norm_squared * step_projection);
    images.noalias() += update.step *
        (update.inverse_step_norm_squared * residual_projection -
         update.residual_step_inner_product *
             update.inverse_step_norm_squared *
             update.inverse_step_norm_squared * step_projection);
  }
  return images;
}

bool SymmetricSecantCorrection::add_pair(
    const Eigen::VectorXd& step,
    const Eigen::VectorXd& target_image) {
  if (step.size() == 0 || target_image.size() != step.size() ||
      !step.allFinite() || !target_image.allFinite()) {
    return false;
  }
  const double step_norm_squared = step.squaredNorm();
  if (!(step_norm_squared > 0.0) || !std::isfinite(step_norm_squared)) {
    return false;
  }
  Eigen::VectorXd residual = target_image - apply(step);
  const double residual_norm = residual.stableNorm();
  const double target_scale = std::max(1.0, target_image.stableNorm());
  if (!std::isfinite(residual_norm) ||
      residual_norm <= std::sqrt(std::numeric_limits<double>::epsilon()) *
          target_scale) {
    return false;
  }

  Update update;
  update.step = step;
  update.residual = std::move(residual);
  update.inverse_step_norm_squared = 1.0 / step_norm_squared;
  update.residual_step_inner_product =
      update.residual.dot(update.step);
  updates_.push_back(std::move(update));
  return true;
}

int SymmetricSecantCorrection::size() const noexcept {
  return static_cast<int>(updates_.size());
}

SecantCorrectedCoreHvp::SecantCorrectedCoreHvp(
    ExactReducedHvp* exact_hvp,
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size)
    : exact_hvp_(exact_hvp) {
  if (exact_hvp_ == nullptr) {
    throw std::invalid_argument(
        "secant-corrected core HVP requires a core operator");
  }
  const auto pairs = transport_nonredundant_secant_pairs(
      current_space,
      packed_secant_history,
      max_history_size);
  if (pairs.empty()) {
    return;
  }

  const Eigen::Index reduced_size = current_space.reduced_size();
  Eigen::MatrixXd steps(reduced_size, pairs.size());
  Eigen::MatrixXd gradient_changes(reduced_size, pairs.size());
  Eigen::Index n_valid = 0;
  for (const auto& pair : pairs) {
    if (pair.step.size() != reduced_size ||
        pair.gradient_change.size() != reduced_size ||
        !pair.step.allFinite() || !pair.gradient_change.allFinite() ||
        !(pair.step.squaredNorm() > 0.0)) {
      continue;
    }
    steps.col(n_valid) = pair.step;
    gradient_changes.col(n_valid) = pair.gradient_change;
    ++n_valid;
  }
  if (n_valid == 0) {
    return;
  }
  steps.conservativeResize(Eigen::NoChange, n_valid);
  gradient_changes.conservativeResize(Eigen::NoChange, n_valid);
  const Eigen::MatrixXd core_images = exact_hvp_->apply_core_batch(steps);
  for (Eigen::Index column = 0; column < n_valid; ++column) {
    correction_.add_pair(
        steps.col(column),
        gradient_changes.col(column) - core_images.col(column));
  }
}

Eigen::VectorXd SecantCorrectedCoreHvp::apply(
    const Eigen::VectorXd& reduced_direction) {
  return exact_hvp_->apply_core(reduced_direction) +
      correction_.apply(reduced_direction);
}

Eigen::MatrixXd SecantCorrectedCoreHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_hvp_->apply_core_batch(reduced_directions) +
      correction_.apply_batch(reduced_directions);
}

int SecantCorrectedCoreHvp::correction_size() const noexcept {
  return correction_.size();
}

}  // namespace xmvb::vb
