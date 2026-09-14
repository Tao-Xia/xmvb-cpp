#include "vbscf/optimization/trust_region/retraction.hpp"

#include <cmath>

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  const Eigen::VectorXd full_tangent =
      space.expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  Eigen::VectorXd packed_tangent = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(parameter_view.size()));
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_tangent.size();
       ++packed_index) {
    packed_tangent[packed_index] =
        full_tangent[differentiable_indices[packed_index]];
  }
  return packed_tangent;
}

double nonredundant_step_norm(const Eigen::VectorXd& reduced_step) {
  if (reduced_step.size() == 0) {
    return 0.0;
  }
  // OrbitalChart whitens reduced coordinates in the accepted-point
  // normalized-orbital AO metric. Their Euclidean norm is therefore the
  // physical trust-region norm; the raw packed tangent norm is coordinate-
  // scale dependent and must not control globalization.
  const double tangent_norm = reduced_step.stableNorm();
  return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

Eigen::VectorXd NonredundantRetractionMetric::tangent(
    const Eigen::VectorXd& reduced_step) const {
  return reduced_step;
}

double NonredundantRetractionMetric::norm(
    const Eigen::VectorXd& reduced_step) const {
  const double tangent_norm = reduced_step.stableNorm();
  return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

Eigen::VectorXd NonredundantRetractionMetric::clip_to_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius) const {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm = norm(reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm <= trust_radius) {
    return reduced_step;
  }
  return (trust_radius / tangent_norm) * reduced_step;
}

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  constexpr double kInitialStepSafetyFraction = 0.95;
  const double target_radius = kInitialStepSafetyFraction * trust_radius;
  if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm = nonredundant_step_norm(reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm < target_radius) {
    return reduced_step;
  }
  return (target_radius / tangent_norm) * reduced_step;
}

}  // namespace xmvb::vb
