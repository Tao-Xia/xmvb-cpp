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

NonredundantRetractionMetric::NonredundantRetractionMetric(
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const OrbitalPreparationInput& input)
    : space_(space),
      parameter_view_(parameter_view),
      physical_metric_(input) {}

Eigen::VectorXd NonredundantRetractionMetric::tangent(
    const Eigen::VectorXd& reduced_step) const {
  return reduced_step;
}

Eigen::VectorXd NonredundantRetractionMetric::apply(
    const Eigen::VectorXd& reduced_step) const {
  return space_.project_reduced_gradient(
      physical_metric_.apply(
          parameter_view_, space_.expand_step(reduced_step)));
}

double NonredundantRetractionMetric::norm(
    const Eigen::VectorXd& reduced_step) const {
  const double squared_norm = physical_metric_.squared_norm(
      parameter_view_, space_.expand_step(reduced_step));
  const double tangent_norm = std::sqrt(squared_norm);
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
    double trust_radius,
    const NonredundantRetractionMetric& metric) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  constexpr double kInitialStepSafetyFraction = 0.95;
  const double target_radius = kInitialStepSafetyFraction * trust_radius;
  if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm = metric.norm(reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm < target_radius) {
    return reduced_step;
  }
  return (target_radius / tangent_norm) * reduced_step;
}

}  // namespace xmvb::vb
