#pragma once

#include <Eigen/Core>

#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

double compute_nonredundant_retract_tangent_norm(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

class NonredundantRetractionMetric {
public:
  NonredundantRetractionMetric(
      const OrbitalPreparationInput& orbital_preparation_input,
      const OrbitalChart& space,
      const SparseParameterLayout& parameter_view);

  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const;
  double norm(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const;

private:
  const OrbitalPreparationInput& orbital_preparation_input_;
  const OrbitalChart& space_;
  const SparseParameterLayout& parameter_view_;
};

Eigen::VectorXd shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step,
    double trust_radius);

}  // namespace xmvb::vb
