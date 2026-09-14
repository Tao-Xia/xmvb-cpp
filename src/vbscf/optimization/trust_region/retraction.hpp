#pragma once

#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

double nonredundant_step_norm(const Eigen::VectorXd& reduced_step);

class NonredundantRetractionMetric {
public:
  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const;
  double norm(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const;

};

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius);

}  // namespace xmvb::vb
