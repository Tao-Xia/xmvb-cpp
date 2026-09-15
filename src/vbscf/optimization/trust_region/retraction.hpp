#pragma once

#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalChart& space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

/**
 * @brief Accepted-point pullback metric of the additive sparse retraction.
 *
 * The algebraic quotient chart supplies the horizontal raw-coefficient lift.
 * This object measures that same lift after the inactive-subspace and active-
 * ray physical map; it never stores a full reduced metric matrix.
 */
class NonredundantRetractionMetric {
public:
  NonredundantRetractionMetric(
      const OrbitalChart& space,
      const SparseParameterLayout& parameter_view,
      const OrbitalPreparationInput& input);

  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_step) const;
  double norm(const Eigen::VectorXd& reduced_step) const;
  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const;
private:
  const OrbitalChart& space_;
  const SparseParameterLayout& parameter_view_;
  OrbitalPhysicalMetric physical_metric_;
};

Eigen::VectorXd shrink_reduced_step_inside_trust_radius(
    const Eigen::VectorXd& reduced_step,
    double trust_radius,
    const NonredundantRetractionMetric& metric);

}  // namespace xmvb::vb
