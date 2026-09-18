#include "vbscf/optimization/coupled/model_reduction.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace xmvb::vb {

CoupledModelReduction evaluate_coupled_model_reduction(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& accepted_orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& structure_residual,
    const Eigen::Ref<const Eigen::VectorXd>& response_correction,
    const Eigen::Ref<const Eigen::VectorXd>& remaining_structure_residual,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_step,
    const Eigen::Ref<const Eigen::VectorXd>& response_step,
    const Eigen::Ref<const Eigen::VectorXd>& incremental_operator_image) {
  const int n_orbitals = coupled_operator.n_orbital_coordinates();
  const int n_response = coupled_operator.n_response_coordinates();
  if (accepted_orbital_gradient.size() != n_orbitals ||
      orbital_step.size() != n_orbitals ||
      structure_residual.size() != n_response ||
      response_correction.size() != n_response ||
      remaining_structure_residual.size() != n_response ||
      response_step.size() != n_response ||
      incremental_operator_image.size() != coupled_operator.size() ||
      !accepted_orbital_gradient.allFinite() ||
      !orbital_step.allFinite() || !structure_residual.allFinite() ||
      !response_correction.allFinite() ||
      !remaining_structure_residual.allFinite() ||
      !response_step.allFinite() ||
      !incremental_operator_image.allFinite()) {
    throw std::invalid_argument(
        "coupled model-reduction inputs have inconsistent dimensions or values");
  }

  CoupledModelReduction result;
  result.corrected_orbital_gradient = accepted_orbital_gradient +
      coupled_operator.apply_response_to_orbital(
          response_correction).col(0);
  const Eigen::VectorXd response_hessian_correction =
      remaining_structure_residual - structure_residual;
  result.baseline_defect_model_change =
      structure_residual.dot(response_correction) +
      0.5 * response_correction.dot(response_hessian_correction);
  result.remaining_structure_linear_change =
      remaining_structure_residual.dot(response_step);
  result.incremental_quadratic_change = 0.5 * (
      orbital_step.dot(incremental_operator_image.head(n_orbitals)) +
      response_step.dot(incremental_operator_image.tail(n_response)));
  result.incremental_model_change =
      result.corrected_orbital_gradient.dot(orbital_step) +
      result.remaining_structure_linear_change +
      result.incremental_quadratic_change;
  result.total_model_change =
      result.baseline_defect_model_change + result.incremental_model_change;
  result.baseline_defect_predicted_decrease =
      -result.baseline_defect_model_change;
  result.incremental_predicted_decrease =
      -result.incremental_model_change;
  result.total_predicted_decrease = -result.total_model_change;

  const std::array<double, 8> scalars{
      result.baseline_defect_model_change,
      result.incremental_model_change,
      result.total_model_change,
      result.baseline_defect_predicted_decrease,
      result.incremental_predicted_decrease,
      result.total_predicted_decrease,
      result.remaining_structure_linear_change,
      result.incremental_quadratic_change};
  for (const double value : scalars) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "coupled model-reduction evaluation produced a non-finite value");
    }
  }
  if (!result.corrected_orbital_gradient.allFinite()) {
    throw std::runtime_error(
        "coupled model-reduction gradient is non-finite");
  }
  return result;
}

}  // namespace xmvb::vb
