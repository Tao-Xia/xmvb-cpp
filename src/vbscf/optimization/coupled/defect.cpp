#include "vbscf/optimization/coupled/defect.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

void validate_options(const MinresOptions& options) {
  if (!(options.relative_residual_tolerance >= 0.0) ||
      !std::isfinite(options.relative_residual_tolerance) ||
      !(options.absolute_residual_tolerance >= 0.0) ||
      !std::isfinite(options.absolute_residual_tolerance) ||
      (options.relative_residual_tolerance == 0.0 &&
       options.absolute_residual_tolerance == 0.0) ||
      options.maximum_iterations < 0) {
    throw std::invalid_argument(
        "invalid structure-defect MINRES convergence options");
  }
}

MinresResult exact_zero_result(int size, const MinresOptions& options) {
  MinresResult result;
  result.solution = Eigen::VectorXd::Zero(size);
  result.operator_image = Eigen::VectorXd::Zero(size);
  result.residual = Eigen::VectorXd::Zero(size);
  result.stop_reason = MinresStopReason::Converged;
  result.residual_target = options.absolute_residual_tolerance;
  return result;
}

}  // namespace

StructureDefectCorrection correct_structure_defect(
    const CoupledNewtonOperator& coupled_operator,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& structure_residual,
    const MinresOptions& options,
    const SymmetricOperatorAction& apply_inverse_preconditioner) {
  if (orbital_gradient.size() !=
          coupled_operator.n_orbital_coordinates() ||
      structure_residual.size() !=
          coupled_operator.n_response_coordinates() ||
      !orbital_gradient.allFinite() || !structure_residual.allFinite()) {
    throw std::invalid_argument(
        "structure-defect correction inputs are inconsistent");
  }
  validate_options(options);

  StructureDefectCorrection correction;
  if (structure_residual.isZero(0.0)) {
    correction.response_correction = Eigen::VectorXd::Zero(
        coupled_operator.n_response_coordinates());
    correction.corrected_orbital_gradient = orbital_gradient;
    correction.remaining_structure_residual = Eigen::VectorXd::Zero(
        coupled_operator.n_response_coordinates());
    correction.linear_result = exact_zero_result(
        coupled_operator.n_response_coordinates(), options);
    return correction;
  }

  correction.linear_result = solve_symmetric_minres(
      [&coupled_operator](const Eigen::VectorXd& direction) {
        return coupled_operator.apply_response_hessian(direction).col(0).eval();
      },
      -structure_residual,
      options,
      apply_inverse_preconditioner);
  correction.response_correction = correction.linear_result.solution;
  correction.remaining_structure_residual = structure_residual +
      correction.linear_result.operator_image;
  correction.stationary_limit_model_change = 0.5 *
      structure_residual.dot(correction.response_correction);
  correction.exact_model_change =
      structure_residual.dot(correction.response_correction) +
      0.5 * correction.response_correction.dot(
          correction.linear_result.operator_image);
  if (correction.converged()) {
    correction.corrected_orbital_gradient = orbital_gradient +
        coupled_operator.apply_response_to_orbital(
            correction.response_correction).col(0);
  }
  return correction;
}

}  // namespace xmvb::vb
