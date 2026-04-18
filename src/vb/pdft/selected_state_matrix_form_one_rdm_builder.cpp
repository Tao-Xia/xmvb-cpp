#include "vb/pdft/selected_state_matrix_form_one_rdm_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

namespace xmvb::vb {

namespace {


constexpr double kUnitWeightTolerance = 1.0e-10;

double max_abs_skew_part(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("skew-part check requires a square matrix");
  }

  double max_abs_value = 0.0;
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      max_abs_value = std::max(
          max_abs_value,
          std::abs(matrix(row, column) - matrix(column, row)));
    }
  }
  return max_abs_value;
}

double frobenius_inner_product(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("Frobenius inner product size mismatch");
  }
  return left.cwiseProduct(right).sum();
}

void validate_single_state_gradient_result(
    const CppActiveSpaceGradientResult& gradient_result) {
  if (gradient_result.scf_result.selected_state_indices.size() != 1 ||
      gradient_result.scf_result.state_average_weights.size() != 1) {
    throw std::invalid_argument(
        "state-specific one-RDM build requires exactly one selected state");
  }
  const double state_weight =
      gradient_result.scf_result.state_average_weights.front();
  if (std::abs(state_weight - 1.0) > kUnitWeightTolerance) {
    throw std::invalid_argument(
        "state-specific one-RDM build requires a unit selected-state weight");
  }
}

SelectedStateMatrixFormOneRdmResult assemble_selected_state_matrix_form_one_rdm(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) {
  validate_single_state_gradient_result(gradient_result);

  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "selected-state one-RDM build requires a positive basis and active dimension");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid inactive/active orbital partition");
  }

  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  const std::size_t active_matrix_size =
      xmvb::to_size(n_active_orbitals) * n_active_orbitals;
  if (gradient_result.active_one_electron_gradient.size() != active_matrix_size ||
      gradient_result.orbital_preparation_result.active_orbital_overlap_matrix.size() !=
          active_matrix_size ||
      gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.size() !=
          ao_matrix_size ||
      gradient_result.orbital_preparation_result.inactive_density_matrix.size() !=
          ao_matrix_size ||
      input.orbital_preparation_input.active_orbital_overlap_matrix.size() !=
          ao_matrix_size) {
    throw std::invalid_argument(
        "selected-state one-RDM build received inconsistent matrix dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> raw_active_one_rdm(
      gradient_result.active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd symmetrized_active_one_rdm =
      0.5 * (raw_active_one_rdm + raw_active_one_rdm.transpose());
  const Eigen::Map<const Eigen::MatrixXd> active_overlap_matrix(
      gradient_result.orbital_preparation_result.active_orbital_overlap_matrix.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_orbital_matrix(
      gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd ao_active_density_matrix =
      active_auxiliary_orbitals *
      symmetrized_active_one_rdm *
      active_auxiliary_orbitals.transpose();
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_matrix(
      gradient_result.orbital_preparation_result.inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd ao_total_density_matrix =
      2.0 * inactive_density_matrix + ao_active_density_matrix;
  const Eigen::Map<const Eigen::MatrixXd> ao_overlap_matrix(
      input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  // The current matrix-form active-space density lives in the same nonorthogonal
  // active basis that contracts with `HHO`, so particle counts are recovered by
  // a metric trace rather than by the raw Euclidean trace.
  SelectedStateMatrixFormOneRdmResult result;
  result.state_index = gradient_result.scf_result.selected_state_indices.front();
  result.matrix_form_active_one_rdm.assign(
      symmetrized_active_one_rdm.data(),
      symmetrized_active_one_rdm.data() + symmetrized_active_one_rdm.size());
  result.active_orbital_overlap_matrix =
      gradient_result.orbital_preparation_result.active_orbital_overlap_matrix;
  result.matrix_form_ao_active_density_matrix.assign(
      ao_active_density_matrix.data(),
      ao_active_density_matrix.data() + ao_active_density_matrix.size());
  result.matrix_form_ao_total_density_matrix.assign(
      ao_total_density_matrix.data(),
      ao_total_density_matrix.data() + ao_total_density_matrix.size());
  result.active_one_rdm_asymmetry_max_abs = max_abs_skew_part(raw_active_one_rdm);
  result.active_metric_trace =
      frobenius_inner_product(symmetrized_active_one_rdm, active_overlap_matrix);
  result.ao_active_metric_trace =
      frobenius_inner_product(ao_active_density_matrix, ao_overlap_matrix);
  result.ao_total_metric_trace =
      frobenius_inner_product(ao_total_density_matrix, ao_overlap_matrix);
  return result;
}

}  // namespace

SelectedStateMatrixFormOneRdmBuilder::SelectedStateMatrixFormOneRdmBuilder(
    VBSCFAlgorithm algorithm)
    : gradient_evaluator_(algorithm) {}

SelectedStateMatrixFormOneRdmBuilder::SelectedStateMatrixFormOneRdmBuilder(
    CppActiveSpaceGradientEvaluator gradient_evaluator)
    : gradient_evaluator_(std::move(gradient_evaluator)) {}

SelectedStateMatrixFormOneRdmResult
SelectedStateMatrixFormOneRdmBuilder::build(
    const CppVbInput& input,
    int state_index,
    double nuclear_repulsion_energy) const {
  const CppActiveSpaceGradientResult gradient_result =
      gradient_evaluator_.evaluate(
          input,
          std::vector<int>{state_index},
          std::vector<double>{1.0},
          nuclear_repulsion_energy);
  return build_from_state_specific_gradient(input, gradient_result);
}

SelectedStateMatrixFormOneRdmResult
SelectedStateMatrixFormOneRdmBuilder::build_from_state_specific_gradient(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) const {
  return assemble_selected_state_matrix_form_one_rdm(input, gradient_result);
}

}  // namespace xmvb::vb
