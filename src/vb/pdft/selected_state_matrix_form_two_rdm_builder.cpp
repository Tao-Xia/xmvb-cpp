#include "vb/pdft/selected_state_matrix_form_two_rdm_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

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

void validate_single_state_gradient_result(
    const CppActiveSpaceGradientResult& gradient_result) {
  if (gradient_result.scf_result.selected_state_indices.size() != 1 ||
      gradient_result.scf_result.state_average_weights.size() != 1) {
    throw std::invalid_argument(
        "state-specific two-RDM build requires exactly one selected state");
  }
  const double state_weight =
      gradient_result.scf_result.state_average_weights.front();
  if (std::abs(state_weight - 1.0) > kUnitWeightTolerance) {
    throw std::invalid_argument(
        "state-specific two-RDM build requires a unit selected-state weight");
  }
}

Eigen::MatrixXd build_full_active_pair_kernel_matrix(
    const std::vector<double>& packed_active_two_electron_integrals,
    int n_active_pairs) {
  Eigen::MatrixXd pair_kernel_matrix =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  for (int column_pair_index = 0;
       column_pair_index < n_active_pairs;
       ++column_pair_index) {
    for (int row_pair_index = 0;
         row_pair_index < n_active_pairs;
         ++row_pair_index) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_pair_index,
              column_pair_index);
      pair_kernel_matrix(row_pair_index, column_pair_index) =
          packed_active_two_electron_integrals[xmvb::to_size(packed_index)];
    }
  }
  return pair_kernel_matrix;
}

Eigen::MatrixXd build_full_active_pair_density_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    int n_active_pairs) {
  Eigen::MatrixXd pair_density_matrix =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  for (int column_pair_index = 0;
       column_pair_index < n_active_pairs;
       ++column_pair_index) {
    for (int row_pair_index = 0;
         row_pair_index < n_active_pairs;
         ++row_pair_index) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_pair_index,
              column_pair_index);
      double pair_density_value =
          packed_active_two_electron_gradient[xmvb::to_size(packed_index)];
      if (row_pair_index == column_pair_index) {
        pair_density_value *= 2.0;
      }
      pair_density_matrix(row_pair_index, column_pair_index) =
          pair_density_value;
    }
  }
  return pair_density_matrix;
}

SelectedStateMatrixFormTwoRdmResult assemble_selected_state_matrix_form_two_rdm(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) {
  validate_single_state_gradient_result(gradient_result);

  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "selected-state two-RDM build requires a positive active dimension");
  }

  const std::size_t expected_packed_size =
      packed_active_two_electron_integral_count(n_active_orbitals);
  if (gradient_result.packed_active_two_electron_gradient.size() !=
      expected_packed_size) {
    throw std::invalid_argument(
        "selected-state two-RDM build received inconsistent packed gradient dimensions");
  }

  const std::vector<double> packed_active_two_electron_integrals =
      gradient_result.active_space_two_electron_result.packed_active_two_electron_integrals.empty()
          ? reconstruct_packed_active_two_electron_integrals(
                make_active_space_two_electron_view(
                    gradient_result.active_space_two_electron_result),
                n_active_orbitals)
          : gradient_result.active_space_two_electron_result
                .packed_active_two_electron_integrals;
  if (packed_active_two_electron_integrals.size() != expected_packed_size) {
    throw std::invalid_argument(
        "selected-state two-RDM build received inconsistent packed integral dimensions");
  }

  const int n_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  const Eigen::MatrixXd pair_density_matrix =
      build_full_active_pair_density_matrix(
          gradient_result.packed_active_two_electron_gradient,
          n_active_pairs);
  const Eigen::MatrixXd pair_kernel_matrix =
      build_full_active_pair_kernel_matrix(
          packed_active_two_electron_integrals,
          n_active_pairs);

  double packed_energy = 0.0;
  for (std::size_t index = 0;
       index < expected_packed_size;
       ++index) {
    packed_energy +=
        packed_active_two_electron_integrals[index] *
        gradient_result.packed_active_two_electron_gradient[index];
  }

  // The unpacked pair basis duplicates off-diagonal packed entries into both
  // `(P,Q)` and `(Q,P)`. Doubling diagonal density entries therefore makes the
  // half-trace over the full symmetric pair matrix exactly match the packed
  // contraction used by the current VBSCF energy path.
  const double unpacked_energy =
      0.5 * pair_kernel_matrix.cwiseProduct(pair_density_matrix).sum();

  SelectedStateMatrixFormTwoRdmResult result;
  result.state_index = gradient_result.scf_result.selected_state_indices.front();
  result.n_active_pairs = n_active_pairs;
  result.packed_matrix_form_active_two_rdm =
      gradient_result.packed_active_two_electron_gradient;
  result.matrix_form_active_pair_density_matrix.assign(
      pair_density_matrix.data(),
      pair_density_matrix.data() + pair_density_matrix.size());
  result.active_pair_density_symmetry_max_abs =
      max_abs_skew_part(pair_density_matrix);
  result.packed_active_two_electron_energy = packed_energy;
  result.unpacked_active_pair_energy = unpacked_energy;
  return result;
}

}  // namespace

SelectedStateMatrixFormTwoRdmBuilder::SelectedStateMatrixFormTwoRdmBuilder(
    VBSCFAlgorithm algorithm)
    : gradient_evaluator_(algorithm) {}

SelectedStateMatrixFormTwoRdmBuilder::SelectedStateMatrixFormTwoRdmBuilder(
    CppActiveSpaceGradientEvaluator gradient_evaluator)
    : gradient_evaluator_(std::move(gradient_evaluator)) {}

SelectedStateMatrixFormTwoRdmResult
SelectedStateMatrixFormTwoRdmBuilder::build(
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

SelectedStateMatrixFormTwoRdmResult
SelectedStateMatrixFormTwoRdmBuilder::build_from_state_specific_gradient(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) const {
  return assemble_selected_state_matrix_form_two_rdm(input, gradient_result);
}

}  // namespace xmvb::vb
