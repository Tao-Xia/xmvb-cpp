#include "vbscf/derivatives/hessian/context/response_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/eigen_response.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"

namespace xmvb::vb {
namespace {

/**
 * @brief Checks accepted outer-response buffers for NaN/Inf contamination.
 *
 * This file hosts the frozen accepted-point generalized-eigen response cache.
 * Keeping a small local finite-value guard here avoids dragging unrelated
 * exact_ctx implementation helpers across translation units just to validate
 * these matrix responses.
 */
template <typename Derived>
void throw_if_nonfinite_matrix(
    const Eigen::MatrixBase<Derived>& matrix,
    const char* label) {
  if (!matrix.allFinite()) {
    throw std::runtime_error(
        std::string(label) + " contains non-finite values");
  }
}

void throw_if_nonfinite_vector(
    const std::vector<double>& values,
    const char* label) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          std::string(label) + " contains non-finite values");
    }
  }
}

AcceptedSelectedStateGeneralizedEigenResponseOperator
build_accepted_selected_state_generalized_eigen_response_operator(
    const AcceptedPointContext& accepted_point_context) {
  const int n_structures = accepted_point_context.n_structures;
  const int n_selected_states =
      static_cast<int>(accepted_point_context.selected_state_indices.size());
  if (n_structures <= 0 || n_selected_states <= 0) {
    throw std::invalid_argument(
        "accepted selected-state eigen-response operator requires positive dimensions");
  }
  const std::size_t selected_state_count =
      static_cast<std::size_t>(n_selected_states);
  if (!accepted_point_context.structure_action.has_value()) {
    throw std::invalid_argument(
        "accepted-point matrix-free structure action is unavailable");
  }
  if (accepted_point_context.selected_state_energies.size() !=
          selected_state_count ||
      accepted_point_context.selected_state_eigenvectors.rows() !=
          n_structures ||
      accepted_point_context.selected_state_eigenvectors.cols() !=
          n_selected_states) {
    throw std::invalid_argument(
        "accepted-point selected eigensystem dimensions are inconsistent");
  }
  if (accepted_point_context.normalized_state_weights.size() !=
      selected_state_count) {
    throw std::invalid_argument(
        "accepted-point normalized_state_weights do not match selected states");
  }

  AcceptedSelectedStateGeneralizedEigenResponseOperator response_operator;
  response_operator.structure_action =
      &accepted_point_context.structure_action.value();
  response_operator.selected_eigenvalues = Eigen::Map<const Eigen::VectorXd>(
      accepted_point_context.selected_state_energies.data(),
      n_selected_states);
  response_operator.selected_eigenvectors =
      accepted_point_context.selected_state_eigenvectors;
  response_operator.overlap_selected =
      response_operator.structure_action
          ->apply(response_operator.selected_eigenvectors)
          .overlap;
  accepted_point_context.structure_solve_accuracy.validate();
  // A first-order eigensystem response feeds a second-order orbital model.
  // Reusing the outer gradient threshold directly can leave the Rayleigh
  // curvature inaccurate when large core and response terms cancel. Balance
  // the response error against the accepted Ritz-energy backward error: its
  // square root is the natural first-order accuracy associated with a
  // second-order energy target. The outer gradient contract remains an upper
  // bound, so no independent empirical response threshold is introduced.
  const double selected_energy_scale = std::max(
      1.0,
      response_operator.selected_eigenvalues.cwiseAbs().maxCoeff());
  const double relative_energy_accuracy =
      accepted_point_context.structure_solve_accuracy.energy_tolerance /
      selected_energy_scale;
  response_operator.relative_residual_tolerance =
      std::min(
          accepted_point_context.structure_solve_accuracy.gradient_tolerance,
          std::sqrt(relative_energy_accuracy));
  for (const int state_index : accepted_point_context.selected_state_indices) {
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }

  return response_operator;
}

}  // namespace

AcceptedOuterResponseContext build_accepted_outer_response_context(
    const VbScfInput* input,
    const AcceptedPointContext* accepted_point_context,
    const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks) {
  if (input == nullptr || accepted_point_context == nullptr ||
      structure_coefficient_blocks == nullptr) {
    throw std::invalid_argument(
        "accepted outer-response cache inputs must not be null");
  }

  AcceptedOuterResponseContext context;
  context.input = input;
  context.accepted_point_context = accepted_point_context;
  context.structure_coefficient_blocks = structure_coefficient_blocks;
  context.selected_state_eigen_response_operator =
      build_accepted_selected_state_generalized_eigen_response_operator(
          *accepted_point_context);
  context.structure_factors = build_accepted_structure_response_factors(
      *input,
      *accepted_point_context,
      context.selected_state_eigen_response_operator.selected_eigenvectors);
  return context;
}

SelectedStateGeneralizedEigenDirectionalResponse
AcceptedSelectedStateGeneralizedEigenResponseOperator::apply(
    const SelectedStateDirectionalStructureImages& directional_images) const {
  if (structure_action == nullptr) {
    throw std::invalid_argument(
        "accepted selected-state eigen-response operator has no structure action");
  }
  const int n_structures = structure_action->n_structures();
  const int n_selected_states =
      static_cast<int>(selected_eigenvalues.size());
  if (n_selected_states <= 0 ||
      selected_eigenvectors.rows() != n_structures ||
      selected_eigenvectors.cols() != n_selected_states) {
    throw std::invalid_argument(
        "accepted selected-state eigen-response operator is inconsistent");
  }
  if (directional_images.delta_hamiltonian_selected.rows() != n_structures ||
      directional_images.delta_hamiltonian_selected.cols() !=
          n_selected_states ||
      directional_images.delta_overlap_selected.rows() != n_structures ||
      directional_images.delta_overlap_selected.cols() != n_selected_states) {
    throw std::invalid_argument(
        "directional structure images do not match the selected-state response operator");
  }
  const StructureDiagonal& diagonal = structure_action->diagonal();
  const xmvb::core::GeneralizedEigenAction action =
      [this](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        StructureActionResult images = structure_action->apply(vectors);
        return xmvb::core::GeneralizedEigenActionResult{
            std::move(images.hamiltonian),
            std::move(images.overlap)};
      };
  const xmvb::core::EigenResponseResult response =
      xmvb::core::solve_generalized_eigen_response(
          action,
          diagonal.hamiltonian,
          diagonal.overlap,
          selected_eigenvalues,
          selected_eigenvectors,
          overlap_selected,
          directional_images.delta_hamiltonian_selected,
          directional_images.delta_overlap_selected,
          xmvb::core::EigenResponseOptions{
              n_structures + 1,
              relative_residual_tolerance});

  SelectedStateGeneralizedEigenDirectionalResponse result;
  result.delta_selected_eigenvector_matrix = response.eigenvector_response;
  result.delta_selected_eigenvalues.assign(
      response.eigenvalue_response.data(),
      response.eigenvalue_response.data() + response.eigenvalue_response.size());
  result.linear_iterations = response.iterations;
  result.block_actions = response.block_actions;
  result.max_relative_residual =
      response.relative_residual_norms.maxCoeff();

  throw_if_nonfinite_matrix(
      result.delta_selected_eigenvector_matrix,
      "exact outer-response directional selected-state eigenvectors");
  throw_if_nonfinite_vector(
      result.delta_selected_eigenvalues,
      "exact outer-response directional selected-state energies");
  return result;
}

}  // namespace xmvb::vb
