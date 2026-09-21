#include "vbscf/derivatives/hessian/context/response_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
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

void throw_if_invalid_selected_matrix_responses(
    const std::vector<Eigen::MatrixXd>& responses,
    int n_directions,
    int n_selected_states,
    const char* label) {
  if (responses.size() != static_cast<std::size_t>(n_directions)) {
    throw std::runtime_error(
        std::string(label) + " has the wrong direction count");
  }
  for (const Eigen::MatrixXd& response : responses) {
    if (response.rows() != n_selected_states ||
        response.cols() != n_selected_states ||
        !response.allFinite()) {
      throw std::runtime_error(
          std::string(label) + " has invalid dimensions or values");
    }
  }
}

bool weights_are_equal(const std::vector<double>& weights) {
  if (weights.size() < 2) return false;
  const double reference = weights.front();
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(reference));
  return std::all_of(
      weights.begin() + 1,
      weights.end(),
      [&](double weight) {
        return std::abs(weight - reference) <= tolerance;
      });
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
  StructureActionResult selected_images = response_operator.structure_action
      ->apply(response_operator.selected_eigenvectors);
  response_operator.selected_residuals = selected_images.hamiltonian -
      selected_images.overlap * response_operator.selected_eigenvalues.asDiagonal();
  response_operator.overlap_selected = std::move(selected_images.overlap);
  response_operator.use_equal_weight_subspace_response = weights_are_equal(
      accepted_point_context.normalized_state_weights);
  response_operator.response_recycle_spaces.resize(selected_state_count);
  if (!accepted_point_context.full_structure_eigenvalues.empty()) {
    if (accepted_point_context.full_structure_eigenvalues.size() !=
            static_cast<std::size_t>(n_structures) ||
        accepted_point_context.root_eigenvectors.rows() != n_structures ||
        accepted_point_context.root_eigenvectors.cols() != n_structures) {
      throw std::invalid_argument(
          "accepted point lacks a complete structure eigenspectrum");
    }
    response_operator.full_eigenvalues =
        &accepted_point_context.full_structure_eigenvalues;
    response_operator.full_eigenvectors =
        &accepted_point_context.root_eigenvectors;
    response_operator.selected_root_indices =
        accepted_point_context.selected_state_indices;
  }
  accepted_point_context.structure_solve_accuracy.validate();
  const double selected_energy_scale = std::max(
      1.0,
      response_operator.selected_eigenvalues.cwiseAbs().maxCoeff());
  response_operator.relative_residual_tolerance =
      accepted_point_context.structure_solve_accuracy
          .response_backward_error_tolerance(selected_energy_scale);
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
    const AcceptedPointContext* accepted_point_context) {
  if (input == nullptr || accepted_point_context == nullptr) {
    throw std::invalid_argument(
        "accepted outer-response cache inputs must not be null");
  }

  AcceptedOuterResponseContext context;
  context.input = input;
  context.accepted_point_context = accepted_point_context;
  context.selected_state_eigen_response_operator =
      build_accepted_selected_state_generalized_eigen_response_operator(
          *accepted_point_context);
  if (!context.selected_state_eigen_response_operator.structure_action
           ->supports_integral_direction()) {
    context.structure_factors = build_accepted_structure_response_factors(
        *input,
        *accepted_point_context,
        context.selected_state_eigen_response_operator.selected_eigenvectors);
  }
  return context;
}

std::uint64_t
AcceptedSelectedStateGeneralizedEigenResponseOperator::revision() const noexcept {
  std::uint64_t result = 0;
  for (const auto& space : response_recycle_spaces) result += space.revision();
  return result;
}

SelectedStateGeneralizedEigenDirectionalResponse
AcceptedSelectedStateGeneralizedEigenResponseOperator::apply(
    const SelectedStateDirectionalStructureImages& directional_images,
    double requested_relative_residual_tolerance,
    bool frozen) const {
  return apply_direction_block(
      directional_images.delta_hamiltonian_selected,
      directional_images.delta_overlap_selected,
      requested_relative_residual_tolerance,
      frozen);
}

SelectedStateGeneralizedEigenDirectionalResponse
AcceptedSelectedStateGeneralizedEigenResponseOperator::apply_direction_block(
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double requested_relative_residual_tolerance,
    bool frozen) const {
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
  const int n_rhs = static_cast<int>(delta_hamiltonian_selected.cols());
  if (delta_hamiltonian_selected.rows() != n_structures ||
      n_rhs <= 0 || n_rhs % n_selected_states != 0 ||
      delta_overlap_selected.rows() != n_structures ||
      delta_overlap_selected.cols() != n_rhs) {
    throw std::invalid_argument(
        "directional structure image block does not match the selected-state response operator");
  }
  const int n_directions = n_rhs / n_selected_states;
  Eigen::VectorXd block_eigenvalues(n_rhs);
  Eigen::MatrixXd block_eigenvectors(n_structures, n_rhs);
  Eigen::MatrixXd block_overlap_selected(n_structures, n_rhs);
  Eigen::MatrixXd block_residuals(n_structures, n_rhs);
  std::vector<int> block_root_indices;
  if (full_eigenvalues != nullptr && full_eigenvectors != nullptr) {
    block_root_indices.resize(n_rhs);
  }
  for (int direction = 0; direction < n_directions; ++direction) {
    const int first = direction * n_selected_states;
    block_eigenvalues.segment(first, n_selected_states) =
        selected_eigenvalues;
    block_eigenvectors.middleCols(first, n_selected_states) =
        selected_eigenvectors;
    block_overlap_selected.middleCols(first, n_selected_states) =
        overlap_selected;
    block_residuals.middleCols(first, n_selected_states) = selected_residuals;
    if (!block_root_indices.empty()) {
      std::copy(
          selected_root_indices.begin(),
          selected_root_indices.end(),
          block_root_indices.begin() + first);
    }
  }
  const StructureDiagonal& diagonal =
      structure_action->preconditioner_diagonal();
  const xmvb::core::GeneralizedEigenAction action =
      [this](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        StructureActionResult images = structure_action->apply(vectors);
        return xmvb::core::GeneralizedEigenActionResult{
            std::move(images.hamiltonian),
            std::move(images.overlap)};
      };
  if (!std::isfinite(requested_relative_residual_tolerance) ||
      requested_relative_residual_tolerance < 0.0 ||
      requested_relative_residual_tolerance >= 1.0) {
    throw std::invalid_argument(
        "requested eigen-response tolerance must be finite in [0, 1)");
  }
  const double effective_relative_residual_tolerance =
      requested_relative_residual_tolerance > 0.0
          ? requested_relative_residual_tolerance
          : relative_residual_tolerance;
  const auto evaluate = [&](bool frozen_model) {
    if (use_equal_weight_subspace_response) {
      SelectedStateGeneralizedEigenDirectionalResponse result;
      result.delta_selected_eigenvector_matrix.resize(n_structures, n_rhs);
      result.selected_matrix_responses.reserve(n_directions);
      result.linear_iterations.reserve(n_rhs);
      std::vector<xmvb::core::EigenResponseRecycleSpace*> recycle_spaces(
          static_cast<std::size_t>(n_selected_states));
      for (std::size_t state = 0;
           state < static_cast<std::size_t>(n_selected_states); ++state) {
        recycle_spaces[state] = &response_recycle_spaces[state];
      }
      for (int direction = 0; direction < n_directions; ++direction) {
        const int first = direction * n_selected_states;
        const auto delta_hamiltonian =
            delta_hamiltonian_selected.middleCols(first, n_selected_states);
        const auto delta_overlap =
            delta_overlap_selected.middleCols(first, n_selected_states);
        xmvb::core::EigenSubspaceResponseResult response;
        if (!block_root_indices.empty()) {
          response = xmvb::core::
                solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
                    action,
                    Eigen::Map<const Eigen::VectorXd>(
                        full_eigenvalues->data(), n_structures),
                    *full_eigenvectors,
                    selected_root_indices,
                    selected_eigenvalues,
                    selected_eigenvectors,
                    overlap_selected,
                    delta_hamiltonian,
                    delta_overlap,
                    effective_relative_residual_tolerance);
        } else if (frozen_model) {
          const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces(
              recycle_spaces.begin(), recycle_spaces.end());
          response = xmvb::core::
              evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
                  action, selected_eigenvalues, selected_eigenvectors,
                  overlap_selected, delta_hamiltonian, delta_overlap, spaces,
                  &selected_residuals);
        } else {
          response = xmvb::core::solve_equal_weight_generalized_eigen_subspace_response(
                  action,
                  diagonal.hamiltonian,
                  diagonal.overlap,
                  selected_eigenvalues,
                  selected_eigenvectors,
                  overlap_selected,
                  delta_hamiltonian,
                  delta_overlap,
                  xmvb::core::EigenResponseOptions{
                      n_structures + 1,
                      effective_relative_residual_tolerance},
                  recycle_spaces,
                  &selected_residuals);
        }
        result.delta_selected_eigenvector_matrix.middleCols(
            first, n_selected_states) = response.eigenvector_response;
        result.selected_matrix_responses.push_back(
            response.selected_matrix_response);
        result.linear_iterations.insert(
            result.linear_iterations.end(),
            response.iterations.begin(),
            response.iterations.end());
        result.block_actions += response.block_actions;
        result.max_relative_residual = std::max(
            result.max_relative_residual,
            response.relative_residual_norms.maxCoeff());
      }
      throw_if_nonfinite_matrix(
          result.delta_selected_eigenvector_matrix,
          "equal-weight outer-response directional selected subspace");
      throw_if_invalid_selected_matrix_responses(
          result.selected_matrix_responses,
          n_directions,
          n_selected_states,
          "equal-weight outer-response directional selected matrices");
      return result;
    }
    std::vector<xmvb::core::EigenResponseRecycleSpace*> block_recycle_spaces(
        static_cast<std::size_t>(n_rhs));
    for (int direction = 0; direction < n_directions; ++direction) {
      const int first = direction * n_selected_states;
      for (int state = 0; state < n_selected_states; ++state) {
        block_recycle_spaces[static_cast<std::size_t>(first + state)] =
            &response_recycle_spaces[static_cast<std::size_t>(state)];
      }
    }
    xmvb::core::EigenResponseResult response;
    if (!block_root_indices.empty()) {
      response = xmvb::core::solve_generalized_eigen_response_from_full_spectrum(
            action,
            Eigen::Map<const Eigen::VectorXd>(
                full_eigenvalues->data(), n_structures),
            *full_eigenvectors,
            block_root_indices,
            block_eigenvalues,
            block_eigenvectors,
            block_overlap_selected,
            delta_hamiltonian_selected,
            delta_overlap_selected,
            effective_relative_residual_tolerance);
    } else if (frozen_model) {
      const std::vector<const xmvb::core::EigenResponseRecycleSpace*> spaces(
          block_recycle_spaces.begin(), block_recycle_spaces.end());
      response = xmvb::core::evaluate_frozen_generalized_eigen_response(
          action, block_eigenvalues, block_eigenvectors, block_overlap_selected,
          delta_hamiltonian_selected, delta_overlap_selected, spaces,
          &block_residuals);
    } else {
      response = xmvb::core::solve_generalized_eigen_response(
            action,
            diagonal.hamiltonian,
            diagonal.overlap,
            block_eigenvalues,
            block_eigenvectors,
            block_overlap_selected,
            delta_hamiltonian_selected,
            delta_overlap_selected,
            xmvb::core::EigenResponseOptions{
                n_structures + 1,
                effective_relative_residual_tolerance},
            block_recycle_spaces,
            &block_residuals);
    }

    SelectedStateGeneralizedEigenDirectionalResponse result;
    result.delta_selected_eigenvector_matrix = response.eigenvector_response;
    result.selected_matrix_responses.reserve(n_directions);
    for (int direction = 0; direction < n_directions; ++direction) {
      result.selected_matrix_responses.push_back(
          response.eigenvalue_response
              .segment(direction * n_selected_states, n_selected_states)
              .asDiagonal());
    }
    result.linear_iterations = response.iterations;
    result.block_actions = response.block_actions;
    result.max_relative_residual =
        response.relative_residual_norms.maxCoeff();

    throw_if_nonfinite_matrix(
        result.delta_selected_eigenvector_matrix,
        "exact outer-response directional selected-state eigenvectors");
    throw_if_invalid_selected_matrix_responses(
        result.selected_matrix_responses,
        n_directions,
        n_selected_states,
        "exact outer-response directional selected-state matrices");
    return result;
  };

  if (frozen || !block_root_indices.empty()) return evaluate(true);

  // Independently stopped MINRES solutions train a shared space; they are not
  // returned as mutually compatible Hessian samples. Reproject every RHS in
  // the final space, and certify that approximation before publishing it.
  std::vector<int> iterations(static_cast<std::size_t>(n_rhs), 0);
  int block_actions = 0;
  for (;;) {
    const std::uint64_t before = revision();
    const auto trained = evaluate(false);
    block_actions += trained.block_actions;
    for (int rhs = 0; rhs < n_rhs; ++rhs) {
      iterations[static_cast<std::size_t>(rhs)] +=
          trained.linear_iterations[static_cast<std::size_t>(rhs)];
    }
    auto projected = evaluate(true);
    block_actions += projected.block_actions;
    if (projected.max_relative_residual <= effective_relative_residual_tolerance) {
      projected.block_actions = block_actions;
      projected.linear_iterations = std::move(iterations);
      return projected;
    }
    if (revision() == before) {
      std::ostringstream message;
      message << "common structure-response space cannot meet the requested residual: "
              << projected.max_relative_residual << " > "
              << effective_relative_residual_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

}  // namespace xmvb::vb
