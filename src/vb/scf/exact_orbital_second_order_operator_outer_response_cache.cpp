#include "vb/scf/exact_orbital_second_order_operator_outer_response_internal.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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
    const CppActiveSpaceSecondOrderContext& accepted_point_context) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  const int n_selected_states =
      static_cast<int>(accepted_point_context.selected_state_indices.size());
  if (n_structures <= 0 || n_selected_states <= 0) {
    throw std::invalid_argument(
        "accepted selected-state eigen-response operator requires positive dimensions");
  }
  const std::size_t structure_count = static_cast<std::size_t>(n_structures);
  const std::size_t selected_state_count =
      static_cast<std::size_t>(n_selected_states);
  const std::size_t expected_eigenvector_size =
      structure_count * structure_count;
  if (accepted_point_context.eigen_result.eigenvalues.size() !=
          structure_count ||
      accepted_point_context.eigen_result.eigenvector_matrix.size() !=
          expected_eigenvector_size) {
    throw std::invalid_argument(
        "accepted-point generalized eigensystem dimensions are inconsistent");
  }
  if (accepted_point_context.normalized_state_weights.size() !=
      selected_state_count) {
    throw std::invalid_argument(
        "accepted-point normalized_state_weights do not match selected states");
  }

  AcceptedSelectedStateGeneralizedEigenResponseOperator response_operator;
  response_operator.accepted_eigenvector_matrix_storage =
      &accepted_point_context.eigen_result.eigenvector_matrix;
  response_operator.n_structures = n_structures;
  response_operator.accepted_eigenvalues =
      Eigen::Map<const Eigen::VectorXd>(
          accepted_point_context.eigen_result.eigenvalues.data(),
          n_structures);
  response_operator.selected_columns.reserve(selected_state_count);
  const auto accepted_eigenvector_matrix =
      response_operator.accepted_eigenvector_matrix_view();

  Eigen::VectorXd selected_state_weights = Eigen::VectorXd::Zero(n_structures);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
    selected_state_weights[state_index] =
        accepted_point_context.normalized_state_weights[selected_state_offset];
  }

  constexpr double kDegeneracyToleranceScale = 1.0e6;
  constexpr double kRelativeGapTolerance = 1.0e-8;
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int column_state =
        accepted_point_context.selected_state_indices[static_cast<std::size_t>(
            selected_state_offset)];
    const double column_energy =
        response_operator.accepted_eigenvalues[column_state];
    const double column_weight = selected_state_weights[column_state];

    AcceptedSelectedStateEigenResponseColumnCache column_cache;
    column_cache.selected_state_index = column_state;
    column_cache.selected_state_energy = column_energy;
    column_cache.energy_gaps = Eigen::VectorXd::Zero(n_structures);
    column_cache.gap_tolerances = Eigen::VectorXd::Zero(n_structures);
    column_cache.uses_equal_weight_gauge = Eigen::ArrayXi::Zero(n_structures);

    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double row_energy =
          response_operator.accepted_eigenvalues[row_state];
      column_cache.energy_gaps[row_state] = column_energy - row_energy;
      column_cache.gap_tolerances[row_state] =
          std::max(
              kRelativeGapTolerance *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}),
              kDegeneracyToleranceScale *
                  std::numeric_limits<double>::epsilon() *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}));
      column_cache.uses_equal_weight_gauge[row_state] =
          std::abs(selected_state_weights[row_state] - column_weight) <= 1.0e-12
              ? 1
              : 0;
    }
    response_operator.selected_columns.push_back(std::move(column_cache));
  }

  return response_operator;
}

}  // namespace

AcceptedOuterResponseLinearResponseCache
build_accepted_outer_response_linear_response_cache(
    const CppVbInput* input,
    const CppActiveSpaceSecondOrderContext* accepted_point_context,
    const std::vector<StructureCoefficientBlock>* structure_coefficient_blocks) {
  if (input == nullptr || accepted_point_context == nullptr ||
      structure_coefficient_blocks == nullptr) {
    throw std::invalid_argument(
        "accepted outer-response cache inputs must not be null");
  }

  AcceptedOuterResponseLinearResponseCache cache;
  cache.input = input;
  cache.accepted_point_context = accepted_point_context;
  cache.structure_coefficient_blocks = structure_coefficient_blocks;
  cache.selected_state_indices = accepted_point_context->selected_state_indices;
  cache.selected_state_eigen_response_operator =
      build_accepted_selected_state_generalized_eigen_response_operator(
          *accepted_point_context);
  const int n_structures =
      cache.selected_state_eigen_response_operator.n_structures;
  const int n_selected_states =
      static_cast<int>(cache.selected_state_indices.size());
  const auto accepted_eigenvector_matrix =
      cache.selected_state_eigen_response_operator.accepted_eigenvector_matrix_view();
  cache.accepted_selected_eigenvector_columns =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int state_index =
        cache.selected_state_indices[static_cast<std::size_t>(
            selected_state_offset)];
    cache.accepted_selected_eigenvector_columns.col(selected_state_offset) =
        accepted_eigenvector_matrix.col(state_index);
  }
  return cache;
}

SelectedStateGeneralizedEigenDirectionalResponse
AcceptedSelectedStateGeneralizedEigenResponseOperator::apply(
    const SelectedStateProjectedDirectionalMatrices&
        projected_directional_structure_matrices) const {
  const auto accepted_eigenvector_matrix = accepted_eigenvector_matrix_view();
  const int n_structures = accepted_eigenvector_matrix.rows();
  const int n_selected_states = static_cast<int>(selected_columns.size());
  if (accepted_eigenvector_matrix.cols() != n_structures ||
      accepted_eigenvalues.size() != n_structures) {
    throw std::invalid_argument(
        "accepted selected-state eigen-response operator is inconsistent");
  }
  if (projected_directional_structure_matrices
              .transformed_delta_hamiltonian_selected.rows() != n_structures ||
      projected_directional_structure_matrices
              .transformed_delta_hamiltonian_selected.cols() !=
          n_selected_states ||
      projected_directional_structure_matrices
              .transformed_delta_overlap_selected.rows() != n_structures ||
      projected_directional_structure_matrices
              .transformed_delta_overlap_selected.cols() !=
          n_selected_states) {
    throw std::invalid_argument(
        "projected directional structure dimensions do not match the accepted-point selected-state response operator");
  }

  SelectedStateGeneralizedEigenDirectionalResponse result;
  result.delta_selected_eigenvector_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  result.delta_selected_eigenvalues.assign(
      static_cast<std::size_t>(n_selected_states),
      0.0);

  // This is the frozen accepted-point map
  //   (delta H_tilde_sel, delta S_tilde_sel) -> (delta C_sel, delta E_sel).
  // Gap tolerances and gauge handling depend only on the accepted eigensystem,
  // so each apply injects only the fresh directional projected columns.
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const auto& column_cache =
        selected_columns[static_cast<std::size_t>(selected_state_offset)];
    const auto transformed_delta_hamiltonian_column =
        projected_directional_structure_matrices
            .transformed_delta_hamiltonian_selected.col(selected_state_offset);
    const auto transformed_delta_overlap_column =
        projected_directional_structure_matrices
            .transformed_delta_overlap_selected.col(selected_state_offset);

    Eigen::VectorXd eigenvector_rotation_column =
        Eigen::VectorXd::Zero(n_structures);
    const int column_state = column_cache.selected_state_index;
    const double column_energy = column_cache.selected_state_energy;
    const double transformed_overlap_diagonal =
        transformed_delta_overlap_column[column_state];
    result.delta_selected_eigenvalues[static_cast<std::size_t>(
        selected_state_offset)] =
        transformed_delta_hamiltonian_column[column_state] -
        column_energy * transformed_overlap_diagonal;
    eigenvector_rotation_column[column_state] =
        -0.5 * transformed_overlap_diagonal;

    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double numerator =
          transformed_delta_hamiltonian_column[row_state] -
          column_energy * transformed_delta_overlap_column[row_state];
      const double overlap_gauge_rotation =
          -0.5 * transformed_delta_overlap_column[row_state];
      const double gap = column_cache.energy_gaps[row_state];
      const double gap_tolerance = column_cache.gap_tolerances[row_state];
      if (std::abs(gap) <= gap_tolerance) {
        if (column_cache.uses_equal_weight_gauge[row_state] != 0) {
          eigenvector_rotation_column[row_state] = overlap_gauge_rotation;
          continue;
        }
        const double safe_gap =
            std::copysign(
                gap_tolerance,
                gap != 0.0 ? gap : (numerator != 0.0 ? numerator : 1.0));
        const double regularized_rotation = numerator / safe_gap;
        eigenvector_rotation_column[row_state] =
            std::isfinite(regularized_rotation)
                ? regularized_rotation
                : overlap_gauge_rotation;
        continue;
      }
      eigenvector_rotation_column[row_state] = numerator / gap;
    }

    result.delta_selected_eigenvector_matrix.col(selected_state_offset).noalias() =
        accepted_eigenvector_matrix * eigenvector_rotation_column;
  }

  throw_if_nonfinite_matrix(
      result.delta_selected_eigenvector_matrix,
      "exact outer-response directional selected-state eigenvectors");
  throw_if_nonfinite_vector(
      result.delta_selected_eigenvalues,
      "exact outer-response directional selected-state energies");
  return result;
}

}  // namespace xmvb::vb
