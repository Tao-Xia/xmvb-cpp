#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vb/matrices/biorthogonal_spin_pair.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/structure_expansion_term.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

#ifndef XMVB_CPP_ENABLE_DETERMINANT_LOW_RANK_UPDATES
#define XMVB_CPP_ENABLE_DETERMINANT_LOW_RANK_UPDATES 1
#endif

constexpr double kLowRankUpdateStabilityThreshold = 1.0e-10;

using ColumnMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct StructurePairAdjoints {
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

struct SpinDeterminantTraversalState {
  bool initialized = false;
  std::vector<int> occupied_orbitals_right;
  std::vector<double> overlap_submatrix;
  DeterminantOverlapResult overlap_result;
  ColumnMajorMatrixXd inverse_matrix;
};

struct SingleOrbitalRowUpdatePlan {
  std::vector<int> row_permutation;
  int replacement_row = -1;
};

int permutation_sign(const std::vector<int>& permutation) {
  std::vector<bool> visited(permutation.size(), false);
  int sign = 1;
  for (std::size_t start_index = 0; start_index < permutation.size(); ++start_index) {
    if (visited[start_index]) {
      continue;
    }

    int cycle_length = 0;
    int current_index = static_cast<int>(start_index);
    while (!visited[static_cast<std::size_t>(current_index)]) {
      visited[static_cast<std::size_t>(current_index)] = true;
      current_index = permutation[static_cast<std::size_t>(current_index)];
      ++cycle_length;
    }
    if (cycle_length > 0 && ((cycle_length - 1) % 2 != 0)) {
      sign = -sign;
    }
  }
  return sign;
}

bool is_valid_permutation(const std::vector<int>& permutation) {
  std::vector<bool> seen(permutation.size(), false);
  for (const int value : permutation) {
    if (value < 0 || static_cast<std::size_t>(value) >= permutation.size()) {
      return false;
    }
    if (seen[static_cast<std::size_t>(value)]) {
      return false;
    }
    seen[static_cast<std::size_t>(value)] = true;
  }
  return true;
}

std::vector<double> permute_rows_of_overlap_submatrix(
    const std::vector<double>& overlap_submatrix,
    int n_electrons,
    const std::vector<int>& row_permutation) {
  std::vector<double> permuted_overlap_submatrix(overlap_submatrix.size(), 0.0);
  const auto source_matrix = map_column_major_matrix(overlap_submatrix, n_electrons);
  auto permuted_matrix = Eigen::Map<ColumnMajorMatrixXd>(
      permuted_overlap_submatrix.data(),
      n_electrons,
      n_electrons);
  for (int row_index = 0; row_index < n_electrons; ++row_index) {
    permuted_matrix.row(row_index) =
        source_matrix.row(row_permutation[static_cast<std::size_t>(row_index)]);
  }
  return permuted_overlap_submatrix;
}

ColumnMajorMatrixXd permute_columns_of_inverse_matrix(
    const ColumnMajorMatrixXd& inverse_matrix,
    const std::vector<int>& row_permutation) {
  ColumnMajorMatrixXd permuted_inverse(inverse_matrix.rows(), inverse_matrix.cols());
  for (int column_index = 0; column_index < inverse_matrix.cols(); ++column_index) {
    permuted_inverse.col(column_index) =
        inverse_matrix.col(row_permutation[static_cast<std::size_t>(column_index)]);
  }
  return permuted_inverse;
}

bool build_single_orbital_row_update_plan(
    const std::vector<int>& previous_occupied_orbitals_right,
    const std::vector<int>& next_occupied_orbitals_right,
    SingleOrbitalRowUpdatePlan& update_plan) {
  if (previous_occupied_orbitals_right.size() != next_occupied_orbitals_right.size()) {
    return false;
  }

  const int n_electrons = static_cast<int>(previous_occupied_orbitals_right.size());
  update_plan.row_permutation.assign(static_cast<std::size_t>(n_electrons), -1);
  update_plan.replacement_row = -1;

  std::vector<bool> used_previous_rows(static_cast<std::size_t>(n_electrons), false);
  for (int next_row = 0; next_row < n_electrons; ++next_row) {
    const int next_orbital =
        next_occupied_orbitals_right[static_cast<std::size_t>(next_row)];
    int matched_previous_row = -1;
    for (int previous_row = 0; previous_row < n_electrons; ++previous_row) {
      if (used_previous_rows[static_cast<std::size_t>(previous_row)]) {
        continue;
      }
      if (previous_occupied_orbitals_right[static_cast<std::size_t>(previous_row)] != next_orbital) {
        continue;
      }
      matched_previous_row = previous_row;
      break;
    }

    if (matched_previous_row >= 0) {
      update_plan.row_permutation[static_cast<std::size_t>(next_row)] = matched_previous_row;
      used_previous_rows[static_cast<std::size_t>(matched_previous_row)] = true;
      continue;
    }

    if (update_plan.replacement_row >= 0) {
      return false;
    }
    update_plan.replacement_row = next_row;
  }

  int removed_previous_row = -1;
  for (int previous_row = 0; previous_row < n_electrons; ++previous_row) {
    if (used_previous_rows[static_cast<std::size_t>(previous_row)]) {
      continue;
    }
    if (removed_previous_row >= 0) {
      return false;
    }
    removed_previous_row = previous_row;
  }

  if (update_plan.replacement_row < 0) {
    return removed_previous_row < 0;
  }
  if (removed_previous_row < 0) {
    return false;
  }

  update_plan.row_permutation[static_cast<std::size_t>(update_plan.replacement_row)] =
      removed_previous_row;
  return is_valid_permutation(update_plan.row_permutation);
}

int count_position_mismatches(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right) {
  if (occupied_orbitals_left.size() != occupied_orbitals_right.size()) {
    throw std::invalid_argument("occupied orbital list sizes must match");
  }

  int mismatch_count = 0;
  for (std::size_t orbital_index = 0;
       orbital_index < occupied_orbitals_left.size();
       ++orbital_index) {
    if (occupied_orbitals_left[orbital_index] != occupied_orbitals_right[orbital_index]) {
      ++mismatch_count;
    }
  }
  return mismatch_count;
}

std::vector<int> build_greedy_determinant_traversal_order(
    const std::vector<std::vector<int>>& alpha_occupied_orbitals_by_determinant,
    const std::vector<std::vector<int>>& beta_occupied_orbitals_by_determinant) {
  const int n_determinants = static_cast<int>(alpha_occupied_orbitals_by_determinant.size());
  std::vector<int> traversal_order;
  traversal_order.reserve(static_cast<std::size_t>(n_determinants));
  if (n_determinants == 0) {
    return traversal_order;
  }

  std::vector<bool> visited(static_cast<std::size_t>(n_determinants), false);
  int current_index = 0;
  traversal_order.push_back(current_index);
  visited[static_cast<std::size_t>(current_index)] = true;

  for (int visited_count = 1; visited_count < n_determinants; ++visited_count) {
    int best_index = -1;
    int best_distance = std::numeric_limits<int>::max();

    for (int candidate_index = 0; candidate_index < n_determinants; ++candidate_index) {
      if (visited[static_cast<std::size_t>(candidate_index)]) {
        continue;
      }

      const int candidate_distance =
          count_position_mismatches(
              alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(current_index)],
              alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(candidate_index)]) +
          count_position_mismatches(
              beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(current_index)],
              beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(candidate_index)]);
      if (candidate_distance < best_distance) {
        best_distance = candidate_distance;
        best_index = candidate_index;
      }
    }

    traversal_order.push_back(best_index);
    visited[static_cast<std::size_t>(best_index)] = true;
    current_index = best_index;
  }

  return traversal_order;
}

void store_spin_determinant_traversal_state(
    const std::vector<int>& occupied_orbitals_right,
    std::vector<double> overlap_submatrix,
    DeterminantOverlapResult overlap_result,
    SpinDeterminantTraversalState& state) {
  state.initialized = true;
  state.occupied_orbitals_right = occupied_orbitals_right;
  state.overlap_submatrix = std::move(overlap_submatrix);
  state.overlap_result = std::move(overlap_result);
  state.inverse_matrix.resize(0, 0);

  if (state.overlap_result.n_electrons == 0) {
    return;
  }
  if (state.overlap_result.nullity != 0 ||
      std::abs(state.overlap_result.overlap_determinant) <=
          kLowRankUpdateStabilityThreshold) {
    return;
  }

  state.inverse_matrix = build_inverse_overlap_submatrix_from_result(state.overlap_result);
}

bool try_apply_single_row_overlap_update(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& basis_overlap_matrix,
    int n_orbitals,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    SpinDeterminantTraversalState& state) {
  if (!state.initialized || state.overlap_result.n_electrons == 0) {
    return occupied_orbitals_left.empty();
  }
  if (state.overlap_result.nullity != 0) {
    return false;
  }
  if (occupied_orbitals_right.size() != state.occupied_orbitals_right.size()) {
    return false;
  }
  if (state.inverse_matrix.rows() != static_cast<int>(occupied_orbitals_left.size())) {
    return false;
  }

  if (occupied_orbitals_right == state.occupied_orbitals_right) {
    return true;
  }

  SingleOrbitalRowUpdatePlan update_plan;
  if (!build_single_orbital_row_update_plan(
          state.occupied_orbitals_right,
          occupied_orbitals_right,
          update_plan) ||
      !is_valid_permutation(update_plan.row_permutation) ||
      update_plan.replacement_row < 0) {
    return false;
  }

  int changed_rows_after_permutation = 0;
  for (std::size_t row_index = 0; row_index < occupied_orbitals_right.size(); ++row_index) {
    const int permuted_previous_orbital =
        state.occupied_orbitals_right
            [static_cast<std::size_t>(update_plan.row_permutation[row_index])];
    if (permuted_previous_orbital != occupied_orbitals_right[row_index]) {
      ++changed_rows_after_permutation;
      if (static_cast<int>(row_index) != update_plan.replacement_row) {
        return false;
      }
    }
  }
  if (changed_rows_after_permutation != 1) {
    return false;
  }

  const int n_electrons = static_cast<int>(occupied_orbitals_left.size());
  std::vector<double> updated_overlap_submatrix = permute_rows_of_overlap_submatrix(
      state.overlap_submatrix,
      n_electrons,
      update_plan.row_permutation);
  auto updated_overlap_matrix = Eigen::Map<ColumnMajorMatrixXd>(
      updated_overlap_submatrix.data(),
      n_electrons,
      n_electrons);
  const int permuted_determinant_sign = permutation_sign(update_plan.row_permutation);
  const double permuted_determinant =
      static_cast<double>(permuted_determinant_sign) * state.overlap_result.overlap_determinant;
  if (!std::isfinite(permuted_determinant)) {
    return false;
  }

  ColumnMajorMatrixXd permuted_inverse = permute_columns_of_inverse_matrix(
      state.inverse_matrix,
      update_plan.row_permutation);
  Eigen::RowVectorXd row_delta(n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occupied_orbitals_left[static_cast<std::size_t>(left_column)];
    const double updated_value =
        basis_overlap_matrix[static_cast<std::size_t>(orbital_index_left) * n_orbitals +
                             occupied_orbitals_right[static_cast<std::size_t>(update_plan.replacement_row)]];
    row_delta(left_column) =
        updated_value - updated_overlap_matrix(update_plan.replacement_row, left_column);
    updated_overlap_matrix(update_plan.replacement_row, left_column) = updated_value;
  }

  const Eigen::VectorXd inverse_column =
      permuted_inverse.col(update_plan.replacement_row);
  const Eigen::RowVectorXd row_times_inverse = row_delta * permuted_inverse;
  const double denominator = 1.0 + row_delta.dot(inverse_column);
  if (!std::isfinite(denominator) ||
      std::abs(denominator) <= kLowRankUpdateStabilityThreshold) {
    return false;
  }

  const double updated_determinant = permuted_determinant * denominator;
  if (!std::isfinite(updated_determinant)) {
    return false;
  }

  ColumnMajorMatrixXd updated_inverse = std::move(permuted_inverse);
  updated_inverse.noalias() -=
      (inverse_column * row_times_inverse) / denominator;
  if (!updated_inverse.allFinite()) {
    return false;
  }

  state.occupied_orbitals_right = occupied_orbitals_right;
  state.overlap_submatrix = std::move(updated_overlap_submatrix);
  state.overlap_result = determinant_overlap_resolver.resolve(
      state.overlap_submatrix,
      n_electrons);
  if (state.overlap_result.nullity != 0 ||
      std::abs(state.overlap_result.overlap_determinant) <=
          kLowRankUpdateStabilityThreshold) {
    return false;
  }
  state.inverse_matrix = std::move(updated_inverse);
  return true;
}

const SpinDeterminantTraversalState& resolve_spin_determinant_overlap_with_cache(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& basis_overlap_matrix,
    int n_orbitals,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    SpinDeterminantTraversalState& traversal_state) {
  if (occupied_orbitals_left.empty()) {
    DeterminantOverlapResult overlap_result;
    overlap_result.overlap_determinant = 1.0;
    store_spin_determinant_traversal_state(
        occupied_orbitals_right,
        {},
        std::move(overlap_result),
        traversal_state);
    return traversal_state;
  }

#if XMVB_CPP_ENABLE_DETERMINANT_LOW_RANK_UPDATES
  if (!try_apply_single_row_overlap_update(
          occupied_orbitals_left,
          occupied_orbitals_right,
          basis_overlap_matrix,
          n_orbitals,
          determinant_overlap_resolver,
          traversal_state)) {
#endif
    auto overlap_submatrix = build_overlap_submatrix(
        occupied_orbitals_left,
        occupied_orbitals_right,
        basis_overlap_matrix,
        n_orbitals);
    auto overlap_result = determinant_overlap_resolver.resolve(
        overlap_submatrix,
        static_cast<int>(occupied_orbitals_left.size()));
    store_spin_determinant_traversal_state(
        occupied_orbitals_right,
        std::move(overlap_submatrix),
        std::move(overlap_result),
        traversal_state);
#if XMVB_CPP_ENABLE_DETERMINANT_LOW_RANK_UPDATES
  }
#endif

  return traversal_state;
}

std::vector<double> normalize_state_average_weights(
    const std::vector<double>& state_average_weights) {
  double weight_sum = 0.0;
  for (const double state_weight : state_average_weights) {
    if (state_weight < 0.0) {
      throw std::invalid_argument("state_average_weights must be non-negative");
    }
    weight_sum += state_weight;
  }
  if (weight_sum <= 0.0) {
    throw std::invalid_argument("state_average_weights must sum to a positive value");
  }

  std::vector<double> normalized_weights = state_average_weights;
  for (double& state_weight : normalized_weights) {
    state_weight /= weight_sum;
  }
  return normalized_weights;
}

void validate_state_selection(
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    int n_structures) {
  if (selected_state_indices.empty()) {
    throw std::invalid_argument("selected_state_indices must not be empty");
  }
  if (selected_state_indices.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and state_average_weights must have the same length");
  }
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }
}

double compute_average_structure_overlap(
    const std::vector<double>& overlap_matrix,
    int n_structures) {
  double diagonal_sum = 0.0;
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    diagonal_sum += overlap_matrix[static_cast<std::size_t>(structure_index) * n_structures +
                                   structure_index];
  }
  return diagonal_sum / static_cast<double>(n_structures);
}

double structure_upper_weight(
    const std::vector<double>& eigenvector_matrix,
    int n_structures,
    int structure_row,
    int structure_column,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  double weight = 0.0;

  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index = selected_state_indices[selected_state_offset];
    const double state_weight = state_average_weights[selected_state_offset];
    const double coefficient_row =
        eigenvector_matrix[static_cast<std::size_t>(state_index) * n_structures + structure_row];
    const double coefficient_column =
        eigenvector_matrix[static_cast<std::size_t>(state_index) * n_structures + structure_column];

    if (structure_row == structure_column) {
      weight += state_weight * coefficient_row * coefficient_column;
    } else {
      weight += 2.0 * state_weight * coefficient_row * coefficient_column;
    }
  }

  return weight;
}

double structure_upper_overlap_weight(
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures,
    int structure_row,
    int structure_column,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  double weight = 0.0;

  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index = selected_state_indices[selected_state_offset];
    const double state_weight = state_average_weights[selected_state_offset];
    const double state_energy = eigenvalues[static_cast<std::size_t>(state_index)];
    const double coefficient_row =
        eigenvector_matrix[static_cast<std::size_t>(state_index) * n_structures + structure_row];
    const double coefficient_column =
        eigenvector_matrix[static_cast<std::size_t>(state_index) * n_structures + structure_column];

    if (structure_row == structure_column) {
      weight -= state_weight * state_energy * coefficient_row * coefficient_column;
    } else {
      weight -= 2.0 * state_weight * state_energy * coefficient_row * coefficient_column;
    }
  }

  return weight;
}

double determinant_pair_structure_weight(
    const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
    const std::vector<double>& eigenvector_matrix,
    int n_structures,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  double pair_weight = 0.0;

  for (const auto& left_term : determinant_to_structures_left) {
    for (const auto& right_term : determinant_to_structures_right) {
      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }
      pair_weight +=
          left_term.coefficient * right_term.coefficient *
          structure_upper_weight(
              eigenvector_matrix,
              n_structures,
              left_term.structure_index,
              right_term.structure_index,
              selected_state_indices,
              state_average_weights);
    }
  }

  return pair_weight;
}

StructurePairAdjoints determinant_pair_structure_adjoints(
    const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  StructurePairAdjoints adjoints;

  for (const auto& left_term : determinant_to_structures_left) {
    for (const auto& right_term : determinant_to_structures_right) {
      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }
      const double coefficient_product =
          left_term.coefficient * right_term.coefficient;
      adjoints.hamiltonian_weight +=
          coefficient_product *
          structure_upper_weight(
              eigenvector_matrix,
              n_structures,
              left_term.structure_index,
              right_term.structure_index,
              selected_state_indices,
              state_average_weights);
      adjoints.overlap_weight +=
          coefficient_product *
          structure_upper_overlap_weight(
              eigenvector_matrix,
              eigenvalues,
              n_structures,
              left_term.structure_index,
              right_term.structure_index,
              selected_state_indices,
              state_average_weights);
    }
  }

  return adjoints;
}


double selected_state_average_energy(
    const std::vector<double>& eigenvalues,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  double energy = 0.0;
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    energy += state_average_weights[selected_state_offset] *
              eigenvalues[static_cast<std::size_t>(selected_state_indices[selected_state_offset])];
  }
  return energy;
}

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_one_electron_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  double one_electron_reference_energy = 0.0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          static_cast<std::size_t>(column) * n_basis_functions + row;
      one_electron_reference_energy +=
          inactive_density_matrix[index] *
          (ao_effective_one_electron_matrix[index] + ao_core_hamiltonian_matrix[index]);
    }
  }
  return one_electron_reference_energy;
}

}  // namespace

CppActiveSpaceGradientEvaluator::CppActiveSpaceGradientEvaluator(
    VbScfAlgorithm algorithm)
    : orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      active_space_one_electron_builder_(),
      active_space_two_electron_builder_(),
      structure_builder_(algorithm),
      generalized_eigensolver_(),
      algorithm_(algorithm) {}

CppActiveSpaceGradientEvaluator::CppActiveSpaceGradientEvaluator(
    ActiveSpaceOrbitalPreparer orbital_preparer,
    AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
    ActiveSpaceOneElectronBuilder active_space_one_electron_builder,
    ActiveSpaceTwoElectronBuilder active_space_two_electron_builder,
    FullDeterminantStructureHamiltonianOverlapBuilder structure_builder,
    xmvb::core::GeneralizedEigensolver generalized_eigensolver,
    VbScfAlgorithm algorithm)
    : orbital_preparer_(std::move(orbital_preparer)),
      ao_effective_one_electron_builder_(std::move(ao_effective_one_electron_builder)),
      active_space_one_electron_builder_(std::move(active_space_one_electron_builder)),
      active_space_two_electron_builder_(std::move(active_space_two_electron_builder)),
      structure_builder_(std::move(structure_builder)),
      generalized_eigensolver_(std::move(generalized_eigensolver)),
      algorithm_(algorithm) {}

CppActiveSpaceGradientResult CppActiveSpaceGradientEvaluator::evaluate(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

CppActiveSpaceGradientResult CppActiveSpaceGradientEvaluator::evaluate(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.structure_data.n_structures <= 0) {
    throw std::invalid_argument("input.structure_data.n_structures must be positive");
  }
  validate_state_selection(
      selected_state_indices,
      state_average_weights,
      input.structure_data.n_structures);
  const std::vector<double> normalized_weights =
      normalize_state_average_weights(state_average_weights);

  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  CppActiveSpaceGradientResult result;

  auto stage_start_time = std::chrono::steady_clock::now();
  const auto orbital_result =
      orbital_preparer_.prepare(input.orbital_preparation_input);
  result.orbital_preparation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  const auto ao_effective_one_electron_result =
      ao_effective_one_electron_builder_.build(
          orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.ao_integral_input.n_basis_functions);
  result.ao_effective_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_one_electron_result =
      active_space_one_electron_builder_.build(
          ao_effective_one_electron_result.ao_effective_one_electron_matrix,
          orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  result.active_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  const auto active_space_two_electron_result =
      active_space_two_electron_builder_.build(
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          input.ao_integral_input.n_basis_functions,
          input.orbital_preparation_input.n_active_orbitals);
  result.active_two_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  const auto structure_matrices = structure_builder_.build(
      input.structure_data.alpha_occupied_orbitals_by_determinant,
      input.structure_data.beta_occupied_orbitals_by_determinant,
      input.structure_data.determinant_to_structure_terms,
      orbital_result.active_orbital_overlap_matrix,
      active_space_one_electron_result.active_one_electron_matrix,
      input.orbital_preparation_input.n_active_orbitals,
      active_space_two_electron_result.packed_active_two_electron_integrals,
      input.structure_data.n_structures);
  result.structure_matrix_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  const auto eigen_result = generalized_eigensolver_.solve(
      structure_matrices.hamiltonian_matrix,
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  result.eigensolver_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  result.orbital_preparation_result = orbital_result;
  result.ao_effective_one_electron_result = ao_effective_one_electron_result;
  result.active_orbital_overlap_matrix = orbital_result.active_orbital_overlap_matrix;
  result.active_space_one_electron_result = active_space_one_electron_result;
  result.active_space_two_electron_result = active_space_two_electron_result;
  result.active_orbital_overlap_gradient.assign(
      orbital_result.active_orbital_overlap_matrix.size(),
      0.0);
  result.active_one_electron_gradient.assign(
      active_space_one_electron_result.active_one_electron_matrix.size(),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      active_space_two_electron_result.packed_active_two_electron_integrals.size(),
      0.0);

  result.scf_result.n_structures = input.structure_data.n_structures;
  result.scf_result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  result.scf_result.selected_state_indices = selected_state_indices;
  result.scf_result.state_average_weights = normalized_weights;
  result.scf_result.structure_matrices = structure_matrices;
  result.scf_result.average_structure_overlap = compute_average_structure_overlap(
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  result.scf_result.electronic_state_energies = eigen_result.eigenvalues;
  result.scf_result.eigenvector_matrix = eigen_result.eigenvector_matrix;
  result.scf_result.one_electron_reference_energy = compute_one_electron_reference_energy(
      orbital_result.inactive_density_matrix,
      ao_effective_one_electron_result.ao_effective_one_electron_matrix,
      input.ao_integral_input.ao_core_hamiltonian_matrix,
      input.ao_integral_input.n_basis_functions);
  result.scf_result.electronic_energy = selected_state_average_energy(
      eigen_result.eigenvalues,
      selected_state_indices,
      normalized_weights);
  result.scf_result.total_energy =
      result.scf_result.one_electron_reference_energy +
      result.scf_result.electronic_energy +
      nuclear_repulsion_energy;
  result.scf_result.selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    result.scf_result.selected_state_total_energies[selected_state_offset] =
        eigen_result.eigenvalues[static_cast<std::size_t>(selected_state_indices[selected_state_offset])] +
        nuclear_repulsion_energy;
  }

  DeterminantOverlapResolver determinant_overlap_resolver;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_occupied_orbitals_by_determinant.size());
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

  stage_start_time = std::chrono::steady_clock::now();
  for (int determinant_index_left = 0; determinant_index_left < n_determinants; ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right < n_determinants;
         ++determinant_index_right) {
      const auto pair_adjoints = determinant_pair_structure_adjoints(
          input.structure_data.determinant_to_structure_terms[static_cast<std::size_t>(determinant_index_left)],
          input.structure_data.determinant_to_structure_terms[static_cast<std::size_t>(determinant_index_right)],
          eigen_result.eigenvector_matrix,
          eigen_result.eigenvalues,
          input.structure_data.n_structures,
          selected_state_indices,
          normalized_weights);
      if (pair_adjoints.hamiltonian_weight == 0.0 &&
          pair_adjoints.overlap_weight == 0.0) {
        continue;
      }

      const auto alpha_overlap_submatrix = build_overlap_submatrix(
          input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          orbital_result.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto alpha_result = determinant_overlap_resolver.resolve(
          alpha_overlap_submatrix,
          static_cast<int>(
              input.structure_data.alpha_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_left)]
                  .size()));
      const auto beta_overlap_submatrix = build_overlap_submatrix(
          input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          orbital_result.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto beta_result = determinant_overlap_resolver.resolve(
          beta_overlap_submatrix,
          static_cast<int>(
              input.structure_data.beta_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_left)]
                  .size()));
      const ColumnMajorMatrixXd alpha_first_order_cofactor_matrix =
          build_first_order_cofactor_matrix_from_result(alpha_result);
      const ColumnMajorMatrixXd beta_first_order_cofactor_matrix =
          build_first_order_cofactor_matrix_from_result(beta_result);

      const double alpha_weight =
          pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant;
      const double beta_weight =
          pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant;

      const int n_alpha_electrons = static_cast<int>(
          input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)].size());
      for (int left_column = 0; left_column < n_alpha_electrons; ++left_column) {
        const int orbital_index_left =
            input.structure_data.alpha_occupied_orbitals_by_determinant
                [static_cast<std::size_t>(determinant_index_left)]
                [static_cast<std::size_t>(left_column)];
        for (int right_row = 0; right_row < n_alpha_electrons; ++right_row) {
          const int orbital_index_right =
              input.structure_data.alpha_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_right)]
                  [static_cast<std::size_t>(right_row)];
          result.active_one_electron_gradient[static_cast<std::size_t>(orbital_index_left) *
                                                  n_active_orbitals +
                                              orbital_index_right] +=
              alpha_weight *
              alpha_first_order_cofactor_matrix(right_row, left_column);
        }
      }

      const int n_beta_electrons = static_cast<int>(
          input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)].size());
      for (int left_column = 0; left_column < n_beta_electrons; ++left_column) {
        const int orbital_index_left =
            input.structure_data.beta_occupied_orbitals_by_determinant
                [static_cast<std::size_t>(determinant_index_left)]
                [static_cast<std::size_t>(left_column)];
        for (int right_row = 0; right_row < n_beta_electrons; ++right_row) {
          const int orbital_index_right =
              input.structure_data.beta_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_right)]
                  [static_cast<std::size_t>(right_row)];
          result.active_one_electron_gradient[static_cast<std::size_t>(orbital_index_left) *
                                                  n_active_orbitals +
                                              orbital_index_right] +=
              beta_weight *
              beta_first_order_cofactor_matrix(right_row, left_column);
        }
      }

      if (alpha_result.nullity != 0 || beta_result.nullity != 0) {
        throw std::runtime_error("analytic active-space overlap gradient requires nullity == 0");
      }
      ColumnMajorMatrixXd alpha_same_spin_inverse_overlap_gradient =
          ColumnMajorMatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
      ColumnMajorMatrixXd beta_same_spin_inverse_overlap_gradient =
          ColumnMajorMatrixXd::Zero(n_beta_electrons, n_beta_electrons);
      ColumnMajorMatrixXd alpha_opposite_spin_inverse_overlap_gradient =
          ColumnMajorMatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
      ColumnMajorMatrixXd beta_opposite_spin_inverse_overlap_gradient =
          ColumnMajorMatrixXd::Zero(n_beta_electrons, n_beta_electrons);

      double opposite_spin_phi = 0.0;
      SameSpinBiorthogonalPhiResult alpha_phi_result;
      SameSpinBiorthogonalPhiResult beta_phi_result;
      if (algorithm_ == VbScfAlgorithm::Biorthogonal) {
        alpha_phi_result = compute_same_spin_biorthogonal_phi(
            input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
            input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
            active_space_one_electron_result.active_one_electron_matrix,
            n_active_orbitals,
            active_space_two_electron_result.packed_active_two_electron_integrals,
            alpha_result,
            &alpha_same_spin_inverse_overlap_gradient);
        beta_phi_result = compute_same_spin_biorthogonal_phi(
            input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
            input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
            active_space_one_electron_result.active_one_electron_matrix,
            n_active_orbitals,
            active_space_two_electron_result.packed_active_two_electron_integrals,
            beta_result,
            &beta_same_spin_inverse_overlap_gradient);
        if (n_alpha_electrons > 0 && n_beta_electrons > 0) {
          opposite_spin_phi = compute_opposite_spin_biorthogonal_phi(
              input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
              input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
              alpha_result,
              input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
              input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
              beta_result,
              active_space_two_electron_result.packed_active_two_electron_integrals,
              &alpha_opposite_spin_inverse_overlap_gradient,
              &beta_opposite_spin_inverse_overlap_gradient);
        }
      } else {
        alpha_phi_result = compute_same_spin_original_phi(
            input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
            input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
            active_space_one_electron_result.active_one_electron_matrix,
            n_active_orbitals,
            active_space_two_electron_result.packed_active_two_electron_integrals,
            alpha_result,
            &alpha_same_spin_inverse_overlap_gradient);
        beta_phi_result = compute_same_spin_original_phi(
            input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
            input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
            active_space_one_electron_result.active_one_electron_matrix,
            n_active_orbitals,
            active_space_two_electron_result.packed_active_two_electron_integrals,
            beta_result,
            &beta_same_spin_inverse_overlap_gradient);
        if (n_alpha_electrons > 0 && n_beta_electrons > 0) {
          opposite_spin_phi = compute_opposite_spin_original_phi(
              input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
              input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
              alpha_result,
              input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
              input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
              beta_result,
              active_space_two_electron_result.packed_active_two_electron_integrals,
              &alpha_opposite_spin_inverse_overlap_gradient,
              &beta_opposite_spin_inverse_overlap_gradient);
        }
      }
      const ColumnMajorMatrixXd alpha_inverse_overlap_gradient =
          alpha_same_spin_inverse_overlap_gradient +
          alpha_opposite_spin_inverse_overlap_gradient;
      const ColumnMajorMatrixXd beta_inverse_overlap_gradient =
          beta_same_spin_inverse_overlap_gradient +
          beta_opposite_spin_inverse_overlap_gradient;
      const double alpha_phi = alpha_phi_result.total_phi;
      const double beta_phi = beta_phi_result.total_phi;
      const double alpha_determinant_weight =
          pair_adjoints.overlap_weight * beta_result.overlap_determinant +
          pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
              (alpha_phi + beta_phi + opposite_spin_phi);
      const double beta_determinant_weight =
          pair_adjoints.overlap_weight * alpha_result.overlap_determinant +
          pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
              (alpha_phi + beta_phi + opposite_spin_phi);
      accumulate_spin_overlap_gradient(
          input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          input.structure_data.alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          alpha_result,
          alpha_determinant_weight,
          pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
              alpha_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);
      accumulate_spin_overlap_gradient(
          input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          input.structure_data.beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          beta_result,
          beta_determinant_weight,
          pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
              beta_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);

      if (alpha_result.nullity == 0) {
        for (int left_first = 0; left_first < n_alpha_electrons - 1; ++left_first) {
          const int orbital_index_left_first =
              input.structure_data.alpha_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_left)]
                  [static_cast<std::size_t>(left_first)];
          for (int right_first = 0; right_first < n_alpha_electrons - 1; ++right_first) {
            const int orbital_index_right_first =
                input.structure_data.alpha_occupied_orbitals_by_determinant
                    [static_cast<std::size_t>(determinant_index_right)]
                    [static_cast<std::size_t>(right_first)];
            const double cofactor_11 =
                alpha_first_order_cofactor_matrix(right_first, left_first);
            for (int left_second = left_first + 1; left_second < n_alpha_electrons; ++left_second) {
              const int orbital_index_left_second =
                  input.structure_data.alpha_occupied_orbitals_by_determinant
                      [static_cast<std::size_t>(determinant_index_left)]
                      [static_cast<std::size_t>(left_second)];
              const double cofactor_12 =
                  alpha_first_order_cofactor_matrix(right_first, left_second);
              for (int right_second = right_first + 1; right_second < n_alpha_electrons; ++right_second) {
                const int orbital_index_right_second =
                    input.structure_data.alpha_occupied_orbitals_by_determinant
                        [static_cast<std::size_t>(determinant_index_right)]
                        [static_cast<std::size_t>(right_second)];
                const double cofactor_22 =
                    alpha_first_order_cofactor_matrix(right_second, left_second);
                const double cofactor_21 =
                    alpha_first_order_cofactor_matrix(right_second, left_first);
                const double second_order_cofactor =
                    (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21) /
                    alpha_result.overlap_determinant;

                const int direct_index = TwoElectronIndexer::two_electron_storage_index(
                    orbital_index_right_first,
                    orbital_index_left_first,
                    orbital_index_right_second,
                    orbital_index_left_second);
                const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
                    orbital_index_right_first,
                    orbital_index_left_second,
                    orbital_index_right_second,
                    orbital_index_left_first);
                result.packed_active_two_electron_gradient[static_cast<std::size_t>(direct_index)] +=
                    alpha_weight * second_order_cofactor;
                result.packed_active_two_electron_gradient[static_cast<std::size_t>(exchange_index)] -=
                    alpha_weight * second_order_cofactor;
              }
            }
          }
        }
      }

      if (beta_result.nullity == 0) {
        for (int left_first = 0; left_first < n_beta_electrons - 1; ++left_first) {
          const int orbital_index_left_first =
              input.structure_data.beta_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_left)]
                  [static_cast<std::size_t>(left_first)];
          for (int right_first = 0; right_first < n_beta_electrons - 1; ++right_first) {
            const int orbital_index_right_first =
                input.structure_data.beta_occupied_orbitals_by_determinant
                    [static_cast<std::size_t>(determinant_index_right)]
                    [static_cast<std::size_t>(right_first)];
            const double cofactor_11 =
                beta_first_order_cofactor_matrix(right_first, left_first);
            for (int left_second = left_first + 1; left_second < n_beta_electrons; ++left_second) {
              const int orbital_index_left_second =
                  input.structure_data.beta_occupied_orbitals_by_determinant
                      [static_cast<std::size_t>(determinant_index_left)]
                      [static_cast<std::size_t>(left_second)];
              const double cofactor_12 =
                  beta_first_order_cofactor_matrix(right_first, left_second);
              for (int right_second = right_first + 1; right_second < n_beta_electrons; ++right_second) {
                const int orbital_index_right_second =
                    input.structure_data.beta_occupied_orbitals_by_determinant
                        [static_cast<std::size_t>(determinant_index_right)]
                        [static_cast<std::size_t>(right_second)];
                const double cofactor_22 =
                    beta_first_order_cofactor_matrix(right_second, left_second);
                const double cofactor_21 =
                    beta_first_order_cofactor_matrix(right_second, left_first);
                const double second_order_cofactor =
                    (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21) /
                    beta_result.overlap_determinant;

                const int direct_index = TwoElectronIndexer::two_electron_storage_index(
                    orbital_index_right_first,
                    orbital_index_left_first,
                    orbital_index_right_second,
                    orbital_index_left_second);
                const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
                    orbital_index_right_first,
                    orbital_index_left_second,
                    orbital_index_right_second,
                    orbital_index_left_first);
                result.packed_active_two_electron_gradient[static_cast<std::size_t>(direct_index)] +=
                    beta_weight * second_order_cofactor;
                result.packed_active_two_electron_gradient[static_cast<std::size_t>(exchange_index)] -=
                    beta_weight * second_order_cofactor;
              }
            }
          }
        }
      }

      if (alpha_result.nullity < 2 && beta_result.nullity < 2 &&
          n_alpha_electrons > 0 && n_beta_electrons > 0) {
        for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
          const int alpha_orbital_left =
              input.structure_data.alpha_occupied_orbitals_by_determinant
                  [static_cast<std::size_t>(determinant_index_left)]
                  [static_cast<std::size_t>(alpha_left_column)];
          for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
            const int alpha_orbital_right =
                input.structure_data.alpha_occupied_orbitals_by_determinant
                    [static_cast<std::size_t>(determinant_index_right)]
                    [static_cast<std::size_t>(alpha_right_row)];
            const double alpha_cofactor =
                alpha_first_order_cofactor_matrix(alpha_right_row, alpha_left_column);
            for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
              const int beta_orbital_left =
                  input.structure_data.beta_occupied_orbitals_by_determinant
                      [static_cast<std::size_t>(determinant_index_left)]
                      [static_cast<std::size_t>(beta_left_column)];
              for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
                const int beta_orbital_right =
                    input.structure_data.beta_occupied_orbitals_by_determinant
                        [static_cast<std::size_t>(determinant_index_right)]
                        [static_cast<std::size_t>(beta_right_row)];
                const double beta_cofactor =
                    beta_first_order_cofactor_matrix(beta_right_row, beta_left_column);

                const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
                    beta_orbital_right,
                    beta_orbital_left,
                    alpha_orbital_right,
                    alpha_orbital_left);
                result.packed_active_two_electron_gradient
                    [static_cast<std::size_t>(two_electron_index)] +=
                    pair_adjoints.hamiltonian_weight * alpha_cofactor * beta_cofactor;
              }
            }
          }
        }
      }
    }
  }
  result.adjoint_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();

  return result;
}

}  // namespace xmvb::vb
