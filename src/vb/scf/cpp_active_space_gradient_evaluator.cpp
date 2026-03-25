#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixMap = Eigen::Map<Matrix>;

struct StructurePairAdjoints {
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

struct ActiveSpaceGradientForwardContext {
  TimedPreparedActiveSpaceContext timed_active_space_context;
  FullDeterminantStructureBuildResult structure_build_result;
  xmvb::core::GeneralizedEigenResult eigen_result;
  double structure_matrix_wall_time_seconds = 0.0;
  double eigensolver_wall_time_seconds = 0.0;
};

void accumulate_one_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Matrix& cofactor_1st,
    double weight,
    Eigen::Ref<Matrix> active_one_electron_gradient) {
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[static_cast<std::size_t>(right_row)];
      active_one_electron_gradient(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_same_spin_two_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Matrix& cofactor_1st,
    double overlap_determinant,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double cofactor_scale = weight / overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[static_cast<std::size_t>(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[static_cast<std::size_t>(right_first)];
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[static_cast<std::size_t>(left_second)];
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[static_cast<std::size_t>(right_second)];
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double second_order_cofactor =
              cofactor_scale * (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21);

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
          (*packed_active_two_electron_gradient)[static_cast<std::size_t>(direct_index)] +=
              second_order_cofactor;
          (*packed_active_two_electron_gradient)[static_cast<std::size_t>(exchange_index)] -=
              second_order_cofactor;
        }
      }
    }
  }
}

void accumulate_opposite_spin_two_electron_gradient_contribution(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const Matrix& alpha_cofactor_1st,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const Matrix& beta_cofactor_1st,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[static_cast<std::size_t>(alpha_right_row)];
      const double weighted_alpha_cofactor =
          weight * alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[static_cast<std::size_t>(beta_right_row)];
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          (*packed_active_two_electron_gradient)[static_cast<std::size_t>(two_electron_index)] +=
              weighted_alpha_cofactor * beta_cofactor_1st(beta_right_row, beta_left_column);
        }
      }
    }
  }
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

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::core::GeneralizedEigensolver& generalized_eigensolver) {
  ActiveSpaceGradientForwardContext context;
  context.timed_active_space_context =
      prepare_timed_active_space_context(
          input,
          orbital_preparer,
          ao_effective_one_electron_builder,
          active_space_one_electron_builder,
          active_space_two_electron_builder);

  const auto& prepared_active_space =
      context.timed_active_space_context.prepared_active_space;
  auto stage_start_time = std::chrono::steady_clock::now();
  context.structure_build_result = structure_builder.build_with_pair_evaluations(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result.packed_active_two_electron_integrals,
      input.structure_data.n_structures);
  context.structure_matrix_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.eigen_result = generalized_eigensolver.solve(
      context.structure_build_result.structure_matrices.hamiltonian_matrix,
      context.structure_build_result.structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  context.eigensolver_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  return context;
}

void populate_scf_result(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double nuclear_repulsion_energy,
    const ActiveSpaceGradientForwardContext& forward_context,
    CppVbScfResult* scf_result) {
  const auto& prepared_active_space =
      forward_context.timed_active_space_context.prepared_active_space;
  const auto& structure_matrices = forward_context.structure_build_result.structure_matrices;
  const auto& eigen_result = forward_context.eigen_result;

  scf_result->n_structures = input.structure_data.n_structures;
  scf_result->nuclear_repulsion_energy = nuclear_repulsion_energy;
  scf_result->selected_state_indices = selected_state_indices;
  scf_result->state_average_weights = normalized_weights;
  scf_result->structure_matrices = structure_matrices;
  scf_result->average_structure_overlap = compute_average_structure_overlap(
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  scf_result->electronic_state_energies = eigen_result.eigenvalues;
  scf_result->eigenvector_matrix = eigen_result.eigenvector_matrix;
  scf_result->one_electron_reference_energy =
      prepared_active_space.one_electron_reference_energy;
  scf_result->electronic_energy = selected_state_average_energy(
      eigen_result.eigenvalues,
      selected_state_indices,
      normalized_weights);
  scf_result->total_energy =
      scf_result->one_electron_reference_energy +
      scf_result->electronic_energy +
      nuclear_repulsion_energy;
  scf_result->selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    scf_result->selected_state_total_energies[selected_state_offset] =
        eigen_result.eigenvalues[static_cast<std::size_t>(
            selected_state_indices[selected_state_offset])] +
        nuclear_repulsion_energy;
  }
}

void initialize_active_space_gradient_result(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double nuclear_repulsion_energy,
    const ActiveSpaceGradientForwardContext& forward_context,
    CppActiveSpaceGradientResult* result) {
  const auto& timings = forward_context.timed_active_space_context.timings;
  const auto& prepared_active_space =
      forward_context.timed_active_space_context.prepared_active_space;

  result->orbital_preparation_wall_time_seconds =
      timings.orbital_preparation_wall_time_seconds;
  result->ao_effective_one_electron_wall_time_seconds =
      timings.ao_effective_one_electron_wall_time_seconds;
  result->active_one_electron_wall_time_seconds =
      timings.active_one_electron_wall_time_seconds;
  result->active_two_electron_wall_time_seconds =
      timings.active_two_electron_wall_time_seconds;
  result->structure_matrix_wall_time_seconds =
      forward_context.structure_matrix_wall_time_seconds;
  result->eigensolver_wall_time_seconds =
      forward_context.eigensolver_wall_time_seconds;

  result->orbital_preparation_result = prepared_active_space.orbital_result;
  result->ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  result->active_orbital_overlap_matrix =
      prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  result->active_space_one_electron_result =
      prepared_active_space.active_space_one_electron_result;
  result->active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  result->active_orbital_overlap_gradient.assign(
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.size(),
      0.0);
  result->active_one_electron_gradient.assign(
      prepared_active_space.active_space_one_electron_result.h1e_act.size(),
      0.0);
  result->packed_active_two_electron_gradient.assign(
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals.size(),
      0.0);

  populate_scf_result(
      input,
      selected_state_indices,
      normalized_weights,
      nuclear_repulsion_energy,
      forward_context,
      &result->scf_result);
}

void accumulate_active_space_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    const ActiveSpaceGradientForwardContext& forward_context,
    CppActiveSpaceGradientResult* result) {
  const auto& prepared_active_space =
      forward_context.timed_active_space_context.prepared_active_space;
  const auto& active_space_one_electron_result =
      prepared_active_space.active_space_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  const auto& structure_build_result = forward_context.structure_build_result;
  const auto& eigen_result = forward_context.eigen_result;

  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  MatrixMap active_one_electron_gradient(
      result->active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  auto& packed_active_two_electron_gradient =
      result->packed_active_two_electron_gradient;

  const auto stage_start_time = std::chrono::steady_clock::now();
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right < n_determinants;
         ++determinant_index_right) {
      const auto pair_adjoints = determinant_pair_structure_adjoints(
          input.structure_data.determinant_to_structure_terms[static_cast<std::size_t>(
              determinant_index_left)],
          input.structure_data.determinant_to_structure_terms[static_cast<std::size_t>(
              determinant_index_right)],
          eigen_result.eigenvector_matrix,
          eigen_result.eigenvalues,
          input.structure_data.n_structures,
          selected_state_indices,
          normalized_weights);
      if (pair_adjoints.hamiltonian_weight == 0.0 &&
          pair_adjoints.overlap_weight == 0.0) {
        continue;
      }

      const auto& determinant_pair_evaluation =
          structure_build_result.pair_evaluation(
              determinant_index_left,
              determinant_index_right);
      const auto& alpha_occ_L =
          input.structure_data.alpha_det[static_cast<std::size_t>(
              determinant_index_left)];
      const auto& alpha_occ_R =
          input.structure_data.alpha_det[static_cast<std::size_t>(
              determinant_index_right)];
      const auto& beta_occ_L =
          input.structure_data.beta_det[static_cast<std::size_t>(
              determinant_index_left)];
      const auto& beta_occ_R =
          input.structure_data.beta_det[static_cast<std::size_t>(
              determinant_index_right)];
      const auto& alpha_result = determinant_pair_evaluation.alpha.overlap_result;
      const auto& beta_result = determinant_pair_evaluation.beta.overlap_result;
      const Matrix alpha_cofactor_1st =
          calc_cofactor_1st(alpha_result);
      const Matrix beta_cofactor_1st =
          calc_cofactor_1st(beta_result);

      const double alpha_weight =
          pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant;
      const double beta_weight =
          pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant;

      const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
      const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
      accumulate_one_electron_gradient_contribution(
          alpha_occ_L,
          alpha_occ_R,
          alpha_cofactor_1st,
          alpha_weight,
          active_one_electron_gradient);
      accumulate_one_electron_gradient_contribution(
          beta_occ_L,
          beta_occ_R,
          beta_cofactor_1st,
          beta_weight,
          active_one_electron_gradient);

      if (alpha_result.nullity != 0 || beta_result.nullity != 0) {
        throw std::runtime_error(
            "analytic active-space overlap gradient requires nullity == 0");
      }
      Matrix alpha_same_spin_inverse_overlap_gradient =
          Matrix::Zero(n_alpha_electrons, n_alpha_electrons);
      Matrix beta_same_spin_inverse_overlap_gradient =
          Matrix::Zero(n_beta_electrons, n_beta_electrons);
      Matrix alpha_opposite_spin_inverse_overlap_gradient =
          Matrix::Zero(n_alpha_electrons, n_alpha_electrons);
      Matrix beta_opposite_spin_inverse_overlap_gradient =
          Matrix::Zero(n_beta_electrons, n_beta_electrons);

      double opposite_spin_phi = 0.0;
      const SameSpinPhiResult alpha_phi_result = compute_same_spin_original_phi(
          alpha_occ_L,
          alpha_occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          active_space_two_electron_result.packed_active_two_electron_integrals,
          alpha_result,
          &alpha_same_spin_inverse_overlap_gradient);
      const SameSpinPhiResult beta_phi_result = compute_same_spin_original_phi(
          beta_occ_L,
          beta_occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          active_space_two_electron_result.packed_active_two_electron_integrals,
          beta_result,
          &beta_same_spin_inverse_overlap_gradient);
      if (n_alpha_electrons > 0 && n_beta_electrons > 0) {
        opposite_spin_phi = compute_opposite_spin_original_phi(
            alpha_occ_L,
            alpha_occ_R,
            alpha_result,
            beta_occ_L,
            beta_occ_R,
            beta_result,
            active_space_two_electron_result.packed_active_two_electron_integrals,
            &alpha_opposite_spin_inverse_overlap_gradient,
            &beta_opposite_spin_inverse_overlap_gradient);
      }
      const Matrix alpha_inverse_overlap_gradient =
          alpha_same_spin_inverse_overlap_gradient +
          alpha_opposite_spin_inverse_overlap_gradient;
      const Matrix beta_inverse_overlap_gradient =
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
          alpha_occ_L,
          alpha_occ_R,
          alpha_result,
          alpha_determinant_weight,
          pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
              alpha_inverse_overlap_gradient,
          n_active_orbitals,
          &result->active_orbital_overlap_gradient);
      accumulate_spin_overlap_gradient(
          beta_occ_L,
          beta_occ_R,
          beta_result,
          beta_determinant_weight,
          pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
              beta_inverse_overlap_gradient,
          n_active_orbitals,
          &result->active_orbital_overlap_gradient);

      if (alpha_result.nullity == 0) {
        accumulate_same_spin_two_electron_gradient_contribution(
            alpha_occ_L,
            alpha_occ_R,
            alpha_cofactor_1st,
            alpha_result.overlap_determinant,
            alpha_weight,
            &packed_active_two_electron_gradient);
      }

      if (beta_result.nullity == 0) {
        accumulate_same_spin_two_electron_gradient_contribution(
            beta_occ_L,
            beta_occ_R,
            beta_cofactor_1st,
            beta_result.overlap_determinant,
            beta_weight,
            &packed_active_two_electron_gradient);
      }

      if (alpha_result.nullity < 2 && beta_result.nullity < 2 &&
          n_alpha_electrons > 0 && n_beta_electrons > 0) {
        accumulate_opposite_spin_two_electron_gradient_contribution(
            alpha_occ_L,
            alpha_occ_R,
            alpha_cofactor_1st,
            beta_occ_L,
            beta_occ_R,
            beta_cofactor_1st,
            pair_adjoints.hamiltonian_weight,
            &packed_active_two_electron_gradient);
      }
    }
  }
  result->adjoint_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time)
          .count();
}

}  // namespace

CppActiveSpaceGradientEvaluator::CppActiveSpaceGradientEvaluator(
    VBSCFAlgorithm algorithm)
    : orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      active_space_one_electron_builder_(),
      active_space_two_electron_builder_(),
      structure_builder_(algorithm),
      generalized_eigensolver_() {}

CppActiveSpaceGradientEvaluator::CppActiveSpaceGradientEvaluator(
    ActiveSpaceOrbitalPreparer orbital_preparer,
    AoEffectiveOneElectronBuilder ao_effective_one_electron_builder,
    ActiveSpaceOneElectronBuilder active_space_one_electron_builder,
    ActiveSpaceTwoElectronBuilder active_space_two_electron_builder,
    FullDeterminantStructureHamiltonianOverlapBuilder structure_builder,
    xmvb::core::GeneralizedEigensolver generalized_eigensolver)
    : orbital_preparer_(std::move(orbital_preparer)),
      ao_effective_one_electron_builder_(std::move(ao_effective_one_electron_builder)),
      active_space_one_electron_builder_(std::move(active_space_one_electron_builder)),
      active_space_two_electron_builder_(std::move(active_space_two_electron_builder)),
      structure_builder_(std::move(structure_builder)),
      generalized_eigensolver_(std::move(generalized_eigensolver)) {}

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
  const auto forward_context = build_active_space_gradient_forward_context(
      input,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      active_space_two_electron_builder_,
      structure_builder_,
      generalized_eigensolver_);
  CppActiveSpaceGradientResult result;
  initialize_active_space_gradient_result(
      input,
      selected_state_indices,
      normalized_weights,
      nuclear_repulsion_energy,
      forward_context,
      &result);
  accumulate_active_space_gradient(
      input,
      selected_state_indices,
      normalized_weights,
      forward_context,
      &result);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();

  return result;
}

}  // namespace xmvb::vb
