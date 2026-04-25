#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
#include "vb/matrices/determinant_pair_storage_utils.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/cpp_active_space_gradient_result_utils.hpp"
#include "vb/scf/exact_ctx_memory_accounting.hpp"
#include "vb/scf/opposite_spin_matrix_backward.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb {

namespace {


struct StructurePairAdjoints {
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

struct StructurePairWeightTables {
  std::vector<double> hamiltonian_upper_weights;
  std::vector<double> overlap_upper_weights;
};

struct ActiveSpaceGradientForwardContext {
  TimedPreparedActiveSpaceContext timed_active_space_context;
  SameSpinPairCacheContext same_spin_pair_cache;
  StructureAccumulationResult structure_matrices;
  xmvb::core::GeneralizedEigenResult eigen_result;
  double structure_matrix_wall_time_seconds = 0.0;
  double eigensolver_wall_time_seconds = 0.0;
};

struct ThreadLocalActiveSpaceGradientBuffers {
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> active_one_electron_gradient;
  std::vector<double> packed_active_two_electron_gradient;
};

bool has_nonzero_structure_pair_adjoints(
    const StructurePairAdjoints& adjoints) {
  return adjoints.hamiltonian_weight != 0.0 ||
      adjoints.overlap_weight != 0.0;
}

SameSpinPhiResult compute_same_spin_phi_from_active_space_result(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const DeterminantOverlapResult& overlap_result,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (active_space_two_electron_result.representation ==
          ActiveSpaceTwoElectronRepresentation::PackedExact &&
      !active_space_two_electron_result.packed_active_two_electron_integrals.empty()) {
    return compute_same_spin_original_phi(
        occ_L,
        occ_R,
        h1e_act,
        n_active_orbitals,
        active_space_two_electron_result.packed_active_two_electron_integrals,
        overlap_result,
        inverse_overlap_gradient);
  }

  return compute_same_spin_original_phi(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      overlap_result,
      inverse_overlap_gradient);
}

SameSpinPhiResult evaluate_same_spin_phi_with_optional_cache(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (pair_evaluation.has_same_spin_phi_cache) {
    if (inverse_overlap_gradient != nullptr) {
      *inverse_overlap_gradient = pair_evaluation.same_spin_inverse_overlap_gradient;
    }
    return {
        pair_evaluation.same_spin_one_electron_phi,
        pair_evaluation.same_spin_total_phi,
    };
  }

  return compute_same_spin_phi_from_active_space_result(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      pair_evaluation.overlap_result,
      inverse_overlap_gradient);
}
void accumulate_one_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::Ref<Eigen::MatrixXd> active_one_electron_gradient) {
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      active_one_electron_gradient(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_same_spin_two_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double overlap_determinant,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double cofactor_scale = weight / overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
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
          (*packed_active_two_electron_gradient)[direct_index] +=
              second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              second_order_cofactor;
        }
      }
    }
  }
}

void accumulate_opposite_spin_two_electron_gradient_contribution(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const Eigen::MatrixXd& alpha_cofactor_1st,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const Eigen::MatrixXd& beta_cofactor_1st,
    const OppositeSpinPairCache* beta_pair_cache,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  if (alpha_pair_cache != nullptr && beta_pair_cache != nullptr &&
      has_opposite_spin_first_order_projection(*alpha_pair_cache) &&
      has_opposite_spin_first_order_projection(*beta_pair_cache) &&
      alpha_pair_cache->n_packed_active_pairs ==
          beta_pair_cache->n_packed_active_pairs) {
    const auto& alpha_projection = alpha_pair_cache->first_order_cofactor_projection;
    const auto& beta_projection = beta_pair_cache->first_order_cofactor_projection;
    for (std::size_t alpha_entry = 0;
         alpha_entry < alpha_projection.packed_pair_indices.size();
         ++alpha_entry) {
      const int alpha_packed_pair_index =
          alpha_projection.packed_pair_indices[alpha_entry];
      const double weighted_alpha_value =
          weight * alpha_projection.packed_pair_values[alpha_entry];
      for (std::size_t beta_entry = 0;
           beta_entry < beta_projection.packed_pair_indices.size();
           ++beta_entry) {
        const int beta_packed_pair_index =
            beta_projection.packed_pair_indices[beta_entry];
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                beta_packed_pair_index,
                alpha_packed_pair_index);
        (*packed_active_two_electron_gradient)[
            packed_pair_of_pairs_index] +=
            weighted_alpha_value * beta_projection.packed_pair_values[beta_entry];
      }
    }
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[alpha_left_column];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[alpha_right_row];
      const double weighted_alpha_cofactor =
          weight * alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[beta_left_column];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[beta_right_row];
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          (*packed_active_two_electron_gradient)[two_electron_index] +=
              weighted_alpha_cofactor * beta_cofactor_1st(beta_right_row, beta_left_column);
        }
      }
    }
  }
}

std::vector<double> normalize_state_average_weights_local(
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
    diagonal_sum += overlap_matrix[structure_index * n_structures +
                                   structure_index];
  }
  return diagonal_sum / static_cast<double>(n_structures);
}

std::size_t structure_upper_storage_index(
    int structure_row,
    int structure_column) {
  if (structure_row < 0 || structure_column < 0 ||
      structure_row > structure_column) {
    throw std::invalid_argument("structure upper-triangular index is out of range");
  }
  return structure_column * (structure_column + 1) / 2 +
      structure_row;
}

// Precompute the selected-state structure adjoint weights once after the
// generalized eigensolve so each determinant-pair reverse pass can reuse the
// same upper-triangular structure weights instead of rescanning the selected
// eigenvectors for every structure-term product.
StructurePairWeightTables build_structure_pair_weight_tables(
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights) {
  const std::size_t n_upper_entries =
      n_structures * (n_structures + 1) / 2;
  StructurePairWeightTables weights;
  weights.hamiltonian_upper_weights.assign(n_upper_entries, 0.0);
  weights.overlap_upper_weights.assign(n_upper_entries, 0.0);

  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index = selected_state_indices[selected_state_offset];
    const double state_weight = state_average_weights[selected_state_offset];
    const double state_energy = eigenvalues[state_index];
    const double* eigenvector_column =
        eigenvector_matrix.data() +
        state_index * n_structures;
    for (int structure_column = 0;
         structure_column < n_structures;
         ++structure_column) {
      const double coefficient_column = eigenvector_column[structure_column];
      const std::size_t column_offset =
          structure_column * (structure_column + 1) / 2;
      for (int structure_row = 0;
           structure_row <= structure_column;
           ++structure_row) {
        const double symmetry =
            structure_row == structure_column ? 1.0 : 2.0;
        const double weighted_product =
            symmetry * state_weight * eigenvector_column[structure_row] * coefficient_column;
        const std::size_t storage_index =
            column_offset + structure_row;
        weights.hamiltonian_upper_weights[storage_index] += weighted_product;
        weights.overlap_upper_weights[storage_index] -=
            state_energy * weighted_product;
      }
    }
  }

  return weights;
}

StructurePairAdjoints determinant_pair_structure_adjoints(
    const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
    const StructurePairWeightTables& structure_pair_weights) {
  StructurePairAdjoints adjoints;

  for (const auto& left_term : determinant_to_structures_left) {
    for (const auto& right_term : determinant_to_structures_right) {
      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }
      const double coefficient_product =
          left_term.coefficient * right_term.coefficient;
      const std::size_t storage_index = structure_upper_storage_index(
          left_term.structure_index,
          right_term.structure_index);
      adjoints.hamiltonian_weight +=
          coefficient_product *
          structure_pair_weights.hamiltonian_upper_weights[storage_index];
      adjoints.overlap_weight +=
          coefficient_product *
          structure_pair_weights.overlap_upper_weights[storage_index];
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
              eigenvalues[selected_state_indices[selected_state_offset]];
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
  const FullDeterminantPairEvaluator cache_builder =
      structure_builder.make_pair_evaluator();
  // The active-space objective and adjoint touch the same ordered alpha/beta
  // determinant-pair reuse pattern. Build that cache once here so the forward
  // structure matrices and the later backward sweep share the same payload.
  context.same_spin_pair_cache = build_same_spin_pair_cache_context(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      cache_builder,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result);
  if (context.same_spin_pair_cache.enabled()) {
    populate_same_spin_phi_cache(
        &context.same_spin_pair_cache,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        input.orbital_preparation_input.n_active_orbitals,
        prepared_active_space.active_space_two_electron_result);
  }
  auto stage_start_time = std::chrono::steady_clock::now();
  context.structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result,
      input.structure_data.n_structures,
      context.same_spin_pair_cache);
  context.structure_matrix_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.eigen_result = generalized_eigensolver.solve(
      context.structure_matrices.hamiltonian_matrix,
      context.structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  context.eigensolver_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  return context;
}

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const CppVbInput& input,
    TimedPreparedActiveSpaceContext timed_active_space_context,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::core::GeneralizedEigensolver& generalized_eigensolver) {
  ActiveSpaceGradientForwardContext context;
  context.timed_active_space_context = std::move(timed_active_space_context);

  const auto& prepared_active_space =
      context.timed_active_space_context.prepared_active_space;
  const FullDeterminantPairEvaluator cache_builder =
      structure_builder.make_pair_evaluator();
  // Reuse the accepted orbital/integral layer but rebuild the
  // determinant-topology-dependent same-spin cache and structure matrices for
  // the requested structure space.
  context.same_spin_pair_cache = build_same_spin_pair_cache_context(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      cache_builder,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result);
  if (context.same_spin_pair_cache.enabled()) {
    populate_same_spin_phi_cache(
        &context.same_spin_pair_cache,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        input.orbital_preparation_input.n_active_orbitals,
        prepared_active_space.active_space_two_electron_result);
  }

  auto stage_start_time = std::chrono::steady_clock::now();
  context.structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      prepared_active_space.orbital_result.active_orbital_overlap_matrix,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result,
      input.structure_data.n_structures,
      context.same_spin_pair_cache);
  context.structure_matrix_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.eigen_result = generalized_eigensolver.solve(
      context.structure_matrices.hamiltonian_matrix,
      context.structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  context.eigensolver_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  return context;
}

std::shared_ptr<CppActiveSpaceSecondOrderContext>
finalize_active_space_second_order_context(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    const CppActiveSpaceGradientResult& gradient_result,
    ActiveSpaceGradientForwardContext* forward_context) {
  if (forward_context == nullptr) {
    throw std::invalid_argument("forward_context must not be null");
  }

  auto context = std::make_shared<CppActiveSpaceSecondOrderContext>();
  // Move the heavy accepted-point payload out of the transient forward context
  // once the relaxed gradient has finished using it. This keeps the future
  // second-order cache alive without duplicating the large unique-spin tables.
  context->prepared_active_space =
      std::move(forward_context->timed_active_space_context.prepared_active_space);
  context->same_spin_pair_cache = std::move(forward_context->same_spin_pair_cache);
  context->structure_matrices = std::move(forward_context->structure_matrices);
  context->active_orbital_overlap_gradient =
      gradient_result.active_orbital_overlap_gradient;
  context->active_one_electron_gradient =
      gradient_result.active_one_electron_gradient;
  context->packed_active_two_electron_gradient =
      gradient_result.packed_active_two_electron_gradient;
  context->eigen_result = std::move(forward_context->eigen_result);
  context->selected_state_indices = selected_state_indices;
  context->normalized_state_weights = normalized_weights;
  context->n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  context->use_full_matrix_form_adjoint =
      context->same_spin_pair_cache.enabled();
  context->use_matrix_form_opposite_spin =
      context->same_spin_pair_cache.enabled();
  context->selected_state_energies = gather_selected_state_energies(
      context->eigen_result.eigenvalues,
      selected_state_indices);
  if (context->use_matrix_form_opposite_spin) {
    context->selected_state_matrices =
        build_selected_state_determinant_matrices_from_normalized_weights(
            input.structure_data,
            context->eigen_result.eigenvector_matrix,
            selected_state_indices,
            normalized_weights,
            context->same_spin_pair_cache);
  }
  // Accepted-point exact_ctx memory can become the dominant footprint on
  // medium systems long before the SCF iteration finishes. Keep the breakdown
  // behind an env-gated logger so production runs stay unchanged, while
  // compute-node diagnostics can attribute large resident sets to concrete
  // accepted-context payloads.
  ExactCtxMemoryBreakdown memory_breakdown;
  append_exact_ctx_memory_breakdown(
      "accepted_point_context",
      *context,
      &memory_breakdown);
  maybe_log_exact_ctx_memory_breakdown(
      "accepted_point_context",
      memory_breakdown);
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
  const auto& structure_matrices = forward_context.structure_matrices;
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
        eigen_result.eigenvalues[
            selected_state_indices[selected_state_offset]] +
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
      packed_active_two_electron_integral_count(
          input.orbital_preparation_input.n_active_orbitals),
      0.0);

  populate_scf_result(
      input,
      selected_state_indices,
      normalized_weights,
      nuclear_repulsion_energy,
      forward_context,
      &result->scf_result);
}

void accumulate_additive_vector(
    const std::vector<double>& partial,
    std::vector<double>* total) {
  for (std::size_t index = 0; index < total->size(); ++index) {
    (*total)[index] += partial[index];
  }
}

FullDeterminantPairEvaluation evaluate_active_space_determinant_pair(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const CppVbInput& input,
    const std::vector<double>& active_orbital_overlap_matrix,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals) {
  // Large determinant expansions still avoid an O(n_det^2) full-pair cache,
  // but the gradient path can reuse the same ordered alpha/beta same-spin
  // kernels as the forward build when the determinant space has strong
  // Cartesian-product structure.
  return evaluate_full_determinant_pair_with_optional_same_spin_cache(
      same_spin_pair_cache,
      pair_evaluator,
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      determinant_index_left,
      determinant_index_right,
      active_orbital_overlap_matrix,
      active_space_one_electron_result.h1e_act,
      n_active_orbitals,
      active_space_two_electron_result);
}

void accumulate_active_space_gradient_pair_with_adjoints(
    const StructurePairAdjoints& pair_adjoints,
    const CppVbInput& input,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    Eigen::Ref<Eigen::MatrixXd> active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (pair_adjoints.hamiltonian_weight == 0.0 &&
      pair_adjoints.overlap_weight == 0.0) {
    return;
  }

  const auto& alpha_occ_L =
      input.structure_data.alpha_det[
          determinant_index_left];
  const auto& alpha_occ_R =
      input.structure_data.alpha_det[
          determinant_index_right];
  const auto& beta_occ_L =
      input.structure_data.beta_det[
          determinant_index_left];
  const auto& beta_occ_R =
      input.structure_data.beta_det[
          determinant_index_right];
  const auto& alpha_result = determinant_pair_evaluation.alpha.overlap_result;
  const auto& beta_result = determinant_pair_evaluation.beta.overlap_result;
  const Eigen::MatrixXd alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result);
  const Eigen::MatrixXd beta_cofactor_1st =
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
  Eigen::MatrixXd alpha_same_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
  Eigen::MatrixXd beta_same_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(n_beta_electrons, n_beta_electrons);
  Eigen::MatrixXd alpha_opposite_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
  Eigen::MatrixXd beta_opposite_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(n_beta_electrons, n_beta_electrons);

  double opposite_spin_phi = 0.0;
  const SameSpinPhiResult alpha_phi_result = evaluate_same_spin_phi_with_optional_cache(
      alpha_occ_L,
      alpha_occ_R,
      active_space_one_electron_result.h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      determinant_pair_evaluation.alpha,
      &alpha_same_spin_inverse_overlap_gradient);
  const SameSpinPhiResult beta_phi_result = evaluate_same_spin_phi_with_optional_cache(
      beta_occ_L,
      beta_occ_R,
      active_space_one_electron_result.h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      determinant_pair_evaluation.beta,
      &beta_same_spin_inverse_overlap_gradient);
  if (!skip_opposite_spin && n_alpha_electrons > 0 && n_beta_electrons > 0) {
    opposite_spin_phi = compute_opposite_spin_original_phi(
        alpha_occ_L,
        alpha_occ_R,
        alpha_result,
        &determinant_pair_evaluation.alpha.opposite_spin_pair_cache,
        beta_occ_L,
        beta_occ_R,
        beta_result,
        &determinant_pair_evaluation.beta.opposite_spin_pair_cache,
        active_space_two_electron_result,
        &alpha_opposite_spin_inverse_overlap_gradient,
        &beta_opposite_spin_inverse_overlap_gradient);
  }
  const Eigen::MatrixXd alpha_inverse_overlap_gradient =
      alpha_same_spin_inverse_overlap_gradient +
      alpha_opposite_spin_inverse_overlap_gradient;
  const Eigen::MatrixXd beta_inverse_overlap_gradient =
      beta_same_spin_inverse_overlap_gradient +
      beta_opposite_spin_inverse_overlap_gradient;
  const double alpha_phi = alpha_phi_result.total_phi;
  const double beta_phi = beta_phi_result.total_phi;
  const double phi_sum = alpha_phi + beta_phi + opposite_spin_phi;
  const double alpha_determinant_weight =
      pair_adjoints.overlap_weight * beta_result.overlap_determinant +
      pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
          phi_sum;
  const double beta_determinant_weight =
      pair_adjoints.overlap_weight * alpha_result.overlap_determinant +
      pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
          phi_sum;
  accumulate_spin_overlap_gradient(
      alpha_occ_L,
      alpha_occ_R,
      alpha_result,
      alpha_determinant_weight,
      pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
          alpha_inverse_overlap_gradient,
      n_active_orbitals,
      active_orbital_overlap_gradient);
  accumulate_spin_overlap_gradient(
      beta_occ_L,
      beta_occ_R,
      beta_result,
      beta_determinant_weight,
      pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
          beta_inverse_overlap_gradient,
      n_active_orbitals,
      active_orbital_overlap_gradient);

  if (alpha_result.nullity == 0) {
    accumulate_same_spin_two_electron_gradient_contribution(
        alpha_occ_L,
        alpha_occ_R,
        alpha_cofactor_1st,
        alpha_result.overlap_determinant,
        alpha_weight,
        packed_active_two_electron_gradient);
  }

  if (beta_result.nullity == 0) {
    accumulate_same_spin_two_electron_gradient_contribution(
        beta_occ_L,
        beta_occ_R,
        beta_cofactor_1st,
        beta_result.overlap_determinant,
        beta_weight,
        packed_active_two_electron_gradient);
  }

  if (!skip_opposite_spin &&
      alpha_result.nullity < 2 && beta_result.nullity < 2 &&
      n_alpha_electrons > 0 && n_beta_electrons > 0) {
    accumulate_opposite_spin_two_electron_gradient_contribution(
        alpha_occ_L,
        alpha_occ_R,
        alpha_cofactor_1st,
        &determinant_pair_evaluation.alpha.opposite_spin_pair_cache,
        beta_occ_L,
        beta_occ_R,
        beta_cofactor_1st,
        &determinant_pair_evaluation.beta.opposite_spin_pair_cache,
        pair_adjoints.hamiltonian_weight,
        packed_active_two_electron_gradient);
  }
}

void accumulate_active_space_gradient_pair(
    const CppVbInput& input,
    const StructurePairWeightTables& structure_pair_weights,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    Eigen::Ref<Eigen::MatrixXd> active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  const auto pair_adjoints = determinant_pair_structure_adjoints(
      input.structure_data.determinant_to_structure_terms[
          determinant_index_left],
      input.structure_data.determinant_to_structure_terms[
          determinant_index_right],
      structure_pair_weights);
  accumulate_active_space_gradient_pair_with_adjoints(
      pair_adjoints,
      input,
      active_space_one_electron_result,
      active_space_two_electron_result,
      determinant_pair_evaluation,
      determinant_index_left,
      determinant_index_right,
      n_active_orbitals,
      skip_opposite_spin,
      active_one_electron_gradient,
      active_orbital_overlap_gradient,
      packed_active_two_electron_gradient);
}

void accumulate_active_space_gradient_unordered_pair(
    const CppVbInput& input,
    const StructurePairWeightTables& structure_pair_weights,
    const std::vector<double>& active_orbital_overlap_matrix,
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    Eigen::Ref<Eigen::MatrixXd> active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (determinant_index_left < determinant_index_right) {
    throw std::invalid_argument(
        "unordered determinant pair expects determinant_index_left >= determinant_index_right");
  }
  if (determinant_index_left == determinant_index_right) {
    const auto determinant_pair_evaluation =
        evaluate_active_space_determinant_pair(
            same_spin_pair_cache,
            pair_evaluator,
            input,
            active_orbital_overlap_matrix,
            active_space_one_electron_result,
            active_space_two_electron_result,
            determinant_index_left,
            determinant_index_right,
            n_active_orbitals);
    accumulate_active_space_gradient_pair(
        input,
        structure_pair_weights,
        active_space_one_electron_result,
        active_space_two_electron_result,
        determinant_pair_evaluation,
        determinant_index_left,
        determinant_index_right,
        n_active_orbitals,
        skip_opposite_spin,
        active_one_electron_gradient,
        active_orbital_overlap_gradient,
        packed_active_two_electron_gradient);
    return;
  }

  const auto direct_pair_adjoints = determinant_pair_structure_adjoints(
      input.structure_data.determinant_to_structure_terms[
          determinant_index_left],
      input.structure_data.determinant_to_structure_terms[
          determinant_index_right],
      structure_pair_weights);
  const auto swapped_pair_adjoints = determinant_pair_structure_adjoints(
      input.structure_data.determinant_to_structure_terms[
          determinant_index_right],
      input.structure_data.determinant_to_structure_terms[
          determinant_index_left],
      structure_pair_weights);
  const StructurePairAdjoints combined_pair_adjoints = {
      direct_pair_adjoints.hamiltonian_weight +
          swapped_pair_adjoints.hamiltonian_weight,
      direct_pair_adjoints.overlap_weight +
          swapped_pair_adjoints.overlap_weight,
  };
  if (!has_nonzero_structure_pair_adjoints(combined_pair_adjoints)) {
    return;
  }

  const auto determinant_pair_evaluation =
      evaluate_active_space_determinant_pair(
          same_spin_pair_cache,
          pair_evaluator,
          input,
          active_orbital_overlap_matrix,
          active_space_one_electron_result,
          active_space_two_electron_result,
          determinant_index_left,
          determinant_index_right,
          n_active_orbitals);
  // The forward structure builder evaluates only the canonical unordered
  // determinant pair `(left >= right)` and folds both structure-term
  // orientations into that single pair value. The backward therefore must
  // combine direct and swapped structure adjoints onto the same canonical pair
  // instead of sending the swapped term through the transposed determinant pair.
  accumulate_active_space_gradient_pair_with_adjoints(
      combined_pair_adjoints,
      input,
      active_space_one_electron_result,
      active_space_two_electron_result,
      determinant_pair_evaluation,
      determinant_index_left,
      determinant_index_right,
      n_active_orbitals,
      skip_opposite_spin,
      active_one_electron_gradient,
      active_orbital_overlap_gradient,
      packed_active_two_electron_gradient);
}

void accumulate_active_space_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const ActiveSpaceGradientForwardContext& forward_context,
    CppActiveSpaceGradientResult* result) {
  const auto& prepared_active_space =
      forward_context.timed_active_space_context.prepared_active_space;
  const auto& active_orbital_overlap_matrix =
      prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  const auto& active_space_one_electron_result =
      prepared_active_space.active_space_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  const StructurePairWeightTables structure_pair_weights =
      build_structure_pair_weight_tables(
          forward_context.eigen_result.eigenvector_matrix,
          forward_context.eigen_result.eigenvalues,
          input.structure_data.n_structures,
          selected_state_indices,
          normalized_weights);
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

  const auto stage_start_time = std::chrono::steady_clock::now();
  int n_threads = 1;
  n_threads = xmvb::effective_openmp_thread_count();
  const std::size_t n_determinant_pairs =
      unordered_determinant_pair_count(n_determinants);
  if (n_threads > n_determinants) {
    n_threads = n_determinants;
  }
  if (n_threads < 1) {
    n_threads = 1;
  }

  const auto& same_spin_pair_cache = forward_context.same_spin_pair_cache;
  const bool use_full_matrix_form_adjoint = same_spin_pair_cache.enabled();
  const bool use_matrix_form_opposite_spin = same_spin_pair_cache.enabled();
  SelectedStateDeterminantMatrices selected_state_matrices;
  OppositeSpinMatrixBackwardContribution opposite_spin_contribution;
  SameSpinMatrixBackwardContribution same_spin_contribution;
  if (use_matrix_form_opposite_spin) {
    selected_state_matrices = build_selected_state_determinant_matrices_from_normalized_weights(
        input.structure_data,
        forward_context.eigen_result.eigenvector_matrix,
        selected_state_indices,
        normalized_weights,
        same_spin_pair_cache);
  }

  if (use_full_matrix_form_adjoint) {
    const std::vector<double> selected_state_energies =
        gather_selected_state_energies(
            forward_context.eigen_result.eigenvalues,
            selected_state_indices);
    same_spin_contribution = build_same_spin_matrix_backward_contribution(
        same_spin_pair_cache,
        selected_state_matrices,
        selected_state_energies,
        n_active_orbitals);
    opposite_spin_contribution = build_opposite_spin_matrix_backward_contribution(
        same_spin_pair_cache,
        selected_state_matrices,
        n_active_orbitals);
    accumulate_additive_vector(
        same_spin_contribution.active_orbital_overlap_gradient,
        &result->active_orbital_overlap_gradient);
    accumulate_additive_vector(
        same_spin_contribution.active_one_electron_gradient,
        &result->active_one_electron_gradient);
    accumulate_additive_vector(
        same_spin_contribution.packed_active_two_electron_gradient,
        &result->packed_active_two_electron_gradient);
    accumulate_additive_vector(
        opposite_spin_contribution.active_orbital_overlap_gradient,
        &result->active_orbital_overlap_gradient);
    accumulate_additive_vector(
        opposite_spin_contribution.packed_active_two_electron_gradient,
        &result->packed_active_two_electron_gradient);
    result->adjoint_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time)
            .count();
    return;
  }

  if (n_threads == 1) {
    const FullDeterminantPairEvaluator pair_evaluator =
        structure_builder.make_pair_evaluator();
    Eigen::Map<Eigen::MatrixXd> active_one_electron_gradient(
        result->active_one_electron_gradient.data(),
        n_active_orbitals,
        n_active_orbitals);
    for (int determinant_index_left = 0;
         determinant_index_left < n_determinants;
         ++determinant_index_left) {
      for (int determinant_index_right = 0;
           determinant_index_right <= determinant_index_left;
           ++determinant_index_right) {
        accumulate_active_space_gradient_unordered_pair(
            input,
            structure_pair_weights,
            active_orbital_overlap_matrix,
            &same_spin_pair_cache,
            pair_evaluator,
            active_space_one_electron_result,
            active_space_two_electron_result,
            determinant_index_left,
            determinant_index_right,
            n_active_orbitals,
            use_matrix_form_opposite_spin,
            active_one_electron_gradient,
            &result->active_orbital_overlap_gradient,
            &result->packed_active_two_electron_gradient);
      }
    }
  } else {
    std::vector<ThreadLocalActiveSpaceGradientBuffers> partial_gradients;
    partial_gradients.reserve(n_threads);
    for (int thread_index = 0; thread_index < n_threads; ++thread_index) {
      ThreadLocalActiveSpaceGradientBuffers buffers;
      buffers.active_orbital_overlap_gradient.assign(
          result->active_orbital_overlap_gradient.size(),
          0.0);
      buffers.active_one_electron_gradient.assign(
          result->active_one_electron_gradient.size(),
          0.0);
      buffers.packed_active_two_electron_gradient.assign(
          result->packed_active_two_electron_gradient.size(),
          0.0);
      partial_gradients.push_back(std::move(buffers));
    }

    std::atomic<bool> failed(false);
    std::exception_ptr first_exception;
    const std::size_t pair_chunk_size =
        unordered_determinant_pair_parallel_chunk_size(
            n_determinant_pairs,
            n_threads);
    const std::size_t pair_chunk_stride =
        pair_chunk_size * static_cast<std::size_t>(n_threads);

#pragma omp parallel num_threads(n_threads)
    {
      const FullDeterminantPairEvaluator pair_evaluator =
          structure_builder.make_pair_evaluator();
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      auto& local_gradients =
          partial_gradients[thread_index];
      Eigen::Map<Eigen::MatrixXd> local_active_one_electron_gradient(
          local_gradients.active_one_electron_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      // Each pair contributes to thread-local gradient buffers.  Cyclic chunk
      // ownership balances sparse/zero-adjoint determinant rows without
      // introducing non-deterministic dynamic scheduling into the reduction.
      for (std::size_t pair_begin =
               static_cast<std::size_t>(thread_index) * pair_chunk_size;
           pair_begin < n_determinant_pairs;
           pair_begin += pair_chunk_stride) {
        const std::size_t pair_end =
            std::min(n_determinant_pairs, pair_begin + pair_chunk_size);
        auto determinant_pair =
            determinant_pair_from_storage_index(pair_begin);
        for (std::size_t pair_storage_index = pair_begin;
             pair_storage_index < pair_end;
             ++pair_storage_index) {
          if (failed.load(std::memory_order_relaxed)) {
            break;
          }

          try {
            accumulate_active_space_gradient_unordered_pair(
                input,
                structure_pair_weights,
                active_orbital_overlap_matrix,
                &same_spin_pair_cache,
                pair_evaluator,
                active_space_one_electron_result,
                active_space_two_electron_result,
                determinant_pair.left,
                determinant_pair.right,
                n_active_orbitals,
                use_matrix_form_opposite_spin,
                local_active_one_electron_gradient,
                &local_gradients.active_orbital_overlap_gradient,
                &local_gradients.packed_active_two_electron_gradient);
          } catch (...) {
#pragma omp critical
            {
              if (!failed.load(std::memory_order_relaxed)) {
                first_exception = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
              }
            }
          }
          advance_unordered_determinant_pair(&determinant_pair);
        }
      }
    }

    if (first_exception) {
      std::rethrow_exception(first_exception);
    }

    for (const auto& local_gradients : partial_gradients) {
      accumulate_additive_vector(
          local_gradients.active_orbital_overlap_gradient,
          &result->active_orbital_overlap_gradient);
      accumulate_additive_vector(
          local_gradients.active_one_electron_gradient,
          &result->active_one_electron_gradient);
      accumulate_additive_vector(
          local_gradients.packed_active_two_electron_gradient,
          &result->packed_active_two_electron_gradient);
    }
  }

  if (use_matrix_form_opposite_spin) {
    opposite_spin_contribution = build_opposite_spin_matrix_backward_contribution(
        same_spin_pair_cache,
        selected_state_matrices,
        n_active_orbitals);
    accumulate_additive_vector(
        opposite_spin_contribution.active_orbital_overlap_gradient,
        &result->active_orbital_overlap_gradient);
    accumulate_additive_vector(
        opposite_spin_contribution.packed_active_two_electron_gradient,
        &result->packed_active_two_electron_gradient);
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
      normalize_state_average_weights_local(state_average_weights);
  auto forward_context = build_active_space_gradient_forward_context(
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
      structure_builder_,
      forward_context,
      &result);
  result.second_order_context = finalize_active_space_second_order_context(
      input,
      selected_state_indices,
      normalized_weights,
      result,
      &forward_context);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();

  return result;
}

CppActiveSpaceGradientResult CppActiveSpaceGradientEvaluator::evaluate(
    const CppVbInput& input,
    TimedPreparedActiveSpaceContext timed_prepared_active_space_context,
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
      normalize_state_average_weights_local(state_average_weights);

  auto forward_context = build_active_space_gradient_forward_context(
      input,
      std::move(timed_prepared_active_space_context),
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
      structure_builder_,
      forward_context,
      &result);
  result.second_order_context = finalize_active_space_second_order_context(
      input,
      selected_state_indices,
      normalized_weights,
      result,
      &forward_context);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();

  return result;
}

CppActiveSpaceGradientResult
CppActiveSpaceGradientEvaluator::evaluate_with_fixed_active_space_adjoint(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    double nuclear_repulsion_energy) const {
  const auto total_start_time = std::chrono::steady_clock::now();
  if (input.structure_data.n_structures <= 0) {
    throw std::invalid_argument("input.structure_data.n_structures must be positive");
  }

  auto timed_active_space_context =
      prepare_timed_active_space_context(
          input,
          orbital_preparer_,
          ao_effective_one_electron_builder_,
          active_space_one_electron_builder_,
          active_space_two_electron_builder_);

  CppActiveSpaceGradientResult result;
  initialize_active_space_gradient_probe_result(
      input,
      std::move(timed_active_space_context),
      &result);
  if (result.active_orbital_overlap_gradient.size() !=
          accepted_point_context.active_orbital_overlap_gradient.size() ||
      result.active_one_electron_gradient.size() !=
          accepted_point_context.active_one_electron_gradient.size() ||
      result.packed_active_two_electron_gradient.size() !=
          accepted_point_context.packed_active_two_electron_gradient.size()) {
    throw std::invalid_argument(
        "accepted-point active-space adjoint dimensions do not match the trial orbital space");
  }

  // The accepted-point active-space adjoint is held fixed while the lower
  // orbital-preparation and integral layers are rebuilt at the probe point.
  result.active_orbital_overlap_gradient =
      accepted_point_context.active_orbital_overlap_gradient;
  result.active_one_electron_gradient =
      accepted_point_context.active_one_electron_gradient;
  result.packed_active_two_electron_gradient =
      accepted_point_context.packed_active_two_electron_gradient;
  result.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  (void)nuclear_repulsion_energy;
  return result;
}

TimedPreparedActiveSpaceContext
CppActiveSpaceGradientEvaluator::prepare_timed_active_space_context_for_probe(
    const CppVbInput& input) const {
  return prepare_timed_active_space_context(
      input,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      active_space_two_electron_builder_);
}

}  // namespace xmvb::vb
