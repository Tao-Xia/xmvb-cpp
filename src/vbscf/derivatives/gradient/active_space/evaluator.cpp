#include "vbscf/derivatives/gradient/active_space/evaluator.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/preparation/space.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/gradient/active_space/helpers.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {

namespace {

struct ActiveSpaceGradientForwardContext {
  TimedPreparedActiveSpaceContext timed_active_space_context;
  SameSpinPairCacheContext same_spin_pair_cache;
  std::optional<StructureAction> structure_action;
  StructureAccumulationResult structure_matrices;
  xmvb::core::GeneralizedEigenResult eigen_result;
  Eigen::VectorXd structure_overlap_diagonal;
  Eigen::MatrixXd overlap_eigenvectors;
  std::optional<DavidsonDiagnostics> davidson_diagnostics;
  Eigen::MatrixXd selected_state_eigenvectors;
  std::vector<double> selected_state_energies;
  double average_structure_overlap = 0.0;
  double structure_matrix_wall_time_seconds = 0.0;
  double eigensolver_wall_time_seconds = 0.0;
};

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

int required_root_count(const std::vector<int>& selected_state_indices) {
  return *std::max_element(
             selected_state_indices.begin(),
             selected_state_indices.end()) +
      1;
}

void select_structure_states(
    int n_structures,
    const std::vector<int>& selected_state_indices,
    ActiveSpaceGradientForwardContext* context) {
  const int n_roots = static_cast<int>(context->eigen_result.eigenvalues.size());
  if (n_roots < required_root_count(selected_state_indices) ||
      context->eigen_result.eigenvector_matrix.size() !=
          static_cast<std::size_t>(n_structures) * n_roots) {
    throw std::runtime_error(
        "structure eigensolver returned inconsistent selected-root dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> roots(
      context->eigen_result.eigenvector_matrix.data(),
      n_structures,
      n_roots);
  context->selected_state_eigenvectors.resize(
      n_structures,
      static_cast<int>(selected_state_indices.size()));
  context->selected_state_energies.resize(selected_state_indices.size());
  for (std::size_t state = 0; state < selected_state_indices.size(); ++state) {
    const int root = selected_state_indices[state];
    context->selected_state_eigenvectors.col(static_cast<int>(state)) =
        roots.col(root);
    context->selected_state_energies[state] =
        context->eigen_result.eigenvalues[static_cast<std::size_t>(root)];
  }
}

void solve_structure_problem(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::core::GeneralizedEigensolver& generalized_eigensolver,
    ActiveSpaceGradientForwardContext* context) {
  const auto& prepared = context->timed_active_space_context.prepared_active_space;
  const int n_structures = input.structure_data.n_structures;
  const int n_roots = required_root_count(selected_state_indices);
  auto stage_start_time = std::chrono::steady_clock::now();

  if (structure_eigensolver == StructureEigensolver::Dense) {
    context->structure_matrices = structure_builder.build(
        input.structure_data.alpha_det,
        input.structure_data.beta_det,
        input.structure_data.determinant_to_structure_terms,
        prepared.orbital_result.active_orbital_overlap_matrix,
        prepared.active_space_one_electron_result.h1e_act,
        input.orbital_preparation_input.n_active_orbitals,
        prepared.active_space_two_electron_result,
        n_structures,
        context->same_spin_pair_cache);
    context->structure_matrix_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stage_start_time)
            .count();
    context->average_structure_overlap = compute_average_structure_overlap(
        context->structure_matrices.overlap_matrix,
        n_structures);
    context->structure_overlap_diagonal.resize(n_structures);
    for (int structure = 0; structure < n_structures; ++structure) {
      context->structure_overlap_diagonal[structure] =
          context->structure_matrices.overlap_matrix[
              static_cast<std::size_t>(structure) * n_structures + structure];
    }
    stage_start_time = std::chrono::steady_clock::now();
    context->eigen_result = generalized_eigensolver.solve_dense(
        context->structure_matrices.hamiltonian_matrix,
        context->structure_matrices.overlap_matrix,
        n_structures);
  } else {
    context->structure_action.emplace(
        input.structure_data.determinant_to_structure_terms,
        n_structures,
        context->same_spin_pair_cache,
        prepared.active_space_two_electron_result,
        input.orbital_preparation_input.n_active_orbitals);
    context->structure_matrix_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stage_start_time)
            .count();
    const StructureAction& structure_action = *context->structure_action;
    context->average_structure_overlap =
        structure_action.diagonal().overlap.mean();
    context->structure_overlap_diagonal =
        structure_action.diagonal().overlap;
    const xmvb::core::GeneralizedEigenAction action =
        [&structure_action](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
          StructureActionResult images = structure_action.apply(vectors);
          return xmvb::core::GeneralizedEigenActionResult{
              std::move(images.hamiltonian),
              std::move(images.overlap)};
        };
    const xmvb::core::DavidsonOptions options =
        xmvb::core::make_davidson_options(n_structures, n_roots);
    stage_start_time = std::chrono::steady_clock::now();
    const bool can_recycle =
        initial_eigenvectors.rows() == n_structures &&
        initial_eigenvectors.cols() >= n_roots &&
        initial_eigenvectors.allFinite();
    xmvb::core::DavidsonResult davidson = can_recycle
        ? generalized_eigensolver.solve_davidson(
              action,
              structure_action.diagonal().hamiltonian,
              structure_action.diagonal().overlap,
              initial_eigenvectors.leftCols(n_roots),
              options)
        : generalized_eigensolver.solve_davidson(
              action,
              structure_action.diagonal().hamiltonian,
              structure_action.diagonal().overlap,
              options);
    context->davidson_diagnostics = DavidsonDiagnostics{
        davidson.iterations,
        davidson.block_actions,
        davidson.peak_subspace_dimension,
        *std::max_element(
            davidson.relative_residual_norms.begin(),
            davidson.relative_residual_norms.end())};
    context->eigen_result = std::move(davidson.eigenpairs);
    context->overlap_eigenvectors =
        std::move(davidson.overlap_eigenvectors);
  }
  context->eigensolver_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - stage_start_time)
          .count();
  select_structure_states(n_structures, selected_state_indices, context);
}

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
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
  const DeterminantPairEvaluator cache_builder =
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
  solve_structure_problem(
      input,
      selected_state_indices,
      structure_eigensolver,
      initial_eigenvectors,
      structure_builder,
      generalized_eigensolver,
      &context);
  return context;
}

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const VbScfInput& input,
    TimedPreparedActiveSpaceContext timed_active_space_context,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::core::GeneralizedEigensolver& generalized_eigensolver) {
  ActiveSpaceGradientForwardContext context;
  context.timed_active_space_context = std::move(timed_active_space_context);

  const auto& prepared_active_space =
      context.timed_active_space_context.prepared_active_space;
  const DeterminantPairEvaluator cache_builder =
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

  solve_structure_problem(
      input,
      selected_state_indices,
      structure_eigensolver,
      initial_eigenvectors,
      structure_builder,
      generalized_eigensolver,
      &context);
  return context;
}

std::shared_ptr<AcceptedPointContext>
finalize_active_space_second_order_context(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    const ActiveSpaceGradientResult& gradient_result,
    ActiveSpaceGradientForwardContext* forward_context) {
  if (forward_context == nullptr) {
    throw std::invalid_argument("forward_context must not be null");
  }

  auto context = std::make_shared<AcceptedPointContext>();
  // Move the heavy accepted-point payload out of the transient forward context
  // once the relaxed gradient has finished using it. This keeps the future
  // second-order cache alive without duplicating the large unique-spin tables.
  context->prepared_active_space =
      std::move(forward_context->timed_active_space_context.prepared_active_space);
  context->same_spin_pair_cache = std::move(forward_context->same_spin_pair_cache);
  if (forward_context->structure_action.has_value()) {
    context->structure_action = std::move(forward_context->structure_action);
  } else {
    context->structure_action.emplace(
        input.structure_data.determinant_to_structure_terms,
        input.structure_data.n_structures,
        context->same_spin_pair_cache,
        context->prepared_active_space.active_space_two_electron_result,
        input.orbital_preparation_input.n_active_orbitals);
  }
  context->active_orbital_overlap_gradient =
      gradient_result.active_orbital_overlap_gradient;
  context->active_one_electron_gradient =
      gradient_result.active_one_electron_gradient;
  context->packed_active_two_electron_gradient =
      gradient_result.packed_active_two_electron_gradient;
  context->selected_state_indices = selected_state_indices;
  context->normalized_state_weights = normalized_weights;
  context->n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  context->n_structures = input.structure_data.n_structures;
  context->use_full_matrix_form_adjoint =
      context->same_spin_pair_cache.enabled();
  context->use_matrix_form_opposite_spin =
      context->same_spin_pair_cache.enabled();
  context->selected_state_energies =
      std::move(forward_context->selected_state_energies);
  context->selected_state_eigenvectors =
      std::move(forward_context->selected_state_eigenvectors);
  const int n_roots =
      static_cast<int>(forward_context->eigen_result.eigenvalues.size());
  const Eigen::Map<const Eigen::MatrixXd> accepted_roots(
      forward_context->eigen_result.eigenvector_matrix.data(),
      context->n_structures,
      n_roots);
  context->root_eigenvectors = accepted_roots;
  if (context->use_matrix_form_opposite_spin) {
    context->selected_state_matrices =
        build_selected_state_determinant_matrices_from_selected_columns(
            input.structure_data,
            context->selected_state_eigenvectors,
            selected_state_indices,
            normalized_weights,
            context->same_spin_pair_cache);
  }
  return context;
}

void populate_scf_result(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double nuclear_repulsion_energy,
    const ActiveSpaceGradientForwardContext& forward_context,
    VbScfResult* scf_result) {
  const auto& prepared_active_space =
      forward_context.timed_active_space_context.prepared_active_space;
  const auto& structure_matrices = forward_context.structure_matrices;
  const auto& eigen_result = forward_context.eigen_result;

  scf_result->n_structures = input.structure_data.n_structures;
  scf_result->nuclear_repulsion_energy = nuclear_repulsion_energy;
  scf_result->selected_state_indices = selected_state_indices;
  scf_result->state_average_weights = normalized_weights;
  scf_result->structure_matrices = structure_matrices;
  scf_result->average_structure_overlap =
      forward_context.average_structure_overlap;
  scf_result->davidson_diagnostics =
      forward_context.davidson_diagnostics;
  scf_result->electronic_state_energies = eigen_result.eigenvalues;
  scf_result->eigenvector_matrix = eigen_result.eigenvector_matrix;
  scf_result->structure_overlap_diagonal.assign(
      forward_context.structure_overlap_diagonal.data(),
      forward_context.structure_overlap_diagonal.data() +
          forward_context.structure_overlap_diagonal.size());
  scf_result->overlap_eigenvector_matrix.clear();
  if (forward_context.overlap_eigenvectors.size() != 0) {
    scf_result->overlap_eigenvector_matrix.assign(
        forward_context.overlap_eigenvectors.data(),
        forward_context.overlap_eigenvectors.data() +
            forward_context.overlap_eigenvectors.size());
  }
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
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double nuclear_repulsion_energy,
    const ActiveSpaceGradientForwardContext& forward_context,
    ActiveSpaceGradientResult* result) {
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


void accumulate_active_space_gradient(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    const ActiveSpaceGradientForwardContext& forward_context,
    ActiveSpaceGradientResult* result) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const auto stage_start_time = std::chrono::steady_clock::now();
  const auto& same_spin_pair_cache = forward_context.same_spin_pair_cache;
  if (!same_spin_pair_cache.enabled()) {
    throw std::runtime_error(
        "active-space gradient requires the matrix-form same-spin cache");
  }

  const SelectedStateDeterminantMatrices selected_state_matrices =
      build_selected_state_determinant_matrices_from_selected_columns(
          input.structure_data,
          forward_context.selected_state_eigenvectors,
          selected_state_indices,
          normalized_weights,
          same_spin_pair_cache);
  const std::vector<double>& selected_state_energies =
      forward_context.selected_state_energies;
  const SameSpinMatrixBackwardContribution same_spin_contribution =
      build_same_spin_matrix_backward_contribution(
          same_spin_pair_cache,
          selected_state_matrices,
          selected_state_energies,
          n_active_orbitals);
  const OppositeSpinMatrixBackwardContribution opposite_spin_contribution =
      build_opposite_spin_matrix_backward_contribution(
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
}

}  // namespace

ActiveSpaceGradientEvaluator::ActiveSpaceGradientEvaluator()
    : orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      active_space_one_electron_builder_(),
      active_space_two_electron_builder_(),
      structure_builder_(),
      generalized_eigensolver_() {}

ActiveSpaceGradientEvaluator::ActiveSpaceGradientEvaluator(
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

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
    double nuclear_repulsion_energy) const {
  return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  const Eigen::MatrixXd no_initial_eigenvectors;
  return evaluate(
      input,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      StructureEigensolver::Dense,
      no_initial_eigenvectors);
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const {
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
      selected_state_indices,
      structure_eigensolver,
      initial_eigenvectors,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      active_space_two_electron_builder_,
      structure_builder_,
      generalized_eigensolver_);
  ActiveSpaceGradientResult result;
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

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
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

  const Eigen::MatrixXd no_initial_eigenvectors;
  auto forward_context = build_active_space_gradient_forward_context(
      input,
      std::move(timed_prepared_active_space_context),
      selected_state_indices,
      StructureEigensolver::Dense,
      no_initial_eigenvectors,
      structure_builder_,
      generalized_eigensolver_);
  ActiveSpaceGradientResult result;
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

ActiveSpaceGradientResult
ActiveSpaceGradientEvaluator::evaluate_with_fixed_active_space_adjoint(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
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

  ActiveSpaceGradientResult result;
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
ActiveSpaceGradientEvaluator::prepare_timed_active_space_context_for_probe(
    const VbScfInput& input) const {
  return prepare_timed_active_space_context(
      input,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      active_space_two_electron_builder_);
}

}  // namespace xmvb::vb
