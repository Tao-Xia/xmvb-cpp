#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_gradient.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_one_electron_transform_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_two_electron_transform_backward.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

std::vector<int> build_full_structure_index_range(int n_structures) {
  if (n_structures < 0) {
    throw std::invalid_argument("n_structures must be non-negative");
  }

  std::vector<int> indices(xmvb::to_size(n_structures));
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    indices[xmvb::to_size(structure_index)] = structure_index;
  }
  return indices;
}

std::vector<double> dense_matrix_to_vector(
    const Eigen::MatrixXd& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

double compute_average_structure_overlap(
    const Eigen::MatrixXd& overlap_matrix) {
  if (overlap_matrix.rows() != overlap_matrix.cols()) {
    throw std::invalid_argument("overlap_matrix must be square");
  }
  if (overlap_matrix.rows() <= 0) {
    throw std::invalid_argument("overlap_matrix must be non-empty");
  }
  double diagonal_sum = 0.0;
  for (int structure_index = 0;
       structure_index < overlap_matrix.rows();
       ++structure_index) {
    diagonal_sum += overlap_matrix(structure_index, structure_index);
  }
  return diagonal_sum / static_cast<double>(overlap_matrix.rows());
}

void accumulate_additive_vector(
    const std::vector<double>& partial,
    std::vector<double>* total) {
  if (total == nullptr) {
    throw std::invalid_argument("total must not be null");
  }
  if (partial.size() != total->size()) {
    throw std::invalid_argument("vector sizes do not match for accumulation");
  }

  for (std::size_t index = 0; index < total->size(); ++index) {
    (*total)[index] += partial[index];
  }
}

int parallel_task_thread_count(
    int n_tasks) {
  if (n_tasks <= 1) {
    return 1;
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads > n_tasks) {
    n_threads = n_tasks;
  }
  if (n_threads < 1) {
    n_threads = 1;
  }
  return n_threads;
}

void capture_parallel_exception(
    std::atomic<bool>* failed,
    std::exception_ptr* first_exception) {
  if (failed == nullptr || first_exception == nullptr) {
    throw std::invalid_argument("parallel failure outputs must not be null");
  }
#pragma omp critical
  {
    if (!failed->load(std::memory_order_relaxed)) {
      *first_exception = std::current_exception();
      failed->store(true, std::memory_order_relaxed);
    }
  }
}

xmvb::core::GeneralizedEigenResult build_exact_selected_eigen_result(
    const BiorthogonalExactSelectedStructureScfResult& exact_result) {
  xmvb::core::GeneralizedEigenResult eigen_result;
  eigen_result.eigenvalues = exact_result.exact_evaluation.eigenvalues;
  eigen_result.eigenvector_matrix =
      dense_matrix_to_vector(exact_result.exact_evaluation.structure_coefficient_matrix);
  return eigen_result;
}

BiorthogonalExactSelectedStructureScfResult
build_exact_selected_scf_result_from_matrix_build(
    const BiorthogonalExactSelectedStructureMatrixBuildResult& matrix_result,
    const std::vector<double>& eigenvalues,
    const Eigen::MatrixXd& structure_coefficient_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    double nuclear_repulsion_energy,
    double one_electron_reference_energy) {
  if (selected_state_indices.size() != normalized_state_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and normalized_state_weights must be aligned");
  }

  BiorthogonalExactSelectedStructureScfResult result;
  result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  result.one_electron_reference_energy = one_electron_reference_energy;
  result.selected_state_indices = selected_state_indices;
  result.state_average_weights = normalized_state_weights;
  result.average_structure_overlap =
      compute_average_structure_overlap(
          matrix_result.physical_structure_overlap);
  result.structure_matrices.n_structures = matrix_result.n_selected_structures;
  result.structure_matrices.overlap_matrix =
      dense_matrix_to_vector(matrix_result.physical_structure_overlap);
  result.structure_matrices.hamiltonian_matrix =
      dense_matrix_to_vector(matrix_result.physical_structure_hamiltonian);

  result.exact_evaluation.n_active_orbitals = matrix_result.n_active_orbitals;
  result.exact_evaluation.n_full_structures = matrix_result.n_full_structures;
  result.exact_evaluation.n_selected_structures = matrix_result.n_selected_structures;
  result.exact_evaluation.n_determinants = matrix_result.n_determinants;
  result.exact_evaluation.selected_structure_indices =
      matrix_result.selected_structure_indices;
  result.exact_evaluation.physical_structure_overlap =
      matrix_result.physical_structure_overlap;
  result.exact_evaluation.physical_structure_hamiltonian =
      matrix_result.physical_structure_hamiltonian;
  result.exact_evaluation.eigenvalues = eigenvalues;
  result.exact_evaluation.structure_coefficient_matrix =
      structure_coefficient_matrix;

  result.selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
  for (std::size_t selection_index = 0;
       selection_index < selected_state_indices.size();
       ++selection_index) {
    const int state_index = selected_state_indices[selection_index];
    const double state_energy = eigenvalues[xmvb::to_size(state_index)];
    result.electronic_energy +=
        normalized_state_weights[selection_index] * state_energy;
    result.selected_state_total_energies[selection_index] =
        state_energy + nuclear_repulsion_energy;
  }
  result.total_energy =
      result.one_electron_reference_energy +
      result.electronic_energy +
      nuclear_repulsion_energy;
  return result;
}

std::shared_ptr<CppActiveSpaceSecondOrderContext>
build_exact_selected_second_order_context(
    const CppVbInput& input,
    TimedPreparedActiveSpaceContext&& timed_active_space_context,
    SameSpinPairCacheContext&& same_spin_pair_cache,
    const CppActiveSpaceGradientResult& gradient_result,
    const BiorthogonalExactSelectedStructureScfResult& exact_result) {
  auto context = std::make_shared<CppActiveSpaceSecondOrderContext>();

  // Keep the accepted-point orbital and active-space tensors alive so the
  // lower orbital pullback and any future accepted-point probes see the same
  // `SSO/HHO/GGO` state used by the exact biorthogonal selected-space build.
  context->prepared_active_space =
      std::move(timed_active_space_context.prepared_active_space);
  context->same_spin_pair_cache = std::move(same_spin_pair_cache);
  context->structure_matrices = exact_result.structure_matrices;
  context->active_orbital_overlap_gradient =
      gradient_result.active_orbital_overlap_gradient;
  context->active_one_electron_gradient =
      gradient_result.active_one_electron_gradient;
  context->packed_active_two_electron_gradient =
      gradient_result.packed_active_two_electron_gradient;
  context->eigen_result = build_exact_selected_eigen_result(exact_result);
  context->selected_state_indices = exact_result.selected_state_indices;
  context->normalized_state_weights = exact_result.state_average_weights;
  context->selected_state_energies = gather_selected_state_energies(
      context->eigen_result.eigenvalues,
      context->selected_state_indices);
  context->n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

  // HVP is intentionally out of scope for `tbvbscf` right now. Keep the exact
  // accepted-point payload available, but do not advertise the nonorthogonal
  // matrix-form second-order operators on this biorthogonal context yet.
  context->use_full_matrix_form_adjoint = false;
  context->use_matrix_form_opposite_spin = false;
  return context;
}

CppActiveSpaceGradientResult evaluate_active_space_gradient_impl(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  (void) algorithm;
  if (validation_tolerance < 0.0) {
    throw std::invalid_argument("validation_tolerance must be non-negative");
  }

  const auto total_start_time = std::chrono::steady_clock::now();

  CppActiveSpaceGradientEvaluator probe_evaluator(VBSCFAlgorithm::Original);
  auto timed_active_space_context =
      probe_evaluator.prepare_timed_active_space_context_for_probe(input);
  const auto& prepared_active_space =
      timed_active_space_context.prepared_active_space;

  const auto exact_forward_start_time = std::chrono::steady_clock::now();
  const xmvb::vb::FullDeterminantStructureData full_structure_data =
      build_biorthogonal_full_structure_data(
          input,
          prepared_active_space);
  const BiorthogonalExactSelectedStructureMatrixBuildResult matrix_result =
      build_biorthogonal_exact_selected_structure_matrices(
          full_structure_data,
          selected_structure_indices);
  const std::vector<double> normalized_state_weights =
      xmvb::vb::normalize_state_average_weights(state_average_weights);
  const xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  const xmvb::core::GeneralizedEigenResult eigen_result =
      generalized_eigensolver.solve(
          dense_matrix_to_vector(matrix_result.physical_structure_hamiltonian),
          dense_matrix_to_vector(matrix_result.physical_structure_overlap),
          matrix_result.n_selected_structures);
  const Eigen::Map<const Eigen::MatrixXd> mapped_structure_coefficients(
      eigen_result.eigenvector_matrix.data(),
      matrix_result.n_selected_structures,
      matrix_result.n_selected_structures);
  const BiorthogonalExactSelectedStructureScfResult exact_result =
      build_exact_selected_scf_result_from_matrix_build(
          matrix_result,
          eigen_result.eigenvalues,
          mapped_structure_coefficients,
          selected_state_indices,
          normalized_state_weights,
          nuclear_repulsion_energy,
          prepared_active_space.one_electron_reference_energy);
  const double exact_forward_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - exact_forward_start_time)
          .count();

  const auto cache_start_time = std::chrono::steady_clock::now();
  const FullDeterminantPairEvaluator pair_evaluator;
  const xmvb::vb::SameSpinPairCacheBuildOptions cache_build_options{
      false,
  };
  SameSpinPairCacheContext same_spin_pair_cache =
      build_same_spin_pair_cache_context(
          input.structure_data.alpha_det,
          input.structure_data.beta_det,
          pair_evaluator,
          prepared_active_space.orbital_result.active_orbital_overlap_matrix,
          prepared_active_space.active_space_one_electron_result.h1e_act,
          input.orbital_preparation_input.n_active_orbitals,
          prepared_active_space.active_space_two_electron_result,
          cache_build_options);
  populate_same_spin_phi_cache(
      &same_spin_pair_cache,
      prepared_active_space.active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      prepared_active_space.active_space_two_electron_result);
  const double same_spin_cache_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - cache_start_time)
          .count();

  const auto adjoint_start_time = std::chrono::steady_clock::now();
  const BiorthogonalOrbitalIntegrals orbital_integrals =
      build_biorthogonal_orbital_integrals(
          input.orbital_preparation_input.n_active_orbitals,
          prepared_active_space.orbital_result.active_orbital_overlap_matrix,
          prepared_active_space.active_space_one_electron_result.h1e_act);
  const ActiveSpaceTwoElectronView right_right_two_electron_view =
      make_active_space_two_electron_view(
          prepared_active_space.active_space_two_electron_result);
  const BiorthogonalSelectedStateMatrices selected_state_matrices =
      build_biorthogonal_selected_state_matrices_from_structure_problem(
          full_structure_data,
          selected_structure_indices,
          exact_result.exact_evaluation.structure_coefficient_matrix,
          exact_result.exact_evaluation.eigenvalues,
          same_spin_pair_cache,
          exact_result.selected_state_indices,
          exact_result.state_average_weights,
          orbital_integrals,
          right_right_two_electron_view);
  const BiorthogonalForwardSpinPairTileProvider alpha_biorthogonal_provider(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      true,
      orbital_integrals,
      right_right_two_electron_view);
  const BiorthogonalForwardSpinPairTileProvider beta_biorthogonal_provider(
      same_spin_pair_cache.beta_reuse_table.unique_determinants,
      false,
      orbital_integrals,
      right_right_two_electron_view);
  SameSpinMatrixBackwardContribution same_spin_contribution;
  OppositeSpinMatrixBackwardContribution opposite_spin_contribution;
  BiorthogonalOneElectronTransformBackwardContribution
      one_electron_transform_contribution;
  BiorthogonalTwoElectronTransformBackwardContribution
      two_electron_transform_contribution;
  const int adjoint_task_threads = parallel_task_thread_count(4);
  std::atomic<bool> adjoint_failed(false);
  std::exception_ptr adjoint_exception;

  // At a fixed accepted-point forward state, the four exact biorthogonal
  // pullbacks depend only on the shared `L/Q/R` bundle and prepared kernels, so
  // they can run concurrently before the final vector accumulation.
#pragma omp parallel sections num_threads(adjoint_task_threads) if(adjoint_task_threads > 1)
  {
#pragma omp section
    {
      try {
        same_spin_contribution =
            build_biorthogonal_same_spin_matrix_backward_contribution(
                same_spin_pair_cache,
                alpha_biorthogonal_provider,
                beta_biorthogonal_provider,
                selected_state_matrices,
                input.orbital_preparation_input.n_active_orbitals);
      } catch (...) {
        capture_parallel_exception(&adjoint_failed, &adjoint_exception);
      }
    }
#pragma omp section
    {
      try {
        opposite_spin_contribution =
            build_biorthogonal_opposite_spin_matrix_backward_contribution(
                same_spin_pair_cache,
                selected_state_matrices,
                right_right_two_electron_view,
                input.orbital_preparation_input.n_active_orbitals);
      } catch (...) {
        capture_parallel_exception(&adjoint_failed, &adjoint_exception);
      }
    }
#pragma omp section
    {
      try {
        one_electron_transform_contribution =
            build_biorthogonal_one_electron_transform_backward_contribution(
                same_spin_pair_cache,
                selected_state_matrices,
                orbital_integrals,
                prepared_active_space.active_space_one_electron_result.h1e_act);
      } catch (...) {
        capture_parallel_exception(&adjoint_failed, &adjoint_exception);
      }
    }
#pragma omp section
    {
      try {
        two_electron_transform_contribution =
            build_biorthogonal_two_electron_transform_backward_contribution(
                input,
                selected_state_matrices,
                orbital_integrals,
                prepared_active_space.active_space_two_electron_result);
      } catch (...) {
        capture_parallel_exception(&adjoint_failed, &adjoint_exception);
      }
    }
  }
  if (adjoint_exception) {
    std::rethrow_exception(adjoint_exception);
  }
  const double adjoint_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - adjoint_start_time)
          .count();

  CppActiveSpaceGradientResult result;
  initialize_active_space_gradient_probe_result(
      input,
      timed_active_space_context,
      &result);
  result.scf_result =
      build_cpp_vb_scf_result_from_exact_selected_structure_result(exact_result);

  // The exact biorthogonal active-space adjoint closes in four blocks:
  // 1. same-spin selected-state contraction on ordered unique-spin pairs,
  // 2. opposite-spin selected-state contraction on packed pair channels,
  // 3. one-electron `X^{-1} HHO` transform pullback,
  // 4. two-electron left-leg transform pullback.
  result.active_orbital_overlap_gradient =
      same_spin_contribution.active_orbital_overlap_gradient;
  result.active_one_electron_gradient =
      one_electron_transform_contribution.active_one_electron_gradient;
  result.packed_active_two_electron_gradient =
      two_electron_transform_contribution.packed_active_two_electron_gradient;
  accumulate_additive_vector(
      opposite_spin_contribution.active_orbital_overlap_gradient,
      &result.active_orbital_overlap_gradient);
  accumulate_additive_vector(
      one_electron_transform_contribution.active_orbital_overlap_gradient,
      &result.active_orbital_overlap_gradient);
  accumulate_additive_vector(
      two_electron_transform_contribution.active_orbital_overlap_gradient,
      &result.active_orbital_overlap_gradient);

  // The exact selected-space forward currently exports only one combined stage.
  // Keep that time visible instead of hiding it inside `total_wall_time`.
  result.structure_matrix_wall_time_seconds = exact_forward_wall_time_seconds;
  result.eigensolver_wall_time_seconds = 0.0;
  result.adjoint_wall_time_seconds =
      same_spin_cache_wall_time_seconds + adjoint_wall_time_seconds;
  result.second_order_context =
      build_exact_selected_second_order_context(
          input,
          std::move(timed_active_space_context),
          std::move(same_spin_pair_cache),
          result,
          exact_result);
  result.total_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - total_start_time)
          .count();
  return result;
}

CppOrbitalGradientResult evaluate_orbital_gradient_impl(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  CppActiveSpaceGradientResult active_space_gradient_result =
      evaluate_active_space_gradient_impl(
          input,
          selected_structure_indices,
          selected_state_indices,
          state_average_weights,
          algorithm,
          nuclear_repulsion_energy,
          validation_tolerance);
  CppOrbitalGradientEvaluator gradient_evaluator(VBSCFAlgorithm::Original);
  return gradient_evaluator.evaluate(
      input,
      std::move(active_space_gradient_result));
}

}  // namespace

CppActiveSpaceGradientResult
evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  return evaluate_active_space_gradient_impl(
      input,
      selected_structure_indices,
      {0},
      {1.0},
      algorithm,
      nuclear_repulsion_energy,
      validation_tolerance);
}

CppActiveSpaceGradientResult
evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  return evaluate_active_space_gradient_impl(
      input,
      selected_structure_indices,
      selected_state_indices,
      state_average_weights,
      algorithm,
      nuclear_repulsion_energy,
      validation_tolerance);
}

CppOrbitalGradientResult
evaluate_biorthogonal_exact_selected_structure_orbital_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  return evaluate_orbital_gradient_impl(
      input,
      selected_structure_indices,
      {0},
      {1.0},
      algorithm,
      nuclear_repulsion_energy,
      validation_tolerance);
}

CppOrbitalGradientResult
evaluate_biorthogonal_exact_selected_structure_orbital_gradient(
    const CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance) {
  return evaluate_orbital_gradient_impl(
      input,
      selected_structure_indices,
      selected_state_indices,
      state_average_weights,
      algorithm,
      nuclear_repulsion_energy,
      validation_tolerance);
}

}  // namespace xmvb::vb::biorthogonal_vbscf
