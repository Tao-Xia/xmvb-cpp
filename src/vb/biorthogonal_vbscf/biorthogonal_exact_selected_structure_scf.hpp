#pragma once

#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/scf/cpp_vb_scf_result.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief SCF-style wrapper around the exact selected-structure biorthogonal projection.
 *
 * This object keeps the exact selected-subspace biorthogonal matrices and
 * determinant coefficients, but also exposes the state-selection metadata and
 * energy bookkeeping expected by higher-level VBSCF callers.
 */
struct BiorthogonalExactSelectedStructureScfResult {
  double nuclear_repulsion_energy = 0.0;
  double one_electron_reference_energy = 0.0;
  double electronic_energy = 0.0;
  double total_energy = 0.0;
  double average_structure_overlap = 0.0;

  std::vector<int> selected_state_indices;
  std::vector<double> state_average_weights;
  std::vector<double> selected_state_total_energies;

  xmvb::vb::StructureAccumulationResult structure_matrices;
  BiorthogonalExactSelectedStructureEvaluationResult exact_evaluation;
};

/**
 * @brief Evaluates one exact selected-structure subspace for the ground state.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const PreparedBiorthogonalInput& prepared_input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy = 0.0,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Evaluates one exact selected-structure subspace for chosen states.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const PreparedBiorthogonalInput& prepared_input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience ground-state overload that prepares the active-space tensors internally.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy = 0.0,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload for callers that already prepared the active-space tensors once.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience ground-state overload that prepares the active-space tensors internally.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double nuclear_repulsion_energy = 0.0,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Convenience overload that prepares the active-space tensors internally.
 */
BiorthogonalExactSelectedStructureScfResult
evaluate_biorthogonal_exact_selected_structure_scf(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    double symmetry_tolerance = 1.0e-8);

/**
 * @brief Converts one exact selected-space SCF result into the standard C++ SCF payload.
 *
 * Higher-level optimizers and gradient evaluators consume the common
 * `CppVbScfResult` layout. This helper preserves the exact selected-space
 * structure matrices, eigenpairs, and energy bookkeeping while exporting them
 * through that shared interface.
 */
xmvb::vb::CppVbScfResult
build_cpp_vb_scf_result_from_exact_selected_structure_result(
    const BiorthogonalExactSelectedStructureScfResult& exact_result);

}  // namespace xmvb::vb::biorthogonal_vbscf
