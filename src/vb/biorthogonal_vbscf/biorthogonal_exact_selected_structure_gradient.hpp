#pragma once

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/scf/cpp_orbital_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Evaluates the exact selected-space active-space gradient.
 *
 * The exact selected-space forward keeps the full determinant topology,
 * contracts the physical selected-space matrices
 *
 * `S_str = T_sel^T S_det T_sel`,
 * `H_str = T_sel^T H_det T_sel`,
 *
 * and then solves the resulting generalized eigenproblem on the selected
 * structure columns. This routine differentiates that exact biorthogonal
 * objective directly:
 *
 * 1. exact same-spin selected-state contraction,
 * 2. exact opposite-spin selected-state contraction,
 * 3. one-electron transform pullback,
 * 4. two-electron transform pullback.
 *
 * The returned SCF payload is therefore the real `tbvbscf` forward result, not
 * a nonorthogonal wrapper with overwritten metadata.
 */
CppActiveSpaceGradientResult
evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original,
    double nuclear_repulsion_energy = 0.0,
    double validation_tolerance = 0.0);

/**
 * @brief Evaluates the state-averaged exact selected-space active-space gradient.
 */
CppActiveSpaceGradientResult
evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    xmvb::vb::VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance = 0.0);

/**
 * @brief Evaluates the exact selected-space orbital gradient.
 *
 * This is the orbital-level counterpart of the exact selected-space active
 * gradient above. The active-space adjoint is assembled with the exact
 * biorthogonal backward, then passed into the existing lower orbital
 * backpropagation stack so the sparse-orbital gradient matches the exact
 * `tbvbscf` objective while reusing the validated AO/orbital pullback.
 */
CppOrbitalGradientResult
evaluate_biorthogonal_exact_selected_structure_orbital_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original,
    double nuclear_repulsion_energy = 0.0,
    double validation_tolerance = 0.0);

/**
 * @brief Evaluates the state-averaged exact selected-space orbital gradient.
 */
CppOrbitalGradientResult
evaluate_biorthogonal_exact_selected_structure_orbital_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    xmvb::vb::VBSCFAlgorithm algorithm,
    double nuclear_repulsion_energy,
    double validation_tolerance = 0.0);

}  // namespace xmvb::vb::biorthogonal_vbscf
