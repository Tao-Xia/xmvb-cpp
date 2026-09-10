#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct AoIntegralInput;
struct OrbitalPreparationInput;

namespace detail {

int choose_exact_ao_h1e_thread_count(
    const OrbitalPreparationInput& orbital_preparation_input);

/**
 * Applies the AO effective-one-electron forward derivative and its transpose
 * pullback in one pass over the accepted AO integral representation.
 */
void apply_fused_exact_ao_one_electron_response(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    Eigen::MatrixXd* symmetrized_pullback_source,
    std::vector<double>* delta_ao_effective_h1e_storage,
    std::vector<double>* inactive_density_gradient_storage,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e_workspaces,
    std::vector<Eigen::MatrixXd>* partial_density_gradient_workspaces);

}  // namespace detail
}  // namespace xmvb::vb
