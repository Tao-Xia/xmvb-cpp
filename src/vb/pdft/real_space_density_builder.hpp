#pragma once

#include <Eigen/Core>

#include "vb/pdft/real_space_densities.hpp"
#include "vb/pdft/ao_grid_values.hpp"
#include "vb/pdft/selected_state_exact_physical_on_top_pair_density_builder.hpp"

namespace xmvb::vb::pdft {

/**
 * @brief Builds real-space densities from RDMs and AO values.
 *
 * This builder evaluates the total electron density rho(r) and on-top
 * pair density Pi(r) at grid points using:
 * - Physical AO 1-RDM (gamma_{mu,nu})
 * - On-top pair density contraction context
 * - AO basis function values at grid points
 *
 * The implementation uses efficient matrix operations to evaluate
 * densities at many grid points simultaneously.
 */
class RealSpaceDensityBuilder {
public:
  /**
   * @brief Builds densities from physical 1-RDM and on-top context.
   *
   * Evaluates:
   *   rho(r) = sum_{mu,nu} gamma_{mu,nu} chi_mu(r) chi_nu(r)
   *   Pi(r) = evaluated from on-top context
   *
   * @param ao_density_matrix Physical AO 1-RDM (n_basis x n_basis, symmetric).
   * @param on_top_context On-top pair density contraction context.
   * @param ao_values AO values at grid points.
   * @return Real-space densities at each grid point.
   */
  RealSpaceDensities build(
      const Eigen::MatrixXd& ao_density_matrix,
      const SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context,
      const AoGridValues& ao_values) const;

private:
  /**
   * @brief Evaluates total electron density rho(r) at grid points.
   *
   * Uses the formula:
   *   rho(r_g) = sum_{mu,nu} gamma_{mu,nu} chi_mu(r_g) chi_nu(r_g)
   *
   * Implemented as:
   *   rho = diag(chi * gamma * chi^T)
   *
   * where chi is the n_points x n_basis AO value matrix.
   */
  Eigen::VectorXd evaluate_electron_density(
      const Eigen::MatrixXd& ao_density_matrix,
      const AoGridValues& ao_values) const;

  /**
   * @brief Evaluates on-top pair density Pi(r) at grid points.
   *
   * Uses the on-top context to evaluate Pi(r) point-by-point without
   * materializing the full 4-index 2-RDM.
   */
  Eigen::VectorXd evaluate_on_top_pair_density(
      const SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context,
      const AoGridValues& ao_values) const;
};

}  // namespace xmvb::vb::pdft
