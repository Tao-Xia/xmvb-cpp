#pragma once

#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Real-space densities evaluated on a grid.
 *
 * Contains the total electron density rho(r) and on-top pair density Pi(r)
 * evaluated at grid points. These are the fundamental quantities needed
 * for PDFT energy evaluation.
 */
struct RealSpaceDensities {
  /**
   * @brief Total electron density rho(r) at grid points.
   *
   * Length: n_points.
   * rho(r) = sum_{mu,nu} gamma_{mu,nu} chi_mu(r) chi_nu(r)
   */
  Eigen::VectorXd rho;

  /**
   * @brief On-top pair density Pi(r) at grid points.
   *
   * Length: n_points.
   * Pi(r) = (1/2) sum_{mu,nu,lambda,kappa} Gamma_{mu,nu,lambda,kappa}
   *         chi_mu(r) chi_nu(r) chi_lambda(r) chi_kappa(r)
   */
  Eigen::VectorXd pi;

  /**
   * @brief Number of grid points.
   */
  int n_points() const { return static_cast<int>(rho.size()); }
};

}  // namespace xmvb::vb::pdft
