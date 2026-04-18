#pragma once

#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief AO basis function values evaluated on a grid.
 *
 * Stores chi_mu(r_g) for all basis functions mu and grid points g.
 * The values matrix is organized as n_points x n_basis_functions
 * in column-major order for efficient access patterns.
 */
struct AoGridValues {
  /**
   * @brief AO values at grid points.
   *
   * Column-major matrix: n_points x n_basis_functions.
   * Entry (g, mu) = chi_mu(r_g).
   */
  Eigen::MatrixXd values;

  /**
   * @brief Optional AO gradient values for GGA functionals.
   *
   * Column-major matrix: n_points x (3 * n_basis_functions).
   * Entry (g, 3*mu + d) = d chi_mu / d x_d at r_g,
   * where d = 0 (x), 1 (y), 2 (z).
   *
   * Empty if gradients were not requested.
   */
  Eigen::MatrixXd gradients;

  /**
   * @brief Number of grid points.
   */
  int n_points() const { return static_cast<int>(values.rows()); }

  /**
   * @brief Number of basis functions.
   */
  int n_basis_functions() const { return static_cast<int>(values.cols()); }

  /**
   * @brief Whether gradient values are available.
   */
  bool has_gradients() const { return gradients.size() > 0; }
};

}  // namespace xmvb::vb::pdft
