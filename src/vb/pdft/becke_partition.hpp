#pragma once

#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Becke partition weights for molecular integration.
 *
 * The Becke partition scheme constructs smooth partition weights w_A(r)
 * for each atom A such that:
 *   - sum_A w_A(r) = 1 at every point r
 *   - w_A(r) is smooth and differentiable
 *   - w_A(r) → 1 near atom A, → 0 far from A
 *
 * The partition is based on confocal elliptical coordinates and iterated
 * cell functions that ensure smoothness across atomic boundaries.
 *
 * Reference: Becke, J. Chem. Phys. 88, 2547 (1988).
 */
class BeckePartition {
public:
  /**
   * @brief Computes Becke partition weights for all atoms at grid points.
   *
   * For each grid point r_g, computes weights w_A(r_g) for all atoms A
   * such that sum_A w_A(r_g) = 1.
   *
   * @param grid_points Grid point coordinates (n_points x 3, column-major).
   * @param atomic_coords Atomic coordinates (n_atoms x 3, column-major).
   * @param atomic_numbers Atomic numbers (length n_atoms).
   * @param exponent Becke partition exponent k (typically 3).
   * @return Partition weights (n_points x n_atoms, column-major).
   *         Each row sums to 1.
   */
  static Eigen::MatrixXd compute_weights(
      const Eigen::MatrixXd& grid_points,
      const Eigen::MatrixXd& atomic_coords,
      const Eigen::VectorXi& atomic_numbers,
      int exponent = 3);

private:
  /**
   * @brief Becke cell function s(mu).
   *
   * s(mu) = (3/2) * mu - (1/2) * mu^3  for |mu| <= 1
   *       = sign(mu)                    for |mu| > 1
   *
   * This is the polynomial form that ensures smoothness.
   */
  static double cell_function(double mu);

  /**
   * @brief Iterated cell function s_k(mu) = s(s(...s(mu)...)).
   *
   * Applies the cell function k times to achieve higher smoothness.
   */
  static double iterated_cell_function(double mu, int k);

  /**
   * @brief Heteronuclear Becke size-adjustment parameter.
   *
   * For the Bragg-Slater ratio `chi = R_A / R_B`, the adjusted confocal
   * coordinate is
   *
   *   mu' = mu + a_AB (1 - mu^2),   a_AB = (1 - chi^2) / (4 chi)
   *
   * with `a_AB` clipped to `[-0.5, 0.5]` as in Becke's original recipe.
   */
  static double size_adjustment_factor(int atomic_number_A, int atomic_number_B);
};

}  // namespace xmvb::vb::pdft
