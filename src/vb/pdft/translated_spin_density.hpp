#pragma once

#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Translated spin densities for on-top functionals.
 *
 * The translation scheme maps the total density rho and on-top pair
 * density Pi to effective spin densities rho_alpha and rho_beta that
 * can be used with standard spin-polarized DFT functionals.
 *
 * Reference: Li Manni et al., J. Chem. Theory Comput. 10, 3669 (2014).
 */
struct TranslatedSpinDensity {
  /**
   * @brief Translated alpha spin density.
   *
   * rho_alpha(r) = rho(r) * (1 + zeta_t(r)) / 2
   */
  Eigen::VectorXd rho_alpha;

  /**
   * @brief Translated beta spin density.
   *
   * rho_beta(r) = rho(r) * (1 - zeta_t(r)) / 2
   */
  Eigen::VectorXd rho_beta;

  /**
   * @brief On-top ratio R(r) = 4 * Pi(r) / rho(r)^2.
   *
   * This ratio measures the deviation from a single-determinant
   * reference. R = 0 for single determinant, R = 1 for maximum
   * on-top density.
   */
  Eigen::VectorXd on_top_ratio;

  /**
   * @brief Translated spin polarization zeta_t(r) = sqrt(max(0, 1 - R(r))).
   *
   * This is the effective spin polarization that accounts for
   * multireference character through the on-top density.
   */
  Eigen::VectorXd zeta_translated;

  /**
   * @brief Number of grid points.
   */
  int n_points() const { return static_cast<int>(rho_alpha.size()); }
};

/**
 * @brief Computes translated spin densities from rho and Pi.
 *
 * Implements the translation scheme:
 *   R(r) = 4 * Pi(r) / rho(r)^2
 *   zeta_t(r) = sqrt(max(0, 1 - R(r)))
 *   rho_alpha(r) = rho(r) * (1 + zeta_t(r)) / 2
 *   rho_beta(r) = rho(r) * (1 - zeta_t(r)) / 2
 *
 * Numerical stability:
 * - Applies density threshold to avoid division by zero
 * - Clamps R to [0, 1] range
 * - Ensures rho_alpha, rho_beta >= 0
 *
 * @param rho Total electron density at grid points.
 * @param pi On-top pair density at grid points.
 * @param density_threshold Minimum density for numerical stability (default 1e-12).
 * @return Translated spin densities.
 */
TranslatedSpinDensity compute_translated_spin_density(
    const Eigen::VectorXd& rho,
    const Eigen::VectorXd& pi,
    double density_threshold = 1.0e-12);

}  // namespace xmvb::vb::pdft
