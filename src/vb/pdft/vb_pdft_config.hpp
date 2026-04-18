#pragma once

#include <string>
#include <Eigen/Core>

#include "vb/pdft/molecular_grid.hpp"

namespace xmvb::vb::pdft {

/**
 * @brief Configuration for VB-PDFT energy evaluation.
 */
struct VbPdftConfig {
  /**
   * @brief Grid configuration.
   */
  MolecularGridConfig grid_config;

  /**
   * @brief Libxc functional ID for on-top functional.
   *
   * Common choices:
   * - 1: XC_LDA_X (Slater exchange)
   * - 7: XC_LDA_C_VWN (VWN correlation)
   * - 101: XC_GGA_X_PBE (PBE exchange)
   * - 130: XC_GGA_C_PBE (PBE correlation)
   */
  int functional_id = 1;  // Default: LDA exchange

  /**
   * @brief Density threshold for numerical stability.
   *
   * Points with rho < threshold are treated as zero density.
   */
  double density_threshold = 1.0e-12;

  /**
   * @brief Whether to use GGA (requires AO gradients).
   */
  bool use_gga = false;
};

/**
 * @brief VB-PDFT energy evaluation result.
 */
struct VbPdftEnergyResult {
  /**
   * @brief Nuclear repulsion energy.
   */
  double nuclear_repulsion_energy = 0.0;

  /**
   * @brief One-electron energy: Tr(h * gamma).
   */
  double one_electron_energy = 0.0;

  /**
   * @brief Classical Coulomb energy: (1/2) Tr(gamma * gamma * J).
   */
  double coulomb_energy = 0.0;

  /**
   * @brief On-top functional energy: integral of eps_ot[rho, Pi].
   */
  double on_top_energy = 0.0;

  /**
   * @brief Total VB-PDFT energy.
   *
   * E_total = V_nn + E_one + E_coulomb + E_ot
   */
  double total_energy = 0.0;

  /**
   * @brief Number of electrons from density integration.
   *
   * Should equal the expected number of electrons.
   * Useful for checking numerical accuracy.
   */
  double integrated_electron_count = 0.0;

  /**
   * @brief Number of grid points used.
   */
  int n_grid_points = 0;

  /**
   * @brief Selected state index.
   */
  int state_index = 0;
};

}  // namespace xmvb::vb::pdft
