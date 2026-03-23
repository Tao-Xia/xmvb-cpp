#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief AO effective one-electron matrices built from the inactive density.
 */
struct AoEffectiveOneElectronResult {
  /**
   * @brief Column-major AO Coulomb-exchange contribution `G11`.
   */
  std::vector<double> ao_coulomb_exchange_matrix;

  /**
   * @brief Column-major AO effective one-electron matrix `F11 = HHF + G11`.
   */
  std::vector<double> ao_effective_one_electron_matrix;
};

}  // namespace xmvb::vb
