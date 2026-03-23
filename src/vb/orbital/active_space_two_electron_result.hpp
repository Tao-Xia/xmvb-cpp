#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Active-space two-electron integrals derived from AO ERIs.
 */
struct ActiveSpaceTwoElectronResult {
  /**
   * @brief Packed active-space two-electron integrals `GGO`.
   */
  std::vector<double> packed_active_two_electron_integrals;
};

}  // namespace xmvb::vb
