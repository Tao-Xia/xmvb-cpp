#pragma once

#include <array>
#include <cstdint>

namespace xmvb::mcscf::detail {

/** @brief Shared exact Pascal coefficients for the supported 0..63 orbitals.
 * All coefficients fit uint64_t. The read-only table is shared across spin
 * sectors and workers, with no per-cursor or per-orbital-point allocation.
 */
inline constexpr auto spin_combinations=[] {
  std::array<std::array<std::uint64_t,64>,64> table{};
  table[0][0]=1;
  for (int n=1;n<64;++n) {
    table[n][0]=1;
    for (int k=1;k<=n;++k) table[n][k]=table[n-1][k-1]+table[n-1][k];
  }
  return table;
}();

constexpr std::uint64_t spin_choose(int n,int k) noexcept {
  return n<0 || k<0 || k>n ? 0:spin_combinations[n][k];
}

} // namespace xmvb::mcscf::detail
