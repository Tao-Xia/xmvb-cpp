#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mcscf/ci/strings.hpp"

namespace xmvb::mcscf {

/** @brief One exact single replacement, in packed-pair order.
 * The current contract has at most63 orbitals, so pair/orbital indices fit
 * uint16/uint8. Keep the numerical sign in double: node-wise kernels consume
 * it directly. String addresses retain int; no floating precision is narrowed.
 */
struct SpinLink {
  int source;
  std::uint16_t pair;
  /** Source-to-row operator c†_created c_annihilated, not the reverse. */
  std::uint8_t created,annihilated;
  double sign;
};

/** @brief Borrowed immutable string row; no per-field maps or allocation. */
struct DensityConnections {
  const SpinLink* data;
  int count;
  int size() const noexcept { return count; }
  const SpinLink& operator[](int i) const noexcept { return data[i]; }
  const SpinLink* begin() const noexcept { return data; }
  const SpinLink* end() const noexcept { return count ? data+count:data; }
};

/** @brief Fixed-stride topology for a complete one-spin sector.
 * Each string has ne*(norb-ne) links. A flat record array stores those rows;
 * one immutable bit-string basis supplies occupations and index lookup.
 * Sigma directions and identical alpha/beta sectors share this immutable data.
 */
struct SpinConnections {
  int strings=0,degree=0;
  std::vector<SpinLink> links;
  /** Occupation bit strings and packed-link prefix per highest orbital.
   * Both are single-spin metadata, shared by every action at this point.
   * With at most 63 orbitals, a row has at most 992 singles (uint16_t).
   */
  SpinStrings basis;
  std::vector<std::uint16_t> orbital_offsets;
  int orbitals=0;

  int n_strings() const noexcept { return strings; }
  int n_occupied() const noexcept { return basis.electrons(); }
  int n_singles() const noexcept { return degree; }

  DensityConnections singles(int string) const noexcept {
    return {degree ? links.data()+static_cast<std::size_t>(degree)*string:nullptr,degree};
  }
  OccupiedOrbitals occupied(int string) const noexcept { return basis.occupied(string); }
  /** @brief Count links before a packed channel, using its high orbital and
   * low-orbital prefix mask. Links already follow high/low packed order.
   * Only opposite occupations connect; addresses/signs remain precomputed.
   */
  int link_prefix(int string,int high,std::uint64_t low_mask) const noexcept {
    if (high==orbitals) return degree;
    const auto mask=basis.masks()[string];
    const auto opposite=(mask>>high)&1 ? ~mask:mask;
    return orbital_offsets[static_cast<std::size_t>(string)*(orbitals+1)+high]+
        std::popcount(opposite&low_mask);
  }
  /** @brief Visit only sources repairing a certified prefix deficit.
   * Source support requires p<=first<=last<q, with p empty and q occupied
   * in this target string. Those p are exactly the first links in q's
   * existing packed-pair block, so no rank search or new excitation map is
   * needed. Callback arguments are link index and local envelope channel.
   */
  template<class Function>
  void visit_lower_sources(int string,int first,int last,Function&& consume) const {
    const auto occupied=basis.masks()[string];
    auto holes=~occupied&((std::uint64_t{1}<<(first+1))-1);
    auto high=occupied&~((std::uint64_t{1}<<(last+1))-1);
    if (!holes || !high) return;
    std::array<int,63> low;
    int count=0;
    while (holes) {
      low[count++]=std::countr_zero(holes);
      holes&=holes-1;
    }
    const auto* offsets=orbital_offsets.data()+static_cast<std::size_t>(string)*(orbitals+1);
    while (high) {
      const int q=std::countr_zero(high),channel=(q-last-1)*(first+1);
      const int begin=offsets[q];
      for (int i=0;i<count;++i) consume(begin+i,channel+low[i]);
      high&=high-1;
    }
  }

  std::size_t dynamic_bytes() const noexcept {
    return links.capacity()*sizeof(SpinLink)+basis.dynamic_bytes()+
        orbital_offsets.capacity()*sizeof(std::uint16_t);
  }
};

} // namespace xmvb::mcscf
