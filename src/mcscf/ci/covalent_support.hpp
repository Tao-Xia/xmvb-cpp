#pragma once

#include <cstdint>
#include <bit>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "mcscf/ci/covalent_columns.hpp"

namespace xmvb::mcscf {

/** @brief Exact support of an upper-triangular transform of singly occupied sites.
 * Prefix electron counts must be at least the prefix length. A single
 * one-body excitation can lower a prefix by one: depth=1 is its exact
 * structural closure. Metadata is linear in the number of spin strings;
 * no product-space mask is retained here.
 */
class CovalentSupport {
public:
  CovalentSupport(int n_orbitals, const std::vector<std::uint64_t>& alpha,
                  const std::vector<std::uint64_t>& beta)
      : chunks_((n_orbitals + 7) / 8), n_(n_orbitals), alpha_(prefixes(alpha, n_orbitals)),
        beta_(prefixes(beta, n_orbitals)), lengths_(chunks_, 0),
        columns_(n_orbitals,alpha,beta) {
    for (int k = 0; k < n_orbitals; ++k)
      lengths_[k / 8] |= static_cast<std::uint64_t>(k + 1) << (8 * (k % 8));
  }

  /** @brief Emit sorted support/closure alpha indices from a reusable prefix DAG.
   * Entire legal subtrees are colex intervals. Scratch consists only of the
   * bounded traversal stack and the caller's current tile row list. Persistent
   * storage is O(U_beta * n_orbitals * (n_alpha_electrons+1)), never U_alpha*U_beta.
   */
  void append_column_rows(int beta,std::vector<int>& rows,int depth=0,
      std::size_t upper=std::numeric_limits<std::size_t>::max()) const {
    columns_.append(beta,rows,depth,upper);
  }

  bool contains(int alpha, int beta, int depth = 0) const noexcept {
    const auto* a = alpha_.data() + static_cast<std::size_t>(alpha) * chunks_;
    const auto* b = beta_.data() + static_cast<std::size_t>(beta) * chunks_;
    constexpr std::uint64_t high = 0x8080808080808080ULL;
    constexpr std::uint64_t one = 0x0101010101010101ULL;
    // Eight exact byte-lane comparisons per word. At half filling with
    // <=63 orbitals and depth 0/1, sums stay <128, so lanes cannot carry.
    // The high-bit guard prevents subtraction borrows between byte lanes.
    for (int k = 0; k < chunks_; ++k) {
      const auto sum = a[k] + b[k] + static_cast<std::uint64_t>(depth) * one;
      if ((((sum | high) - lengths_[k]) & high) != high) return false;
    }
    return true;
  }

  /** @brief First/last deficient prefix; {-1,-1} means depth-zero support.
   * A frontier node can be reached only by a raising replacement crossing
   * every deficient prefix. Byte-lane arithmetic shares contains()'s proof.
   */
  std::pair<int,int> deficient_prefixes(int alpha,int beta) const noexcept {
    constexpr std::uint64_t high=0x8080808080808080ULL;
    const auto* a=alpha_.data()+static_cast<std::size_t>(alpha)*chunks_;
    const auto* b=beta_.data()+static_cast<std::size_t>(beta)*chunks_;
    int first=-1,last=-1;
    for (int k=0;k<chunks_;++k) {
      const auto bad=~(((a[k]+b[k])|high)-lengths_[k])&high;
      if (bad) {
        if (first<0) first=8*k+std::countr_zero(bad)/8;
        last=8*k+(63-std::countl_zero(bad))/8;
      }
    }
    return {first,last};
  }

  /** @brief Bit k marks zero electron surplus after orbital k (zero-based).
   * Eight exact byte-lane equalities share the stored spin prefixes. Counts
   * stay below 128, so adding 127 cannot carry into an adjacent byte.
   */
  std::uint64_t zero_prefixes(int alpha,int beta) const noexcept {
    constexpr std::uint64_t high=0x8080808080808080ULL;
    constexpr std::uint64_t bias=0x7f7f7f7f7f7f7f7fULL;
    const auto* a=alpha_.data()+static_cast<std::size_t>(alpha)*chunks_;
    const auto* b=beta_.data()+static_cast<std::size_t>(beta)*chunks_;
    std::uint64_t result=0;
    for (int k=0;k<chunks_;++k) {
      const auto equal=~(((a[k]+b[k])^lengths_[k])+bias)&high;
      result|=(((equal>>7)*0x0102040810204080ULL)>>56)<<(8*k);
    }
    return result&((std::uint64_t{1}<<n_)-1);
  }

  std::size_t dynamic_bytes() const noexcept {
    return (alpha_.capacity()+beta_.capacity()+lengths_.capacity())*
        sizeof(std::uint64_t)+columns_.dynamic_bytes();
  }

private:
  static std::vector<std::uint64_t> prefixes(
      const std::vector<std::uint64_t>& masks, int n_orbitals) {
    const int chunks = (n_orbitals + 7) / 8;
    std::vector<std::uint64_t> result(masks.size() * chunks, 0);
    for (std::size_t string = 0; string < masks.size(); ++string) {
      int count = 0;
      for (int k = 0; k < n_orbitals; ++k) {
        count += (masks[string] >> k) & 1ULL;
        result[string * chunks + k / 8] |=
            static_cast<std::uint64_t>(count) << (8 * (k % 8));
      }
    }
    return result;
  }
  int chunks_,n_;
  std::vector<std::uint64_t> alpha_, beta_, lengths_;
  CovalentColumns columns_;
};

}  // namespace xmvb::mcscf
