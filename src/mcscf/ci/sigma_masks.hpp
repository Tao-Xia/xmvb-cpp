#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <Eigen/Core>

#include "core/workspace.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf::detail {

/** @brief Current-tile threshold bitsets for certified frontier channels.
 * Source rows have first=n-1 and last=-1. The certified envelope admits
 * (p,q) iff first>=p and last<q; the opposite source orientation
 * permits only source rows. No global product mask or link plan is stored.
 */
class SigmaRowMasks {
  workspace::Temporary temporary_;
  Eigen::Matrix<std::uint64_t,Eigen::Dynamic,Eigen::Dynamic> planes_;
  int orbitals_=0;
public:
  template<class Rows>
  void reset(int orbitals,const Rows& rows,int workers) {
    orbitals_=orbitals;
    const auto words=(rows.size()+63)/64;
    planes_.resize(words,2*orbitals);
    temporary_.resize(sizeof(std::uint64_t)*planes_.size());
    cooperative_openmp_for(words,workers,[&](std::ptrdiff_t word) {
      for (int col=0;col<2*orbitals;++col) planes_(word,col)=0;
      const auto end=std::min<std::size_t>(rows.size(),64*(word+1));
      for (std::size_t row=64*word;row<end;++row) {
        const int group=rows[row].channel_group;
        const int first=group ? (group-1)/orbitals:orbitals-1;
        const int last=group ? (group-1)%orbitals:-1;
        const auto bit=std::uint64_t{1}<<(row%64);
        planes_(word,first)|=bit;
        planes_(word,orbitals+last+1)|=bit;
      }
      for (int p=orbitals-2;p>=0;--p) planes_(word,p)|=planes_(word,p+1);
      for (int q=1;q<orbitals;++q) planes_(word,orbitals+q)|=planes_(word,orbitals+q-1);
    });
  }

  /** @brief Visit surviving tile row indices in their original order. */
  template<class Function>
  void visit(int first,int last,int p,int q,bool raises_source,Function&& consume) const {
    for (std::size_t word=static_cast<std::size_t>(first)/64;
         word<(static_cast<std::size_t>(last)+63)/64;++word) {
      auto bits=raises_source ? planes_(word,p)&planes_(word,orbitals_+q):
          planes_(word,orbitals_-1);
      if (word==first/64) bits&=~std::uint64_t{0}<<(first%64);
      if (word==(last-1)/64 && last%64) bits&=(std::uint64_t{1}<<(last%64))-1;
      while (bits) {
        consume(static_cast<int>(word*64+std::countr_zero(bits)));
        bits&=bits-1;
      }
    }
  }
};

} // namespace xmvb::mcscf::detail
