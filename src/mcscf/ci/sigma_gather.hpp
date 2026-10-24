#pragma once

#include <algorithm>
#include <span>

#include "mcscf/ci/channel_product.hpp"

namespace xmvb::mcscf::detail {

/** @brief One tile-local target with its composed grouped-image address.
 * RHS increments live once in the selector layout. All responses share this
 * composed size_t address; no channel values or excitation-edge arrays are cached.
 */
struct SigmaGatherRow {
  int alpha,channel_group=0;
  std::size_t image;
  std::size_t address(int rhs,int channel,int rhs_stride,std::size_t channel_stride) const noexcept {
    return image+static_cast<std::size_t>(rhs)*rhs_stride+
        static_cast<std::size_t>(channel)*channel_stride;
  }
};

/** @brief Borrowed image with action-invariant addressing resolved at entry.
 * FirstRhs is valid for a single-RHS action. Its full-channel load needs
 * neither a selector nor a row-group stride lookup: the RHS offset is zero.
 */
template<bool Projected,bool FirstRhs>
struct SigmaGatherImage {
  const double* values;
  const int* rhs_strides;
  int element_stride=1;
  const ChannelSelectors* projected=nullptr;
  int pair=0,width=1;
  bool raises_source=true;

  int channel(const SigmaGatherRow& row) const noexcept {
    if constexpr (!Projected) return 0;
    if (row.channel_group && !raises_source) return -1;
    return projected->local(row.channel_group,pair);
  }
  double load(const SigmaGatherRow& row,int rhs,int channel) const noexcept {
    if constexpr (!Projected && FirstRhs)
      return values[row.image*element_stride];
    else {
      const int stride=rhs_strides[row.channel_group];
      if constexpr (Projected)
        return values[row.address(FirstRhs ? 0:rhs,channel,stride,
            static_cast<std::size_t>(width)*stride)];
      else return values[row.address(rhs,0,stride,0)*element_stride];
    }
  }
};

/** @brief Contract one beta edge over existing target rows only.
 * Join the sparse destination before decoding channels or loading coefficients.
 * The alpha order permits one monotone merge.
 */
template<bool Projected,bool FirstRhs>
inline void gather_beta_rows(const SigmaGatherImage<Projected,FirstRhs>& image,int rhs,double sign,
    std::span<const SigmaGatherRow> targets,int first,int last,
    double* destination,const int* rows,int n_rows) {
  const auto before=[](const SigmaGatherRow& row,int alpha) { return row.alpha<alpha; };
  auto target=std::lower_bound(targets.begin(),targets.end(),first,before);
  const auto end=std::lower_bound(target,targets.end(),last,before);
  if (target==end) return;
  int output=0;
  if (rows) output=std::lower_bound(rows,rows+n_rows,target->alpha)-rows;
  const auto consume=[&](const SigmaGatherRow& row) {
    // The sparse output join is cheaper than decoding a channel address.
    // Every skipped output row has exactly zero contribution to the consumer.
    if (rows) {
      while (output<n_rows && rows[output]<row.alpha) ++output;
      if (output==n_rows || rows[output]!=row.alpha) return;
    }
    const int channel=image.channel(row);
    if (channel>=0)
      destination[rows ? output:row.alpha]+=sign*image.load(row,rhs,channel);
  };
  for (;target!=end;++target) consume(*target);
}

} // namespace xmvb::mcscf::detail
