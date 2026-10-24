#pragma once

#include <algorithm>
#include <span>

#include "mcscf/ci/channel_product.hpp"
#include "mcscf/ci/sigma_masks.hpp"

namespace xmvb::mcscf::detail {

/** @brief One tile-local target with its composed grouped-image address.
 * An RHS advances by stride image rows. All temporal responses share this
 * symbolic address; no channel values or excitation-edge arrays are cached.
 */
struct SigmaGatherRow {
  int alpha;
  std::size_t image;
  int stride,channel_group=0;
  std::size_t address(int rhs,int channel,std::size_t channel_stride) const noexcept {
    return image+static_cast<std::size_t>(rhs)*stride+
        static_cast<std::size_t>(channel)*channel_stride;
  }
};

/** @brief Borrowed numerical image and its exact symbolic addressing. */
struct SigmaGatherImage {
  const double* values;
  int element_stride=1;
  const ChannelSelectors* projected=nullptr;
  const SigmaRowMasks* masks=nullptr;
  int pair=0,row_offset=0,low=0,high=0,width=1;
  bool raises_source=true;

  int channel(const SigmaGatherRow& row) const noexcept {
    if (!projected) return 0;
    if (row.channel_group && !raises_source) return -1;
    return projected->local(row.channel_group,pair);
  }
  double load(const SigmaGatherRow& row,int rhs,int channel) const noexcept {
    return values[projected ? row.address(rhs,channel,static_cast<std::size_t>(width)*row.stride):
        row.address(rhs,0,0)*element_stride];
  }
};

/** @brief Contract one beta edge over existing target rows only.
 * Exact word masks skip impossible rows while retaining alpha order and one
 * monotone destination merge. The caller owns a disjoint destination range.
 */
inline void gather_beta_rows(const SigmaGatherImage& image,int rhs,double sign,
    std::span<const SigmaGatherRow> targets,int first,int last,
    double* destination,const int* rows,int n_rows) {
  const auto before=[](const SigmaGatherRow& row,int alpha) { return row.alpha<alpha; };
  auto target=std::lower_bound(targets.begin(),targets.end(),first,before);
  const auto end=std::lower_bound(target,targets.end(),last,before);
  if (target==end) return;
  int output=0;
  if (rows) output=std::lower_bound(rows,rows+n_rows,target->alpha)-rows;
  const auto consume=[&](const SigmaGatherRow& row) {
    const int channel=image.channel(row);
    if (channel<0) return;
    if (!rows) { destination[row.alpha]+=sign*image.load(row,rhs,channel);return; }
    while (output<n_rows && rows[output]<row.alpha) ++output;
    if (output<n_rows && rows[output]==row.alpha)
      destination[output]+=sign*image.load(row,rhs,channel);
  };
  if (image.masks) {
    image.masks->visit(image.row_offset+static_cast<int>(target-targets.begin()),
        image.row_offset+static_cast<int>(end-targets.begin()),image.low,image.high,image.raises_source,
        [&](int index) { consume(targets[index-image.row_offset]); });
  } else for (;target!=end;++target) consume(*target);
}

} // namespace xmvb::mcscf::detail
