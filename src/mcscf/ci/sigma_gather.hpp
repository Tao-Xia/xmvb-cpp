#pragma once

#include <algorithm>
#include <span>

#include "mcscf/ci/channel_product.hpp"

namespace xmvb::mcscf::detail {

/** @brief One tile-local target with its composed grouped-image address.
 * An RHS advances by stride image rows. All temporal responses share this
 * symbolic address; no channel values or excitation-edge arrays are cached.
 */
struct SigmaGatherRow { int alpha,image,stride; int channel_group=0; };

/** @brief Contract one beta edge over existing target rows only.
 * Targets are sorted alpha addresses with the layout permutation composed once.
 * `values` points at one channel; `element_stride` separates image rows.
 * Sparse destination rows are intersected exactly, without a product mask.
 * The caller owns a disjoint destination interval, so no atomics are needed.
 */
inline void gather_beta_rows(const double* values,int element_stride,int rhs,double sign,
    std::span<const SigmaGatherRow> targets,int first,int last,
    double* destination,const int* rows,int n_rows,
    const ChannelSelectors* projected=nullptr,int pair=0,std::size_t image_rows=0) {
  const auto load=[&](const SigmaGatherRow& row) {
    if (!projected) return values[
        static_cast<std::size_t>(row.image+rhs*row.stride)*element_stride];
    const int local=projected->local(row.channel_group,pair);
    return local<0 ? 0.0:values[row.image+rhs*row.stride+local*image_rows];
  };
  const auto before=[](const SigmaGatherRow& row,int alpha) { return row.alpha<alpha; };
  auto target=std::lower_bound(targets.begin(),targets.end(),first,before);
  const auto end=std::lower_bound(target,targets.end(),last,before);
  if (target==end) return;
  if (!rows) {
    for (;target!=end;++target)
      destination[target->alpha]+=sign*load(*target);
    return;
  }
  int output=std::lower_bound(rows,rows+n_rows,target->alpha)-rows;
  for (;target!=end;++target) {
    const int alpha=target->alpha;
    while (output<n_rows && rows[output]<alpha) ++output;
    if (output==n_rows) break;
    if (rows[output]==alpha)
      destination[output]+=sign*load(*target);
  }
}

} // namespace xmvb::mcscf::detail
