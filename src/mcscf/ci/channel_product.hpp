#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include "core/workspace.hpp"

namespace xmvb::mcscf::detail {

using ChannelRows=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

/** @brief Shared exact source-channel selectors and corresponding K columns.
 * Group zero is unrestricted. Group 1+first*n+last permits packed pairs
 * (p,q) with p<=first<=last<q.
 */
class ChannelSelectors {
  struct Block { std::vector<int> columns,inverse; int first=-1,last=-1; };
  int pairs_;
  std::vector<Block> blocks_;
public:
  ChannelSelectors(int pairs,int orbitals,bool restricted):pairs_(pairs) {
    blocks_.resize(restricted ? 1+orbitals*orbitals:1);
    if (!restricted) return;
    for (int first=0;first<orbitals;++first) for (int last=first;last<orbitals-1;++last) {
      auto& block=blocks_[1+first*orbitals+last];
      block.first=first;block.last=last;
      block.inverse.assign(pairs,-1);
      for (int q=last+1;q<orbitals;++q) for (int p=0;p<=first;++p) {
        const int pair=q*(q+1)/2+p;
        block.inverse[pair]=block.columns.size();
        block.columns.push_back(pair);
      }
    }
  }

  int groups() const noexcept { return blocks_.size(); }
  std::pair<int,int> frontier(int group) const noexcept { return {blocks_[group].first,blocks_[group].last}; }
  int channels(int group) const noexcept { return group ? blocks_[group].columns.size():pairs_; }
  int local(int group,int pair) const noexcept { return group ? blocks_[group].inverse[pair]:pair; }
  int global(int group,int channel) const noexcept { return group ? blocks_[group].columns[channel]:channel; }
};

/** @brief K[:,J] or K[J,J] for unrestricted or source-supported left images. */
class ChannelProduct : public ChannelSelectors {
  const Eigen::MatrixXd& kernel_;
  workspace::Temporary temporary_;
  std::vector<Eigen::MatrixXd> kernels_;
public:
  ChannelProduct(const Eigen::MatrixXd& kernel,int orbitals,bool restricted,
      bool projected_image=false)
      :ChannelSelectors(kernel.rows(),orbitals,restricted),kernel_(kernel),kernels_(groups()) {
    for (int group=1;group<groups();++group) {
      auto& panel=kernels_[group];
      panel.resize(projected_image ? channels(group):kernel.rows(),channels(group));
      for (int k=0;k<channels(group);++k) {
        if (projected_image) for (int j=0;j<channels(group);++j)
          panel(j,k)=kernel(global(group,j),global(group,k));
        else panel.col(k)=kernel.col(global(group,k));
      }
    }
    std::size_t bytes=0;
    for (const auto& panel:kernels_) bytes+=sizeof(double)*panel.size();
    temporary_.resize(bytes);
  }
  const Eigen::MatrixXd& kernel(int group) const noexcept { return group ? kernels_[group]:kernel_; }
};

/** @brief Ragged row-major sources and column-major channel images in group order.
 * The row maps are shared by every response. Numerical panels are never
 * copied to rearrange rows. Unrestricted tiles use implicit identity maps.
 */
class ChannelLayout {
  std::vector<int> offsets_,image_rows_,channels_,rhs_strides_;
  std::vector<std::size_t> packed_offsets_;
  bool identity_=true,packed_images_=false;
  int pairs_=0,rows_=0;
public:
  void reset(const ChannelSelectors& basis,std::span<const int> groups,int count,int width,
      bool store_images=true,bool packed_images=false) {
    pairs_=basis.channels(0);rows_=count*width;packed_images_=packed_images;
    offsets_.assign(basis.groups()+1,0);
    packed_offsets_.assign(basis.groups()+1,0);
    channels_.resize(basis.groups());rhs_strides_.resize(basis.groups());
    identity_=groups.empty() || std::all_of(groups.begin(),groups.end(),[](int g) { return g==0; });
    if (identity_) offsets_[1]=count*width;
    else for (const int group:groups) offsets_[group+1]+=width;
    for (int group=0;group<basis.groups();++group) {
      offsets_[group+1]+=offsets_[group];
      channels_[group]=basis.channels(group);
      rhs_strides_[group]=(offsets_[group+1]-offsets_[group])/width;
      packed_offsets_[group+1]=packed_offsets_[group]+
          static_cast<std::size_t>(offsets_[group+1]-offsets_[group])*channels_[group];
    }
    if (identity_ || !store_images) { image_rows_.clear(); return; }
    image_rows_.resize(count*width);
    auto position=offsets_;
    for (int rhs=0;rhs<width;++rhs) for (int row=0;row<count;++row)
      image_rows_[rhs*count+row]=position[groups[row]]++;
  }

  /** @brief Reuse the tile's diagonal-index storage as sorted RHS-zero images.
   * Invoke once after reset(). No coefficients or per-RHS index copies are made.
   */
  void prepare_diagonals(std::span<int> rows) const {
    for (int& row:rows) row=image(row);
    std::sort(rows.begin(),rows.end());
  }

  /** @brief Diagonals owned by one group and local RHS-zero row interval. */
  std::span<const int> diagonals(std::span<const int> images,int group,int first,int last) const {
    first=std::max(0,first);last=std::min(rhs_stride(group),last);
    if (first>=last || images.empty()) return {};
    const auto begin=std::lower_bound(images.begin(),images.end(),this->begin(group)+first);
    const auto end=std::lower_bound(begin,images.end(),this->begin(group)+last);
    return {begin,end};
  }
  struct Panel { int first,rows; std::size_t packed; };

  /** Largest consecutive row panel fitting its exact ragged source copies
   * plus the requested images. Only scalar boundaries are returned; source
   * rows remain in their existing group order, without padding or repacking.
   */
  Panel panel(int first,int source_copies,std::size_t entries,int image_copies=1) const {
    if (first<0 || first>=offsets_.back() || source_copies<0 || image_copies<1)
      throw std::invalid_argument("invalid channel row panel");
    Panel result{first,0,0};
    std::size_t used=0;
    for (int g=0;g<static_cast<int>(channels_.size());++g) {
      const int start=std::max(first+result.rows,offsets_[g]);
      if (start>=offsets_[g+1]) continue;
      const auto row_entries=static_cast<std::size_t>(source_copies)*channels_[g]+
          static_cast<std::size_t>(image_copies)*(packed_images_ ? channels_[g]:pairs_);
      const int count=static_cast<int>(std::min<std::size_t>(offsets_[g+1]-start,
          (entries-used)/row_entries));
      result.rows+=count;result.packed+=static_cast<std::size_t>(count)*channels_[g];
      used+=static_cast<std::size_t>(count)*row_entries;
      if (start+count<offsets_[g+1]) break;
    }
    if (!result.rows) throw std::length_error("channel panel cannot hold one row");
    return result;
  }

  /** Column-major images share packed group extents only for K[J,J]. */
  std::size_t output_offset(int group) const noexcept {
    return packed_images_ ? packed_offsets_[group]:static_cast<std::size_t>(offsets_[group]);
  }
  int output_stride(int group) const noexcept {
    return packed_images_ ? size(group):rows_;
  }
  std::size_t output_index(int image,int group,int channel) const noexcept {
    return output_offset(group)+image-offsets_[group]+
        static_cast<std::size_t>(channel)*output_stride(group);
  }

  const int* rhs_strides() const noexcept { return rhs_strides_.data(); }
  int rhs_stride(int group) const noexcept { return rhs_strides_[group]; }
  int begin(int group) const noexcept { return offsets_[group]; }
  int size(int group) const noexcept { return offsets_[group+1]-offsets_[group]; }
  std::size_t offset(int group) const noexcept { return packed_offsets_[group]; }
  std::size_t entries() const noexcept { return packed_offsets_.back(); }
  std::size_t packed(int row,int group) const noexcept {
    return identity_ ? static_cast<std::size_t>(row)*pairs_:
        packed_offsets_[group]+static_cast<std::size_t>(image_rows_[row]-offsets_[group])*channels_[group];
  }
  int image(int row) const noexcept { return identity_ ? row:image_rows_[row]; }
};

/** @brief Multiply without packing; each worker owns a slice of every group.
 * Source and destination must be disjoint. Addition fuses directional forcing
 * into an existing image without materializing a second full-channel image.
 * Exchange weights are fused into BLAS alpha, with a diagonal-row correction.
 */
inline void multiply_channels(const ChannelProduct& basis,const ChannelLayout& layout,
    const double* source,double* destination,int thread,int team,bool add,
    std::span<const int> diagonal_images,int width,bool exchange) {
  const double scale=exchange ? 2.0:1.0;
  for (int group=0;group<basis.groups();++group) {
    const int size=layout.size(group),first=size*thread/team,last=size*(thread+1)/team;
    if (first==last) continue;
    const int channels=basis.channels(group);
    Eigen::Map<const ChannelRows> input(source+layout.offset(group)+static_cast<std::size_t>(first)*channels,
        last-first,channels);
    // Projected images pack each group's actual K[J,J] output. Full images
    // keep their shared count*width stride; neither form repacks coefficients.
    Eigen::Map<Eigen::MatrixXd,0,Eigen::OuterStride<>> output(
        destination+layout.output_offset(group)+first,last-first,basis.kernel(group).rows(),
        Eigen::OuterStride<>(layout.output_stride(group)));
    const auto scale_diagonals=[&](double factor) {
      for (int v=0;v<width;++v) {
        const int shift=v*layout.rhs_stride(group);
        for (const int image:layout.diagonals(diagonal_images,group,first-shift,last-shift))
          output.row(image-layout.begin(group)+shift-first)*=factor;
      }
    };
    // Preserve an existing diagonal image across accumulation: halving
    // 2*old+2*increment gives old+increment. No row GEMM is repeated.
    if (exchange && add) scale_diagonals(2.0);
    if (add) output.noalias()+=scale*input*basis.kernel(group).transpose();
    else output.noalias()=scale*input*basis.kernel(group).transpose();
    if (exchange) scale_diagonals(0.5);
  }
}

/** @brief Sum grouped A^T B using consumed halo scratch, never channel copies.
 * All team members call this function. Large halos hold small per-worker
 * channel Grams; tiny halos instead assign independent global Gram entries.
 * The latter needs no scratch and has no per-entry team synchronization.
 */
inline void contract_channel_gram(const ChannelProduct& basis,const ChannelLayout& layout,
    const double* left,const double* right,Eigen::MatrixXd& result,
    std::span<const int> diagonal_images,int count,int width,bool exchange,
    double* scratch,std::size_t capacity,int thread,int team) {
  const int pairs=basis.channels(0),parts=std::min(team,count*width);
  const auto area=static_cast<std::size_t>(pairs)*pairs;
  if (area<=capacity/(2*parts)) {
    if (thread<parts) {
      Eigen::Map<Eigen::MatrixXd> partial(scratch+2*thread*area,pairs,pairs);
      partial.setZero();
      for (int group=0;group<basis.groups();++group) {
        const int size=layout.size(group),first=size*thread/parts,last=size*(thread+1)/parts;
        if (first==last) continue;
        const int k=basis.channels(group);
        Eigen::Map<const ChannelRows> a(left+layout.offset(group)+static_cast<std::size_t>(first)*k,last-first,k);
        Eigen::Map<const ChannelRows> b(right+layout.offset(group)+static_cast<std::size_t>(first)*k,last-first,k);
        Eigen::Map<Eigen::MatrixXd> gram(scratch+(2*thread+1)*area,k,k);
        gram.noalias()=(exchange ? 2.0:1.0)*a.transpose()*b;
        if (exchange) for (int v=0;v<width;++v) {
          const int shift=v*layout.rhs_stride(group);
          for (const int base:layout.diagonals(diagonal_images,group,first-shift,last-shift)) {
            const int local=base-layout.begin(group)+shift-first;
            gram.noalias()-=a.row(local).transpose()*b.row(local);
          }
        }
        if (!group) partial+=gram;
        else for (int j=0;j<k;++j) for (int i=0;i<k;++i)
          partial(basis.global(group,i),basis.global(group,j))+=gram(i,j);
      }
    }
#pragma omp barrier
#pragma omp for schedule(static)
    for (int entry=0;entry<pairs*pairs;++entry) {
      double value=0;
      for (int part=0;part<parts;++part) value+=scratch[2*part*area+entry];
      result.data()[entry]+=value;
    }
  } else {
#pragma omp for schedule(static)
    for (int entry=0;entry<pairs*pairs;++entry) {
      const int row=entry%pairs,col=entry/pairs;
      double value=0;
      for (int group=0;group<basis.groups();++group) {
        const int size=layout.size(group);
        if (!size) continue;
        const int i=basis.local(group,row),j=basis.local(group,col),k=basis.channels(group);
        if (i<0 || j<0) continue;
        using Vector=Eigen::Map<const Eigen::VectorXd,0,Eigen::InnerStride<>>;
        const Vector a(left+layout.offset(group)+i,size,Eigen::InnerStride<>(k));
        const Vector b(right+layout.offset(group)+j,size,Eigen::InnerStride<>(k));
        value+=(exchange ? 2.0:1.0)*a.dot(b);
        if (exchange) for (int v=0;v<width;++v)
          for (const int base:layout.diagonals(diagonal_images,group,0,layout.rhs_stride(group))) {
            const int image=base-layout.begin(group)+v*layout.rhs_stride(group);
            value-=a[image]*b[image];
          }
      }
      result.data()[entry]+=value;
    }
  }
}

} // namespace xmvb::mcscf::detail
