#pragma once

#include <algorithm>
#include <span>
#include <vector>
#include <Eigen/Core>
#include "core/workspace.hpp"

namespace xmvb::mcscf::detail {

using ChannelRows=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

/** @brief Shared exact source-channel selectors and corresponding K columns.
 * Group zero is unrestricted. Group 1+first*n+last permits packed pairs
 * (p,q) with p<=first<=last<q. Output channels are never restricted here.
 */
class ChannelSelectors {
  struct Block { std::vector<int> columns,inverse; };
  int pairs_;
  std::vector<Block> blocks_;
public:
  ChannelSelectors(int pairs,int orbitals,bool restricted):pairs_(pairs) {
    blocks_.resize(restricted ? 1+orbitals*orbitals:1);
    if (!restricted) return;
    for (int first=0;first<orbitals;++first) for (int last=first;last<orbitals-1;++last) {
      auto& block=blocks_[1+first*orbitals+last];
      block.inverse.assign(pairs,-1);
      for (int q=last+1;q<orbitals;++q) for (int p=0;p<=first;++p) {
        const int pair=q*(q+1)/2+p;
        block.inverse[pair]=block.columns.size();
        block.columns.push_back(pair);
      }
    }
  }

  int groups() const noexcept { return blocks_.size(); }
  int channels(int group) const noexcept { return group ? blocks_[group].columns.size():pairs_; }
  int local(int group,int pair) const noexcept { return group ? blocks_[group].inverse[pair]:pair; }
  int global(int group,int channel) const noexcept { return group ? blocks_[group].columns[channel]:channel; }
};

/** @brief Host numerical panels on the shared symbolic channel selectors. */
class ChannelProduct : public ChannelSelectors {
  const Eigen::MatrixXd& kernel_;
  workspace::Temporary temporary_;
  std::vector<Eigen::MatrixXd> kernels_;
public:
  ChannelProduct(const Eigen::MatrixXd& kernel,int orbitals,bool restricted)
      :ChannelSelectors(kernel.rows(),orbitals,restricted),kernel_(kernel),kernels_(groups()) {
    for (int group=1;group<groups();++group) {
      auto& panel=kernels_[group];
      panel.resize(kernel.rows(),channels(group));
      for (int k=0;k<channels(group);++k) panel.col(k)=kernel.col(global(group,k));
    }
    std::size_t bytes=0;
    for (const auto& panel:kernels_) bytes+=sizeof(double)*panel.size();
    temporary_.resize(bytes);
  }
  const Eigen::MatrixXd& kernel(int group) const noexcept { return group ? kernels_[group]:kernel_; }
};

/** @brief Ragged row-major sources and full-channel column-major images in group order.
 * The row maps are shared by every response. Numerical panels are never
 * copied to rearrange rows. Unrestricted tiles use implicit identity maps.
 */
class ChannelLayout {
  std::vector<int> offsets_,image_rows_,channels_;
  std::vector<std::size_t> packed_offsets_;
  bool identity_=true;
  int pairs_=0;
public:
  void reset(const ChannelSelectors& basis,std::span<const int> groups,int count,int width) {
    pairs_=basis.channels(0);
    offsets_.assign(basis.groups()+1,0);
    packed_offsets_.assign(basis.groups()+1,0);
    channels_.resize(basis.groups());
    identity_=groups.empty() || std::all_of(groups.begin(),groups.end(),[](int g) { return g==0; });
    if (identity_) offsets_[1]=count*width;
    else for (const int group:groups) offsets_[group+1]+=width;
    for (int group=0;group<basis.groups();++group) {
      offsets_[group+1]+=offsets_[group];
      channels_[group]=basis.channels(group);
      packed_offsets_[group+1]=packed_offsets_[group]+
          static_cast<std::size_t>(offsets_[group+1]-offsets_[group])*channels_[group];
    }
    if (identity_) { image_rows_.clear(); return; }
    image_rows_.resize(count*width);
    auto position=offsets_;
    for (int row=0;row<count*width;++row) {
      const int group=groups[row%count],image=position[group]++;
      image_rows_[row]=image;
    }
  }
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
    std::span<const int> diagonal_rows,int count,int width,bool exchange) {
  const int pairs=basis.channels(0);
  const double scale=exchange ? 2.0:1.0;
  for (int group=0;group<basis.groups();++group) {
    const int size=layout.size(group),first=size*thread/team,last=size*(thread+1)/team;
    if (first==last) continue;
    const int channels=basis.channels(group);
    Eigen::Map<const ChannelRows> input(source+layout.offset(group)+static_cast<std::size_t>(first)*channels,
        last-first,channels);
    // The full image is column-major with leading dimension count*width.
    // A fixed output channel is then contiguous across all image rows.
    Eigen::Map<Eigen::MatrixXd,0,Eigen::OuterStride<>> output(
        destination+layout.begin(group)+first,last-first,pairs,
        Eigen::OuterStride<>(count*width));
    if (add) output.noalias()+=scale*input*basis.kernel(group).transpose();
    else output.noalias()=scale*input*basis.kernel(group).transpose();
    // Exchange orbits have weight two except on alpha=beta. Correct only
    // those rows; a separate read/write pass over every full image is avoided.
    if (exchange) for (int v=0;v<width;++v) for (const int row:diagonal_rows) {
      const int local=layout.image(v*count+row)-layout.begin(group)-first;
      if (local>=0 && local<last-first)
        output.row(local).noalias()-=input.row(local)*basis.kernel(group).transpose();
    }
  }
}

/** @brief Sum grouped A^T B using consumed halo scratch, never channel copies.
 * All team members call this function. Large halos hold small per-worker
 * channel Grams; tiny halos instead assign independent global Gram entries.
 * The latter needs no scratch and has no per-entry team synchronization.
 */
inline void contract_channel_gram(const ChannelProduct& basis,const ChannelLayout& layout,
    const double* left,const double* right,Eigen::MatrixXd& result,
    std::span<const int> diagonal_rows,int count,int width,bool exchange,
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
        if (exchange) for (int v=0;v<width;++v) for (const int local:diagonal_rows) {
          const int image=layout.image(v*count+local)-layout.begin(group);
          if (image>=first && image<last)
            gram.noalias()-=a.row(image-first).transpose()*b.row(image-first);
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
        if (exchange) for (int v=0;v<width;++v) for (const int local:diagonal_rows) {
          const int image=layout.image(v*count+local)-layout.begin(group);
          if (image>=0 && image<size) value-=a[image]*b[image];
        }
      }
      result.data()[entry]+=value;
    }
  }
}

} // namespace xmvb::mcscf::detail
