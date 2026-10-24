#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "mcscf/ci/links.hpp"
#include "mcscf/ci/product_support.hpp"

namespace xmvb::mcscf::detail {

/** @brief Numeric halo, channel and target-index budget; immutable spin links are shared. */
struct SigmaTileBudget {
  int width,pairs,slabs;
  std::size_t bytes;
  int packed_slabs=0;
  int halo_slabs=1;
  /** Consumer panel extent; zero leaves only resource-byte admission. */
  int max_groups=0;
  /** Tile-local channel addresses for a source-supported image. */
  bool projected_image=false;
  bool operator==(const SigmaTileBudget&) const=default;
};

/** Exact target boundary for the same support, links and source layout.
 * Callers certify that the replayed tile fits their current storage budget. */
struct SigmaTileBoundary {
  int targets;
  std::size_t payload;
  bool full_alpha,full_beta;
};

struct SigmaTargetGroup { int beta,begin,end,column=0; };

/** Current bounded tile only; contains no cursors or borrowed support objects. */
struct SigmaTile {
  std::vector<SigmaTargetGroup> groups;
  std::vector<int> active,group_index,halo,columns,halo_index,column_index,released;
  std::vector<unsigned char> active_support;
  std::vector<int> channel_groups,diagonal_rows;
  std::size_t unsupported_targets=0,bulk_targets=0;
};

/** @brief Stream each target once, admitting its actual shared neighborhoods.
 * One pending target crosses a budget boundary without rewinding the support
 * cursor. Storage is single-spin metadata plus the current tile, never U^2.
 */
class SigmaTilePlan : public SigmaTile {
public:
  SigmaTilePlan(const SpinConnections& alpha,const SpinConnections& beta,
      const ProductSupport* support,int parity,SigmaTileBudget budget,bool release);
  bool next();
  bool next_known(const SigmaTileBoundary& boundary);
  SigmaTileBoundary boundary() const noexcept;
  std::size_t payload_bytes() const;
  /** The most recent byte-admission pass stopped at its resource bound. */
  bool byte_limited() const noexcept { return byte_limited_; }
  bool exhausted() const noexcept { return !pending_ && next_beta_==beta_targets_.size(); }

private:
  struct Node { int alpha,beta; bool supported; int group,channels; };
  bool prepare_column();
  Node column_node(int row) const;
  bool pull(Node& node);
  void reset_tile();
  void append_node(const Node& node,bool new_group,bool expand_alpha);
  bool append_column_tail(const Node& first);
  bool finish_tile(bool sorted_halo=false,bool sorted_columns=false);
  const SpinConnections& alpha_;
  const SpinConnections& beta_;
  const ProductSupport* support_;
  int parity_;
  SigmaTileBudget budget_;
  std::vector<int> beta_targets_,seen_alpha_,new_alpha_,new_beta_,last_use_,release_order_;
  std::vector<int> column_rows_;
  std::vector<unsigned char> column_support_;
  bool column_ready_=false,byte_limited_=false;
  std::vector<unsigned char> alpha_seen_;
  std::optional<Node> pending_;
  std::size_t next_beta_=0,next_release_=0;
  // Byte admission is integer arithmetic. A rejected candidate may exceed
  // size_t; 128 bits cover the products of the int-bounded tile dimensions.
  using Bytes=__uint128_t;
  Bytes row_bytes_=0,entry_bytes_=0,column_bytes_=0,group_bytes_=0,payload_=0;
  int next_alpha_=0;
};

} // namespace xmvb::mcscf::detail
