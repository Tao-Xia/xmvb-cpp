#include "mcscf/ci/sigma_plan.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "mcscf/ci/sigma_gather.hpp"

namespace xmvb::mcscf::detail {

SigmaTilePlan::SigmaTilePlan(const SpinConnections& alpha,const SpinConnections& beta,
    const ProductSupport* support,int parity,SigmaTileBudget budget,bool release)
    : alpha_(alpha),beta_(beta),support_(support),parity_(parity),budget_(budget),
      alpha_seen_(alpha.n_strings(),0) {
  halo_index.assign(alpha.n_strings(),-1);column_index.assign(beta.n_strings(),-1);
  if (budget.width<=0 || budget.pairs<=0 || budget.slabs<=0 ||
      budget.halo_slabs<=0 || !budget.bytes)
    throw std::invalid_argument("invalid sigma tile budget");
  if (budget.packed_slabs<0 || budget.packed_slabs>budget.slabs ||
      (budget.packed_slabs && (!support || !support->has_channel_envelopes())))
    throw std::invalid_argument("packed sigma channels require a certified support");
  const auto width=static_cast<Bytes>(budget.width);
  column_bytes_=sizeof(double)*width*budget.halo_slabs;
  group_bytes_=sizeof(int)*width;
  entry_bytes_=sizeof(double)*width*budget.packed_slabs;
  row_bytes_=sizeof(SigmaGatherRow)+sizeof(double)*width*
      (budget.slabs-budget.packed_slabs)*budget.pairs;
  if (budget.packed_slabs)
    row_bytes_+=sizeof(int)+width*(sizeof(int)+sizeof(std::size_t));
  if (support_ && support_->has_channel_envelopes()) {
    // Every beta column belongs to the complete singly occupied family.
    beta_targets_=support_->columns();
  } else if (support_) {
    auto cursor=support_->cursor(1);
    int b;
    while (cursor.next_column(b)) beta_targets_.push_back(b);
  } else {
    beta_targets_.resize(beta.n_strings());
    std::iota(beta_targets_.begin(),beta_targets_.end(),0);
  }
  if (parity_<0) std::erase(beta_targets_,0);
  new_alpha_.reserve(alpha.n_singles()+1);
  new_beta_.reserve(beta.n_singles()+1);
  if (release) {
    last_use_.assign(beta.n_strings(),-1);
    for (int p=0;p<static_cast<int>(beta_targets_.size());++p) {
      const int b=beta_targets_[p];
      last_use_[b]=p;
      for (const auto& edge:beta.singles(b)) last_use_[edge.source]=p;
    }
    release_order_.resize(beta.n_strings());
    std::iota(release_order_.begin(),release_order_.end(),0);
    std::sort(release_order_.begin(),release_order_.end(),[&](int a,int b) {
      return last_use_[a]<last_use_[b] || (last_use_[a]==last_use_[b] && a<b);
    });
  }
}

std::size_t SigmaTilePlan::payload_bytes() const {
  return static_cast<std::size_t>(payload_);
}

SigmaTileBoundary SigmaTilePlan::boundary() const noexcept {
  return {static_cast<int>(active.size()),payload_bytes(),
      halo.size()==static_cast<std::size_t>(alpha_.n_strings()),
      columns.size()==static_cast<std::size_t>(beta_.n_strings())};
}

bool SigmaTilePlan::prepare_column() {
  while (next_beta_<beta_targets_.size()) {
    const int b=beta_targets_[next_beta_];
    if (!column_ready_) {
      if (support_ && support_->has_channel_envelopes()) {
        const int upper=parity_ ? b+(parity_>0):alpha_.n_strings();
        support_->append_column_rows(b,column_rows_,1,upper);
      } else if (support_) {
        auto cursor=support_->column(b,1);
        int a,ignored; bool label;
        while (cursor.next(a,ignored,label)) {
          if (parity_ && (a>b || (a==b && parity_<0))) continue;
          column_rows_.push_back(a);column_support_.push_back(label);
        }
      }
      column_ready_=true;
    }
    const int size=support_ ? static_cast<int>(column_rows_.size()):
        parity_ ? b+(parity_>0):alpha_.n_strings();
    if (next_alpha_<size) return true;
    column_rows_.clear();column_support_.clear();column_ready_=false;
    next_alpha_=0;++next_beta_;
  }
  return false;
}

SigmaTilePlan::Node SigmaTilePlan::column_node(int row) const {
  const int b=beta_targets_[next_beta_],a=support_ ? column_rows_[row]:row;
  if (support_ && support_->has_channel_envelopes()) {
    if (budget_.packed_slabs) {
      const auto [first,last]=support_->channel_envelope(a,b);
      return {a,b,first<0,first<0 ? 0:1+first*support_->n_orbitals()+last,
          first<0 ? budget_.pairs:(first+1)*(support_->n_orbitals()-1-last)};
    }
    return {a,b,support_->contains(a,b),0,budget_.pairs};
  }
  return {a,b,!support_ || column_support_[row]!=0,0,budget_.pairs};
}

bool SigmaTilePlan::pull(Node& node) {
  if (pending_) { node=*pending_; pending_.reset(); return true; }
  if (!prepare_column()) return false;
  node=column_node(next_alpha_++);
  return true;
}

void SigmaTilePlan::reset_tile() {
  for (const int a:halo) halo_index[a]=-1;
  for (const int b:columns) column_index[b]=-1;
  for (const int a:seen_alpha_) alpha_seen_[a]=0;
  groups.clear(); active.clear(); group_index.clear(); halo.clear(); columns.clear();
  active_support.clear(); seen_alpha_.clear(); released.clear(); channel_groups.clear(); payload_=0;
  diagonal_rows.clear(); unsupported_targets=0;
}

void SigmaTilePlan::append_node(const Node& node,bool new_group,bool expand_alpha) {
  if (expand_alpha) { alpha_seen_[node.alpha]=1; seen_alpha_.push_back(node.alpha); }
  for (const int a:new_alpha_) { halo_index[a]=0; halo.push_back(a); }
  for (const int b:new_beta_) { column_index[b]=0; columns.push_back(b); }
  if (new_group) groups.push_back({node.beta,static_cast<int>(active.size()),static_cast<int>(active.size())});
  if (parity_ && node.alpha==node.beta) diagonal_rows.push_back(active.size());
  unsupported_targets+=!node.supported;
  active.push_back(node.alpha);
  active_support.push_back(node.supported);
  if (budget_.packed_slabs) channel_groups.push_back(node.group);
  group_index.push_back(groups.size()-1);
  groups.back().end=active.size();
}

bool SigmaTilePlan::finish_tile(bool sorted_halo,bool sorted_columns) {
  while (next_release_<release_order_.size() &&
      last_use_[release_order_[next_release_]]<static_cast<int>(next_beta_))
    released.push_back(release_order_[next_release_++]);
  if (!sorted_halo) std::sort(halo.begin(),halo.end());
  if (!sorted_columns) std::sort(columns.begin(),columns.end());
  for (int i=0;i<static_cast<int>(halo.size());++i) halo_index[halo[i]]=i;
  for (int i=0;i<static_cast<int>(columns.size());++i) column_index[columns[i]]=i;
  return !active.empty();
}

bool SigmaTilePlan::next() {
  reset_tile();
  Node node;
  while (pull(node)) {
    new_alpha_.clear(); new_beta_.clear();
    const bool new_group=groups.empty() || groups.back().beta!=node.beta;
    const auto include=[](int id,const std::vector<int>& index,std::vector<int>& added) {
      if (index[id]<0) added.push_back(id);
    };
    // Once a halo is the complete single-spin sector, every further neighbor
    // is already present. This is set saturation, not selected-space closure.
    const bool expand_alpha=halo.size()<static_cast<std::size_t>(alpha_.n_strings()) &&
        !alpha_seen_[node.alpha];
    if (expand_alpha) {
      include(node.alpha,halo_index,new_alpha_);
      for (const auto& edge:alpha_.singles(node.alpha)) include(edge.source,halo_index,new_alpha_);
    }
    if (new_group && columns.size()<static_cast<std::size_t>(beta_.n_strings())) {
      include(node.beta,column_index,new_beta_);
      for (const auto& edge:beta_.singles(node.beta)) include(edge.source,column_index,new_beta_);
    }
    // B = row_bytes*T + entry_bytes*entries + h*(column_bytes*b+group_bytes*g).
    // Only halo/column/group growth changes the rectangular term. Most nodes
    // add just their row and channel entries; never remeasure the entire tile.
    auto bytes=payload_+row_bytes_+entry_bytes_*node.channels;
    if (!new_alpha_.empty())
      bytes+=new_alpha_.size()*(column_bytes_*(columns.size()+new_beta_.size())+
          group_bytes_*(groups.size()+new_group));
    if (!new_beta_.empty() || new_group)
      bytes+=halo.size()*(column_bytes_*new_beta_.size()+group_bytes_*new_group);
    if (bytes>budget_.bytes || active.size()>=static_cast<std::size_t>(std::numeric_limits<int>::max()/budget_.width)) {
      if (active.empty()) throw std::length_error("one direct-CI target exceeds the tile workspace");
      pending_=node;
      break;
    }
    append_node(node,new_group,expand_alpha);
    payload_=bytes;
  }
  return finish_tile();
}

bool SigmaTilePlan::next_known(const SigmaTileBoundary& boundary) {
  if (boundary.targets<=0) throw std::invalid_argument("invalid sigma tile boundary");
  reset_tile();
  // A rejected/lookahead node is the last row read from the current column.
  // Rewind that one scalar cursor, not the support traversal or a product plan.
  if (pending_) { --next_alpha_;pending_.reset(); }
  active.reserve(boundary.targets);group_index.reserve(boundary.targets);
  active_support.reserve(boundary.targets);
  if (budget_.packed_slabs) channel_groups.reserve(boundary.targets);
  int remaining=boundary.targets;
  while (remaining) {
    if (!prepare_column()) throw std::invalid_argument("sigma tile boundary exceeds target stream");
    const int b=beta_targets_[next_beta_];
    const int size=support_ ? static_cast<int>(column_rows_.size()):
        parity_ ? b+(parity_>0):alpha_.n_strings();
    const int count=std::min(remaining,size-next_alpha_);
    const int begin=active.size(),end=begin+count,group=groups.size();
    // A tile's intersection with one beta column is one contiguous segment.
    // Its group and beta-link neighborhood are constant across all targets.
    groups.push_back({b,begin,end});
    if (!boundary.full_beta && columns.size()<static_cast<std::size_t>(beta_.n_strings())) {
      const auto include=[&](int id) {
        if (column_index[id]<0) { column_index[id]=0;columns.push_back(id); }
      };
      include(b);
      for (const auto& edge:beta_.singles(b)) include(edge.source);
    }
    active.resize(end);group_index.insert(group_index.end(),count,group);
    active_support.resize(end);
    if (budget_.packed_slabs) channel_groups.resize(end);
    for (int i=begin;i<end;++i) {
      const auto node=column_node(next_alpha_++);
      active[i]=node.alpha;active_support[i]=node.supported;
      unsupported_targets+=!node.supported;
      if (budget_.packed_slabs) channel_groups[i]=node.group;
      if (parity_ && node.alpha==b) diagonal_rows.push_back(i);
      if (!boundary.full_alpha && halo.size()<static_cast<std::size_t>(alpha_.n_strings()) &&
          !alpha_seen_[node.alpha]) {
        alpha_seen_[node.alpha]=1;seen_alpha_.push_back(node.alpha);
        const auto include=[&](int id) {
          if (halo_index[id]<0) { halo_index[id]=0;halo.push_back(id); }
        };
        include(node.alpha);
        for (const auto& edge:alpha_.singles(node.alpha)) include(edge.source);
      }
    }
    remaining-=count;
  }
  // Original next() pulls the first rejected target before finalizing a tile.
  // Reproduce that lookahead so both the pending node and beta release cursor
  // match, including when the target stream is exhausted.
  Node node;
  if (pull(node)) pending_=node;
  payload_=boundary.payload;
  if (boundary.full_alpha) {
    halo.resize(alpha_.n_strings());
    std::iota(halo.begin(),halo.end(),0);
  }
  if (boundary.full_beta) {
    columns.resize(beta_.n_strings());
    std::iota(columns.begin(),columns.end(),0);
  }
  return finish_tile(boundary.full_alpha,boundary.full_beta);
}

} // namespace xmvb::mcscf::detail
