#include "mcscf/ci/strings.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace xmvb::mcscf {

std::size_t string_count(int n, int electrons) {
  if (n<0 || electrons<0 || electrons>n) return 0;
  std::size_t count = 1;
  const int k = std::min(electrons, n - electrons);
  for (int i = 1; i <= k; ++i) {
    std::size_t factor=n-k+i,divisor=i;
    const auto common=std::gcd(factor,divisor);
    factor/=common;divisor/=common;
    // Cancel before multiplication: a sparse sector may have a large, but
    // representable, universe that must never be allocated to index its masks.
    count/=divisor;
    if (count>std::numeric_limits<std::size_t>::max()/factor)
      throw std::length_error("spin-string count overflows");
    count*=factor;
  }
  return count;
}

namespace {
void validate_sector(int n,int electrons) {
  if (n<1 || n>63 || electrons<0 || electrons>n)
    throw std::invalid_argument("invalid fixed-particle spin space");
}
std::vector<std::uint64_t> encode(int n,const std::vector<std::vector<int>>& rows) {
  if (n<1 || n>63) throw std::invalid_argument("invalid spin orbital count");
  std::vector<std::uint64_t> masks;
  masks.reserve(rows.size());
  for (const auto& row:rows) {
    std::uint64_t mask=0;
    int previous=-1;
    for (const int p:row) {
      if (p<=previous || p<0 || p>=n)
        throw std::invalid_argument("invalid spin occupation");
      mask|=std::uint64_t{1}<<p;previous=p;
    }
    masks.push_back(mask);
  }
  return masks;
}
} // namespace

struct SpinStrings::Data {
  int n,electrons;
  std::vector<std::uint64_t> masks,choose,ranks;
  bool complete;
  std::vector<int> indices;
  Data(int orbitals,std::span<const std::uint64_t> input)
      :n(orbitals),electrons(input.empty()?0:std::popcount(input.front())),
       masks(input.begin(),input.end()) {
    validate_sector(n,electrons);
    if (masks.empty() || masks.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid spin-string count");
    complete=masks.size()==string_count(n,electrons);
    choose.assign((n+1)*(electrons+1),0);
    for (int p=0;p<=n;++p) {
      choose[p*(electrons+1)]=1;
      for (int k=1;k<=std::min(p,electrons);++k)
        choose[p*(electrons+1)+k]=choose[(p-1)*(electrons+1)+k-1]+
            choose[(p-1)*(electrons+1)+k];
    }
    const auto full=(std::uint64_t{1}<<n)-1;
    bool canonical=complete;
    for (std::size_t i=0;i<masks.size();++i) {
      if ((masks[i]&~full) || std::popcount(masks[i])!=electrons)
        throw std::invalid_argument("invalid fixed-particle spin mask");
      canonical&=rank(masks[i])==i;
    }
    if (complete && !canonical) {
      indices.assign(masks.size(),-1);
      for (int i=0;i<static_cast<int>(masks.size());++i) {
        auto& index=indices[rank(masks[i])];
        if (index>=0) throw std::invalid_argument("duplicate spin string");
        index=i;
      }
    } else if (!complete) {
      // A sparse spin sector never allocates a full-universe rank table.
      std::vector<std::pair<std::uint64_t,int>> sorted;
      sorted.reserve(masks.size());
      for (int i=0;i<static_cast<int>(masks.size());++i) sorted.emplace_back(rank(masks[i]),i);
      std::sort(sorted.begin(),sorted.end());
      ranks.reserve(sorted.size());indices.reserve(sorted.size());
      for (const auto& [r,i]:sorted) {
        if (!ranks.empty() && ranks.back()==r) throw std::invalid_argument("duplicate spin string");
        ranks.push_back(r);indices.push_back(i);
      }
    }
  }

  std::size_t rank(std::uint64_t mask) const noexcept {
    std::size_t result=0;
    int k=1;
    while (mask) {
      const int p=std::countr_zero(mask);
      result+=choose[p*(electrons+1)+k++];
      mask&=mask-1;
    }
    return result;
  }
};

SpinStrings::SpinStrings(int n,int electrons) {
  validate_sector(n,electrons);
  const auto count=string_count(n,electrons);
  if (count>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("spin-string count exceeds index range");
  std::vector<std::uint64_t> masks;
  masks.reserve(count);
  std::uint64_t mask=electrons ? (std::uint64_t{1}<<electrons)-1:0;
  for (std::size_t i=0;i<count;++i) {
    masks.push_back(mask);
    if (i+1==count) break;
    const auto low=mask&-mask,next=mask+low;
    mask=next|(((next^mask)>>2)/low);
  }
  data_=std::make_shared<const Data>(n,masks);
}
SpinStrings::SpinStrings(int n,std::span<const std::uint64_t> masks)
    :data_(std::make_shared<const Data>(n,masks)) {}
SpinStrings::SpinStrings(int n,const std::vector<std::vector<int>>& rows)
    :SpinStrings(n,encode(n,rows)) {}
const std::vector<std::uint64_t>& SpinStrings::masks() const noexcept {
  static const std::vector<std::uint64_t> empty;
  return data_ ? data_->masks:empty;
}
int SpinStrings::size() const noexcept { return masks().size(); }
int SpinStrings::orbitals() const noexcept { return data_ ? data_->n:0; }
int SpinStrings::electrons() const noexcept { return data_ ? data_->electrons:0; }
bool SpinStrings::complete() const noexcept { return data_ && data_->complete; }
int SpinStrings::index(std::uint64_t mask) const {
  if (!data_ || (mask>>data_->n) || std::popcount(mask)!=data_->electrons)
    throw std::out_of_range("spin mask is outside its sector");
  const auto rank=data_->rank(mask);
  if (data_->complete)
    return data_->indices.empty() ? static_cast<int>(rank):data_->indices[rank];
  const auto found=std::lower_bound(data_->ranks.begin(),data_->ranks.end(),rank);
  if (found==data_->ranks.end() || *found!=rank)
    throw std::out_of_range("spin mask is absent from the sparse sector");
  return data_->indices[found-data_->ranks.begin()];
}
std::vector<std::vector<int>> SpinStrings::decode() const {
  std::vector<std::vector<int>> rows(size());
  for (int i=0;i<size();++i) {
    rows[i].reserve(electrons());
    for (const int p:occupied(i)) rows[i].push_back(p);
  }
  return rows;
}
std::size_t SpinStrings::dynamic_bytes() const noexcept {
  return data_ ? sizeof(Data)+sizeof(std::uint64_t)*(data_->masks.capacity()+data_->choose.capacity()+data_->ranks.capacity())+
      sizeof(int)*data_->indices.capacity():0;
}

} // namespace xmvb::mcscf
