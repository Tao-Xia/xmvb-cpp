#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mcscf/ci/covalent_support.hpp"
#include "mcscf/ci/product_paths.hpp"
#include "mcscf/ci/product_index.hpp"

namespace xmvb::mcscf {

/** @brief Index adapter for exact selected-product support and its halo.
 *
 * ProductPaths generates occupation masks without a product-space table.
 * These per-spin maps translate each generated mask to the caller's existing
 * spin-string index. The supplied spin-string lists must contain every mask
 * generated in their fixed-electron sectors.
 */
class ProductSupport {
public:
  using Mask = ProductPaths::Mask;
  using Seed = ProductPaths::Seed;

  /** @brief Complete singly occupied family, without materialized seeds. */
  ProductSupport(int n_orbitals, const std::vector<Mask>& alpha,
                 const std::vector<Mask>& beta)
      : n_(n_orbitals), alpha_(alpha), beta_(beta), complete_covalent_(true) {
    initialize_strings();
    if (na_ + nb_ != n_)
      throw std::invalid_argument("covalent product support requires half filling");
    covalent_.emplace(n_, alpha_, beta_);
    paths_.emplace(n_, na_, nb_, std::vector<Seed>{}, true);
    prepare_columns();
  }

  /** @brief Seeds are flattened alpha*n_beta+beta indices, not CI amplitudes. */
  ProductSupport(int n_orbitals, const std::vector<Mask>& alpha,
                 const std::vector<Mask>& beta, const std::vector<ProductIndex>& products)
      : n_(n_orbitals), alpha_(alpha), beta_(beta) {
    initialize_strings();
    const Mask full = (Mask{1} << n_) - 1;
    complete_covalent_ = products.size() == alpha_.size() &&
                         products.size() == beta_.size();
    std::vector<Seed> seeds;
    seeds.reserve(products.size());
    for (const ProductIndex product : products) {
      if (product < 0 || static_cast<std::size_t>(product) >=
                             alpha_.size() * beta_.size())
        throw std::invalid_argument("product support seed is outside the spin spaces");
      const auto a = alpha_[product / beta_.size()];
      const auto b = beta_[product % beta_.size()];
      seeds.emplace_back(a, b);
      full_ |= a == highest(na_) && b == highest(nb_);
      complete_covalent_ &= (a & b) == 0 && (a | b) == full;
    }
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    complete_covalent_ &= na_ + nb_ == n_ && seeds.size() == alpha_.size();
    if (complete_covalent_) covalent_.emplace(n_, alpha_, beta_);
    if (!full_ && !complete_covalent_) seeds_ = seeds;
    paths_.emplace(n_, na_, nb_, std::move(seeds), complete_covalent_);
    prepare_columns();
  }

  /** @brief Pulls products as indices in the original spin-string lists. */
  class Cursor {
  public:
    Cursor(const ProductSupport& support, ProductPaths::Cursor paths)
        : support_(&support), paths_(std::move(paths)) {}

    bool next(int& alpha, int& beta, bool& in_support) {
      Mask a, b;
      if (!paths_.next(a, b, in_support)) return false;
      const auto ai = support_->alpha_index_.find(a);
      if (ai == support_->alpha_index_.end())
        throw std::logic_error("generated product is absent from the spin strings");
      alpha = ai->second;
      beta = beta_index(b);
      return true;
    }

    /** @brief Emit each reachable beta index without generating alpha products.
     * This traversal requires its own cursor, separate from next().
     */
    bool next_column(int& beta) {
      Mask b;
      if (!paths_.next_column(b)) return false;
      beta = beta_index(b);
      return true;
    }

    ProductPaths::Counts counts() const noexcept { return paths_.counts(); }

  private:
    int beta_index(Mask mask) {
      if (!cached_beta_ || *cached_beta_ != mask) {
        const auto found = support_->beta_index_.find(mask);
        if (found == support_->beta_index_.end())
          throw std::logic_error("generated product is absent from the spin strings");
        cached_beta_ = mask;
        cached_beta_index_ = found->second;
      }
      return cached_beta_index_;
    }

    const ProductSupport* support_;
    ProductPaths::Cursor paths_;
    std::optional<Mask> cached_beta_;
    int cached_beta_index_ = -1;
  };

  Cursor cursor(int depth) const { return Cursor(*this, paths_->cursor(depth)); }

  Cursor column(int beta_index, int depth) const {
    if (beta_index < 0 || static_cast<std::size_t>(beta_index) >= beta_.size())
      throw std::invalid_argument("product support beta index is outside the spin space");
    return Cursor(*this, paths_->column(beta_[beta_index], depth));
  }

  bool full() const noexcept { return full_; }

  /** @brief Reachable depth-zero beta columns in caller index order.
   * Built once with the immutable support, shared by CI and derivative tiles.
   */
  const std::vector<int>& columns() const noexcept { return columns_; }

  /** @brief Append support or one-generator closure in sorted caller index order.
   * Complete covalent support uses its shared prefix DAG. General selections
   * use their maximal-generator paths; neither stores a product-space table.
   */
  void append_column_rows(int beta,std::vector<int>& rows,int depth=0,
      std::size_t upper=std::numeric_limits<std::size_t>::max()) const {
    if (beta<0 || static_cast<std::size_t>(beta)>=beta_.size())
      throw std::invalid_argument("product support column is outside the spin space");
    if (covalent_) {
      covalent_->append_column_rows(beta,rows,depth,upper);
      return;
    }
    const auto begin=rows.size();
    auto cursor=column(beta,depth);
    int a,b; bool label;
    while (cursor.next(a,b,label))
      if (static_cast<std::size_t>(a)<upper) rows.push_back(a);
    std::sort(rows.begin()+begin,rows.end());
  }

  /** @brief Certified frontier channel envelope, or unrestricted {-1,-1}.
   * The covalent prefix certificate applies to the complete singly occupied
   * seed family, not to arbitrary selected structures. Other families retain
   * all channels; no unsupported closure assumption is made.
   */
  std::pair<int,int> channel_envelope(int alpha,int beta) const noexcept {
    return covalent_ ? covalent_->deficient_prefixes(alpha,beta):std::pair{-1,-1};
  }
  bool has_channel_envelopes() const noexcept { return covalent_.has_value(); }
  int n_orbitals() const noexcept { return n_; }
  /** Immutable support identity, shared by copies without retaining the support
   * itself. Prepared schedules use it rather than borrowed object addresses. */
  const std::shared_ptr<const char>& identity() const noexcept { return identity_; }

  /** @brief Small-query membership test; traversal should use cursor/column. */
  bool contains(int alpha, int beta, int depth = 0) const noexcept {
    if (alpha < 0 || beta < 0 || static_cast<std::size_t>(alpha) >= alpha_.size() ||
        static_cast<std::size_t>(beta) >= beta_.size() || depth < 0 || depth > 1)
      return false;
    if (full_) return true;
    if (complete_covalent_) return covalent_->contains(alpha, beta, depth);
    const Mask a = alpha_[alpha], b = beta_[beta];
    for (const auto& [sa, sb] : seeds_) {
      const int da = prefix_deficit(a, sa, depth);
      if (da <= depth && prefix_deficit(b, sb, depth - da) <= depth - da)
        return true;
    }
    return false;
  }

  std::size_t dynamic_bytes() const noexcept {
    return columns_.capacity()*sizeof(int)+(alpha_.capacity() + beta_.capacity()) * sizeof(Mask) +
        seeds_.capacity() * sizeof(Seed) +
        (alpha_index_.bucket_count() + beta_index_.bucket_count()) * sizeof(void*) +
        (alpha_index_.size() + beta_index_.size()) *
            sizeof(std::pair<const Mask, int>) +
        (paths_ ? paths_->metadata_bytes() : 0) +
        (covalent_ ? covalent_->dynamic_bytes() : 0);
  }

private:
  std::shared_ptr<const char> identity_=std::make_shared<const char>(0);
  void prepare_columns() {
    if (full_ || complete_covalent_) {
      columns_.resize(beta_.size());
      std::iota(columns_.begin(),columns_.end(),0);
    } else {
      auto c=cursor(0); int b;
      while (c.next_column(b)) columns_.push_back(b);
      std::sort(columns_.begin(),columns_.end());
    }
  }
  Mask highest(int electrons) const noexcept {
    const Mask full = (Mask{1} << n_) - 1;
    return full ^ ((Mask{1} << (n_ - electrons)) - 1);
  }

  /** @brief Maximum shortfall of target prefix counts against one seed. */
  int prefix_deficit(Mask target, Mask seed, int limit) const noexcept {
    int difference = 0, deficit = 0;
    for (int k = 0; k < n_; ++k) {
      difference += static_cast<int>((seed >> k) & 1) -
                    static_cast<int>((target >> k) & 1);
      deficit = std::max(deficit, difference);
      if (deficit > limit) break;
    }
    return deficit;
  }

  void initialize_strings() {
    if (n_ < 1 || n_ > 63 || alpha_.empty() || beta_.empty())
      throw std::invalid_argument("invalid product support spin sector");
    const Mask full = (Mask{1} << n_) - 1;
    na_ = __builtin_popcountll(alpha_.front());
    nb_ = __builtin_popcountll(beta_.front());
    alpha_index_.reserve(alpha_.size());
    beta_index_.reserve(beta_.size());
    for (std::size_t i = 0; i < alpha_.size(); ++i)
      if ((alpha_[i] & ~full) || __builtin_popcountll(alpha_[i]) != na_ ||
          !alpha_index_.emplace(alpha_[i], static_cast<int>(i)).second)
        throw std::invalid_argument("invalid product support alpha strings");
    for (std::size_t i = 0; i < beta_.size(); ++i)
      if ((beta_[i] & ~full) || __builtin_popcountll(beta_[i]) != nb_ ||
          !beta_index_.emplace(beta_[i], static_cast<int>(i)).second)
        throw std::invalid_argument("invalid product support beta strings");
  }

  int n_, na_ = 0, nb_ = 0;
  bool complete_covalent_ = false, full_ = false;
  std::vector<Mask> alpha_, beta_;
  std::vector<int> columns_;
  std::unordered_map<Mask, int> alpha_index_, beta_index_;
  std::vector<Seed> seeds_;
  std::optional<CovalentSupport> covalent_;
  std::optional<ProductPaths> paths_;
};

} // namespace xmvb::mcscf
