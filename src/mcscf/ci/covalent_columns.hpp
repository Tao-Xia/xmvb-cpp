#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xmvb::mcscf {

/** @brief Shared prefix DAG for complete-covalent support and one-body closure.
 * Nodes record empty/occupied edges and complete colex intervals, not emitted
 * products. Membership-only views may have incomplete spin lists; they do not
 * prepare this generator or expand a full spin universe.
 */
class CovalentColumns {
    enum Edge : std::uint8_t { Empty = 1, Occupied = 2, Complete = 4 };

  public:
    CovalentColumns(int n,
                    const std::vector<std::uint64_t>& alpha,
                    const std::vector<std::uint64_t>& beta)
        : n_(n), electrons_(std::popcount(alpha.front())), stride_(electrons_ + 1),
          choose_((n_ + 1) * stride_, 0) {
        for (int k = 0; k <= n_; ++k) {
            choose_[k * stride_] = 1;
            for (int e = 1; e <= std::min(k, electrons_); ++e)
                choose_[k * stride_ + e] =
                    choose_[(k - 1) * stride_ + e - 1] + choose_[(k - 1) * stride_ + e];
        }
        if (choose_[n_ * stride_ + electrons_] != alpha.size())
            return;
        alpha_order_.resize(alpha.size());
        branches_.resize(2 * beta.size() * choose_.size(), 0);
        for (int i = 0; i < static_cast<int>(alpha.size()); ++i) {
            std::size_t rank = 0;
            int e = 0;
            for (auto mask = alpha[i]; mask; mask &= mask - 1)
                rank += choose_[std::countr_zero(mask) * stride_ + (++e)];
            alpha_order_[rank] = i;
        }
        ordered_ = std::is_sorted(alpha_order_.begin(), alpha_order_.end());
        std::vector<std::uint64_t> counts(choose_.size());
        for (std::size_t b = 0; b < beta.size(); ++b)
            for (int depth = 0; depth <= 1; ++depth) {
                std::fill(counts.begin(), counts.end(), 0);
                counts[0] = 1;
                auto* nodes = branches_.data() + (2 * b + depth) * choose_.size();
                nodes[0] = Complete;
                int beta_count = 0;
                for (int k = 1; k <= n_; ++k) {
                    beta_count += (beta[b] >> (k - 1)) & 1;
                    for (int e = 0; e <= std::min(k, electrons_); ++e) {
                        if (e + beta_count + depth < k)
                            continue;
                        const auto zero = counts[(k - 1) * stride_ + e];
                        const auto one = e ? counts[(k - 1) * stride_ + e - 1] : 0;
                        counts[k * stride_ + e] = zero + one;
                        nodes[k * stride_ + e] =
                            (zero ? Empty : 0) | (one ? Occupied : 0) |
                            (zero + one == choose_[k * stride_ + e] ? Complete : 0);
                    }
                }
            }
    }

    /** @brief Append sorted original-alpha indices below an exclusive caller bound.
   * Persistent storage is O(U_beta*n_act*(n_alpha_electrons+1)+U_alpha).
   * Generating columns requires a complete alpha carrier; prefix membership
   * queries on a sampled carrier remain independent of this operation.
   */
    void append(int beta,
                std::vector<int>& rows,
                int depth = 0,
                std::size_t upper = std::numeric_limits<std::size_t>::max()) const {
        if (depth != 0 && depth != 1)
            throw std::invalid_argument("covalent column depth must be zero or one");
        if (alpha_order_.empty())
            throw std::logic_error("covalent column generation requires complete alpha strings");
        upper = std::min(upper, alpha_order_.size());
        if (!upper)
            return;
        const auto* nodes =
            branches_.data() + (2 * static_cast<std::size_t>(beta) + depth) * choose_.size();
        // A one-column bitset restores a permuted caller order without sorting.
        std::vector<std::uint64_t> indices(ordered_ ? 0 : (upper + 63) / 64, 0);
        const auto visit = [&](auto&& self, int k, int e, std::size_t rank) -> void {
            // A colex subtree is a contiguous rank interval. Only an identity caller
            // permutation lets us prune it using the caller's exclusive index bound.
            if (ordered_ && rank >= upper)
                return;
            const auto edges = nodes[k * stride_ + e];
            if (edges & Complete) {
                const auto count = choose_[k * stride_ + e];
                if (ordered_) {
                    const auto end = std::min(rank + count, upper);
                    for (auto i = rank; i < end; ++i)
                        rows.push_back(i);
                } else
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto index = alpha_order_[rank + i];
                        if (static_cast<std::size_t>(index) < upper)
                            indices[index / 64] |= std::uint64_t{1} << (index % 64);
                    }
                return;
            }
            if (edges & Empty)
                self(self, k - 1, e, rank);
            if (edges & Occupied)
                self(self, k - 1, e - 1, rank + choose_[(k - 1) * stride_ + e]);
        };
        visit(visit, n_, electrons_, 0);
        for (std::size_t w = 0; w < indices.size(); ++w)
            for (auto bits = indices[w]; bits; bits &= bits - 1)
                rows.push_back(64 * w + std::countr_zero(bits));
    }

    std::size_t dynamic_bytes() const noexcept {
        return choose_.capacity() * sizeof(std::uint64_t) +
               branches_.capacity() * sizeof(std::uint8_t) + alpha_order_.capacity() * sizeof(int);
    }

  private:
    int n_, electrons_, stride_;
    std::vector<std::uint64_t> choose_;
    std::vector<std::uint8_t> branches_;
    std::vector<int> alpha_order_;
    bool ordered_ = false;
};

} // namespace xmvb::mcscf
