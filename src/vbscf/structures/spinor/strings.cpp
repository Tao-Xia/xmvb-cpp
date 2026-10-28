#include "vbscf/structures/spinor/strings.hpp"
#include "mcscf/ci/strings.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <stdexcept>

namespace xmvb::vb::spinor {
namespace {

template <typename Callback> void for_each_mask(int n, int k, Callback&& callback) {
    if (k == 0) {
        callback(std::uint64_t{0});
        return;
    }
    const std::uint64_t limit = std::uint64_t{1} << n;
    std::uint64_t bits = (std::uint64_t{1} << k) - 1;
    while (bits < limit) {
        callback(bits);
        const std::uint64_t low = bits & (~bits + 1);
        const std::uint64_t next_high = bits + low;
        if (next_high >= limit)
            break;
        bits = next_high | (((next_high ^ bits) >> 2) / low);
    }
}

int prefix_parity(std::uint64_t mask, int orbital) {
    const std::uint64_t prefix = (std::uint64_t{1} << orbital) - 1;
    return __builtin_popcountll(mask & prefix) & 1;
}

} // namespace

Strings::Strings(int n_spinor, int n_electrons, int n_a, std::size_t max_states)
    : n_spinor_(n_spinor), n_electrons_(n_electrons), n_a_(n_a) {
    if (n_spinor < 0 || n_spinor > 63 || n_electrons < 0 || n_electrons > n_spinor || n_a < 0 ||
        n_a > n_spinor) {
        throw std::invalid_argument("spinor strings require 0 <= electrons <= spinors <= 63 and "
                                    "0 <= n_a <= n_spinor");
    }
    const std::size_t count = mcscf::string_count(n_spinor, n_electrons);
    if (count > max_states) {
        throw std::length_error("complete spinor carrier exceeds max_states");
    }
    masks_.reserve(count);
    index_by_mask_.reserve(count);
    const int n_b = n_spinor - n_a;
    const int first_a = std::max(0, n_electrons - n_b);
    const int last_a = std::min(n_a, n_electrons);
    for (int occupied_a = first_a; occupied_a <= last_a; ++occupied_a) {
        const int occupied_b = n_electrons - occupied_a;
        for_each_mask(n_a, occupied_a, [&](std::uint64_t a_mask) {
            for_each_mask(n_b, occupied_b, [&](std::uint64_t b_mask) {
                const std::uint64_t mask = a_mask | (b_mask << n_a);
                index_by_mask_.emplace(mask, core::checked_index<Eigen::Index>(masks_.size()));
                masks_.push_back(mask);
            });
        });
    }
    if (masks_.size() != count) {
        throw std::logic_error("incomplete spinor occupation carrier");
    }
}

Eigen::Index Strings::index(std::uint64_t mask) const {
    const auto found = index_by_mask_.find(mask);
    if (found == index_by_mask_.end()) {
        throw std::invalid_argument("occupation mask is absent from spinor carrier");
    }
    return found->second;
}

Eigen::Index Strings::substitution(Eigen::Index source, int created, int removed, int* sign) const {
    if (source < 0 || source >= dimension() || created < 0 || created >= n_spinor_ || removed < 0 ||
        removed >= n_spinor_ || sign == nullptr) {
        throw std::invalid_argument("invalid spinor substitution source or orbital");
    }
    const std::uint64_t source_mask = masks_[source];
    const std::uint64_t removed_bit = std::uint64_t{1} << removed;
    if ((source_mask & removed_bit) == 0)
        return -1;
    const std::uint64_t after_removed = source_mask ^ removed_bit;
    const std::uint64_t created_bit = std::uint64_t{1} << created;
    if ((after_removed & created_bit) != 0)
        return -1;
    *sign =
        (prefix_parity(source_mask, removed) ^ prefix_parity(after_removed, created)) != 0 ? -1 : 1;
    return index(after_removed | created_bit);
}

Eigen::MatrixXcd Strings::apply_one_body(const Eigen::Ref<const Eigen::MatrixXcd>& block,
                                         const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const {
    if (block.rows() != n_spinor_ || block.cols() != n_spinor_ || matrix.rows() != dimension() ||
        !block.allFinite() || !matrix.allFinite()) {
        throw std::invalid_argument("invalid spinor one-body action dimensions or values");
    }
    return apply_one_body_unchecked(block, matrix);
}

Eigen::MatrixXcd
Strings::apply_one_body_unchecked(const Eigen::Ref<const Eigen::MatrixXcd>& block,
                                  const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dimension(), matrix.cols());
    for (int source = 0; source < dimension(); ++source) {
        const std::uint64_t source_mask = masks_[source];
        for (std::uint64_t occupied = source_mask; occupied != 0; occupied &= occupied - 1) {
            const int q = __builtin_ctzll(occupied);
            for (int p = 0; p < n_spinor_; ++p) {
                if (block(p, q) == std::complex<double>{0.0, 0.0})
                    continue;
                int sign = 0;
                const int target = substitution(source, p, q, &sign);
                if (target < 0)
                    continue;
                const std::complex<double> coefficient = static_cast<double>(sign) * block(p, q);
                if (coefficient == std::complex<double>{0.0, 0.0})
                    continue;
                result.row(target) += coefficient * matrix.row(source);
            }
        }
    }
    return result;
}

} // namespace xmvb::vb::spinor
