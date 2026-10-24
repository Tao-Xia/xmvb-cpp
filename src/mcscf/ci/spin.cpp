#include "mcscf/ci/spin.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::mcscf {
namespace {

double raising_sign(std::uint64_t alpha, std::uint64_t beta, int orbital, int na) {
    const auto below = (std::uint64_t{1} << orbital) - 1;
    return (na + __builtin_popcountll(alpha & below) + __builtin_popcountll(beta & below)) % 2
               ? -1.0
               : 1.0;
}

} // namespace

SpinSpace::SpinSpace(int n, int na, int nb)
    : n_(n), na_(na), nb_(nb), alpha_(n, na), beta_(na == nb ? alpha_ : SpinStrings(n, nb)) {
    if (na < n && nb > 0) {
        raised_alpha_ = std::make_unique<SpinStrings>(n, na + 1);
        raised_beta_ = std::make_unique<SpinStrings>(n, nb - 1);
    }
}

Eigen::MatrixXd SpinSpace::apply_squared(const Eigen::Ref<const Eigen::MatrixXd>& x) const {
    if (x.rows() != alpha_.size() || x.cols() != beta_.size())
        throw std::invalid_argument("invalid spin-product coefficient shape");
    const double ms = 0.5 * (na_ - nb_);
    Eigen::MatrixXd result = ms * (ms + 1) * x;
    if (!raised_alpha_)
        return result;
    Eigen::MatrixXd raised = Eigen::MatrixXd::Zero(raised_alpha_->size(), raised_beta_->size());
    for (int b = 0; b < beta_.size(); ++b)
        for (int a = 0; a < alpha_.size(); ++a) {
            const auto am = alpha_.masks()[a], bm = beta_.masks()[b];
            auto transferable = bm & ~am;
            while (transferable) {
                const int p = __builtin_ctzll(transferable);
                const auto bit = std::uint64_t{1} << p;
                const int ar = raised_alpha_->index(am | bit), br = raised_beta_->index(bm ^ bit);
                raised(ar, br) += raising_sign(am, bm, p, na_) * x(a, b);
                transferable &= transferable - 1;
            }
        }
    // The lowering operation is the exact transpose of the same raising links.
    for (int b = 0; b < beta_.size(); ++b)
        for (int a = 0; a < alpha_.size(); ++a) {
            const auto am = alpha_.masks()[a], bm = beta_.masks()[b];
            auto transferable = bm & ~am;
            while (transferable) {
                const int p = __builtin_ctzll(transferable);
                const auto bit = std::uint64_t{1} << p;
                result(a, b) +=
                    raising_sign(am, bm, p, na_) *
                    raised(raised_alpha_->index(am | bit), raised_beta_->index(bm ^ bit));
                transferable &= transferable - 1;
            }
        }
    return result;
}

int SpinSpace::dimension(int spin) const {
    const int electrons = na_ + nb_;
    if (spin < std::abs(na_ - nb_) || spin > std::min(electrons, 2 * n_ - electrons) ||
        (spin - electrons) % 2)
        throw std::invalid_argument("requested total spin is absent from the CI space");
    const int a = (electrons + spin) / 2, b = electrons - a;
    const auto product = [&](int p, int q) {
        const auto x = string_count(n_, p), y = string_count(n_, q);
        if (y && x > std::numeric_limits<std::size_t>::max() / y)
            throw std::length_error("spin-product dimension overflows");
        return x * y;
    };
    const auto count = product(a, b) - product(a + 1, b - 1);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("pure-spin dimension exceeds index range");
    return static_cast<int>(count);
}

Eigen::MatrixXd SpinSpace::project(const Eigen::Ref<const Eigen::MatrixXd>& x, int spin) const {
    if (x.rows() != alpha_.size() || x.cols() != beta_.size())
        throw std::invalid_argument("invalid spin-product coefficient shape");
    dimension(spin);
    Eigen::MatrixXd result = x;
    const double target = 0.25 * spin * (spin + 2);
    // Descending roots remove the largest spin sectors first. No penalty
    // Hamiltonian or empirical spin-shift parameter is introduced.
    for (int other = std::min(na_ + nb_, 2 * n_ - na_ - nb_); other >= std::abs(na_ - nb_);
         other -= 2) {
        if (other == spin)
            continue;
        const double eigenvalue = 0.25 * other * (other + 2);
        result = (apply_squared(result) - eigenvalue * result) / (target - eigenvalue);
    }
    return result;
}

} // namespace xmvb::mcscf
