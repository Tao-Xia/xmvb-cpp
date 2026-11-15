#pragma once

#include <bit>
#include <cstdint>

#include "core/pair_index.hpp"
#include "mcscf/ci/integrals.hpp"

namespace xmvb::mcscf {

/** The production CI diagonal inverse, shared by root and microresponse solves. */
inline void apply_occupation_preconditioner(
    const Eigen::Ref<const Eigen::VectorXd>& diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& residual,
    double energy,
    Eigen::Ref<Eigen::VectorXd> output) {
    const auto denominator = diagonal.array() - energy;
    output = (residual.array() / (denominator.abs() < 1.0e-8).select(1.0e-8, denominator)).matrix();
}

/** Fixed-Ms average of determinant diagonals for one spatial occupation.
 * One scalar per occupation commutes with S^2 in both CSF and product coordinates.
 * Only orbital-sized integral slices are stored; no product diagonal is cached.
 */
class OccupationDiagonal {
  public:
    OccupationDiagonal(const ActiveIntegrals& integrals, int alpha, int beta)
        : h_(integrals.one_electron.diagonal()), alpha_(alpha), beta_(beta) {
        const auto n = h_.size();
        coulomb_.resize(n, n);
        exchange_.resize(n, n);
        for (Eigen::Index p = 0; p < n; ++p)
            for (Eigen::Index q = 0; q <= p; ++q) {
                const auto pp = core::PairIndex::packed_pair_index(p, p);
                const auto qq = core::PairIndex::packed_pair_index(q, q);
                const auto pq = core::PairIndex::packed_pair_index(p, q);
                coulomb_(p, q) = integrals.pair_kernel(pp, qq);
                exchange_(p, q) = integrals.pair_kernel(pq, pq);
            }
    }

    double operator()(std::uint64_t occupied, std::uint64_t doubled) const noexcept {
        const int doubles = std::popcount(doubled);
        const int a = alpha_ - doubles, b = beta_ - doubles, singles = a + b;
        const double equal_spin = singles < 2 ? 0.0
                                              : static_cast<double>(a * (a - 1) + b * (b - 1)) /
                                                    (singles * (singles - 1));
        double energy = 0;
        for (auto orbitals = occupied; orbitals; orbitals &= orbitals - 1) {
            const int q = std::countr_zero(orbitals);
            const bool double_q = (doubled >> q) & 1;
            const int nq = 1 + double_q;
            energy += nq * h_[q] + double_q * coulomb_(q, q);
            for (auto later = orbitals & (orbitals - 1); later; later &= later - 1) {
                const int p = std::countr_zero(later);
                const bool double_p = (doubled >> p) & 1;
                const double same_spin = double_p || double_q ? double_p + double_q : equal_spin;
                energy += (1 + double_p) * nq * coulomb_(p, q) - same_spin * exchange_(p, q);
            }
        }
        return energy;
    }

  private:
    Eigen::VectorXd h_;
    Eigen::MatrixXd coulomb_, exchange_;
    int alpha_, beta_;
};

} // namespace xmvb::mcscf
