#pragma once

#include <bit>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <Eigen/Core>
#include "core/pair_index.hpp"
#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/strings.hpp"

namespace xmvb::mcscf {
using OccupationMask = std::uint64_t;

/** @brief Borrowed small canonical integral accumulators, owned by one worker. */
struct CanonicalSeed {
    Eigen::MatrixXd* one = nullptr;
    Eigen::MatrixXd* two = nullptr;
};

class SimilarityHamiltonian;

/** @brief One directional operator and its scalar edge contraction. */
struct CanonicalDirection {
    const SimilarityHamiltonian* op = nullptr;
    double weight = 0, value = 0;
    CanonicalSeed seed;
};

/** @brief Canonical two-electron operator T^-1 H T, not a symmetric integral model.
 * Pair exchange survives; interchange within a pair generally does not.
 * The canonical element routine uses only occupation masks and no cofactors.
 */
class SimilarityHamiltonian {
    int n_;
    Eigen::MatrixXd h_, g_, r_, inverse_, physical_h_, physical_g_, change_;
    SimilarityHamiltonian(int n, Eigen::MatrixXd h, Eigen::MatrixXd g)
        : n_(n), h_(std::move(h)), g_(std::move(g)) {}

  public:
    SimilarityHamiltonian(const ActiveIntegrals& integrals,
                          const Eigen::Ref<const Eigen::MatrixXd>& orbital_map);

    /** @brief Analytic tangent of the canonical integral similarity. */
    SimilarityHamiltonian direction(const ActiveIntegrals& integrals,
                                    const Eigen::MatrixXd& orbital_direction) const;
    /** @brief Canonical-to-physical integral VJP and its analytic tangent. */
    ActiveIntegrals pullback(const Eigen::MatrixXd& one_adjoint,
                             const Eigen::MatrixXd& two_adjoint,
                             const Eigen::MatrixXd* orbital_direction = nullptr,
                             const Eigen::MatrixXd* one_adjoint_direction = nullptr,
                             const Eigen::MatrixXd* two_adjoint_direction = nullptr) const;
    /** @brief One signed element traversal for value, tangent and integral VJP. */
    double element_jet(OccupationMask ta,
                       OccupationMask tb,
                       OccupationMask sa,
                       OccupationMask sb,
                       double weight,
                       CanonicalSeed seed,
                       std::span<CanonicalDirection> directions) const;

    int orbitals() const noexcept {
        return n_;
    }

    double g(int p, int q, int r, int s) const noexcept {
        return g_(p * n_ + q, r * n_ + s);
    }
    /** @brief General real canonical Slater--Condon element, without Hermiticity assumptions. */
    double
    element(OccupationMask ta, OccupationMask tb, OccupationMask sa, OccupationMask sb) const;
};

} // namespace xmvb::mcscf
