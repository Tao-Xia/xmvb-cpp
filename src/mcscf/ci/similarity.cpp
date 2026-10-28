#include <cstdint>
#include "mcscf/ci/similarity.hpp"
#include "mcscf/ci/canonical.hpp"

#include <algorithm>
#include <limits>
#include <Eigen/LU>

namespace xmvb::mcscf {

SimilarityHamiltonian::SimilarityHamiltonian(const ActiveIntegrals& integrals,
                                             const Eigen::Ref<const Eigen::MatrixXd>& orbital_map)
    : n_(integrals.one_electron.rows()) {
    if (n_ <= 0 || n_ > 63 || orbital_map.rows() != n_ || orbital_map.cols() != n_ ||
        integrals.one_electron.cols() != n_ || integrals.pair_kernel.rows() != n_ * (n_ + 1) / 2 ||
        integrals.pair_kernel.cols() != integrals.pair_kernel.rows() || !orbital_map.allFinite() ||
        (orbital_map.diagonal().array() == 0).any())
        throw std::invalid_argument("invalid canonical similarity integrals");
    const Eigen::MatrixXd inverse = orbital_map.inverse();
    h_ = inverse * integrals.one_electron * orbital_map;
    Eigen::MatrixXd coulomb(n_ * n_, n_ * n_), change(n_ * n_, n_ * n_);
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            for (int k = 0; k < n_; ++k)
                for (int l = 0; l < n_; ++l) {
                    change(i * n_ + j, k * n_ + l) = inverse(i, k) * orbital_map(l, j);
                    coulomb(i * n_ + j, k * n_ + l) =
                        integrals.pair_kernel(core::PairIndex::packed_pair_index(i, j),
                                              core::PairIndex::packed_pair_index(k, l));
                }
    r_ = orbital_map;
    inverse_ = inverse;
    physical_h_ = integrals.one_electron;
    physical_g_ = coulomb;
    change_ = change;
    const Eigen::MatrixXd first = change * coulomb;
    g_.noalias() = first * change.transpose();
}

bool SimilarityHamiltonian::preserves_physical_operator() const {
    // Compare in the original orthogonal metric. An error bound scaled by the
    // large canonical integrals would hide the cancellation being diagnosed.
    const auto resolved = [](const Eigen::MatrixXd& recovered,
                             const Eigen::MatrixXd& physical,
                             Eigen::Index operations) {
        const double u = std::numeric_limits<double>::epsilon();
        const double gamma = operations * u / (1.0 - operations * u);
        return recovered.allFinite() && (recovered - physical).norm() <= gamma * physical.norm();
    };
    if (!resolved(r_ * h_ * inverse_, physical_h_, 4 * n_))
        return false;
    Eigen::MatrixXd reverse(n_ * n_, n_ * n_);
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            for (int k = 0; k < n_; ++k)
                for (int l = 0; l < n_; ++l)
                    reverse(i * n_ + j, k * n_ + l) = r_(i, k) * inverse_(l, j);
    return resolved(reverse * g_ * reverse.transpose(), physical_g_, 4 * n_ * n_);
}

double SimilarityHamiltonian::element(OccupationMask ta,
                                      OccupationMask tb,
                                      OccupationMask sa,
                                      OccupationMask sb) const {
    double value = 0;
    canonical::terms(
        ta,
        tb,
        sa,
        sb,
        [&](int p, int q, double c) { value += c * h_(p, q); },
        [&](int p, int q, int r, int s, double c) { value += c * g(p, q, r, s); });
    return value;
}

double SimilarityHamiltonian::element_jet(OccupationMask ta,
                                          OccupationMask tb,
                                          OccupationMask sa,
                                          OccupationMask sb,
                                          double weight,
                                          CanonicalSeed seed,
                                          std::span<CanonicalDirection> directions) const {
    double result = 0;
    for (auto& d : directions)
        d.value = 0;
    canonical::terms(
        ta,
        tb,
        sa,
        sb,
        [&](int p, int q, double c) {
            result += c * h_(p, q);
            if (seed.one && weight != 0)
                (*seed.one)(p, q) += c * weight;
            for (auto& d : directions) {
                if (d.op)
                    d.value += c * d.op->h_(p, q);
                if (d.seed.one && d.weight != 0)
                    (*d.seed.one)(p, q) += c * d.weight;
            }
        },
        [&](int p, int q, int r, int s, double c) {
            const std::int64_t a = p * n_ + q, b = r * n_ + s;
            result += c * g_(a, b);
            if (seed.two && weight != 0)
                (*seed.two)(a, b) += c * weight;
            for (auto& d : directions) {
                if (d.op)
                    d.value += c * d.op->g_(a, b);
                if (d.seed.two && d.weight != 0)
                    (*d.seed.two)(a, b) += c * d.weight;
            }
        });
    return result;
}
} // namespace xmvb::mcscf
