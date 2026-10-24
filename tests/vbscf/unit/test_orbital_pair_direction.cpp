#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "integrals/response/orbital_pair.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(const Eigen::MatrixXd& actual,
                   const Eigen::MatrixXd& expected,
                   const char* message) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols() &&
                (actual - expected).norm() <= 3e-12 * std::max(1.0, expected.norm()),
            message);
}

void check_direction(int n_auxiliary,
                     bool ill_conditioned,
                     bool zero_direction,
                     unsigned int seed) {
    constexpr int n_basis = 5;
    constexpr int n_active = 3;
    constexpr int n_pairs = n_active * (n_active + 1) / 2;
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> distribution(-0.8, 0.8);
    const auto draw = [&]() { return distribution(random); };

    std::vector<Eigen::MatrixXd> ao_factors(n_auxiliary, Eigen::MatrixXd::Zero(n_basis, n_basis));
    for (auto& factor : ao_factors) {
        for (int mu = 0; mu < n_basis; ++mu) {
            for (int nu = 0; nu <= mu; ++nu) {
                factor(mu, nu) = factor(nu, mu) = draw();
            }
        }
    }

    Eigen::MatrixXd coefficients(n_basis, n_active);
    Eigen::MatrixXd direction(n_basis, n_active);
    for (int a = 0; a < n_active; ++a) {
        const double scale = ill_conditioned ? (a == 0 ? 1e-4 : (a == 2 ? 1e3 : 1.0)) : 1.0;
        for (int mu = 0; mu < n_basis; ++mu) {
            coefficients(mu, a) = scale * draw();
            direction(mu, a) = zero_direction ? 0.0 : draw();
        }
    }

    std::vector<int> first, second;
    first.reserve(n_pairs);
    second.reserve(n_pairs);
    for (int a = 0; a < n_active; ++a) {
        for (int b = 0; b <= a; ++b) {
            first.push_back(a);
            second.push_back(b);
        }
    }

    // Independent RI oracle: form accepted and directional auxiliary factors
    // from full, symmetric AO matrices, with no pair-map or image helper.
    Eigen::MatrixXd factors(n_auxiliary, n_pairs);
    Eigen::MatrixXd factor_direction(n_auxiliary, n_pairs);
    for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
        for (int pair = 0; pair < n_pairs; ++pair) {
            const int a = first[pair], b = second[pair];
            double value = 0.0, derivative = 0.0;
            for (int mu = 0; mu < n_basis; ++mu) {
                for (int nu = 0; nu < n_basis; ++nu) {
                    const double integral = ao_factors[auxiliary](mu, nu);
                    value += coefficients(mu, a) * integral * coefficients(nu, b);
                    derivative += direction(mu, a) * integral * coefficients(nu, b) +
                                  coefficients(mu, a) * integral * direction(nu, b);
                }
            }
            factors(auxiliary, pair) = value;
            factor_direction(auxiliary, pair) = derivative;
        }
    }

    xmvb::integrals::OrbitalPairMatrix products(n_basis * n_active, n_pairs);
    for (int mu = 0; mu < n_basis; ++mu) {
        for (int a = 0; a < n_active; ++a) {
            for (int pair = 0; pair < n_pairs; ++pair) {
                double value = 0.0;
                for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
                    for (int nu = 0; nu < n_basis; ++nu) {
                        value += ao_factors[auxiliary](mu, nu) * coefficients(nu, a) *
                                 factors(auxiliary, pair);
                    }
                }
                products(mu * n_active + a, pair) = value;
            }
        }
    }

    const xmvb::integrals::OrbitalPairMatrix unsym =
        xmvb::integrals::contract_orbital_pair_direction(
            products, direction, first, second, n_active);
    const Eigen::MatrixXd oracle_unsym = factor_direction.transpose() * factors;
    const Eigen::MatrixXd oracle_full = oracle_unsym + oracle_unsym.transpose();
    require_close(
        unsym, oracle_unsym, "orbital-pair direction differs from independent RI factor direction");
    require_close(unsym + unsym.transpose(),
                  oracle_full,
                  "symmetric orbital-pair direction differs from RI Gram direction");
    const Eigen::VectorXd packed = xmvb::integrals::packed_orbital_pair_direction(
        products, direction, first, second, n_active);
    require(packed.size() == n_pairs * (n_pairs + 1) / 2,
            "packed orbital-pair direction has the wrong size");
    Eigen::VectorXd packed_reference(packed.size());
    for (int p = 0; p < n_pairs; ++p) {
        for (int q = 0; q <= p; ++q) {
            packed_reference[xmvb::core::PairIndex::packed_pair_of_pairs_index(p, q)] =
                oracle_full(p, q);
        }
    }
    require((packed - packed_reference).norm() <= 3e-12 * std::max(1.0, packed_reference.norm()),
            "packed orbital-pair direction differs from RI factor Gram direction");

    // W represents a packed symmetric covector: off-diagonal entries have
    // half weight in the full Frobenius contraction.
    xmvb::integrals::OrbitalPairMatrix weight(n_pairs, n_pairs);
    for (int p = 0; p < n_pairs; ++p) {
        for (int q = 0; q <= p; ++q) {
            const double packed_weight = draw();
            weight(p, q) = weight(q, p) = p == q ? packed_weight : 0.5 * packed_weight;
        }
    }
    const double directional_duality = (weight.array() * oracle_full.array()).sum();
    const xmvb::integrals::OrbitalPairMatrix unsym_adjoint = 2.0 * weight;
    const Eigen::MatrixXd orbital_adjoint =
        xmvb::integrals::pullback_orbital_pairs(products, unsym_adjoint, first, second, n_active);
    const double orbital_duality = (orbital_adjoint.array() * direction.array()).sum();
    const double duality_scale = (weight.cwiseAbs().array() * oracle_full.cwiseAbs().array()).sum();
    require(std::abs(orbital_duality - directional_duality) <= 3e-12 * std::max(1.0, duality_scale),
            "orbital-pair direction and pullback violate bilinear duality");
}

} // namespace

int main() {
    check_direction(2, false, false, 17); // RI rank below active-pair count.
    check_direction(9, false, false, 29); // RI rank above active-pair count.
    check_direction(7, true, false, 43);  // Nonorthogonal, ill-scaled raw orbitals.
    check_direction(4, false, true, 61);  // Exactly zero tangent.
    return 0;
}
