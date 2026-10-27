#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>

#include "core/pair_index.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace {
/** @brief Tiny independent canonical-fermion generator oracle, not a carrier backend. */
std::vector<Eigen::MatrixXd> generators(const xmvb::mcscf::SpinStrings& strings) {
    const int n = strings.orbitals(), u = strings.size();
    std::vector<Eigen::MatrixXd> result(n * n, Eigen::MatrixXd::Zero(u, u));
    for (int j = 0; j < u; ++j) {
        const auto mask = strings.masks()[j];
        for (int q = 0; q < n; ++q) {
            if (!(mask & (std::uint64_t{1} << q)))
                continue;
            const auto removed = mask ^ (std::uint64_t{1} << q);
            const int a = std::popcount(mask & ((std::uint64_t{1} << q) - 1));
            for (int p = 0; p < n; ++p) {
                if (removed & (std::uint64_t{1} << p))
                    continue;
                const auto target = removed | (std::uint64_t{1} << p);
                const auto it = std::find(strings.masks().begin(), strings.masks().end(), target);
                const int b = std::popcount(removed & ((std::uint64_t{1} << p) - 1));
                result[p * n + q](it - strings.masks().begin(), j) = (a + b) % 2 ? -1 : 1;
            }
        }
    }
    return result;
}

/** @brief Verify H T = T Hbar and C^T T^T H T C = C^T M Hbar C.
 * Full product matrices exist only in this n<=6 independent algebra oracle.
 * The potentially useful production ordering must still implement streamed
 * Gaussian projection; passing this test establishes no performance claim.
 */
void check(int n) {
    const int r = n / 2, pairs = n * (n + 1) / 2;
    const xmvb::mcscf::SpinStrings strings(n, r);
    const int u = strings.size();
    Eigen::MatrixXd orbital = Eigen::MatrixXd::Identity(n, n);
    for (int q = 0; q < n; ++q) {
        orbital(q, q) += 0.01 * q;
        for (int p = 0; p < q; ++p)
            orbital(p, q) = 0.07 * std::sin(0.21 * (p + 1) * (q + 2));
    }
    const Eigen::MatrixXd inverse = orbital.inverse();
    xmvb::mcscf::ActiveIntegrals integrals;
    integrals.one_electron = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd factor(pairs, pairs);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            integrals.one_electron(p, q) = std::cos(0.23 * (p + q + 2));
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q < pairs; ++q)
            factor(p, q) = 0.03 * std::sin(0.17 * (p + 1) * (q + 3));
    integrals.pair_kernel = factor.transpose() * factor;
    const xmvb::mcscf::DirectCiSigmaAction sigma(strings, strings, integrals);
    const xmvb::vb::ExteriorOrbitalTransform transform(strings, orbital);
    Eigen::MatrixXd t = Eigen::MatrixXd::Identity(u, u);
    transform.apply_left(&t);
    const auto e = generators(strings);
    const auto apply_e = [&](int p, int q, const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
        return e[p * n + q] * x + x * e[p * n + q].transpose();
    };
    const auto index = [n](int i, int j, int k, int l) { return ((i * n + j) * n + k) * n + l; };
    std::vector<double> g(n * n * n * n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    for (int p = 0; p < n; ++p)
                        for (int q = 0; q < n; ++q)
                            for (int s = 0; s < n; ++s)
                                for (int v = 0; v < n; ++v)
                                    g[index(i, j, k, l)] +=
                                        inverse(i, p) * orbital(q, j) * inverse(k, s) *
                                        orbital(v, l) *
                                        integrals.pair_kernel(
                                            xmvb::core::PairIndex::packed_pair_index(p, q),
                                            xmvb::core::PairIndex::packed_pair_index(s, v));
    Eigen::MatrixXd h = inverse * integrals.one_electron * orbital;
    for (int p = 0; p < n; ++p)
        for (int s = 0; s < n; ++s)
            for (int q = 0; q < n; ++q)
                h(p, s) -= 0.5 * g[index(p, q, q, s)];
    // Covalent and arbitrary sparse selected seeds, without assuming closure.
    for (int selection = 0; selection < 2; ++selection) {
        Eigen::MatrixXd x = Eigen::MatrixXd::Zero(u, u);
        for (int a = 0; a < u; ++a)
            for (int b = 0; b < u; ++b)
                if (selection ? (a * u + b) % 17 == 0
                              : (strings.masks()[a] & strings.masks()[b]) == 0)
                    x(a, b) = std::sin(0.19 * (a + 1) * (b + 2));
        x.normalize();
        Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(u, u);
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q) {
                raw.noalias() += h(p, q) * apply_e(p, q, x);
                for (int s = 0; s < n; ++s)
                    for (int v = 0; v < n; ++v)
                        raw.noalias() +=
                            0.5 * g[index(p, q, s, v)] * apply_e(p, q, apply_e(s, v, x));
            }
        const Eigen::MatrixXd y = t * x * t.transpose();
        const Eigen::MatrixXd reference = sigma.apply(y);
        const Eigen::MatrixXd actual = t * raw * t.transpose();
        const double error = (actual - reference).norm() / std::max(1.0, reference.norm());
        const Eigen::MatrixXd metric = t.transpose() * t;
        const Eigen::MatrixXd projected = metric * raw * metric;
        const Eigen::MatrixXd expected = t.transpose() * reference * t;
        const double projection_error =
            (projected - expected).norm() / std::max(1.0, expected.norm());
        double outside = 0;
        if (!selection)
            for (int a = 0; a < u; ++a)
                for (int b = 0; b < u; ++b)
                    if (std::popcount(strings.masks()[a] & strings.masks()[b]) > 2)
                        outside = std::max(outside, std::abs(raw(a, b)));
        std::cout << std::setprecision(17) << "n=" << n << ",selection=" << selection
                  << ",similarity_error=" << error << ",projection_error=" << projection_error
                  << ",outside_two_ionic_pairs=" << outside << '\n';
        if (error > 2e-11 || projection_error > 2e-11 || outside > 1e-13)
            throw std::runtime_error("similarity-order identity or exact closure failed");
    }
}
} // namespace

int main() {
    try {
        check(4);
        check(6);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
