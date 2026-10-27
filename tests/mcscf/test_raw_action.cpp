#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>

#include "core/pair_index.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"

namespace {
/** @brief Tiny independent exterior matrix; never used by the implementation. */
Eigen::MatrixXd compound(const xmvb::mcscf::SpinStrings& strings, const Eigen::MatrixXd& r) {
    Eigen::MatrixXd result(strings.size(), strings.size());
    if (!strings.electrons())
        return Eigen::MatrixXd::Ones(1, 1);
    const auto occupied = strings.decode();
    for (int a = 0; a < strings.size(); ++a)
        for (int b = 0; b < strings.size(); ++b) {
            Eigen::MatrixXd minor(strings.electrons(), strings.electrons());
            for (int i = 0; i < minor.rows(); ++i)
                for (int j = 0; j < minor.cols(); ++j)
                    minor(i, j) = r(occupied[a][i], occupied[b][j]);
            result(a, b) = minor.determinant();
        }
    return result;
}

void check(int n, int na, int nb, int selection) {
    using namespace xmvb::mcscf;
    const SpinStrings alpha(n, na), beta(n, nb);
    const int ua = alpha.size(), ub = beta.size(), pairs = n * (n + 1) / 2, rhs = 3;
    ActiveIntegrals integrals;
    integrals.one_electron.resize(n, n);
    Eigen::MatrixXd factor(pairs, pairs), r = Eigen::MatrixXd::Identity(n, n);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            integrals.one_electron(p, q) = std::cos(0.27 * (p + q + 1));
            if (p <= q)
                r(p, q) += 0.09 * std::sin(0.13 * (p + 2) * (q + 1));
        }
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q < pairs; ++q)
            factor(p, q) = 0.05 * std::sin(0.17 * (p + 1) * (q + 2));
    integrals.pair_kernel = factor.transpose() * factor;
    const DirectCiSigmaAction sigma(alpha, beta, integrals);
    const auto& ac = sigma.alpha_connections();
    const auto& bc = sigma.beta_connections_view();
    std::vector<ProductIndex> products;
    for (int a = 0; a < ua; ++a)
        for (int b = 0; b < ub; ++b) {
            const bool cov = (alpha.masks()[a] ^ beta.masks()[b]) == (std::uint64_t{1} << n) - 1;
            if (selection == 3 || (selection == 0 && cov) ||
                (selection == 1 && cov && a % 3 == 0) ||
                (selection == 2 && (a * ub + b) % 11 == 0) || (selection == 4 && a == 0 && b == 0))
                products.push_back(product_index(a, b, ub));
        }
    if (products.empty())
        throw std::runtime_error("raw CI oracle has an empty fixture");
    const RawCiPlan plan(ac, bc, products);
    const SimilarityHamiltonian h(integrals, r);
    const auto targets = plan.beta_columns();
    for (int b = 0; b < ub; ++b) {
        bool reachable = false;
        for (const auto product : products)
            reachable |= std::popcount(beta.masks()[b] ^ beta.masks()[product % ub]) <= 4;
        if (reachable != std::binary_search(targets.begin(), targets.end(), b))
            throw std::runtime_error("raw beta closure contains missing or extra columns");
    }
    Eigen::MatrixXd values(products.size(), rhs), source = Eigen::MatrixXd::Zero(ua, ub * rhs);
    for (std::size_t i = 0; i < products.size(); ++i)
        for (int v = 0; v < rhs; ++v) {
            values(i, v) = i == 0 && v == 0 ? 0 : std::sin(0.31 * (i + 1) * (v + 2));
            source(products[i] / ub, v * ub + products[i] % ub) = values(i, v);
        }
    Eigen::MatrixXd packed(ua, targets.size() * rhs), copied = packed;
    const auto work = plan.apply_columns(h, values, targets, packed);
    plan.source_columns(values, targets, copied);
    Eigen::MatrixXd actual = Eigen::MatrixXd::Zero(ua, ub * rhs);
    for (std::size_t j = 0; j < targets.size(); ++j)
        for (int v = 0; v < rhs; ++v) {
            actual.col(v * ub + targets[j]) = packed.col(j * rhs + v);
            if ((copied.col(j * rhs + v) - source.col(v * ub + targets[j])).norm() != 0)
                throw std::runtime_error("raw source scatter changed original coefficients");
        }
    const auto ta = compound(alpha, r), tb = compound(beta, r);
    Eigen::MatrixXd orthogonal(ua, ub * rhs), transformed(ua, ub * rhs);
    for (int v = 0; v < rhs; ++v) {
        orthogonal.middleCols(v * ub, ub) = ta * source.middleCols(v * ub, ub) * tb.transpose();
        transformed.middleCols(v * ub, ub) = ta * actual.middleCols(v * ub, ub) * tb.transpose();
    }
    const auto reference = sigma.apply(orthogonal);
    const double error = (transformed - reference).norm() / std::max(1.0, reference.norm());
    std::cout << "n=" << n << ",alpha=" << na << ",beta=" << nb << ",selection=" << selection
              << ",covalent=" << plan.complete_covalent_support() << ",rhs=" << rhs
              << ",targets=" << targets.size() << ",elements=" << work << ",error=" << error
              << '\n';
    if (error > 2e-11 || !std::isfinite(error))
        throw std::runtime_error("raw sparse action differs from orthogonal sigma");
}
} // namespace

int main() {
    try {
        for (const int selection : {0, 1, 2, 3})
            check(4, 2, 2, selection);
        check(4, 3, 1, 0);
        check(5, 3, 1, 2);
        check(4, 0, 2, 2);
        check(6, 0, 3, 4);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
