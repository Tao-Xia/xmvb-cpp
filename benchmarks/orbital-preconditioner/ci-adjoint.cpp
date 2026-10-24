#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vbscf/structures/orthogonal_ci/sigma.hpp"

/** @brief Synthetic full one-alpha/one-beta space, not a molecular benchmark. */
int main() {
    try {
        constexpr int n_orbitals = 48;
        const int n_pairs = n_orbitals * (n_orbitals + 1) / 2;
        std::vector<std::vector<int>> strings;
        for (int orbital = 0; orbital < n_orbitals; ++orbital)
            strings.push_back({orbital});
        xmvb::vb::OrthogonalActiveIntegrals integrals;
        integrals.one_electron = Eigen::MatrixXd::Identity(n_orbitals, n_orbitals);
        Eigen::MatrixXd factors(7, n_pairs);
        for (int j = 0; j < n_pairs; ++j)
            for (int i = 0; i < factors.rows(); ++i)
                factors(i, j) = 0.007 * std::sin(0.17 * (i + 1) * (j + 2));
        integrals.pair_kernel = factors.transpose() * factors;
        xmvb::vb::DirectCiSigmaAction action(strings, strings, integrals);
        Eigen::MatrixXd coefficients(n_orbitals, n_orbitals);
        for (int j = 0; j < n_orbitals; ++j)
            for (int i = 0; i < n_orbitals; ++i)
                coefficients(i, j) = std::sin(0.031 * (i + 1) * (j + 2));
        coefficients /= coefficients.norm();
        const double expected = coefficients.cwiseProduct(action.apply(coefficients)).sum();
        const auto start = std::chrono::steady_clock::now();
        const auto adjoint = action.integral_adjoint(coefficients, coefficients);
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double contracted = adjoint.one_electron.cwiseProduct(integrals.one_electron).sum() +
                                  adjoint.pair_kernel.cwiseProduct(integrals.pair_kernel).sum();
        if (std::abs(contracted - expected) > 1e-11 * std::max(1.0, std::abs(expected)))
            throw std::runtime_error("adjoint differs from the action bilinear form");
        std::cout << std::setprecision(17)
                  << "n_orbitals,n_strings,n_pairs,adjoint_seconds,bilinear,contracted\n"
                  << n_orbitals << ',' << strings.size() << ',' << n_pairs << ',' << wall << ','
                  << expected << ',' << contracted << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
