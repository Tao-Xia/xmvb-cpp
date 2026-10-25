#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "spinor_reference.hpp"
#include "vbscf/structures/spinor/sigma.hpp"
#include "vbscf/structures/spinor/strings.hpp"

namespace {

using Complex = std::complex<double>;
using Strings = xmvb::vb::spinor::Strings;
using Sigma = xmvb::vb::spinor::Sigma;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Function> void require_invalid(Function&& function, const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    } catch (const std::length_error&) {
        rejected = true;
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    require(rejected, message);
}

void close(const Eigen::MatrixXcd& actual,
           const Eigen::MatrixXcd& expected,
           const std::string& message,
           double tolerance = 1.0e-11) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
            message + ": shape");
    const double scale = std::max(1.0, expected.norm());
    require((actual - expected).norm() <= tolerance * scale, message + ": value");
}

Eigen::MatrixXcd sample_hermitian(int n) {
    Eigen::MatrixXcd h(n, n);
    for (int p = 0; p < n; ++p) {
        h(p, p) = 0.31 * (p + 1) - 0.17;
        for (int q = p + 1; q < n; ++q) {
            const Complex value{0.12 * std::sin(0.73 * (p + 1) * (q + 2)),
                                0.09 * std::cos(0.41 * (p + 2) * (q + 1))};
            h(p, q) = value;
            h(q, p) = std::conj(value);
        }
    }
    return h;
}

std::vector<Eigen::MatrixXcd> sample_factors(int n) {
    std::vector<Eigen::MatrixXcd> factors;
    for (int auxiliary = 0; auxiliary < 3; ++auxiliary) {
        Eigen::MatrixXcd factor(n, n);
        for (int p = 0; p < n; ++p) {
            factor(p, p) =
                0.11 * (auxiliary + 1) + 0.04 * std::cos(0.37 * (p + 1) * (auxiliary + 2));
            for (int q = p + 1; q < n; ++q) {
                const Complex value{0.035 * std::sin(0.29 * (p + 2) * (q + 1) * (auxiliary + 1)),
                                    0.028 * std::cos(0.53 * (p + 1) * (q + 2) * (auxiliary + 2))};
                factor(p, q) = value;
                factor(q, p) = std::conj(value);
            }
        }
        factors.push_back(std::move(factor));
    }
    return factors;
}

std::vector<Complex> reconstruct_eri(const std::vector<Eigen::MatrixXcd>& factors) {
    const int n = static_cast<int>(factors.front().rows());
    std::vector<Complex> eri(static_cast<std::size_t>(n) * n * n * n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    Complex value = 0.0;
                    for (const auto& factor : factors) {
                        value += factor(p, q) * factor(r, s);
                    }
                    eri[spinor_reference::eri_index(n, p, q, r, s)] = value;
                }
            }
        }
    }
    return eri;
}

Eigen::MatrixXcd sample_block(int rows, int columns) {
    Eigen::MatrixXcd block(rows, columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            block(row, column) = Complex{std::sin(0.23 * (row + 1) * (column + 2)),
                                         0.37 * std::cos(0.19 * (row + 2) * (column + 1))};
        }
    }
    return block;
}

Eigen::MatrixXcd reference_to_production(const Eigen::MatrixXcd& reference,
                                         const spinor_reference::Basis& basis,
                                         const Strings& strings) {
    const int d = strings.dimension();
    require(reference.rows() == d && reference.cols() == d,
            "reference/product carrier dimension mismatch");
    Eigen::MatrixXcd result(d, d);
    for (int row = 0; row < d; ++row) {
        const int reference_row =
            spinor_reference::find(basis, spinor_reference::occupation(strings.masks()[row]));
        require(reference_row >= 0, "production mask absent from reference basis");
        for (int column = 0; column < d; ++column) {
            const int reference_column = spinor_reference::find(
                basis, spinor_reference::occupation(strings.masks()[column]));
            require(reference_column >= 0, "production mask absent from reference basis");
            result(row, column) = reference(reference_row, reference_column);
        }
    }
    return result;
}

void check_basis(const Strings& strings) {
    const auto independent = spinor_reference::basis(strings.n_spinor(), strings.n_electrons());
    require(strings.dimension() == static_cast<int>(independent.size()),
            "string dimension is not the complete fixed-N space");
    require(strings.masks().size() == independent.size(),
            "string mask count disagrees with dimension");
    std::vector<std::uint64_t> seen;
    for (int i = 0; i < strings.dimension(); ++i) {
        const auto current = strings.masks()[i];
        require(spinor_reference::find(independent, spinor_reference::occupation(current)) >= 0,
                "invalid or incomplete production occupation");
        require(std::find(seen.begin(), seen.end(), current) == seen.end(),
                "duplicate production occupation");
        seen.push_back(current);
        require(strings.index(current) == i, "mask index round-trip failed");
        for (int removed = 0; removed < strings.n_spinor(); ++removed) {
            for (int created = 0; created < strings.n_spinor(); ++created) {
                auto state = spinor_reference::occupation(current);
                int reference_sign = 1;
                const bool survives =
                    spinor_reference::annihilate(state, removed, reference_sign) &&
                    spinor_reference::create(state, created, reference_sign);
                int production_sign = 0;
                const int target = strings.substitution(i, created, removed, &production_sign);
                if (!survives) {
                    require(target == -1, "vanishing substitution retained a link");
                } else {
                    require(target >= 0 && strings.masks()[target] == spinor_reference::mask(state),
                            "substitution target disagrees with occupied-list algebra");
                    require(production_sign == reference_sign,
                            "substitution sign disagrees with occupied-list algebra");
                }
            }
        }
    }
}

void check_eri_symmetry(const std::vector<Complex>& eri, int n) {
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    const auto value = eri[spinor_reference::eri_index(n, p, q, r, s)];
                    require(std::abs(value - eri[spinor_reference::eri_index(n, r, s, p, q)]) <
                                1.0e-14,
                            "ERI pair exchange symmetry failed");
                    require(std::abs(value -
                                     std::conj(eri[spinor_reference::eri_index(n, q, p, s, r)])) <
                                1.0e-14,
                            "ERI Hermitian reversal failed");
                }
            }
        }
    }
}

void check_actions(int n, int electrons, int n_a) {
    const auto strings = std::make_shared<const Strings>(n, electrons, n_a, 64);
    check_basis(*strings);
    const auto basis = spinor_reference::basis(n, electrons);
    const Eigen::MatrixXcd h = sample_hermitian(n);
    const auto factors = sample_factors(n);
    const auto eri = reconstruct_eri(factors);
    check_eri_symmetry(eri, n);
    const Eigen::MatrixXcd dense = reference_to_production(
        spinor_reference::dense_hamiltonian(basis, h, eri), basis, *strings);
    const Eigen::MatrixXcd one_body =
        reference_to_production(spinor_reference::one_body_matrix(basis, h), basis, *strings);
    close(dense, dense.adjoint().eval(), "reference Hamiltonian Hermiticity");
    if (n == 6 && electrons == 3) {
        bool two_step_transfer = false;
        for (int target = 0; target < strings->dimension(); ++target) {
            for (int source = 0; source < strings->dimension(); ++source) {
                const auto target_occupation =
                    spinor_reference::occupation(strings->masks()[target]);
                const auto source_occupation =
                    spinor_reference::occupation(strings->masks()[source]);
                const int target_a =
                    static_cast<int>(std::count_if(target_occupation.begin(),
                                                   target_occupation.end(),
                                                   [n_a](int orbital) { return orbital < n_a; }));
                const int source_a =
                    static_cast<int>(std::count_if(source_occupation.begin(),
                                                   source_occupation.end(),
                                                   [n_a](int orbital) { return orbital < n_a; }));
                if (std::abs(target_a - source_a) == 2 &&
                    std::abs(dense(target, source) - one_body(target, source)) > 1.0e-8) {
                    two_step_transfer = true;
                }
            }
        }
        require(two_step_transfer, "fixture lacks a two-step A/B transfer");
    }
    const Sigma exact(strings, h, eri);
    const Sigma ri(strings, h, factors);
    const Eigen::MatrixXcd arbitrary_operator = sample_block(n, n);
    const Eigen::MatrixXcd arbitrary_dense = reference_to_production(
        spinor_reference::one_body_matrix(basis, arbitrary_operator), basis, *strings);
    const Eigen::MatrixXcd arbitrary_rhs = sample_block(strings->dimension(), 3);
    close(strings->apply_one_body(arbitrary_operator, arbitrary_rhs),
          arbitrary_dense * arbitrary_rhs,
          "non-Hermitian one-body operator orientation");
    for (int width : {1, 3, 5}) {
        const Eigen::MatrixXcd block = sample_block(strings->dimension(), width);
        close(strings->apply_one_body(h, block),
              one_body * block,
              "one-body action, width " + std::to_string(width));
        close(exact.apply(block), dense * block, "exact action, width " + std::to_string(width));
        close(ri.apply(block), dense * block, "RI action, width " + std::to_string(width));
        close(exact.apply(block),
              ri.apply(block),
              "exact/RI agreement, width " + std::to_string(width));
    }
    const Eigen::MatrixXcd left = sample_block(strings->dimension(), 1);
    const Eigen::MatrixXcd right = sample_block(strings->dimension(), 2).rightCols(1);
    const Complex lhs = (left.adjoint() * exact.apply(right))(0, 0);
    const Complex rhs = (exact.apply(left).adjoint() * right)(0, 0);
    require(std::abs(lhs - rhs) < 1.0e-11 * std::max(1.0, std::abs(lhs)),
            "Hermitian bilinear identity failed");
}

void check_cross_sectors() {
    const Strings strings(4, 2, 2, 6);
    const auto basis = spinor_reference::basis(4, 2);
    Eigen::MatrixXcd h = Eigen::MatrixXcd::Zero(4, 4);
    h(2, 0) = Complex{0.3, 0.2};
    h(0, 2) = std::conj(h(2, 0));
    const Eigen::MatrixXcd dense =
        reference_to_production(spinor_reference::one_body_matrix(basis, h), basis, strings);
    bool has_cross_sector = false;
    for (int target = 0; target < strings.dimension(); ++target) {
        for (int source = 0; source < strings.dimension(); ++source) {
            const int a_target =
                ((strings.masks()[target] & 1U) != 0) + ((strings.masks()[target] & 2U) != 0);
            const int a_source =
                ((strings.masks()[source] & 1U) != 0) + ((strings.masks()[source] & 2U) != 0);
            if (a_target != a_source && std::abs(dense(target, source)) > 0.0) {
                has_cross_sector = true;
            }
        }
    }
    require(has_cross_sector, "cross-A/B fixture did not couple sectors");
    close(strings.apply_one_body(h, Eigen::MatrixXcd::Identity(6, 6)),
          dense,
          "cross-A/B one-body action");
}

void check_counterterm() {
    for (int electrons : {0, 1, 2, 3, 4}) {
        const auto strings = std::make_shared<const Strings>(4, electrons, 2, 6);
        const Eigen::MatrixXcd zero = Eigen::MatrixXcd::Zero(4, 4);
        const std::vector<Eigen::MatrixXcd> factors{Eigen::MatrixXcd::Identity(4, 4)};
        const auto eri = reconstruct_eri(factors);
        const Sigma exact(strings, zero, eri);
        const Sigma ri(strings, zero, factors);
        const Eigen::MatrixXcd identity =
            Eigen::MatrixXcd::Identity(strings->dimension(), strings->dimension());
        const double expected = 0.5 * electrons * (electrons - 1);
        close(exact.apply(identity), expected * identity, "exact repeated-index analytical limit");
        close(ri.apply(identity),
              expected * identity,
              "RI repeated-index counterterm analytical limit");
    }
}

void check_exterior_reference_limits() {
    const Strings two_electrons(4, 2, 2, 6);
    const Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(4, 4);
    close(spinor_reference::wedge_matrix(identity, two_electrons.masks()),
          Eigen::MatrixXcd::Identity(6, 6),
          "exterior identity analytical limit");
    const Strings vacuum(4, 0, 2, 1);
    close(spinor_reference::wedge_matrix(identity, vacuum.masks()),
          Eigen::MatrixXcd::Identity(1, 1),
          "vacuum exterior analytical limit");
    Eigen::MatrixXcd r = identity;
    r(0, 0) = Complex{1.2, 0.1};
    r(0, 1) = Complex{0.2, -0.3};
    r(1, 0) = Complex{-0.1, 0.4};
    r(1, 1) = Complex{0.9, -0.2};
    const Eigen::MatrixXcd wedge = spinor_reference::wedge_matrix(r, two_electrons.masks());
    const int pair_index = two_electrons.index(3);
    require(std::abs(wedge(pair_index, pair_index) - (r(0, 0) * r(1, 1) - r(0, 1) * r(1, 0))) <
                1.0e-14,
            "two-electron minor analytical limit");
}

void check_kramers_one_electron() {
    const auto strings = std::make_shared<const Strings>(4, 1, 2, 4);
    Eigen::MatrixXcd a(2, 2), b = Eigen::MatrixXcd::Zero(2, 2);
    a << 0.2, Complex{0.11, 0.07}, Complex{0.11, -0.07}, -0.3;
    b(0, 1) = Complex{0.13, -0.05};
    b(1, 0) = -b(0, 1);
    Eigen::MatrixXcd h(4, 4);
    h.topLeftCorner(2, 2) = a;
    h.topRightCorner(2, 2) = b;
    h.bottomLeftCorner(2, 2) = -b.conjugate();
    h.bottomRightCorner(2, 2) = a.conjugate();
    Eigen::MatrixXcd j = Eigen::MatrixXcd::Zero(4, 4);
    j.topRightCorner(2, 2).setIdentity();
    j.bottomLeftCorner(2, 2) = -Eigen::MatrixXcd::Identity(2, 2);
    close(h, h.adjoint().eval(), "Kramers h Hermiticity");
    close(h, (j * h.conjugate() * j.adjoint()).eval(), "Kramers h time-reversal symmetry");
    close(
        j * j.conjugate(), -Eigen::MatrixXcd::Identity(4, 4), "odd-electron time reversal square");
    const Sigma sigma(strings, h, std::vector<Complex>(4 * 4 * 4 * 4, 0.0));
    const Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(4, 4);
    const Eigen::MatrixXcd dense =
        reference_to_production(spinor_reference::one_body_matrix(spinor_reference::basis(4, 1), h),
                                spinor_reference::basis(4, 1),
                                *strings);
    close(sigma.apply(identity), dense, "odd-electron reference action");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(dense);
    require(solver.info() == Eigen::Success, "Kramers eigensolve failed");
    require(std::abs(solver.eigenvalues()(0) - solver.eigenvalues()(1)) < 1.0e-12 &&
                std::abs(solver.eigenvalues()(2) - solver.eigenvalues()(3)) < 1.0e-12,
            "one-electron Kramers pair is not degenerate");
}

void check_validation() {
    require_invalid([] { Strings(-1, 0, 0, 1); }, "negative spinors accepted");
    require_invalid([] { Strings(65, 1, 32, 100); }, "65 spinors accepted");
    require_invalid([] { Strings(4, -1, 2, 6); }, "negative electrons accepted");
    require_invalid([] { Strings(4, 5, 2, 6); }, "too many electrons accepted");
    require_invalid([] { Strings(4, 2, 5, 6); }, "invalid A/B split accepted");
    require_invalid([] { Strings(4, 2, 2, 5); }, "undersized max_states accepted");
    const auto strings = std::make_shared<const Strings>(4, 2, 2, 6);
    require_invalid([&] { strings->index(0); }, "absent mask accepted");
    require_invalid([&] { strings->index(std::uint64_t{1} << 5); }, "outside-range mask accepted");
    require_invalid(
        [&] {
            int sign = 0;
            strings->substitution(-1, 0, 0, &sign);
        },
        "invalid substitution source accepted");
    require_invalid([&] { strings->substitution(0, 0, 0, nullptr); },
                    "null substitution sign accepted");
    const Eigen::MatrixXcd h = Eigen::MatrixXcd::Zero(4, 4);
    require_invalid(
        [&] { Sigma(strings, Eigen::MatrixXcd::Zero(3, 3), std::vector<Complex>(256)); },
        "wrong one-body size accepted");
    require_invalid([&] { Sigma(strings, h, std::vector<Complex>(255)); },
                    "wrong ERI size accepted");
    require_invalid(
        [&] { Sigma(strings, h, std::vector<Eigen::MatrixXcd>{Eigen::MatrixXcd::Zero(3, 3)}); },
        "wrong RI factor shape accepted");
    const Sigma sigma(strings, h, std::vector<Complex>(256));
    require_invalid([&] { sigma.apply(Eigen::MatrixXcd::Zero(5, 1)); },
                    "wrong block row count accepted");
    require_invalid([&] { strings->apply_one_body(h, Eigen::MatrixXcd::Zero(5, 1)); },
                    "wrong one-body block row count accepted");
}

} // namespace

int main() {
    try {
        check_actions(4, 2, 2); // (2 spatial orbitals, 2 electrons): dimension 6.
        check_actions(6, 3, 3); // Cross-A/B two-step transfers: dimension 20.
        check_actions(4, 0, 2);
        check_actions(4, 4, 2);
        check_cross_sectors();
        check_counterterm();
        check_exterior_reference_limits();
        check_kramers_one_electron();
        check_validation();
        std::cout << "spinor strings: independent dense P0/P1 checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "spinor strings: " << error.what() << '\n';
        return 1;
    }
}
