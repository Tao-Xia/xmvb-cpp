/** @file Independent small-spin oracle and bounded covalent-support census.
 * --cost never constructs a product-amplitude array or a Hamiltonian.
 */
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/strings.hpp"

namespace {
using Bits = std::uint64_t;
using xmvb::mcscf::ProductSupport;
using xmvb::mcscf::SpinStrings;

void require(bool yes, const char* message) {
    if (!yes)
        throw std::runtime_error(message);
}

int pair(int p, int q) {
    if (p < q)
        std::swap(p, q);
    return p * (p + 1) / 2 + q;
}

int fermion_sign(Bits mask, int orbital) {
    return std::popcount(mask & ((Bits{1} << orbital) - 1)) % 2 ? -1 : 1;
}

// F_pp=n_p; F_pq=a^+_p a_q+a^+_q a_p for p>q.
std::vector<Eigen::MatrixXd> packed_generators(const SpinStrings& spin, int n) {
    const int d = n * (n + 1) / 2, size = spin.size();
    std::vector<Eigen::MatrixXd> result(d, Eigen::MatrixXd::Zero(size, size));
    for (int source = 0; source < size; ++source) {
        const Bits mask = spin.masks()[source];
        for (int q = 0; q < n; ++q)
            if (mask & (Bits{1} << q)) {
                const Bits removed = mask ^ (Bits{1} << q);
                for (int p = 0; p < n; ++p)
                    if (!(removed & (Bits{1} << p))) {
                        const Bits target = removed | (Bits{1} << p);
                        const int sign = fermion_sign(mask, q) * fermion_sign(removed, p);
                        result[pair(p, q)](spin.index(target), source) += sign;
                    }
            }
    }
    for (const auto& f : result)
        require((f - f.transpose()).norm() < 1e-14,
                "independent packed generator is not symmetric");
    return result;
}

Eigen::MatrixXd split_action(const Eigen::MatrixXd& x,
                             const Eigen::MatrixXd& k,
                             const std::vector<Eigen::MatrixXd>& alpha,
                             const std::vector<Eigen::MatrixXd>& beta) {
    Eigen::MatrixXd la = Eigen::MatrixXd::Zero(x.rows(), x.rows());
    Eigen::MatrixXd lb = Eigen::MatrixXd::Zero(x.cols(), x.cols());
    for (int p = 0; p < k.rows(); ++p)
        for (int q = 0; q < k.cols(); ++q) {
            la.noalias() += k(p, q) * alpha[p] * alpha[q];
            lb.noalias() += k(p, q) * beta[p] * beta[q];
        }
    Eigen::MatrixXd out = la * x + x * lb.transpose();
    for (int p = 0; p < k.rows(); ++p)
        for (int q = 0; q < k.cols(); ++q)
            out.noalias() += 2 * k(p, q) * alpha[q] * x * beta[p].transpose();
    return out;
}

// Small independent full-product reference: construct E_p on product indices.
std::vector<Eigen::MatrixXd> product_generators(const std::vector<Eigen::MatrixXd>& alpha,
                                                const std::vector<Eigen::MatrixXd>& beta) {
    const int a = alpha.front().rows(), b = beta.front().rows(), d = alpha.size();
    const Eigen::MatrixXd ia = Eigen::MatrixXd::Identity(a, a);
    const Eigen::MatrixXd ib = Eigen::MatrixXd::Identity(b, b);
    std::vector<Eigen::MatrixXd> e(d, Eigen::MatrixXd::Zero(a * b, a * b));
    for (int p = 0; p < d; ++p)
        for (int i = 0; i < a; ++i)
            for (int j = 0; j < b; ++j)
                for (int u = 0; u < a; ++u)
                    for (int v = 0; v < b; ++v)
                        e[p](i * b + j, u * b + v) =
                            alpha[p](i, u) * ib(j, v) + ia(i, u) * beta[p](j, v);
    return e;
}

Eigen::VectorXd flatten(const Eigen::MatrixXd& x) {
    const int a = x.rows(), b = x.cols();
    Eigen::VectorXd v(a * b);
    for (int i = 0; i < a; ++i)
        for (int j = 0; j < b; ++j)
            v(i * b + j) = x(i, j);
    return v;
}

Eigen::MatrixXd dense_reference(const Eigen::MatrixXd& x,
                                const Eigen::MatrixXd& k,
                                const std::vector<Eigen::MatrixXd>& alpha,
                                const std::vector<Eigen::MatrixXd>& beta) {
    const int a = x.rows(), b = x.cols(), d = k.rows();
    const auto e = product_generators(alpha, beta);
    const auto v = flatten(x);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(a * b);
    for (int p = 0; p < d; ++p)
        for (int q = 0; q < d; ++q)
            y.noalias() += k(p, q) * e[p] * (e[q] * v);
    Eigen::MatrixXd out(a, b);
    for (int i = 0; i < a; ++i)
        for (int j = 0; j < b; ++j)
            out(i, j) = y(i * b + j);
    return out;
}

// Cotangent of <bra,H_K ket> with respect to unrestricted K entries.
// For symmetric directions only its symmetric part is observable.
Eigen::MatrixXd split_cotangent(const Eigen::MatrixXd& bra,
                                const Eigen::MatrixXd& ket,
                                const std::vector<Eigen::MatrixXd>& alpha,
                                const std::vector<Eigen::MatrixXd>& beta) {
    const int d = alpha.size();
    Eigen::MatrixXd c(d, d);
    for (int p = 0; p < d; ++p)
        for (int q = 0; q < d; ++q) {
            const Eigen::MatrixXd local_a = alpha[p] * alpha[q] * ket;
            const Eigen::MatrixXd local_b = ket * beta[q] * beta[p];
            const Eigen::MatrixXd cross = alpha[q] * ket * beta[p].transpose();
            c(p, q) = (bra.array() * (local_a + local_b + 2 * cross).array()).sum();
        }
    return c;
}

Eigen::MatrixXd dense_cotangent(const Eigen::MatrixXd& bra,
                                const Eigen::MatrixXd& ket,
                                const std::vector<Eigen::MatrixXd>& alpha,
                                const std::vector<Eigen::MatrixXd>& beta) {
    const auto e = product_generators(alpha, beta);
    const auto left = flatten(bra), right = flatten(ket);
    const int d = e.size();
    Eigen::MatrixXd c(d, d);
    for (int p = 0; p < d; ++p)
        for (int q = 0; q < d; ++q)
            c(p, q) = left.dot(e[p] * (e[q] * right));
    return c;
}

Eigen::MatrixXd smooth(int rows, int cols, double phase) {
    Eigen::MatrixXd out(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out(i, j) = std::sin(phase + 0.31 * i + 0.17 * j + 0.03 * i * j);
    return out;
}

void near(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, const char* message) {
    require((actual - expected).norm() < 2e-11 * std::max(1.0, expected.norm()), message);
}

void check_split(int na, int nb) {
    constexpr int n = 4, d = n * (n + 1) / 2;
    const SpinStrings alpha(n, na), beta(n, nb);
    const auto fa = packed_generators(alpha, n), fb = packed_generators(beta, n);
    xmvb::mcscf::ActiveIntegrals active;
    const auto h = smooth(n, n, 0.23), g = smooth(d, d, 0.71);
    active.one_electron = 0.5 * (h + h.transpose());
    active.pair_kernel = 0.07 * (g + g.transpose());
    const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, active);
    const Eigen::MatrixXd k = action.channel_kernel();
    near(k, k.transpose(), "physical packed kernel is not symmetric");
    const auto check = [&](const Eigen::MatrixXd& x) {
        const auto split = split_action(x, k, fa, fb);
        near(split, action.apply(x), "spin split differs from DirectCiSigmaAction");
        near(split,
             dense_reference(x, k, fa, fb),
             "spin split differs from full-product generator oracle");
        const Eigen::MatrixXd raw = smooth(d, d, 1.13);
        const Eigen::MatrixXd direction = 0.5 * (raw + raw.transpose());
        const auto dy = split_action(x, direction, fa, fb);
        near(dy,
             dense_reference(x, direction, fa, fb),
             "arbitrary symmetric channel-kernel bilinear direction differs from dense oracle");
        const auto bra = smooth(x.rows(), x.cols(), 0.47);
        const Eigen::MatrixXd analytic = split_cotangent(bra, x, fa, fb);
        const Eigen::MatrixXd direct = dense_cotangent(bra, x, fa, fb);
        near(0.5 * (analytic + analytic.transpose()),
             0.5 * (direct + direct.transpose()),
             "local-plus-cross symmetric channel cotangent differs from direct generator Gram");
        const double bilinear = (bra.array() * dy.array()).sum();
        const double cotangent_direction = (analytic.array() * direction.array()).sum();
        require(std::abs(bilinear - cotangent_direction) <
                    2e-11 * std::max(1.0, std::abs(bilinear)),
                "analytic channel cotangent fails symmetric directional pairing");
        const double finite =
            ((bra.array() * dense_reference(x, k + 1e-5 * direction, fa, fb).array()).sum() -
             (bra.array() * dense_reference(x, k - 1e-5 * direction, fa, fb).array()).sum()) /
            (2e-5);
        require(std::abs(bilinear - finite) < 2e-9 * std::max(1.0, std::abs(bilinear)),
                "channel-kernel bilinear derivative differs from centered dense reference");
    };
    check(smooth(alpha.size(), beta.size(), 0.37));
    // A selected carrier is deliberately not closed under one-spin replacement.
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(alpha.size(), beta.size());
    selected(0, 0) = 0.8;
    selected(alpha.size() - 1, beta.size() - 1) = -0.4;
    if (alpha.size() > 2 && beta.size() > 2)
        selected(1, 2) = 0.21;
    check(selected);
    if (na == nb) {
        // One selected source has a same-spin image outside the intersection of
        // alpha- and beta-active target sets. A cross-support mask cannot gate H X.
        Eigen::MatrixXd one = Eigen::MatrixXd::Zero(alpha.size(), beta.size());
        one(0, 0) = 1;
        const int moved = alpha.index((alpha.masks()[0] ^ (Bits{1} << 0)) | (Bits{1} << 2));
        Eigen::MatrixXd direction = Eigen::MatrixXd::Zero(d, d);
        direction(pair(1, 1), pair(2, 0)) = 1;
        direction(pair(2, 0), pair(1, 1)) = 1;
        const auto image = split_action(one, direction, fa, fb);
        near(image,
             dense_reference(one, direction, fa, fb),
             "single-source support counterexample differs from dense generator action");
        for (const auto& generator : fb) {
            const Eigen::MatrixXd beta_only = one * generator.transpose();
            require(beta_only.row(moved).norm() == 0,
                    "counterexample target unexpectedly belongs to beta-active support");
        }
        require(moved != 0 && std::abs(image(moved, 0)) > 1e-12,
                "same-spin action did not reach outside the cross-channel intersection");
        for (int parity : {-1, 1}) {
            Eigen::MatrixXd x = smooth(alpha.size(), alpha.size(), 0.59);
            x = (x + parity * x.transpose()).eval();
            near(action.apply(x, parity),
                 split_action(x, k, fa, fb),
                 "certified exchange-parity action differs from spin split");
        }
    }
}

struct Neighbor {
    int source;
    int channel;
};
using NeighborTable = std::vector<std::vector<Neighbor>>;
NeighborTable make_neighbors(const SpinStrings& spin, int n) {
    NeighborTable table(spin.size());
    for (int i = 0; i < spin.size(); ++i) {
        const Bits mask = spin.masks()[i];
        for (int occupied = 0; occupied < n; ++occupied)
            if (mask & (Bits{1} << occupied))
                for (int empty = 0; empty < n; ++empty)
                    if (!(mask & (Bits{1} << empty)))
                        table[i].push_back(
                            {spin.index(mask ^ (Bits{1} << occupied) ^ (Bits{1} << empty)),
                             pair(occupied, empty)});
    }
    return table;
}

std::size_t table_bytes(const NeighborTable& table) {
    std::size_t bytes = table.capacity() * sizeof(NeighborTable::value_type);
    for (const auto& row : table)
        bytes += row.capacity() * sizeof(Neighbor);
    return bytes;
}

std::pair<int, int> frontier_formula(Bits a, Bits b, int n, int first, int last) {
    const Bits low = (Bits{1} << (first + 1)) - 1;
    const Bits high = ((Bits{1} << n) - 1) ^ ((Bits{1} << (last + 1)) - 1);
    return {std::popcount((~a) & low) * std::popcount(a & high),
            std::popcount((~b) & low) * std::popcount(b & high)};
}

void check_frontier(int n, int na, int nb) {
    const SpinStrings alpha(n, na), beta(n, nb);
    const ProductSupport support(n, alpha.masks(), beta.masks());
    const auto an = make_neighbors(alpha, n), bn = make_neighbors(beta, n);
    std::vector<std::int64_t> rows;
    std::uint64_t checked = 0;
    for (int b = 0; b < beta.size(); ++b) {
        rows.clear();
        support.append_column_rows(b, rows, 1);
        for (const int a : rows)
            if (!support.contains(a, b)) {
                const auto [first, last] = support.channel_envelope(a, b);
                require(first >= 0 && last >= first,
                        "frontier node lacks a deficient-prefix envelope");
                int ac = 0, bc = 0;
                for (const auto& link : an[a])
                    ac += support.contains(link.source, b);
                for (const auto& link : bn[b])
                    bc += support.contains(a, link.source);
                require(std::pair{ac, bc} ==
                            frontier_formula(alpha.masks()[a], beta.masks()[b], n, first, last),
                        "frontier orientation formula differs from explicit source neighbors");
                ++checked;
            }
    }
    require(checked > 0, "frontier oracle visited no unsupported nodes");
}

std::uint64_t choose(int n, int k) {
    std::uint64_t value = 1;
    for (int i = 1; i <= k; ++i)
        value = value * (n - k + i) / i;
    return value;
}

void cost(int n) {
    if (n != 12 && n != 14)
        throw std::invalid_argument("--cost accepts only 12 or 14");
    const auto started = std::chrono::steady_clock::now();
    const int electrons = n / 2, channels = n * (n + 1) / 2;
    const SpinStrings alpha(n, electrons), beta(n, electrons);
    const ProductSupport support(n, alpha.masks(), beta.masks());
    const auto an = make_neighbors(alpha, n), bn = make_neighbors(beta, n);
    std::vector<std::int64_t> rows;
    std::uint64_t s0 = 0, full_i = 0, upper_i = 0, old_channels = 0, cross_targets = 0,
                  cross_mac = 0, source_alpha_channels = 0, source_beta_channels = 0;
    std::size_t peak_rows = 0;
    for (int b = 0; b < beta.size(); ++b) {
        rows.clear();
        support.append_column_rows(b, rows, 1);
        peak_rows = std::max(peak_rows, rows.capacity() * sizeof(std::int64_t));
        for (const int a : rows) {
            ++full_i;
            const bool source = support.contains(a, b);
            if (source)
                ++s0;
            if (a > b)
                continue;
            ++upper_i;
            int ca = 0, cb = 0;
            if (source) {
                ca = cb = electrons;
                for (const auto& link : an[a])
                    ca += support.contains(link.source, b);
                for (const auto& link : bn[b])
                    cb += support.contains(a, link.source);
                source_alpha_channels += ca;
                source_beta_channels += cb;
                old_channels += channels;
            } else {
                const auto [first, last] = support.channel_envelope(a, b);
                require(first >= 0 && last >= first,
                        "cost frontier node lacks a certified envelope");
                const auto counts =
                    frontier_formula(alpha.masks()[a], beta.masks()[b], n, first, last);
                ca = counts.first;
                cb = counts.second;
                old_channels += static_cast<std::uint64_t>(first + 1) * (n - last - 1);
            }
            const auto products = static_cast<std::uint64_t>(ca) * cb;
            const std::uint64_t multiplicity = a == b ? 1 : 2;
            cross_targets += multiplicity * (products > 0);
            cross_mac += multiplicity * products;
        }
    }
    const std::uint64_t singles = electrons * (n - electrons);
    const std::uint64_t doubles = choose(electrons, 2) * choose(n - electrons, 2);
    const auto old_mac = old_channels * channels;
    const auto sparse_general = 2 * s0 * (1 + singles + doubles);
    const auto sparse_even = s0 * (1 + singles + doubles);
    const auto neighbors = table_bytes(an) + table_bytes(bn);
    const auto metadata = support.dynamic_bytes() + neighbors + peak_rows;
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << std::setprecision(12) << "mode=cost n=" << n << " electrons=" << electrons
              << " spin_strings=" << alpha.size() << " depth0_s0=" << s0
              << " depth1_full_i=" << full_i << " depth1_upper_i=" << upper_i
              << " old_channel_sum_upper=" << old_channels << " old_mac_upper=" << old_mac
              << " source_alpha_channel_sum_upper=" << source_alpha_channels
              << " source_beta_channel_sum_upper=" << source_beta_channels
              << " cross_targets_full=" << cross_targets << " cross_mac_full=" << cross_mac
              << " same_spin_sparse_apply_bound_general=" << sparse_general
              << " same_spin_sparse_apply_bound_even=" << sparse_even
              << " support_bytes=" << support.dynamic_bytes() << " neighbors_bytes=" << neighbors
              << " worker_rows_bytes=" << peak_rows << " metadata_accounted_peak_bytes=" << metadata
              << " workers=1 elapsed_seconds=" << seconds << " maxrss_kib=" << usage.ru_maxrss
              << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cost") {
            cost(std::stoi(argv[2]));
        } else if (argc == 1) {
            check_split(2, 2);
            check_split(1, 3);
            check_split(0, 4);
            check_frontier(4, 2, 2);
            check_frontier(4, 1, 3);
            check_frontier(6, 3, 3);
            std::cout << "spin_split_oracle=PASS frontier_oracle=PASS\n";
        } else
            throw std::invalid_argument("usage: test_spin_split [--cost 12|14]");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
