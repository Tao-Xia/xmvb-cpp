#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "integrals/ao/graph.hpp"
#include "integrals/ao/input.hpp"
#include "integrals/ao/ri.hpp"
#include "integrals/jk/contraction.hpp"

namespace {
using Matrix = Eigen::MatrixXd;
using xmvb::integrals::AoIntegralInput;
using xmvb::integrals::AoPairGraph;
using xmvb::integrals::JkWeights;
using xmvb::integrals::RiAoFactorization;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

int packed_pair(int i, int j) {
    if (i < j)
        std::swap(i, j);
    return i * (i + 1) / 2 + j;
}

/** @brief Expand unique pair-pair values to an independent full ERI tensor. */
std::vector<double> expand_eri(const AoPairGraph& graph, int n) {
    std::vector<double> eri(static_cast<std::size_t>(n) * n * n * n, 0.0);
    const auto index = [n](int i, int j, int k, int l) {
        return ((static_cast<std::size_t>(i) * n + j) * n + k) * n + l;
    };
    graph.for_each_integral([&](double value, int i, int j, int k, int l) {
        const int left[2][2] = {{i, j}, {j, i}};
        const int right[2][2] = {{k, l}, {l, k}};
        for (const auto& a : left)
            for (const auto& b : right) {
                eri[index(a[0], a[1], b[0], b[1])] = value;
                eri[index(b[0], b[1], a[0], a[1])] = value;
            }
    });
    return eri;
}

Matrix brute_jk(const AoPairGraph& graph, const Matrix& density, JkWeights weights) {
    const int n = static_cast<int>(density.rows());
    const auto eri = expand_eri(graph, n);
    const auto index = [n](int i, int j, int k, int l) {
        return ((static_cast<std::size_t>(i) * n + j) * n + k) * n + l;
    };
    Matrix result = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    const double value = eri[index(i, j, k, l)];
                    result(i, j) += weights.coulomb * value * density(k, l);
                    result(i, j) += weights.exchange * eri[index(i, k, j, l)] * density(k, l);
                }
    return result;
}

AoIntegralInput make_ao() {
    constexpr int n = 3;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int p = 0; p < n * (n + 1) / 2; ++p)
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            // Nonseparable signed values exercise exchange, coincident indices and
            // pair exchange without assuming positivity of an arbitrary ERI table.
            values.push_back(std::sin(0.37 * (p + 1) * (q + 2)) - 0.13 * (p == q ? 1.0 : 0.0));
        }
    AoIntegralInput ao;
    ao.n_basis_functions = n;
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(
        std::move(left), std::move(right), std::move(values), n);
    return ao;
}

AoPairGraph make_compact_graph(const AoPairGraph& source) {
    AoPairGraph compact;
    compact.pair_first = source.pair_first;
    compact.pair_second = source.pair_second;
    compact.integral_rows.reserve(source.integral_count());
    compact.integral_columns.reserve(source.integral_count());
    compact.integral_values.reserve(source.integral_count());
    for (std::size_t e = 0; e < source.integral_count(); ++e) {
        const auto integral = source.integral(e);
        compact.integral_rows.push_back(integral.left_pair);
        compact.integral_columns.push_back(integral.right_pair);
        compact.integral_values.push_back(integral.value);
    }
    return compact;
}

RiAoFactorization make_ri() {
    constexpr int n = 3;
    constexpr int npair = n * (n + 1) / 2;
    RiAoFactorization ri;
    ri.n_basis_functions = n;
    ri.n_auxiliary_functions = 4;
    ri.n_packed_ao_pairs = npair;
    ri.metric_whitened_ao_pair_factors.resize(4, npair);
    for (int a = 0; a < 4; ++a)
        for (int p = 0; p < npair; ++p)
            ri.metric_whitened_ao_pair_factors(a, p) =
                std::cos(0.29 * (a + 1) * (p + 2)) - 0.08 * a * p;
    return ri;
}

Matrix brute_ri(const RiAoFactorization& ri, const Matrix& density, JkWeights weights) {
    const int n = ri.n_basis_functions;
    const auto pair_factor = [&](int a, int i, int j) {
        return ri.metric_whitened_ao_pair_factors(a, packed_pair(i, j));
    };
    Matrix result = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    double coulomb = 0.0, exchange = 0.0;
                    for (int a = 0; a < ri.n_auxiliary_functions; ++a) {
                        coulomb += pair_factor(a, i, j) * pair_factor(a, k, l);
                        exchange += pair_factor(a, i, k) * pair_factor(a, j, l);
                    }
                    result(i, j) +=
                        (weights.coulomb * coulomb + weights.exchange * exchange) * density(k, l);
                }
    return result;
}

void compare(const Matrix& actual, const Matrix& expected, const std::string& label) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
            label + ": output dimensions changed");
    require((actual - expected).cwiseAbs().maxCoeff() < 1e-11,
            label + ": differs from explicit four-index contraction");
    require((actual - actual.transpose()).cwiseAbs().maxCoeff() < 1e-12,
            label + ": output is not symmetric");
}

void check_exact_backend(const AoIntegralInput& ao,
                         const Matrix& density,
                         const JkWeights& weights,
                         const std::string& layout) {
    compare(xmvb::integrals::contract_jk(density, ao, weights),
            brute_jk(ao.pair_graph, density, weights),
            layout);
}

void check_ri_backend(const RiAoFactorization& ri,
                      const Matrix& density,
                      const JkWeights& weights,
                      const std::string& label) {
    compare(
        xmvb::integrals::contract_jk(density, ri, weights), brute_ri(ri, density, weights), label);
}

} // namespace

int main() {
    try {
        const AoIntegralInput ao = make_ao();
        AoIntegralInput compact = ao;
        compact.pair_graph = make_compact_graph(ao.pair_graph);
        const RiAoFactorization ri = make_ri();

        Matrix signed_density(3, 3);
        signed_density << 0.7, -0.2, 0.15, -0.2, -0.4, 0.3, 0.15, 0.3, 0.9;
        Matrix rank_one_vector(3, 1);
        rank_one_vector << 0.3, -0.8, 0.5;
        const Matrix rank_one = rank_one_vector * rank_one_vector.transpose();

        const std::vector<JkWeights> weights = {
            {1.0, 0.0}, {0.0, 1.0}, {1.0, -0.5}, {2.0, -1.0}, {0.0, 0.0}};
        const std::vector<std::pair<Matrix, std::string>> densities = {
            {signed_density, "signed density"}, {rank_one, "rank-one density"}};
        for (std::size_t wi = 0; wi < weights.size(); ++wi)
            for (const auto& item : densities) {
                const std::string label = "weight case " + std::to_string(wi) + ", " + item.second;
                check_exact_backend(ao, item.first, weights[wi], "CSR " + label);
                check_exact_backend(compact, item.first, weights[wi], "compact " + label);
                check_ri_backend(ri, item.first, weights[wi], "RI " + label);
            }
        std::cout << "exact CSR/compact and RI J-K contractions: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
