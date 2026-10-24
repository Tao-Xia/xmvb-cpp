#include <cmath>
#include <iostream>
#include <stdexcept>
#include <Eigen/Core>
#include "core/pair_index.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/active/exact.hpp"
#include "integrals/active/pair_map.hpp"
#include "integrals/active/ri.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/directional.hpp"
#include "integrals/response/ri.hpp"

namespace {
using PairIndex = xmvb::core::PairIndex;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief Integer ordering/symmetries and unchanged public error handling. */
void check_pair_indices() {
    require(PairIndex::packed_pair_index(3, 1) == 7 &&
                PairIndex::packed_pair_of_pairs_index(7, 3) == 31 &&
                PairIndex::two_electron_storage_index(3, 1, 2, 0) == 31,
            "packed index examples changed");
    int expected = 0;
    for (int p = 0; p < 64; ++p)
        for (int q = 0; q <= p; ++q, ++expected) {
            require(PairIndex::packed_pair_index(p, q) == expected &&
                        PairIndex::packed_pair_index(q, p) == expected &&
                        PairIndex::packed_pair_of_pairs_index(p, q) == expected &&
                        PairIndex::packed_pair_of_pairs_index(q, p) == expected,
                    "packed index ordering changed");
        }
    int failures = 0;
    try {
        PairIndex::packed_pair_index(-1, 0);
    } catch (const std::invalid_argument&) {
        ++failures;
    }
    try {
        PairIndex::packed_pair_of_pairs_index(0, -1);
    } catch (const std::invalid_argument&) {
        ++failures;
    }
    try {
        PairIndex::two_electron_storage_index(0, 1, -1, 2);
    } catch (const std::invalid_argument&) {
        ++failures;
    }
    require(failures == 3, "packed index validation changed");
}
} // namespace

int main() {
    try {
        check_pair_indices();
        using namespace xmvb::integrals;
        constexpr int n_bf = 3, active = 2, pairs = 6;
        RiAoFactorization ri;
        ri.n_basis_functions = n_bf;
        ri.n_auxiliary_functions = 3;
        ri.n_packed_ao_pairs = pairs;
        ri.metric_whitened_ao_pair_factors.resize(3, pairs);
        for (int i = 0; i < ri.metric_whitened_ao_pair_factors.size(); ++i)
            ri.metric_whitened_ao_pair_factors.data()[i] = std::sin(0.7 * (i + 1));
        const Eigen::MatrixXd kernel =
            ri.metric_whitened_ao_pair_factors.transpose() * ri.metric_whitened_ao_pair_factors;
        std::vector<int> left, right;
        std::vector<double> values;
        for (int i = 0; i < pairs; ++i)
            for (int j = 0; j <= i; ++j) {
                left.push_back(i);
                right.push_back(j);
                values.push_back(kernel(i, j));
            }
        AoIntegralInput ao;
        ao.n_basis_functions = n_bf;
        ao.pair_graph = build_ao_pair_graph(left, right, values, n_bf);
        std::vector<int> offsets{0, 2, 4, 6}, indices{0, 1, 0, 1, 0, 1};
        std::vector<double> coefficients{0.8, 0.2, -0.1, 0.7, 0.4, -0.3};
        auto transform = [&](const Eigen::MatrixXd& c) {
            std::vector<double> rows;
            for (int i = 0; i < n_bf; ++i)
                for (int j = 0; j < active; ++j)
                    rows.push_back(c(i, j));
            return transform_exact_active_integrals(ao, offsets, indices, rows, active);
        };
        const Eigen::MatrixXd c =
            Eigen::Map<const Eigen::Matrix<double, 3, 2, Eigen::RowMajor>>(coefficients.data());
        const auto accepted = transform(c);
        PackedOrbitalPairMapMatrix map;
        build_packed_orbital_pair_map(c, &map);
        const Eigen::MatrixXd expected = map.transpose() * kernel * map;
        const auto view = make_active_space_two_electron_view(accepted);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                require(std::abs(lookup_active_space_two_electron_kernel_value(view, i, j, active) -
                                 expected(i, j)) < 1e-12,
                        "shared exact transform differs from explicit pair contraction");

        std::vector<double> w{0.3, -0.2, 0.7, 0.4, -0.1, 0.6};
        const auto cache = build_exact_packed_active_two_electron_adjoint_cache(
            w, accepted.dense_active_coefficients, ao, active, accepted);
        Eigen::MatrixXd d(n_bf, active);
        d << 0.1, -0.4, 0.2, 0.3, -0.2, 0.1;
        constexpr double step = 1e-5;
        const auto plus = transform(c + step * d), minus = transform(c - step * d);
        ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace work;
        std::vector<double> delta;
        compute_exact_packed_active_two_electron_integral_directional_derivative(
            cache, d, ao, &work, &delta);
        double derivative = 0;
        for (int i = 0; i < 6; ++i) {
            const double fd = (plus.packed_active_two_electron_integrals[i] -
                               minus.packed_active_two_electron_integrals[i]) /
                              (2 * step);
            require(std::abs(delta[i] - fd) < 2e-9,
                    "shared exact integral direction failed finite differences");
            derivative += w[i] * delta[i];
        }
        const auto gradient = backpropagate_exact_packed_active_two_electron_gradient(w, cache);
        require(
            (backpropagate_exact_packed_active_two_electron_gradient(w, ao, accepted) - gradient)
                    .norm() < 1e-11,
            "direct exact pullback differs from the accepted HVP cache");
        auto streamed = accepted;
        streamed.dense_ao_pair_products.resize(0, 0);
        require(
            (backpropagate_exact_packed_active_two_electron_gradient(w, ao, streamed) - gradient)
                    .norm() < 1e-11,
            "streamed exact pullback differs from resident accepted products");
        require(std::abs((gradient.array() * d.array()).sum() - derivative) < 1e-11,
                "shared integral adjoint identity failed");
        auto cache_at = [&](const ActiveSpaceTwoElectronResult& at) {
            return build_exact_packed_active_two_electron_adjoint_cache(
                w, at.dense_active_coefficients, ao, active, at);
        };
        const auto h = apply_exact_packed_active_two_electron_adjoint_hessian_vector(cache, d, ao);
        const Eigen::MatrixXd fd =
            (backpropagate_exact_packed_active_two_electron_gradient(w, cache_at(plus)) -
             backpropagate_exact_packed_active_two_electron_gradient(w, cache_at(minus))) /
            (2 * step);
        require((h - fd).norm() < 2e-8, "shared exact HVP failed finite differences");

        const auto ri_accepted = transform_ri_active_integrals(ri, c);
        require(
            (ri_accepted.ri_active_pair_factors - ri.metric_whitened_ao_pair_factors * map).norm() <
                1e-12,
            "shared RI forward transform changed pair factors");
        const auto ri_cache = build_ri_active_two_electron_response_cache(ri, ri_accepted, active);
        require(
            (backpropagate_ri_packed_active_two_electron_gradient(w, ri_cache) - gradient).norm() <
                1e-10,
            "shared RI adjoint differs from exact factors");
        require((apply_ri_packed_active_two_electron_adjoint_hessian_vector(w, ri_cache, d) - h)
                        .norm() < 1e-10,
                "shared RI HVP differs from exact factors");
        std::cout << "shared exact/RI transformation, adjoint and HVP: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
