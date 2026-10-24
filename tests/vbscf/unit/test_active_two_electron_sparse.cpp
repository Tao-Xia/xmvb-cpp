#include <array>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include "integrals/active/exact.hpp"
#include "vbscf/orbitals/preparation/result.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/directional.hpp"
#include "integrals/response/internal.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/active/pair_map.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/** @brief Linear reference ordering preserves values, duplicates and orientation. */
void check_unique_integral_order() {
    using namespace xmvb::vb;
    const std::vector<int> left{5, 0, 3, 2, 5, 0, 3};
    const std::vector<int> right{0, 5, 3, 1, 0, 0, 2};
    const std::vector<double> values{0.7, -0.2, 0.4, 0.3, 0.8, 0.9, -0.1};
    const auto graph = xmvb::integrals::build_ao_pair_graph(left, right, values, 3);
    std::vector<std::tuple<int, int, double>> expected, actual;
    for (std::size_t i = 0; i < left.size(); ++i)
        expected.emplace_back(left[i], right[i], values[i]);
    require(std::is_sorted(graph.integral_edges.begin(), graph.integral_edges.end()),
            "linear ordering differs from sorted CSR references");
    for (std::size_t i = 0; i < graph.integral_count(); ++i) {
        const auto eri = graph.integral(i);
        actual.emplace_back(eri.left_pair, eri.right_pair, eri.value);
        const int edge = graph.integral_edges[i];
        require(edge >= graph.row_offsets[eri.left_pair] &&
                    edge < graph.row_offsets[eri.left_pair + 1],
                "linear ordering assigned the wrong CSR row");
    }
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    require(actual == expected, "linear ordering changed a unique integral");
    require(std::all_of(
                graph.columns.begin(), graph.columns.end(), [](int column) { return column >= 0; }),
            "temporary primary-edge marks leaked into the published graph");
}

/** @brief Exercises long CSR rows, empty rows, edge remainders and SIMD tails. */
void check_csr_edge_groups() {
    using namespace xmvb::vb;
    constexpr int n_bf = 5;
    constexpr int n_pairs = n_bf * (n_bf + 1) / 2;
    std::vector<int> left, right;
    std::vector<double> values;
    Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
    for (int a = 0; a < n_pairs - 1; ++a) {
        for (int b = a; b < n_pairs - 1; ++b) {
            const double value = std::sin(0.217 * (a + 1) * (b + 3));
            left.push_back(a);
            right.push_back(b);
            values.push_back(value);
            reference(a, b) = reference(b, a) = value;
        }
    }
    const auto graph = xmvb::integrals::build_ao_pair_graph(left, right, values, n_bf);
    for (const int width : {0, 1, 7, 8, 9, 36, 65}) {
        xmvb::integrals::ExactCtxPairMatrix input(n_pairs, width);
        for (int row = 0; row < n_pairs; ++row)
            for (int column = 0; column < width; ++column)
                input(row, column) = std::cos(0.31 * (row + 2) * (column + 1));
        xmvb::integrals::ExactCtxPairMatrix output =
            xmvb::integrals::ExactCtxPairMatrix::Constant(n_pairs, width, 1e20);
        xmvb::integrals::detail::apply_exact_ao_pair_graph(graph, input, n_bf, width, &output);
        require(output.isApprox(reference * input, 2e-13),
                "CSR edge grouping changed a long, empty or remainder row");
    }
}

void check_direction_block(const xmvb::integrals::AoIntegralInput& integrals) {
    using namespace xmvb::vb;
    Eigen::MatrixXd orbitals(2, 2);
    orbitals << 1.0, 0.3, -0.2, 0.9;
    std::vector<Eigen::MatrixXd> directions(3, Eigen::MatrixXd::Zero(2, 2));
    directions[0] << 0.2, -0.4, 0.1, 0.3;
    directions[1] << -0.3, 0.1, 0.5, -0.2;
    xmvb::integrals::ExactCtxPairMatrix pair_map;
    xmvb::integrals::build_packed_orbital_pair_map(orbitals, &pair_map);
    xmvb::integrals::ActiveSpaceTwoElectronResult accepted;
    xmvb::integrals::detail::apply_exact_ao_pair_kernel(
        integrals, pair_map, 2, 3, &accepted.dense_ao_pair_products);
    auto cache = xmvb::integrals::build_exact_packed_active_two_electron_adjoint_cache(
        {0.2, -0.1, 0.3, 0.4, -0.2, 0.5}, orbitals, integrals, 2, accepted);
    std::vector<xmvb::integrals::ExactCtxPairMatrix> products;
    std::vector<Eigen::MatrixXd> fixed;
    const auto batch = xmvb::integrals::
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            cache, directions, integrals, &products, &fixed);
    const auto integral_only = xmvb::integrals::
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            cache, directions, integrals);
    require(batch.isApprox(integral_only, 1e-13), "integral-only batch changed delta ERI");
    for (int column = 0; column < 3; ++column) {
        xmvb::integrals::ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace workspace;
        std::vector<double> scalar;
        xmvb::integrals::compute_exact_packed_active_two_electron_integral_directional_derivative(
            cache, directions[column], integrals, &workspace, &scalar);
        require(batch.col(column).isApprox(
                    Eigen::Map<const Eigen::VectorXd>(scalar.data(), scalar.size()), 1e-13),
                "block exact ERI differs from scalar direction");
        require(products[column].isApprox(workspace.directional_pair_products, 1e-13),
                "block half transform differs from scalar image");
        const auto reference =
            xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                cache, directions[column], integrals);
        if (cache.exchange_integrals.size() != 0 && cache.orbital_pair_products.size() != 0) {
            require(products[column].size() == 0 && fixed[column].isApprox(reference, 1e-13),
                    "accepted exchange image returned an incorrect fixed adjoint");
        } else {
            require(fixed[column].size() == 0,
                    "resident pair batch retained a stale fixed adjoint");
            xmvb::integrals::ExactPackedActiveTwoElectronApplyWorkspace pullback;
            Eigen::MatrixXd fused;
            xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
                cache, directions[column], integrals, products[column], &pullback, &fused);
            require(fused.isApprox(reference, 1e-13), "reusing block images changed orbital HVP");
        }
    }
    // Also exercise the resident pair-image contract independently of whether
    // the accepted exchange image fits this tiny problem's resource budget.
    auto pair_cache = cache;
    pair_cache.exchange_integrals.resize(0, 0);
    const auto pair_batch = xmvb::integrals::
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            pair_cache, directions, integrals, &products, &fixed);
    require(pair_batch.isApprox(batch, 1e-13), "resident pair images changed delta ERI");
    for (int column = 0; column < 3; ++column) {
        require(fixed[column].size() == 0, "resident pair image returned a stale fixed adjoint");
        xmvb::integrals::ExactPackedActiveTwoElectronApplyWorkspace pullback;
        Eigen::MatrixXd fused;
        xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
            pair_cache, directions[column], integrals, products[column], &pullback, &fused);
        require(fused.isApprox(
                    xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                        pair_cache, directions[column], integrals),
                    1e-13),
                "resident pair-image HVP changed");
    }
    if (!integrals.pair_graph.has_compact_integral_stream()) {
        cache.accepted_pair_products = nullptr;
        const auto streamed = xmvb::integrals::
            compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
                cache, directions, integrals);
        require(streamed.isApprox(batch, 1e-13), "shared accepted tiles changed delta ERI");
        const auto with_adjoint = xmvb::integrals::
            compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
                cache, directions, integrals, &products, &fixed);
        require(with_adjoint.isApprox(batch, 1e-13), "streamed orbital batch changed delta ERI");
        for (int column = 0; column < 3; ++column) {
            require(products[column].size() == 0,
                    "streamed batch retained a full directional image");
            require(
                fixed[column].isApprox(
                    xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                        cache, directions[column], integrals),
                    1e-13),
                "streamed batch returned an incorrect fixed adjoint");
        }
    }
    const auto empty = xmvb::integrals::
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            cache, {}, integrals, &products, &fixed);
    require(empty.cols() == 0 && products.empty() && fixed.empty(),
            "empty batch retained stale outputs");
    bool rejected = false;
    try {
        xmvb::integrals::
            compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
                cache, {Eigen::MatrixXd::Zero(1, 2)}, integrals);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "batch accepted a malformed direction");
}

} // namespace

int main() {
    check_unique_integral_order();
    check_csr_edge_groups();
    constexpr int n_active = 11;

    const xmvb::integrals::AoPairGraph ordered_graph =
        xmvb::integrals::build_ao_pair_graph({2, 1, 2}, {0, 1, 2}, {1.25, -0.5, 3.0}, n_active);
    constexpr std::array<int, 3> expected_left = {1, 2, 2};
    constexpr std::array<int, 3> expected_right = {1, 0, 2};
    constexpr std::array<double, 3> expected_values = {-0.5, 1.25, 3.0};
    require(ordered_graph.integral_count() == 3, "AO-pair graph lost an integral");
    for (std::size_t index = 0; index < ordered_graph.integral_count(); ++index) {
        const xmvb::integrals::AoPairIntegral integral = ordered_graph.integral(index);
        require(integral.left_pair == expected_left[index] &&
                    integral.right_pair == expected_right[index] &&
                    integral.value == expected_values[index],
                "AO-pair storage traversal changed an integral or its orientation");
        require(index == 0 ||
                    ordered_graph.integral_edges[index - 1] < ordered_graph.integral_edges[index],
                "unique-integral CSR references are not strictly ordered");
    }

    // The large-AO representation keeps the canonical unique stream and omits
    // the duplicated symmetric CSR edges.  Its block action must remain
    // numerically identical to the fast CSR representation.
    constexpr int compact_n_bf = 2;
    const xmvb::integrals::AoPairGraph csr_graph = xmvb::integrals::build_ao_pair_graph(
        {0, 1, 2}, {0, 0, 1}, {2.0, -0.25, 0.75}, compact_n_bf);
    xmvb::integrals::AoPairGraph compact_graph;
    compact_graph.integral_rows = {0, 1, 2};
    compact_graph.integral_columns = {0, 0, 1};
    compact_graph.integral_values = {2.0, -0.25, 0.75};
    compact_graph.pair_first = csr_graph.pair_first;
    compact_graph.pair_second = csr_graph.pair_second;

    xmvb::integrals::ExactCtxPairMatrix coefficients(3, 2);
    coefficients << 0.5, -1.0, 2.0, 0.25, -0.5, 1.5;
    xmvb::integrals::AoIntegralInput csr_input;
    csr_input.n_basis_functions = compact_n_bf;
    csr_input.pair_graph = csr_graph;
    xmvb::integrals::AoIntegralInput compact_input;
    compact_input.n_basis_functions = compact_n_bf;
    compact_input.pair_graph = compact_graph;
    xmvb::integrals::ExactCtxPairMatrix csr_products;
    xmvb::integrals::ExactCtxPairMatrix compact_products;
    xmvb::integrals::detail::apply_exact_ao_pair_kernel(
        csr_input, coefficients, compact_n_bf, 2, &csr_products);
    xmvb::integrals::detail::apply_exact_ao_pair_kernel(
        compact_input, coefficients, compact_n_bf, 2, &compact_products);
    require(compact_products.isApprox(csr_products, 1.0e-14),
            "compact unique-integral action differs from symmetric CSR action");
    // Exercise SIMD lane tails and overwritten output independently of the
    // integral-transform code, including widths larger than a vector register.
    Eigen::Matrix3d reference_operator = Eigen::Matrix3d::Zero();
    for (int row = 0; row < 3; ++row)
        for (int edge = csr_graph.row_offsets[row]; edge < csr_graph.row_offsets[row + 1]; ++edge)
            reference_operator(row, csr_graph.columns[edge]) += csr_graph.values[edge];
    for (const int width : {0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 36, 64, 65}) {
        xmvb::integrals::ExactCtxPairMatrix inputs(3, width);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < width; ++column)
                inputs(row, column) = std::sin((row + 0.3) * (column + 0.7));
        xmvb::integrals::ExactCtxPairMatrix result =
            xmvb::integrals::ExactCtxPairMatrix::Constant(3, width, 1.0e20);
        xmvb::integrals::detail::apply_exact_ao_pair_kernel(
            csr_input, inputs, compact_n_bf, width, &result);
        require(result.isApprox(reference_operator * inputs, 2.0e-13),
                "SIMD CSR action lost lane tails or output initialization");
    }
    check_direction_block(csr_input);
    check_direction_block(compact_input);

    xmvb::vb::OrbitalPreparationResult orbitals;
    orbitals.active_sparse_row_offsets.resize(n_active + 1);
    orbitals.active_sparse_orbital_indices.resize(n_active);
    orbitals.active_sparse_values.assign(n_active, 1.0);
    for (int index = 0; index < n_active; ++index) {
        orbitals.active_sparse_row_offsets[index] = index;
        orbitals.active_sparse_orbital_indices[index] = index;
    }
    orbitals.active_sparse_row_offsets[n_active] = n_active;

    xmvb::integrals::AoIntegralInput integrals;
    integrals.n_basis_functions = n_active;
    integrals.pair_graph = xmvb::integrals::build_ao_pair_graph({0}, {0}, {2.0}, n_active);

    const auto result =
        xmvb::integrals::transform_exact_active_integrals(integrals,
                                                          orbitals.active_sparse_row_offsets,
                                                          orbitals.active_sparse_orbital_indices,
                                                          orbitals.active_sparse_values,
                                                          n_active);

    require(result.dense_active_coefficients.rows() == n_active &&
                result.dense_active_coefficients.cols() == n_active,
            "sparse active-ERI path omitted dense active coefficients");
    require(result.dense_active_coefficients.isApprox(Eigen::MatrixXd::Identity(n_active, n_active),
                                                      0.0),
            "sparse active-ERI path changed active coefficients");
    require(!result.packed_active_two_electron_integrals.empty() &&
                std::abs(result.packed_active_two_electron_integrals.front() - 2.0) < 1.0e-14,
            "sparse active-ERI contraction returned the wrong integral");
    require(result.dense_ao_pair_products.size() == 0,
            "memory-bounded active-ERI path retained dense AO-pair products");
    // One off-diagonal AO edge contributes through both orientations to a
    // diagonal active-pair element: 2 * K(0,1) * B(0,0) * B(1,0) = 8.
    integrals.pair_graph = xmvb::integrals::build_ao_pair_graph({0}, {1}, {2.0}, n_active);
    orbitals.active_sparse_orbital_indices[1] = 0;
    const auto mixed =
        xmvb::integrals::transform_exact_active_integrals(integrals,
                                                          orbitals.active_sparse_row_offsets,
                                                          orbitals.active_sparse_orbital_indices,
                                                          orbitals.active_sparse_values,
                                                          n_active);
    require(mixed.dense_ao_pair_products.size() == 0,
            "mixed-pair regression did not exercise sparse construction");
    if (std::abs(mixed.packed_active_two_electron_integrals.front() - 8.0) >= 1e-14)
        throw std::runtime_error(
            "sparse active diagonal: expected 8, got " +
            std::to_string(mixed.packed_active_two_electron_integrals.front()));
    // A 64-orbital packed tensor is about 16.5 MiB. Its sparse accumulator
    // must not become 32 full copies just because OMP_NUM_THREADS=32.
    constexpr int large_active = 64;
    xmvb::vb::OrbitalPreparationResult large_orbitals;
    for (int i = 0; i < large_active; ++i) {
        large_orbitals.active_sparse_row_offsets.push_back(i);
        large_orbitals.active_sparse_orbital_indices.push_back(i == 1 ? 0 : i);
        large_orbitals.active_sparse_values.push_back(1.0);
    }
    large_orbitals.active_sparse_row_offsets.push_back(large_active);
    xmvb::integrals::AoIntegralInput large_integrals;
    large_integrals.n_basis_functions = large_active;
    large_integrals.pair_graph =
        xmvb::integrals::build_ao_pair_graph({0}, {1}, {2.0}, large_active);
    const auto large_result = xmvb::integrals::transform_exact_active_integrals(
        large_integrals,
        large_orbitals.active_sparse_row_offsets,
        large_orbitals.active_sparse_orbital_indices,
        large_orbitals.active_sparse_values,
        large_active);
    require(large_result.dense_ao_pair_products.size() == 0 &&
                std::abs(large_result.packed_active_two_electron_integrals.front() - 8.0) < 1e-14,
            "bounded large sparse accumulator changed the active integral");
    return 0;
}
