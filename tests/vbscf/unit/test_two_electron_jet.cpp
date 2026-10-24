#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "integrals/active/pair_map.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "integrals/response/directional.hpp"
#include "integrals/response/ri.hpp"

namespace {

void require_close(const Eigen::MatrixXd& actual,
                   const Eigen::MatrixXd& expected,
                   const char* message) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
        (actual - expected).norm() > 2e-11 * std::max(1.0, expected.norm()))
        throw std::runtime_error(message);
}

Eigen::MatrixXd coefficients(int rows, int columns, double phase) {
    Eigen::MatrixXd result(rows, columns);
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row)
            result(row, column) = 0.4 * std::sin(phase + 0.31 * (row + 1) * (column + 2)) +
                                  0.2 * std::cos(0.19 * (row + column + 1));
    return result;
}

std::vector<double> packed_gradient(int active, double phase) {
    const int pairs = active * (active + 1) / 2;
    std::vector<double> result(pairs * (pairs + 1) / 2);
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = 0.3 * std::cos(phase + 0.17 * (index + 1));
    return result;
}

void check_exact() {
    constexpr int basis = 4;
    constexpr int active = 2;
    constexpr int basis_pairs = basis * (basis + 1) / 2;
    constexpr int active_pairs = active * (active + 1) / 2;
    xmvb::integrals::AoIntegralInput ao;
    ao.n_basis_functions = basis;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int first = 0; first < basis_pairs; ++first)
        for (int second = 0; second <= first; ++second) {
            left.push_back(first);
            right.push_back(second);
            values.push_back(0.06 * std::cos(0.23 * (first + 2) * (second + 1)));
        }
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, basis);

    const Eigen::MatrixXd accepted_coefficients = coefficients(basis, active, 0.2);
    const Eigen::MatrixXd direction = coefficients(basis, active, -0.7);
    const auto accepted_gradient = packed_gradient(active, 0.1);
    const auto gradient_direction = packed_gradient(active, -0.5);
    xmvb::integrals::ExactCtxPairMatrix pair_map;
    xmvb::integrals::build_packed_orbital_pair_map(accepted_coefficients, &pair_map);
    xmvb::integrals::ActiveSpaceTwoElectronResult accepted;
    accepted.dense_active_coefficients = accepted_coefficients;
    xmvb::integrals::detail::apply_exact_ao_pair_kernel(
        ao, pair_map, basis, active_pairs, &accepted.dense_ao_pair_products);
    const auto resident = xmvb::integrals::build_exact_packed_active_two_electron_adjoint_cache(
        accepted_gradient, accepted_coefficients, ao, active, accepted);

    for (bool streamed : {false, true}) {
        auto cache = resident;
        if (streamed) {
            cache.accepted_pair_products = nullptr;
            cache.orbital_pair_products.resize(0, 0);
            cache.exchange_integrals.resize(0, 0);
        }
        const Eigen::MatrixXd reference =
            xmvb::integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                cache, direction, ao) +
            xmvb::integrals::backpropagate_exact_packed_active_two_electron_gradient(
                gradient_direction, cache);
        xmvb::integrals::ExactPackedActiveTwoElectronApplyWorkspace workspace;
        Eigen::MatrixXd jet;
        xmvb::integrals::apply_exact_active_two_electron_jet(
            cache, direction, gradient_direction, ao, &workspace, &jet);
        require_close(jet, reference, "exact combined jet disagrees with separate contractions");

        xmvb::integrals::ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace
            integral_workspace;
        std::vector<double> integral_with_fixed, integral_only;
        xmvb::integrals::compute_exact_packed_active_two_electron_integral_directional_derivative(
            cache, direction, ao, &integral_workspace, &integral_with_fixed);
        xmvb::integrals::compute_exact_packed_active_two_electron_integral_directional_derivative(
            cache, direction, ao, &integral_workspace, &integral_only, false);
        if (integral_with_fixed.size() != integral_only.size() ||
            integral_workspace.dense_fixed_adjoint_direction.size() != 0)
            throw std::runtime_error("integral-only exact direction kept a fixed-adjoint image");
        for (std::size_t index = 0; index < integral_only.size(); ++index)
            if (std::abs(integral_only[index] - integral_with_fixed[index]) >
                2e-12 * std::max(1.0, std::abs(integral_with_fixed[index])))
                throw std::runtime_error("integral-only exact direction changed packed integrals");

        if (!streamed) {
            auto no_exchange = cache;
            no_exchange.exchange_integrals.resize(0, 0);
            xmvb::integrals::ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace
                direction_workspace;
            std::vector<double> integral_direction;
            xmvb::integrals::
                compute_exact_packed_active_two_electron_integral_directional_derivative(
                    no_exchange, direction, ao, &direction_workspace, &integral_direction);
            if (direction_workspace.directional_pair_products.rows() != basis_pairs)
                throw std::runtime_error("exact direction did not retain pair products");
            xmvb::integrals::apply_exact_active_two_electron_jet(
                no_exchange,
                direction,
                gradient_direction,
                ao,
                &workspace,
                &jet,
                &direction_workspace.directional_pair_products);
            require_close(jet,
                          reference,
                          "exact precomputed-direction jet disagrees with separate contractions");
        }
    }
}

void check_ri(int auxiliary, bool expect_resident_plan) {
    constexpr int basis = 4;
    constexpr int active = 2;
    constexpr int basis_pairs = basis * (basis + 1) / 2;
    xmvb::integrals::RiAoFactorization factorization;
    factorization.n_basis_functions = basis;
    factorization.n_auxiliary_functions = auxiliary;
    factorization.n_packed_ao_pairs = basis_pairs;
    factorization.metric_whitened_ao_pair_factors.resize(auxiliary, basis_pairs);
    for (Eigen::Index index = 0; index < factorization.metric_whitened_ao_pair_factors.size();
         ++index)
        factorization.metric_whitened_ao_pair_factors.data()[index] =
            0.4 * std::sin(0.11 * (index + 1));
    const Eigen::MatrixXd accepted_coefficients = coefficients(basis, active, 0.4);
    const Eigen::MatrixXd direction = coefficients(basis, active, -0.3);
    const auto accepted_gradient = packed_gradient(active, 0.9);
    const auto gradient_direction = packed_gradient(active, -0.2);
    xmvb::integrals::PackedOrbitalPairMapMatrix pair_map;
    xmvb::integrals::build_packed_orbital_pair_map(accepted_coefficients, &pair_map);
    xmvb::integrals::ActiveSpaceTwoElectronResult accepted;
    accepted.representation =
        xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    accepted.n_auxiliary_functions = auxiliary;
    accepted.dense_active_coefficients = accepted_coefficients;
    accepted.ri_active_pair_factors = factorization.metric_whitened_ao_pair_factors * pair_map;
    auto resident = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, accepted, active);
    xmvb::integrals::prepare_ri_orbital_pair_products(resident);
    if (!resident.orbital_pair_products ||
        xmvb::integrals::ri_uses_orbital_pairs(resident) != expect_resident_plan)
        throw std::runtime_error("RI resident contraction selected the wrong plan");

    const auto gradient_at = [&](double step) {
        const Eigen::MatrixXd shifted_coefficients = accepted_coefficients + step * direction;
        xmvb::integrals::PackedOrbitalPairMapMatrix shifted_pair_map;
        xmvb::integrals::build_packed_orbital_pair_map(shifted_coefficients, &shifted_pair_map);
        xmvb::integrals::ActiveSpaceTwoElectronResult shifted;
        shifted.representation =
            xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
        shifted.n_auxiliary_functions = auxiliary;
        shifted.dense_active_coefficients = shifted_coefficients;
        shifted.ri_active_pair_factors =
            factorization.metric_whitened_ao_pair_factors * shifted_pair_map;
        const auto shifted_cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
            factorization, shifted, active);
        std::vector<double> shifted_gradient = accepted_gradient;
        for (std::size_t i = 0; i < shifted_gradient.size(); ++i)
            shifted_gradient[i] += step * gradient_direction[i];
        return xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
            shifted_gradient, shifted_cache);
    };
    constexpr double fd_step = 1e-5;
    const Eigen::MatrixXd finite_difference =
        (gradient_at(fd_step) - gradient_at(-fd_step)) / (2 * fd_step);
    for (bool streamed : {false, true}) {
        auto cache = resident;
        if (streamed) {
            cache.accepted_one_index_factors.reset();
            cache.orbital_pair_products.reset();
            cache.auxiliary_tile_size = 1;
        }
        if (xmvb::integrals::ri_uses_orbital_pairs(cache) != (!streamed && expect_resident_plan))
            throw std::runtime_error("RI jet contraction plan did not respect residence");
        const Eigen::MatrixXd factor_direction =
            xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(cache, direction);
        const Eigen::MatrixXd reference =
            xmvb::integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
                accepted_gradient, cache, direction, &factor_direction) +
            xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                gradient_direction, cache);
        require_close(
            xmvb::integrals::apply_ri_active_two_electron_jet(
                accepted_gradient, cache, direction, gradient_direction, &factor_direction),
            reference,
            "RI precomputed-factor jet disagrees with separate contractions");
        require_close(xmvb::integrals::apply_ri_active_two_electron_jet(
                          accepted_gradient, cache, direction, gradient_direction),
                      reference,
                      "RI generated-factor jet disagrees with separate contractions");
        const Eigen::MatrixXd jet = xmvb::integrals::apply_ri_active_two_electron_jet(
            accepted_gradient, cache, direction, gradient_direction, &factor_direction);
        if ((jet - finite_difference).norm() > 2e-7 * std::max(1.0, finite_difference.norm()))
            throw std::runtime_error("RI jet disagrees with independent finite difference");
    }
}

} // namespace

int main() {
    check_exact();
    check_ri(5, false);
    check_ri(7, true);
}
