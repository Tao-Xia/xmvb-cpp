#include "integrals/response/adjoint.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "integrals/response/internal.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "integrals/response/orbital_pair.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::integrals {
namespace {

using detail::accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache;
using detail::accumulate_pair_gradient_rows;
using detail::accumulate_pair_product_adjoint;
using detail::apply_exact_ao_pair_kernel;
using detail::apply_generated_pair_rows;
using detail::build_active_pair_gradient_matrix;
using detail::build_active_pair_list;
using detail::build_ao_pair_component_tables;
using detail::build_mixed_ao_pair_to_active_pair_coefficients_from_cache;
using detail::multiply_pair_coefficients_by_gradient_matrix;

/** @brief Builds shared three-active/one-AO integrals without thread replicas. */
void build_orbital_pair_products(ExactPackedActiveTwoElectronAdjointCache* cache) {
    const Eigen::Index n_bf = cache->n_basis_functions;
    const Eigen::Index n = cache->n_active_orbitals;
    const Eigen::Index pairs = cache->active_pair_first_indices.size();
    constexpr std::size_t budget = 64 * 1024 * 1024;
    const std::size_t columns = static_cast<std::size_t>(n) * pairs;
    if (cache->accepted_pair_products == nullptr ||
        columns > budget / sizeof(double) / static_cast<std::size_t>(n_bf)) {
        return;
    }
    cache->orbital_pair_products.setZero(n_bf * n, pairs);
    const auto& c = *cache->accepted_active_coefficients;
    const auto& products = *cache->accepted_pair_products;
#pragma omp parallel for schedule(static)
    for (Eigen::Index mu = 0; mu < n_bf; ++mu) {
        for (Eigen::Index nu = 0; nu < n_bf; ++nu) {
            const double* source = products.row(detail::ao_pair_index(mu, nu)).data();
            for (Eigen::Index a = 0; a < n; ++a) {
                const double coefficient = c(nu, a);
                double* target = cache->orbital_pair_products.row(mu * n + a).data();
#pragma omp simd
                for (Eigen::Index p = 0; p < pairs; ++p) {
                    target[p] += coefficient * source[p];
                }
            }
        }
    }
}

} // namespace

Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_gradient,
    const AoIntegralInput& ao,
    const ActiveSpaceTwoElectronResult& accepted) {
    if (ao.n_basis_functions <= 0 ||
        accepted.dense_active_coefficients.rows() != ao.n_basis_functions ||
        accepted.dense_active_coefficients.cols() < 1)
        throw std::invalid_argument("invalid exact integral pullback dimensions");
    ExactPackedActiveTwoElectronAdjointCache cache;
    cache.n_basis_functions = ao.n_basis_functions;
    cache.n_active_orbitals = static_cast<int>(accepted.dense_active_coefficients.cols());
    cache.ao_integral_input = &ao;
    cache.accepted_active_coefficients = &accepted.dense_active_coefficients;
    if (accepted.dense_ao_pair_products.size() != 0)
        cache.accepted_pair_products = &accepted.dense_ao_pair_products;
    build_ao_pair_component_tables(
        ao.n_basis_functions, &cache.ao_pair_first_indices, &cache.ao_pair_second_indices);
    for (const auto pair : build_active_pair_list(cache.n_active_orbitals)) {
        cache.active_pair_first_indices.push_back(pair.first);
        cache.active_pair_second_indices.push_back(pair.second);
    }
    return backpropagate_exact_packed_active_two_electron_gradient(packed_gradient, cache);
}

Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache) {
    const int n_bf = accepted_cache.n_basis_functions;
    const int n_ao = accepted_cache.n_active_orbitals;
    const std::size_t n_active_pairs = accepted_cache.active_pair_first_indices.size();
    const std::size_t expected_packed_size = n_active_pairs * (n_active_pairs + 1) / 2;
    const std::size_t n_bf_pairs = static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
    if (n_bf <= 0 || n_ao <= 0 ||
        accepted_cache.active_pair_second_indices.size() != n_active_pairs ||
        packed_active_two_electron_gradient.size() != expected_packed_size ||
        accepted_cache.accepted_active_coefficients == nullptr ||
        accepted_cache.accepted_active_coefficients->rows() != n_bf ||
        accepted_cache.accepted_active_coefficients->cols() != n_ao) {
        throw std::invalid_argument(
            "cached exact active-2e adjoint pullback dimensions are inconsistent");
    }

    const auto active_pairs = build_active_pair_list(n_ao);
    const ExactCtxPairMatrix active_pair_gradient_matrix =
        build_active_pair_gradient_matrix(packed_active_two_electron_gradient, active_pairs);
    if (accepted_cache.orbital_pair_products.size() != 0) {
        return pullback_orbital_pairs(accepted_cache.orbital_pair_products,
                                      active_pair_gradient_matrix,
                                      accepted_cache.active_pair_first_indices,
                                      accepted_cache.active_pair_second_indices,
                                      n_ao);
    }
    Eigen::MatrixXd dense_active_gradient = Eigen::MatrixXd::Zero(n_bf, n_ao);
    if (accepted_cache.accepted_pair_products != nullptr) {
        if (accepted_cache.accepted_pair_products->rows() !=
                static_cast<Eigen::Index>(n_bf_pairs) ||
            accepted_cache.accepted_pair_products->cols() !=
                static_cast<Eigen::Index>(n_active_pairs)) {
            throw std::invalid_argument("accepted pair-product dimensions are inconsistent");
        }
        accumulate_pair_product_adjoint(*accepted_cache.accepted_pair_products,
                                        active_pair_gradient_matrix,
                                        *accepted_cache.accepted_active_coefficients,
                                        accepted_cache,
                                        &dense_active_gradient);
        return dense_active_gradient;
    }
    if (accepted_cache.ao_integral_input == nullptr) {
        throw std::invalid_argument("streamed exact 2e cache is missing its AO operator");
    }

    ExactCtxPairMatrix pair_products_tile;
    ExactCtxPairMatrix pair_gradients_tile;
    const Eigen::Index tile_rows = n_bf;
    for (Eigen::Index row_begin = 0; row_begin < static_cast<Eigen::Index>(n_bf_pairs);
         row_begin += tile_rows) {
        const Eigen::Index row_count =
            std::min(tile_rows, static_cast<Eigen::Index>(n_bf_pairs) - row_begin);
        apply_generated_pair_rows(*accepted_cache.ao_integral_input,
                                  *accepted_cache.accepted_active_coefficients,
                                  nullptr,
                                  accepted_cache,
                                  row_begin,
                                  row_count,
                                  &pair_products_tile,
                                  nullptr);
        pair_gradients_tile.noalias() = pair_products_tile * active_pair_gradient_matrix;
        accumulate_pair_gradient_rows(pair_gradients_tile,
                                      row_begin,
                                      *accepted_cache.accepted_active_coefficients,
                                      accepted_cache,
                                      &dense_active_gradient);
    }
    return dense_active_gradient;
}

ExactPackedActiveTwoElectronAdjointCache build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::MatrixXd& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_ao,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result) {
    const int n_bf = ao_integral_input.n_basis_functions;
    if (n_bf <= 0 || n_ao <= 0) {
        throw std::invalid_argument("exact 2e HVP cache dimensions must be positive");
    }
    if (ao_integral_input.pair_graph.empty()) {
        throw std::invalid_argument("exact 2e HVP cache requires materialized AO integrals");
    }

    if (accepted_dense_active_coefficients.rows() != n_bf ||
        accepted_dense_active_coefficients.cols() != n_ao) {
        throw std::invalid_argument("accepted dense active coefficient shape mismatch");
    }

    const auto active_pairs = build_active_pair_list(n_ao);
    const std::size_t n_active_pairs = active_pairs.size();
    const std::size_t n_bf_pairs = n_bf * (n_bf + 1) / 2;
    const std::size_t expected_packed_gradient_size = n_active_pairs * (n_active_pairs + 1) / 2;
    if (packed_active_two_electron_gradient.size() != expected_packed_gradient_size) {
        throw std::invalid_argument("packed active two-electron gradient size mismatch");
    }

    ExactPackedActiveTwoElectronAdjointCache cache;
    cache.n_basis_functions = n_bf;
    cache.n_active_orbitals = n_ao;
    cache.ao_integral_input = &ao_integral_input;
    cache.accepted_active_coefficients = &accepted_dense_active_coefficients;
    build_ao_pair_component_tables(
        n_bf, &cache.ao_pair_first_indices, &cache.ao_pair_second_indices);
    cache.active_pair_first_indices.reserve(n_active_pairs);
    cache.active_pair_second_indices.reserve(n_active_pairs);
    for (const auto& active_pair : active_pairs) {
        cache.active_pair_first_indices.push_back(active_pair.first);
        cache.active_pair_second_indices.push_back(active_pair.second);
    }
    cache.active_pair_gradient_matrix =
        build_active_pair_gradient_matrix(packed_active_two_electron_gradient, active_pairs);
    if (accepted_active_space_two_electron_result.dense_ao_pair_products.size() != 0) {
        if (accepted_active_space_two_electron_result.dense_ao_pair_products.rows() !=
                static_cast<Eigen::Index>(n_bf_pairs) ||
            accepted_active_space_two_electron_result.dense_ao_pair_products.cols() !=
                static_cast<Eigen::Index>(n_active_pairs)) {
            throw std::invalid_argument("accepted dense AO pair product cache size mismatch");
        }
        cache.accepted_pair_products =
            &accepted_active_space_two_electron_result.dense_ao_pair_products;
    }
    build_orbital_pair_products(&cache);
    detail::build_exchange_integrals(&cache);

    return cache;
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input) {
    ExactPackedActiveTwoElectronApplyWorkspace workspace;
    Eigen::MatrixXd dense_active_gradient_direction;
    apply_exact_packed_active_two_electron_adjoint_hessian_vector(accepted_cache,
                                                                  dense_active_direction,
                                                                  ao_integral_input,
                                                                  &workspace,
                                                                  &dense_active_gradient_direction);
    return dense_active_gradient_direction;
}

namespace {
void apply_exact_active_two_electron_jet_impl(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction,
    const ExactCtxPairMatrix* active_pair_gradient_direction,
    const ExactCtxPairMatrix* directional_pair_products) {
    if (workspace == nullptr) {
        throw std::invalid_argument("exact 2e workspace must not be null");
    }
    const int n_bf = ao_integral_input.n_basis_functions;
    const int n_ao = accepted_cache.n_active_orbitals;
    if (n_bf <= 0 || n_ao <= 0) {
        throw std::invalid_argument("exact two-electron HVP dimensions must be positive");
    }
    if (accepted_cache.n_basis_functions != n_bf) {
        throw std::invalid_argument("exact 2e cache basis dimension mismatch");
    }
    if (accepted_cache.accepted_active_coefficients == nullptr ||
        accepted_cache.accepted_active_coefficients->rows() != n_bf ||
        accepted_cache.accepted_active_coefficients->cols() != n_ao) {
        throw std::invalid_argument("exact 2e cache accepted coefficient mismatch");
    }
    if (ao_integral_input.pair_graph.empty()) {
        throw std::invalid_argument("exact two-electron HVP requires materialized AO integrals");
    }

    if (dense_active_direction.rows() != n_bf || dense_active_direction.cols() != n_ao) {
        throw std::invalid_argument(
            "dense active coefficient shape mismatch in exact two-electron HVP");
    }
    const std::size_t n_active_pairs = accepted_cache.active_pair_first_indices.size();
    if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
        throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
    }
    workspace->dense_active_direction = dense_active_direction;

    if (accepted_cache.active_pair_gradient_matrix.rows() !=
            static_cast<Eigen::Index>(n_active_pairs) ||
        accepted_cache.active_pair_gradient_matrix.cols() !=
            static_cast<Eigen::Index>(n_active_pairs)) {
        throw std::invalid_argument("exact 2e cache pair-gradient size mismatch");
    }
    if (active_pair_gradient_direction != nullptr &&
        (active_pair_gradient_direction->rows() != static_cast<Eigen::Index>(n_active_pairs) ||
         active_pair_gradient_direction->cols() != static_cast<Eigen::Index>(n_active_pairs))) {
        throw std::invalid_argument("exact 2e directional pair-gradient size mismatch");
    }
    const Eigen::Index n_bf_pairs = static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2;
    if (accepted_cache.ao_pair_first_indices.size() != static_cast<std::size_t>(n_bf_pairs) ||
        accepted_cache.ao_pair_second_indices.size() != static_cast<std::size_t>(n_bf_pairs)) {
        throw std::invalid_argument("exact 2e AO-pair index size mismatch");
    }
    if (directional_pair_products != nullptr &&
        (directional_pair_products->rows() != n_bf_pairs ||
         directional_pair_products->cols() != static_cast<Eigen::Index>(n_active_pairs))) {
        throw std::invalid_argument("exact 2e directional pair-product size mismatch");
    }
    if (accepted_cache.accepted_pair_products != nullptr &&
        (accepted_cache.accepted_pair_products->rows() != n_bf_pairs ||
         accepted_cache.accepted_pair_products->cols() !=
             static_cast<Eigen::Index>(n_active_pairs))) {
        throw std::invalid_argument("exact 2e accepted pair-product size mismatch");
    }
    workspace->dense_active_gradient_direction.resize(n_bf, n_ao);
    workspace->dense_active_gradient_direction.setZero();
    if (accepted_cache.accepted_pair_products == nullptr) {
        const Eigen::Index tile_rows = n_bf;
        for (Eigen::Index row_begin = 0; row_begin < n_bf_pairs; row_begin += tile_rows) {
            const Eigen::Index row_count = std::min(tile_rows, n_bf_pairs - row_begin);
            apply_generated_pair_rows(
                ao_integral_input,
                *accepted_cache.accepted_active_coefficients,
                directional_pair_products == nullptr ? &workspace->dense_active_direction : nullptr,
                accepted_cache,
                row_begin,
                row_count,
                &workspace->pair_products_tile,
                directional_pair_products == nullptr ? &workspace->directional_pair_products_tile
                                                     : nullptr);

            workspace->pair_gradients_tile.noalias() =
                workspace->pair_products_tile * accepted_cache.active_pair_gradient_matrix;
            accumulate_pair_gradient_rows(workspace->pair_gradients_tile,
                                          row_begin,
                                          workspace->dense_active_direction,
                                          accepted_cache,
                                          &workspace->dense_active_gradient_direction);

            if (directional_pair_products != nullptr) {
                workspace->pair_gradients_tile.noalias() =
                    directional_pair_products->middleRows(row_begin, row_count) *
                    accepted_cache.active_pair_gradient_matrix;
            } else {
                workspace->pair_gradients_tile.noalias() =
                    workspace->directional_pair_products_tile *
                    accepted_cache.active_pair_gradient_matrix;
            }
            accumulate_pair_gradient_rows(workspace->pair_gradients_tile,
                                          row_begin,
                                          *accepted_cache.accepted_active_coefficients,
                                          accepted_cache,
                                          &workspace->dense_active_gradient_direction);
            if (active_pair_gradient_direction != nullptr) {
                workspace->pair_gradients_tile.noalias() =
                    workspace->pair_products_tile * *active_pair_gradient_direction;
                accumulate_pair_gradient_rows(workspace->pair_gradients_tile,
                                              row_begin,
                                              *accepted_cache.accepted_active_coefficients,
                                              accepted_cache,
                                              &workspace->dense_active_gradient_direction);
            }
        }
        if (dense_active_gradient_direction != nullptr) {
            *dense_active_gradient_direction = workspace->dense_active_gradient_direction;
        }
        return;
    }
    // The generic packed-pair contraction is required here: a full-matrix
    // shortcut does not preserve the sparse mixed-chart differential. Each
    // accepted row tile serves both cotangents (and the optional forward
    // direction), with only an AO-by-active accumulator per worker.
    const Eigen::Index tile_rows = n_bf;
    const Eigen::Index n_tiles = (n_bf_pairs + tile_rows - 1) / tile_rows;
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = std::min(xmvb::effective_openmp_thread_count(), static_cast<int>(n_tiles));
#endif
    std::vector<Eigen::MatrixXd> thread_gradients;
    thread_gradients.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread)
        thread_gradients.emplace_back(Eigen::MatrixXd::Zero(n_bf, n_ao));
#pragma omp parallel num_threads(n_threads)
    {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        Eigen::MatrixXd& local_gradient = thread_gradients[thread];
        ExactCtxPairMatrix pair_gradient_tile;
#pragma omp for schedule(static)
        for (Eigen::Index tile = 0; tile < n_tiles; ++tile) {
            const Eigen::Index row_begin = tile * tile_rows;
            const Eigen::Index row_count = std::min(tile_rows, n_bf_pairs - row_begin);
            const auto products =
                accepted_cache.accepted_pair_products->middleRows(row_begin, row_count);
            pair_gradient_tile.noalias() = products * accepted_cache.active_pair_gradient_matrix;
            accumulate_pair_gradient_rows(pair_gradient_tile,
                                          row_begin,
                                          workspace->dense_active_direction,
                                          accepted_cache,
                                          &local_gradient);
            if (active_pair_gradient_direction != nullptr) {
                pair_gradient_tile.noalias() = products * *active_pair_gradient_direction;
                accumulate_pair_gradient_rows(pair_gradient_tile,
                                              row_begin,
                                              *accepted_cache.accepted_active_coefficients,
                                              accepted_cache,
                                              &local_gradient);
            }
            if (directional_pair_products != nullptr) {
                pair_gradient_tile.noalias() =
                    directional_pair_products->middleRows(row_begin, row_count) *
                    accepted_cache.active_pair_gradient_matrix;
                accumulate_pair_gradient_rows(pair_gradient_tile,
                                              row_begin,
                                              *accepted_cache.accepted_active_coefficients,
                                              accepted_cache,
                                              &local_gradient);
            }
        }
    }
    for (const auto& local_gradient : thread_gradients)
        workspace->dense_active_gradient_direction += local_gradient;
    if (directional_pair_products != nullptr) {
        if (dense_active_gradient_direction != nullptr)
            *dense_active_gradient_direction = workspace->dense_active_gradient_direction;
        return;
    }
    if (accepted_cache.exchange_integrals.size() != 0) {
        workspace->dense_active_gradient_direction +=
            detail::contract_exchange_direction(accepted_cache, dense_active_direction);
        if (dense_active_gradient_direction != nullptr)
            *dense_active_gradient_direction = workspace->dense_active_gradient_direction;
        return;
    }
    build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
        *accepted_cache.accepted_active_coefficients,
        workspace->dense_active_direction,
        accepted_cache,
        &workspace->mixed_pair_coefficients);

    multiply_pair_coefficients_by_gradient_matrix(workspace->mixed_pair_coefficients,
                                                  accepted_cache.active_pair_gradient_matrix,
                                                  &workspace->transformed_pair_coefficients);
    apply_exact_ao_pair_kernel(ao_integral_input,
                               workspace->transformed_pair_coefficients,
                               n_bf,
                               n_active_pairs,
                               &workspace->pair_gradients);
    accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
        workspace->pair_gradients,
        *accepted_cache.accepted_active_coefficients,
        accepted_cache,
        &workspace->dense_active_gradient_direction);

    if (dense_active_gradient_direction != nullptr) {
        *dense_active_gradient_direction = workspace->dense_active_gradient_direction;
    }
}
} // namespace

void apply_exact_active_two_electron_jet(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const std::vector<double>& packed_gradient_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction,
    const ExactCtxPairMatrix* directional_pair_products) {
    const auto pairs = build_active_pair_list(accepted_cache.n_active_orbitals);
    const ExactCtxPairMatrix pair_gradient_direction =
        build_active_pair_gradient_matrix(packed_gradient_direction, pairs);
    apply_exact_active_two_electron_jet_impl(accepted_cache,
                                             dense_active_direction,
                                             ao_integral_input,
                                             workspace,
                                             dense_active_gradient_direction,
                                             &pair_gradient_direction,
                                             directional_pair_products);
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
    apply_exact_active_two_electron_jet_impl(accepted_cache,
                                             dense_active_direction,
                                             ao_integral_input,
                                             workspace,
                                             dense_active_gradient_direction,
                                             nullptr,
                                             nullptr);
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& directional_pair_products,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
    if (accepted_cache.accepted_pair_products == nullptr) {
        throw std::invalid_argument(
            "fused exact 2e HVP requires resident directional pair products");
    }

    apply_exact_active_two_electron_jet_impl(accepted_cache,
                                             dense_active_direction,
                                             ao_integral_input,
                                             workspace,
                                             dense_active_gradient_direction,
                                             nullptr,
                                             &directional_pair_products);
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::MatrixXd& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_ao,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result) {
    const auto accepted_cache = build_exact_packed_active_two_electron_adjoint_cache(
        packed_active_two_electron_gradient,
        dense_active_coefficients,
        ao_integral_input,
        n_ao,
        accepted_active_space_two_electron_result);
    return apply_exact_packed_active_two_electron_adjoint_hessian_vector(
        accepted_cache, dense_active_direction, ao_integral_input);
}

} // namespace xmvb::integrals
