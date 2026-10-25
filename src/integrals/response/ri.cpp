#include "integrals/response/ri.hpp"
#include "integrals/response/ri_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"
#include "core/pair_index.hpp"
#include "core/timing.hpp"

namespace xmvb::integrals {
namespace {

std::size_t packed_pair_count(int dimension) {
    const std::size_t size = static_cast<std::size_t>(dimension);
    return size * (size + 1) / 2;
}

void validate_cache(const RiActiveTwoElectronResponseCache& cache, const char* operation) {
    const int n_bf = cache.n_basis_functions;
    const int n_active = cache.n_active_orbitals;
    const int n_auxiliary = cache.n_auxiliary_functions;
    if (n_bf <= 0 || n_active <= 0 || n_auxiliary <= 0) {
        throw std::invalid_argument(std::string(operation) + " requires positive RI dimensions");
    }
    if (cache.auxiliary_tile_size <= 0 || cache.auxiliary_tile_size > n_auxiliary) {
        throw std::invalid_argument(std::string(operation) + " has an invalid auxiliary tile size");
    }
    const std::size_t n_ao_pairs = packed_pair_count(n_bf);
    const std::size_t n_active_pairs = packed_pair_count(n_active);
    if (cache.active_pair_first_indices.size() != n_active_pairs ||
        cache.active_pair_second_indices.size() != n_active_pairs) {
        throw std::invalid_argument(std::string(operation) + " has inconsistent pair-index tables");
    }
    if (cache.metric_whitened_ao_pair_factors == nullptr ||
        cache.metric_whitened_ao_pair_factors->rows() != n_auxiliary ||
        cache.metric_whitened_ao_pair_factors->cols() != static_cast<Eigen::Index>(n_ao_pairs)) {
        throw std::invalid_argument(std::string(operation) + " has inconsistent AO RI factors");
    }
    if (cache.accepted_active_coefficients == nullptr ||
        cache.accepted_active_coefficients->rows() != n_bf ||
        cache.accepted_active_coefficients->cols() != n_active) {
        throw std::invalid_argument(std::string(operation) +
                                    " has inconsistent accepted coefficients");
    }
    if (cache.accepted_active_pair_factors == nullptr ||
        cache.accepted_active_pair_factors->rows() != n_auxiliary ||
        cache.accepted_active_pair_factors->cols() != static_cast<Eigen::Index>(n_active_pairs)) {
        throw std::invalid_argument(std::string(operation) +
                                    " has inconsistent active-pair factors");
    }
    if (cache.accepted_one_index_factors &&
        (cache.accepted_one_index_factors->rows() !=
             static_cast<Eigen::Index>(n_auxiliary) * n_active ||
         cache.accepted_one_index_factors->cols() != n_bf)) {
        throw std::invalid_argument(std::string(operation) + " has inconsistent one-index factors");
    }
}

/** @brief Auxiliary tile width accounting for shared transforms and AO slices. */
int planned_auxiliary_tile_width(const RiActiveTwoElectronResponseCache& cache) {
    const std::size_t n_bf = cache.n_basis_functions;
    const std::size_t n_active = cache.n_active_orbitals;
    const std::size_t workers =
        std::min(xmvb::effective_openmp_thread_count(), cache.n_basis_functions);
    // Reserve two one-index transforms and one AO slice per worker.
    const std::size_t elements_per_aux = n_bf * (2 * n_active + workers);
    return static_cast<int>(std::min<std::size_t>(
        cache.n_auxiliary_functions,
        std::max<std::size_t>(1, workspace::available() / sizeof(double) / elements_per_aux)));
}

/** @brief Visits shared accepted factors, or generates the same tiled factors. */
template <class Consumer>
void visit_accepted_factors(const RiActiveTwoElectronResponseCache& cache, Consumer consume) {
    if (cache.accepted_one_index_factors) {
        // No copy and no direction-dependent accepted transform.
        consume(*cache.accepted_one_index_factors, 0, cache.n_auxiliary_functions);
        return;
    }
    const int width = std::min(cache.auxiliary_tile_size, planned_auxiliary_tile_width(cache));
    Eigen::MatrixXd factors;
    for (int first = 0; first < cache.n_auxiliary_functions; first += width) {
        const int count = std::min(width, cache.n_auxiliary_functions - first);
        factors.resize(static_cast<Eigen::Index>(count) * cache.n_active_orbitals,
                       cache.n_basis_functions);
        transform_ri_one_index(*cache.metric_whitened_ao_pair_factors,
                               *cache.accepted_active_coefficients,
                               first,
                               count,
                               factors);
        consume(factors, first, count);
    }
}

/** @brief Contracts factor adjoints directly to AO-by-active orbital output.
 *
 * The optional directional term differentiates L_A C, not an AO-pair
 * adjoint. Factor images are shared; output rows have unique owners.
 */
Eigen::MatrixXd contract_factor_adjoint(const RiActiveTwoElectronResponseCache& cache,
                                        const Eigen::MatrixXd& adjoint,
                                        const Eigen::MatrixXd* directional_adjoint,
                                        const Eigen::Ref<const Eigen::MatrixXd>* direction) {
    const int n_bf = cache.n_basis_functions;
    const int n_active = cache.n_active_orbitals;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_bf, n_active);
    visit_accepted_factors(cache, [&](const Eigen::MatrixXd& factors, int first_aux, int count) {
        // Repack only the one-index directional factors, never AO-pair adjoints.
        Eigen::MatrixXd directional_factors;
        const int width = std::min(cache.auxiliary_tile_size, planned_auxiliary_tile_width(cache));
        for (int offset = 0; offset < count; offset += width) {
            const int tile_count = std::min(width, count - offset);
            if (direction != nullptr) {
                directional_factors.resize(static_cast<Eigen::Index>(tile_count) * n_active, n_bf);
                transform_ri_one_index(*cache.metric_whitened_ao_pair_factors,
                                       *direction,
                                       first_aux + offset,
                                       tile_count,
                                       directional_factors);
            }
            if (direction == nullptr) {
                detail::accumulate_ri_factor_tile(cache,
                                                  factors,
                                                  count,
                                                  offset,
                                                  adjoint,
                                                  first_aux + offset,
                                                  tile_count,
                                                  workspace::available(),
                                                  result);
            } else {
                detail::accumulate_ri_fixed_factor_tile(cache,
                                                        factors,
                                                        count,
                                                        offset,
                                                        directional_factors,
                                                        tile_count,
                                                        0,
                                                        adjoint,
                                                        *directional_adjoint,
                                                        first_aux + offset,
                                                        tile_count,
                                                        workspace::available(),
                                                        result);
            }
        }
    });
    return result;
}
} // namespace

namespace detail {

Eigen::MatrixXd build_ri_active_pair_adjoint(const std::vector<double>& packed_gradient,
                                             const RiActiveTwoElectronResponseCache& cache) {
    const Eigen::Index n_active_pairs =
        static_cast<Eigen::Index>(cache.active_pair_first_indices.size());
    const std::size_t expected_size = static_cast<std::size_t>(n_active_pairs) *
                                      (static_cast<std::size_t>(n_active_pairs) + 1) / 2;
    if (packed_gradient.size() != expected_size) {
        throw std::invalid_argument("packed RI active-2e adjoint has the wrong size");
    }
    Eigen::MatrixXd active_pair_adjoint(n_active_pairs, n_active_pairs);
    for (Eigen::Index row = 0; row < n_active_pairs; ++row) {
        for (Eigen::Index column = 0; column < n_active_pairs; ++column) {
            const int first = static_cast<int>(row > column ? row : column);
            const int second = static_cast<int>(row > column ? column : row);
            const int packed_index =
                xmvb::core::PairIndex::packed_pair_of_pairs_index(first, second);
            active_pair_adjoint(row, column) =
                (row == column ? 2.0 : 1.0) * packed_gradient[packed_index];
        }
    }
    return active_pair_adjoint;
}

void fill_ri_factor_direction_tile(const RiActiveTwoElectronResponseCache& cache,
                                   const Eigen::Ref<const Eigen::MatrixXd>& accepted,
                                   const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                   int first_aux,
                                   int count,
                                   Eigen::MatrixXd& result) {
    const Eigen::MatrixXd mixed = accepted * direction;
    for (Eigen::Index pair = 0; pair < result.cols(); ++pair) {
        const int a = cache.active_pair_first_indices[pair];
        const int b = cache.active_pair_second_indices[pair];
        result.col(pair).segment(first_aux, count) =
            mixed.col(b).segment(a * count, count) + mixed.col(a).segment(b * count, count);
    }
}

void accumulate_ri_factor_tile(const RiActiveTwoElectronResponseCache& cache,
                               const Eigen::Ref<const Eigen::MatrixXd>& factors,
                               int auxiliary_stride,
                               int auxiliary_offset,
                               const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
                               int adjoint_offset,
                               int count,
                               std::size_t workspace_bytes,
                               Eigen::MatrixXd& result) {
    const int n_bf = cache.n_basis_functions;
    const int n_active = cache.n_active_orbitals;
    const std::size_t elements_per_aux = static_cast<std::size_t>(n_active) * n_active;
    const std::size_t capacity = workspace_bytes / sizeof(double) / elements_per_aux;
    if (capacity == 0)
        throw std::length_error("minimum RI factor cotangent exceeds workspace");
    const int width = static_cast<int>(std::min<std::size_t>(count, capacity));
    const int workers = std::min(effective_openmp_thread_count(), n_bf);
    for (int first = 0; first < count; first += width) {
        const int size = std::min(width, count - first);
        // W[(a,P),b] is the symmetric factor cotangent. Its diagonal carries
        // the same multiplicity as the packed-pair pullback, including a=b.
        Eigen::MatrixXd weights(size * n_active, n_active);
#pragma omp parallel num_threads(workers)
        {
#pragma omp for schedule(static)
            for (Eigen::Index pair = 0; pair < adjoint.cols(); ++pair) {
                const int a = cache.active_pair_first_indices[pair];
                const int b = cache.active_pair_second_indices[pair];
                weights.col(b).segment(a * size, size) =
                    (a == b ? 2.0 : 1.0) * adjoint.col(pair).segment(adjoint_offset + first, size);
                if (a != b)
                    weights.col(a).segment(b * size, size) = weights.col(b).segment(a * size, size);
            }
            int worker = 0;
            int team_size = 1;
#ifdef _OPENMP
            worker = omp_get_thread_num();
            team_size = omp_get_num_threads();
#endif
            const int begin = n_bf * worker / team_size;
            const int rows = n_bf * (worker + 1) / team_size - begin;
            if (auxiliary_stride == size) {
                // The resident/tile image is already [(a,P),mu]; consume it as
                // one rectangular GEMM. No per-P image gather or reduction.
                result.middleRows(begin, rows).noalias() +=
                    factors.block(auxiliary_offset + first, begin, size * n_active, rows)
                        .transpose() *
                    weights;
            } else {
                // A narrow view of a resident slab has gaps between active blocks.
                // Keep that view: factor copies would exceed the tile contract.
                for (int a = 0; a < n_active; ++a)
                    result.middleRows(begin, rows).noalias() +=
                        factors
                            .block(
                                a * auxiliary_stride + auxiliary_offset + first, begin, size, rows)
                            .transpose() *
                        weights.middleRows(a * size, size);
            }
        }
    }
}

void accumulate_ri_fixed_factor_tile(const RiActiveTwoElectronResponseCache& cache,
                                     const Eigen::Ref<const Eigen::MatrixXd>& accepted,
                                     int accepted_aux_stride,
                                     int accepted_aux_offset,
                                     const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                     int direction_aux_stride,
                                     int direction_aux_offset,
                                     const Eigen::Ref<const Eigen::MatrixXd>& accepted_adjoint,
                                     const Eigen::Ref<const Eigen::MatrixXd>& directional_adjoint,
                                     int adjoint_offset,
                                     int count,
                                     std::size_t worker_budget_bytes,
                                     Eigen::MatrixXd& result) {
    accumulate_ri_factor_tile(cache,
                              direction,
                              direction_aux_stride,
                              direction_aux_offset,
                              accepted_adjoint,
                              adjoint_offset,
                              count,
                              worker_budget_bytes,
                              result);
    accumulate_ri_factor_tile(cache,
                              accepted,
                              accepted_aux_stride,
                              accepted_aux_offset,
                              directional_adjoint,
                              adjoint_offset,
                              count,
                              worker_budget_bytes,
                              result);
}

} // namespace detail

RiActiveTwoElectronResponseCache build_ri_active_two_electron_response_cache(
    const RiAoFactorization& ao_ri_factorization,
    const ActiveSpaceTwoElectronResult& accepted_active_space_result,
    int n_active_orbitals) {
    if (ao_ri_factorization.n_basis_functions <= 0 ||
        ao_ri_factorization.n_auxiliary_functions <= 0 || n_active_orbitals <= 0) {
        throw std::invalid_argument("RI active-2e response cache dimensions must be positive");
    }
    if (accepted_active_space_result.representation !=
        ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
        throw std::invalid_argument(
            "RI active-2e response cache requires an RI active-space result");
    }

    const int n_bf = ao_ri_factorization.n_basis_functions;
    const std::size_t n_ao_pairs = packed_pair_count(n_bf);
    const std::size_t n_active_pairs = packed_pair_count(n_active_orbitals);
    if (ao_ri_factorization.n_packed_ao_pairs != static_cast<int>(n_ao_pairs) ||
        ao_ri_factorization.metric_whitened_ao_pair_factors.rows() !=
            ao_ri_factorization.n_auxiliary_functions ||
        ao_ri_factorization.metric_whitened_ao_pair_factors.cols() !=
            static_cast<Eigen::Index>(n_ao_pairs)) {
        throw std::invalid_argument(
            "RI active-2e response cache AO factor dimensions are inconsistent");
    }
    if (accepted_active_space_result.n_auxiliary_functions !=
            ao_ri_factorization.n_auxiliary_functions ||
        accepted_active_space_result.dense_active_coefficients.rows() != n_bf ||
        accepted_active_space_result.dense_active_coefficients.cols() != n_active_orbitals ||
        accepted_active_space_result.ri_active_pair_factors.rows() !=
            ao_ri_factorization.n_auxiliary_functions ||
        accepted_active_space_result.ri_active_pair_factors.cols() !=
            static_cast<Eigen::Index>(n_active_pairs)) {
        throw std::invalid_argument(
            "RI active-2e response cache accepted dimensions are inconsistent");
    }

    RiActiveTwoElectronResponseCache cache;
    cache.n_basis_functions = n_bf;
    cache.n_active_orbitals = n_active_orbitals;
    cache.n_auxiliary_functions = ao_ri_factorization.n_auxiliary_functions;
    cache.metric_whitened_ao_pair_factors = &ao_ri_factorization.metric_whitened_ao_pair_factors;
    cache.accepted_active_coefficients = &accepted_active_space_result.dense_active_coefficients;
    cache.accepted_active_pair_factors = &accepted_active_space_result.ri_active_pair_factors;

    cache.active_pair_first_indices.reserve(n_active_pairs);
    cache.active_pair_second_indices.reserve(n_active_pairs);
    for (int first = 0; first < n_active_orbitals; ++first) {
        for (int second = 0; second <= first; ++second) {
            cache.active_pair_first_indices.push_back(first);
            cache.active_pair_second_indices.push_back(second);
        }
    }
    cache.accepted_one_index_factors = accepted_active_space_result.ri_one_index_factors;
    if (!cache.accepted_one_index_factors) {
        const auto scratch =
            sizeof(double) * static_cast<std::size_t>(n_bf) *
            (2 * n_active_orbitals + std::min(effective_openmp_thread_count(), n_bf));
        auto factors = workspace::make_matrix<Eigen::MatrixXd>(
            static_cast<std::size_t>(cache.n_auxiliary_functions) * n_active_orbitals,
            n_bf,
            scratch);
        if (factors) {
            fill_ri_one_index_tiled(*cache.metric_whitened_ao_pair_factors,
                                    *cache.accepted_active_coefficients,
                                    *factors);
            cache.accepted_one_index_factors = std::move(factors);
        }
    }
    cache.auxiliary_tile_size = planned_auxiliary_tile_width(cache);
    validate_cache(cache, "RI active-2e response cache");
    return cache;
}

void prepare_ri_orbital_pair_products(RiActiveTwoElectronResponseCache& cache) {
    profiling::ScopedPhase phase(profiling::Phase::RiOrbitalPairBuild);
    validate_cache(cache, "RI orbital-pair transform");
    if (cache.orbital_pair_products)
        return;
    const Eigen::Index rows =
        static_cast<Eigen::Index>(cache.n_basis_functions) * cache.n_active_orbitals;
    const Eigen::Index pairs = cache.active_pair_first_indices.size();
    const auto scratch = sizeof(double) * static_cast<std::size_t>(cache.n_basis_functions) *
                         (2 * cache.n_active_orbitals +
                          std::min(effective_openmp_thread_count(), cache.n_basis_functions));
    auto products = workspace::make_matrix<OrbitalPairMatrix>(rows, pairs, scratch);
    if (!products)
        return;
    products->setZero();
    const int workers = std::min<Eigen::Index>(xmvb::effective_openmp_thread_count(), rows);
    visit_accepted_factors(cache, [&](const Eigen::MatrixXd& factors, int first, int count) {
        // Column-major [(a,A),mu] is already row-major [(mu,a),A]. No repacking.
        const Eigen::Map<const OrbitalPairMatrix> image(factors.data(), rows, count);
        const auto pair_factors = cache.accepted_active_pair_factors->middleRows(first, count);
#pragma omp parallel num_threads(workers)
        {
            int worker = 0;
            int team = 1;
#ifdef _OPENMP
            worker = omp_get_thread_num();
            team = omp_get_num_threads();
#endif
            const Eigen::Index begin = rows * worker / team;
            const Eigen::Index end = rows * (worker + 1) / team;
            products->middleRows(begin, end - begin).noalias() +=
                image.middleRows(begin, end - begin) * pair_factors;
        }
    });
    cache.orbital_pair_products = std::move(products);
}

bool ri_uses_orbital_pairs(const RiActiveTwoElectronResponseCache& cache) noexcept {
    if (!cache.orbital_pair_products ||
        cache.orbital_pair_products->rows() !=
            static_cast<Eigen::Index>(cache.n_basis_functions) * cache.n_active_orbitals ||
        cache.orbital_pair_products->cols() !=
            static_cast<Eigen::Index>(cache.active_pair_first_indices.size()))
        return false;
    const long double pairs = static_cast<long double>(cache.active_pair_first_indices.size());
    // B * delta G costs 2 R P^2; image pullback costs 2 N A^2 P.
    return static_cast<long double>(cache.n_auxiliary_functions) * pairs >
           static_cast<long double>(cache.n_basis_functions) * cache.n_active_orbitals *
               cache.n_active_orbitals;
}

Eigen::MatrixXd compute_ri_active_pair_factor_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction) {
    validate_cache(accepted_cache, "RI active-pair factor direction");
    if (dense_active_direction.rows() != accepted_cache.n_basis_functions ||
        dense_active_direction.cols() != accepted_cache.n_active_orbitals) {
        throw std::invalid_argument("RI active-pair factor direction has the wrong shape");
    }
    Eigen::MatrixXd result(accepted_cache.n_auxiliary_functions,
                           accepted_cache.active_pair_first_indices.size());
    visit_accepted_factors(
        accepted_cache, [&](const Eigen::MatrixXd& factors, int first, int count) {
            detail::fill_ri_factor_direction_tile(
                accepted_cache, factors, dense_active_direction, first, count, result);
        });
    return result;
}

std::vector<Eigen::MatrixXd> compute_ri_active_pair_factor_directional_derivative_batch(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions) {
    validate_cache(accepted_cache, "RI active-pair factor direction block");
    if (dense_active_directions.empty())
        return {};
    for (const auto& direction : dense_active_directions) {
        if (direction.rows() != accepted_cache.n_basis_functions ||
            direction.cols() != accepted_cache.n_active_orbitals || !direction.allFinite()) {
            throw std::invalid_argument(
                "RI active-pair factor direction block has an invalid shape or value");
        }
    }
    std::vector<Eigen::MatrixXd> results;
    results.reserve(dense_active_directions.size());
    for (std::size_t i = 0; i < dense_active_directions.size(); ++i) {
        results.emplace_back(accepted_cache.n_auxiliary_functions,
                             accepted_cache.active_pair_first_indices.size());
    }
    visit_accepted_factors(
        accepted_cache, [&](const Eigen::MatrixXd& factors, int first, int count) {
            for (std::size_t i = 0; i < dense_active_directions.size(); ++i) {
                detail::fill_ri_factor_direction_tile(
                    accepted_cache, factors, dense_active_directions[i], first, count, results[i]);
            }
        });
    return results;
}

std::vector<double> compute_ri_packed_active_two_electron_integral_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& directional_active_pair_factors) {
    validate_cache(accepted_cache, "RI packed active-2e direction");
    const Eigen::MatrixXd& accepted_factors = *accepted_cache.accepted_active_pair_factors;
    if (directional_active_pair_factors.rows() != accepted_factors.rows() ||
        directional_active_pair_factors.cols() != accepted_factors.cols()) {
        throw std::invalid_argument("RI active-pair factor direction has the wrong shape");
    }

    const Eigen::MatrixXd one_sided_contraction =
        accepted_factors.transpose() * directional_active_pair_factors;
    const Eigen::Index n_active_pairs = accepted_factors.cols();
    std::vector<double> packed_direction(static_cast<std::size_t>(n_active_pairs) *
                                             (static_cast<std::size_t>(n_active_pairs) + 1) / 2,
                                         0.0);
    for (Eigen::Index first = 0; first < n_active_pairs; ++first) {
        for (Eigen::Index second = 0; second <= first; ++second) {
            const int packed_index = xmvb::core::PairIndex::packed_pair_of_pairs_index(
                static_cast<int>(first), static_cast<int>(second));
            packed_direction[packed_index] =
                one_sided_contraction(first, second) + one_sided_contraction(second, first);
        }
    }
    return packed_direction;
}

Eigen::MatrixXd compute_ri_packed_active_two_electron_integral_directional_derivative_batch(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& directional_active_pair_factors) {
    validate_cache(accepted_cache, "RI packed active-2e direction block");
    if (directional_active_pair_factors.empty())
        return {};
    const Eigen::MatrixXd& accepted_factors = *accepted_cache.accepted_active_pair_factors;
    const Eigen::Index n_active_pairs = accepted_factors.cols();
    const Eigen::Index n_directions =
        static_cast<Eigen::Index>(directional_active_pair_factors.size());
    Eigen::MatrixXd factor_direction_block(accepted_factors.rows(), n_directions * n_active_pairs);
    for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
        const auto& factor_direction =
            directional_active_pair_factors[static_cast<std::size_t>(direction)];
        if (factor_direction.rows() != accepted_factors.rows() ||
            factor_direction.cols() != n_active_pairs || !factor_direction.allFinite()) {
            throw std::invalid_argument(
                "RI active-pair factor direction block has an invalid shape or value");
        }
        factor_direction_block.middleCols(direction * n_active_pairs, n_active_pairs) =
            factor_direction;
    }
    const Eigen::MatrixXd one_sided_block = accepted_factors.transpose() * factor_direction_block;
    const Eigen::Index n_packed_integrals = n_active_pairs * (n_active_pairs + 1) / 2;
    Eigen::MatrixXd packed_directions(n_packed_integrals, n_directions);
    for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
        const auto one_sided =
            one_sided_block.middleCols(direction * n_active_pairs, n_active_pairs);
        for (Eigen::Index first = 0; first < n_active_pairs; ++first) {
            for (Eigen::Index second = 0; second <= first; ++second) {
                const int packed_index = xmvb::core::PairIndex::packed_pair_of_pairs_index(
                    static_cast<int>(first), static_cast<int>(second));
                packed_directions(packed_index, direction) =
                    one_sided(first, second) + one_sided(second, first);
            }
        }
    }
    return packed_directions;
}

std::vector<Eigen::MatrixXd> backpropagate_ri_packed_active_two_electron_gradient_batch(
    std::span<const std::vector<double>* const> packed_gradients,
    const RiActiveTwoElectronResponseCache& accepted_cache,
    std::size_t workspace_bytes) {
    validate_cache(accepted_cache, "RI packed active-2e pullback block");
    if (packed_gradients.empty())
        return {};
    const int n = accepted_cache.n_basis_functions;
    const int na = accepted_cache.n_active_orbitals;
    const int n_aux = accepted_cache.n_auxiliary_functions;
    const int pairs = static_cast<int>(accepted_cache.active_pair_first_indices.size());
    const std::size_t packed_size = static_cast<std::size_t>(pairs) * (pairs + 1) / 2;
    for (const auto* gradient : packed_gradients)
        if (!gradient || gradient->size() != packed_size)
            throw std::invalid_argument(
                "RI packed active-2e pullback block has an invalid cotangent");
    const std::size_t limit = workspace_bytes / sizeof(double);
    std::vector<Eigen::MatrixXd> result;
    result.reserve(packed_gradients.size());
    for (std::size_t r = 0; r < packed_gradients.size(); ++r)
        result.emplace_back(Eigen::MatrixXd::Zero(n, na));

    if (accepted_cache.orbital_pair_products) {
        profiling::ScopedPhase phase(profiling::Phase::OrbitalPairPullback);
        // Reinterpret the accepted (AO,active)-by-pair image as AO-by-(active,pair).
        // One bounded weight panel makes the accepted image a multi-RHS GEMM;
        // no pair-square matrix or per-request copy of that image is retained.
        const auto& products = *accepted_cache.orbital_pair_products;
        if (products.rows() != static_cast<Eigen::Index>(n) * na || products.cols() != pairs)
            throw std::invalid_argument("RI orbital-pair pullback image has an invalid shape");
        const Eigen::Map<const OrbitalPairMatrix> image(products.data(), n, na * pairs);
        const std::size_t per_request =
            static_cast<std::size_t>(na) * na * pairs + static_cast<std::size_t>(n) * na;
        if (limit < per_request)
            throw std::length_error("one RI orbital-pair cotangent exceeds workspace");
        const std::size_t panel_width = std::min(packed_gradients.size(), limit / per_request);
        for (std::size_t begin = 0; begin < packed_gradients.size(); begin += panel_width) {
            const int width =
                static_cast<int>(std::min(panel_width, packed_gradients.size() - begin));
            Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(na * pairs, na * width);
            for (int r = 0; r < width; ++r) {
                const auto& gradient = *packed_gradients[begin + r];
                for (int p = 0; p < pairs; ++p) {
                    const int a = accepted_cache.active_pair_first_indices[p];
                    const int b = accepted_cache.active_pair_second_indices[p];
                    for (int q = 0; q < pairs; ++q) {
                        const int hi = std::max(p, q), lo = std::min(p, q);
                        const double value =
                            (p == q ? 2.0 : 1.0) *
                            gradient[core::PairIndex::packed_pair_of_pairs_index(hi, lo)];
                        weights(b * pairs + q, r * na + a) += value;
                        weights(a * pairs + q, r * na + b) += value;
                    }
                }
            }
            const Eigen::MatrixXd images = image * weights;
            for (int r = 0; r < width; ++r)
                result[begin + r] = images.middleCols(r * na, na);
        }
        return result;
    }

    // A missing three-active image uses the accepted one-index factors once per
    // auxiliary tile. Contract independent packed cotangents as a bounded
    // P-by-(P*width) RHS, then reuse the same accepted factors for each output.
    const bool resident = static_cast<bool>(accepted_cache.accepted_one_index_factors);
    const int workers = std::min(effective_openmp_thread_count(), n);
    const std::size_t transformed_per_aux =
        resident ? 0 : static_cast<std::size_t>(n) * (na + workers);
    const std::size_t pair_square = static_cast<std::size_t>(pairs) * pairs;
    const std::size_t per_aux = transformed_per_aux + static_cast<std::size_t>(na) * na + pairs;
    if (limit < pair_square + per_aux)
        throw std::length_error("one streamed RI cotangent exceeds workspace");
    const int tile = static_cast<int>(
        std::min<std::size_t>({static_cast<std::size_t>(n_aux),
                               static_cast<std::size_t>(accepted_cache.auxiliary_tile_size),
                               (limit - pair_square) / per_aux}));
    for (int first = 0; first < n_aux; first += tile) {
        const int count = std::min(tile, n_aux - first);
        Eigen::MatrixXd owned_factors;
        const Eigen::MatrixXd* factors = accepted_cache.accepted_one_index_factors.get();
        if (!factors) {
            owned_factors.resize(static_cast<Eigen::Index>(count) * na, n);
            transform_ri_one_index(*accepted_cache.metric_whitened_ao_pair_factors,
                                   *accepted_cache.accepted_active_coefficients,
                                   first,
                                   count,
                                   owned_factors);
            factors = &owned_factors;
        }
        const int stride = resident ? n_aux : count;
        const int offset = resident ? first : 0;
        const std::size_t fixed =
            transformed_per_aux * count + static_cast<std::size_t>(count) * na * na;
        const std::size_t per_request = pair_square + static_cast<std::size_t>(count) * pairs;
        const std::size_t panel_width =
            std::min(packed_gradients.size(), (limit - fixed) / per_request);
        for (std::size_t begin = 0; begin < packed_gradients.size(); begin += panel_width) {
            const int width =
                static_cast<int>(std::min(panel_width, packed_gradients.size() - begin));
            Eigen::MatrixXd pair_adjoint(pairs, pairs * width);
            for (int r = 0; r < width; ++r) {
                const auto& gradient = *packed_gradients[begin + r];
                for (int p = 0; p < pairs; ++p)
                    for (int q = 0; q < pairs; ++q) {
                        const int hi = std::max(p, q), lo = std::min(p, q);
                        pair_adjoint(q, r * pairs + p) =
                            (p == q ? 2.0 : 1.0) *
                            gradient[core::PairIndex::packed_pair_of_pairs_index(hi, lo)];
                    }
            }
            Eigen::MatrixXd factor_adjoint(count, pairs * width);
            factor_adjoint.noalias() =
                accepted_cache.accepted_active_pair_factors->middleRows(first, count) *
                pair_adjoint;
            for (int r = 0; r < width; ++r)
                detail::accumulate_ri_factor_tile(accepted_cache,
                                                  *factors,
                                                  stride,
                                                  offset,
                                                  factor_adjoint.middleCols(r * pairs, pairs),
                                                  0,
                                                  count,
                                                  sizeof(double) * static_cast<std::size_t>(count) *
                                                      na * na,
                                                  result[begin + r]);
        }
    }
    return result;
}

Eigen::MatrixXd backpropagate_ri_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache) {
    const std::array<const std::vector<double>*, 1> gradients{&packed_active_two_electron_gradient};
    auto result =
        backpropagate_ri_packed_active_two_electron_gradient_batch(gradients, accepted_cache);
    return std::move(result.front());
}

namespace {
Eigen::MatrixXd apply_ri_active_two_electron_jet_impl(
    const std::vector<double>& packed_accepted_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const std::vector<double>* packed_gradient_direction,
    const Eigen::MatrixXd* directional_active_pair_factors) {
    validate_cache(accepted_cache, "RI active-2e jet");
    if (dense_active_direction.rows() != accepted_cache.n_basis_functions ||
        dense_active_direction.cols() != accepted_cache.n_active_orbitals) {
        throw std::invalid_argument("RI fixed-adjoint HVP direction has the wrong orbital shape");
    }

    Eigen::MatrixXd owned_factor_direction;
    const Eigen::MatrixXd* factor_direction = directional_active_pair_factors;
    if (factor_direction == nullptr) {
        owned_factor_direction = compute_ri_active_pair_factor_directional_derivative(
            accepted_cache, dense_active_direction);
        factor_direction = &owned_factor_direction;
    }
    if (factor_direction->rows() != accepted_cache.n_auxiliary_functions ||
        factor_direction->cols() !=
            static_cast<Eigen::Index>(accepted_cache.active_pair_first_indices.size())) {
        throw std::invalid_argument(
            "precomputed RI active-pair factor direction has the wrong shape");
    }

    const Eigen::MatrixXd active_pair_adjoint =
        detail::build_ri_active_pair_adjoint(packed_accepted_gradient, accepted_cache);
    const Eigen::MatrixXd accepted_factor_adjoint =
        *accepted_cache.accepted_active_pair_factors * active_pair_adjoint;
    Eigen::MatrixXd directional_factor_adjoint = *factor_direction * active_pair_adjoint;
    const bool use_orbital_pairs =
        packed_gradient_direction != nullptr && ri_uses_orbital_pairs(accepted_cache);
    Eigen::MatrixXd directional_pair_adjoint;
    if (packed_gradient_direction != nullptr) {
        directional_pair_adjoint =
            detail::build_ri_active_pair_adjoint(*packed_gradient_direction, accepted_cache);
        if (!use_orbital_pairs) {
            directional_factor_adjoint.noalias() +=
                *accepted_cache.accepted_active_pair_factors * directional_pair_adjoint;
        }
    }
    Eigen::MatrixXd result = contract_factor_adjoint(accepted_cache,
                                                     accepted_factor_adjoint,
                                                     &directional_factor_adjoint,
                                                     &dense_active_direction);
    if (use_orbital_pairs) {
        const OrbitalPairMatrix pair_adjoint = directional_pair_adjoint;
        result += pullback_orbital_pairs(*accepted_cache.orbital_pair_products,
                                         pair_adjoint,
                                         accepted_cache.active_pair_first_indices,
                                         accepted_cache.active_pair_second_indices,
                                         accepted_cache.n_active_orbitals);
    }
    return result;
}
} // namespace

Eigen::MatrixXd
apply_ri_active_two_electron_jet(const std::vector<double>& packed_accepted_gradient,
                                 const RiActiveTwoElectronResponseCache& accepted_cache,
                                 const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
                                 const std::vector<double>& packed_gradient_direction,
                                 const Eigen::MatrixXd* directional_active_pair_factors) {
    return apply_ri_active_two_electron_jet_impl(packed_accepted_gradient,
                                                 accepted_cache,
                                                 dense_active_direction,
                                                 &packed_gradient_direction,
                                                 directional_active_pair_factors);
}

Eigen::MatrixXd apply_ri_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const Eigen::MatrixXd* directional_active_pair_factors) {
    return apply_ri_active_two_electron_jet_impl(packed_active_two_electron_gradient,
                                                 accepted_cache,
                                                 dense_active_direction,
                                                 nullptr,
                                                 directional_active_pair_factors);
}

} // namespace xmvb::integrals
