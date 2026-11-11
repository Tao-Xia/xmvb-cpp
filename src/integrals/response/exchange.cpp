#include "integrals/response/internal.hpp"
#include "integrals/response/adjoint.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "integrals/ao/input.hpp"
#include "integrals/ao/exact.hpp"

namespace xmvb::integrals::detail {

namespace {

/** Build a half panel once per unordered AO pair. Its two AO orientations
 * contribute to the same packed exchange image used by the row contraction.
 * A worker owns its panel and takes at most one output AO stripe lock at a time.
 */
void build_tiled_exchange(const AoPairGraph& graph,
                          const ExactCtxPairMatrix& coefficients,
                          int threads,
                          ExactCtxPairMatrix& result) {
    const int n_bf = coefficients.rows(), n = coefficients.cols();
    const auto& tiles = *graph.tiles;
    // Across the team, the half panels fit in the existing single AO slice.
    const int panel_rows = static_cast<int>(n_bf / threads);
    const Eigen::Index stride = static_cast<Eigen::Index>(n_bf) * n;
    const Eigen::Index width = static_cast<Eigen::Index>(n) * n;
    std::vector<std::mutex> stripe_locks(n_bf);
#pragma omp parallel num_threads(threads)
    {
        ExactCtxPairMatrix half(panel_rows, stride);
#pragma omp for schedule(static)
        for (Eigen::Index pair = 0; pair < result.rows(); ++pair)
            result.row(pair).setZero();

#pragma omp for schedule(dynamic, 1)
        for (std::size_t block = 0; block < tiles.row_blocks(); ++block) {
            for (int begin = 0; begin < tiles.row_count(block); begin += panel_rows) {
                const int rows = std::min(panel_rows, tiles.row_count(block) - begin);
                form_exact_pair_half_panel(
                    graph, block, begin, rows, n_bf, n, coefficients.data(), half.data());
                for (int row = 0; row < rows; ++row) {
                    const auto pair = tiles.row_begin(block) + begin + row;
                    const int i = graph.pair_first[pair], j = graph.pair_second[pair];
                    const double* panel = half.row(row).data();
                    const auto accumulate = [&](int mu, int nu) {
                        const double* c = coefficients.row(nu).data();
                        const std::lock_guard lock(stripe_locks[mu]);
                        double* target = result.row(ao_pair_index(mu, 0)).data();
                        for (int k = 0; k <= mu; ++k) {
                            const double* source = panel + k * n;
                            for (Eigen::Index a = 0; a < n; ++a) {
                                const double coefficient = c[a];
#pragma omp simd
                                for (Eigen::Index b = 0; b < n; ++b)
                                    target[k * width + a * n + b] += coefficient * source[b];
                            }
                        }
                    };
                    accumulate(i, j);
                    if (i != j)
                        accumulate(j, i);
                }
            }
        }
    }
}

} // namespace

void build_exchange_integrals(ExactPackedActiveTwoElectronAdjointCache* cache) {
    const Eigen::Index n_bf = cache->n_basis_functions;
    const Eigen::Index n = cache->n_active_orbitals;
    const std::size_t budget =
        std::min(workspace::available(),
                 cache->ao_integral_input->exact_exchange_memory_budget_bytes.value_or(
                     std::numeric_limits<std::size_t>::max()));
    const std::size_t width = static_cast<std::size_t>(n) * n;
    const std::size_t rows = static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
    const auto& graph = cache->ao_integral_input->pair_graph;
    // Keep the large-space streamed representation. The additional integral
    // image has its own memory bound and needs resident accepted pair products.
    if (cache->accepted_pair_products == nullptr || !graph.has_rows() || rows == 0 ||
        width == 0 || width > budget / sizeof(double) / rows ||
        width > cache->optional_retained_budget_bytes / sizeof(double) / rows)
        return;
    // Include the shared active weights and mixed-direction workspace in the
    // admission bound, rather than bounding only the resident integral tensor.
    const std::size_t slice_values = static_cast<std::size_t>(n_bf) * n_bf * n;
    const std::size_t coefficients_values = static_cast<std::size_t>(n_bf) * n;
    const std::size_t action_values =
        (static_cast<std::size_t>(n_bf) + n) * n * width + 2 * coefficients_values;
    const std::size_t panel_row = static_cast<std::size_t>(n_bf) * width;
    const std::size_t scratch =
        std::max(slice_values + coefficients_values, action_values + panel_row);
    const std::size_t available = budget / sizeof(double) - rows * width;
    if (scratch > available)
        return;
    if (!cache->exchange_reservation.try_resize(rows * width * sizeof(double)))
        return;
    // A single AO slice already exposes n_bf independent rows to OpenMP. A
    // larger tile only increases retained peak scratch without reducing sweeps.
    workspace::Reservation build_scratch;
    const std::size_t build_scratch_bytes = (slice_values + coefficients_values) * sizeof(double);
    if (!build_scratch.try_resize(build_scratch_bytes)) {
        cache->exchange_reservation.try_resize(0);
        return;
    }
    workspace::Temporary build_temporary(build_scratch_bytes);
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ExchangeBuild);
    const ExactCtxPairMatrix coefficients = *cache->accepted_active_coefficients;
    cache->exchange_integrals.resize(rows, n * n);
    const int threads = std::min<int>(effective_openmp_thread_count(), n_bf);
    if (graph.tiles) {
        build_tiled_exchange(graph, coefficients, threads, cache->exchange_integrals);
        return;
    }
    const Eigen::Index tile = 1;
    ExactCtxPairMatrix half;
    // Transform one active index in the integral sweep, then the second with
    // GEMM. The half-transformed AO slices are shared and released per tile;
    // no worker owns a full image and the ERI loop has width n, not n*n.
    for (Eigen::Index begin = 0; begin < n_bf; begin += tile) {
        const Eigen::Index count = std::min(tile, n_bf - begin);
        half.resize(count * n_bf, n_bf * n);
#pragma omp parallel num_threads(threads)
        {
#pragma omp for schedule(dynamic, 1)
            for (Eigen::Index item = 0; item < count * n_bf; ++item) {
                const Eigen::Index mu = begin + item / n_bf;
                const Eigen::Index nu = item % n_bf;
                const auto pair = ao_pair_index(mu, nu);
                double* target = half.row(item).data();
                std::fill_n(target, n_bf * n, 0.0);
                graph.visit_row(pair, [&](std::size_t right, double value) {
                    const int k = graph.pair_first[right], l = graph.pair_second[right];
                    const double* c_k = coefficients.row(k).data();
                    const double* c_l = coefficients.row(l).data();
#pragma omp simd
                    for (Eigen::Index b = 0; b < n; ++b)
                        target[l * n + b] += value * c_k[b];
                    if (k != l) {
#pragma omp simd
                        for (Eigen::Index b = 0; b < n; ++b)
                            target[k * n + b] += value * c_l[b];
                    }
                });
            }
#pragma omp for schedule(static)
            for (Eigen::Index item = 0; item < count * n_bf; ++item) {
                const Eigen::Index local_mu = item / n_bf;
                const Eigen::Index k = item % n_bf;
                const Eigen::Index mu = begin + local_mu;
                if (k > mu)
                    continue;
                Eigen::Map<ExactCtxPairMatrix> target(
                    cache->exchange_integrals.row(ao_pair_index(mu, k)).data(), n, n);
                target.noalias() = coefficients.transpose() *
                                   half.middleRows(local_mu * n_bf, n_bf).middleCols(k * n, n);
            }
        }
    }
}

bool can_contract_exchange_direction(const ExactPackedActiveTwoElectronAdjointCache& cache) {
    const Eigen::Index n_bf = cache.n_basis_functions;
    const Eigen::Index n = cache.n_active_orbitals;
    if (cache.ao_integral_input == nullptr || n_bf <= 0 || n <= 0 ||
        cache.exchange_integrals.rows() != n_bf * (n_bf + 1) / 2 ||
        cache.exchange_integrals.cols() != n * n)
        return false;
    const std::size_t image_bytes =
        static_cast<std::size_t>(cache.exchange_integrals.size()) * sizeof(double);
    const std::size_t explicit_cap =
        cache.ao_integral_input->exact_exchange_memory_budget_bytes.value_or(
            std::numeric_limits<std::size_t>::max());
    const std::size_t capacity =
        std::min(workspace::available(), explicit_cap - std::min(explicit_cap, image_bytes)) /
        sizeof(double);
    const std::size_t width = static_cast<std::size_t>(n) * n;
    const std::size_t row_values = static_cast<std::size_t>(n_bf) * width;
    const std::size_t occupied = width * width + static_cast<std::size_t>(n_bf) * n * width +
                                 2 * static_cast<std::size_t>(n_bf) * n;
    return occupied <= capacity && capacity - occupied >= row_values;
}

Eigen::MatrixXd contract_exchange_direction(const ExactPackedActiveTwoElectronAdjointCache& cache,
                                            const Eigen::Ref<const Eigen::MatrixXd>& direction) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ExchangeAction);
    const Eigen::Index n_bf = cache.n_basis_functions;
    const Eigen::Index n = cache.n_active_orbitals;
    if (direction.rows() != n_bf || direction.cols() != n ||
        cache.exchange_integrals.rows() != n_bf * (n_bf + 1) / 2 ||
        cache.exchange_integrals.cols() != n * n)
        throw std::invalid_argument("exchange integral direction dimensions mismatch");
    if (!can_contract_exchange_direction(cache))
        throw std::logic_error("exchange action exceeds its admitted memory budget");
    const std::size_t image_bytes =
        static_cast<std::size_t>(cache.exchange_integrals.size()) * sizeof(double);
    const std::size_t explicit_cap =
        cache.ao_integral_input->exact_exchange_memory_budget_bytes.value_or(
            std::numeric_limits<std::size_t>::max());
    const std::size_t capacity =
        std::min(workspace::available(), explicit_cap - std::min(explicit_cap, image_bytes)) /
        sizeof(double);
    const std::size_t width = static_cast<std::size_t>(n) * n;
    const std::size_t row_values = static_cast<std::size_t>(n_bf) * width;
    const std::size_t occupied = width * width + static_cast<std::size_t>(n_bf) * n * width +
                                 2 * static_cast<std::size_t>(n_bf) * n;
    const int threads = std::min<int>(effective_openmp_thread_count(), n_bf);
    const Eigen::Index tile = std::min<std::size_t>(threads, (capacity - occupied) / row_values);
    const std::size_t scratch_bytes = (occupied + tile * row_values) * sizeof(double);
    workspace::Reservation action_scratch;
    if (!action_scratch.try_resize(scratch_bytes))
        throw std::logic_error("exchange action cannot reserve its peak scratch");
    workspace::Temporary action_temporary(scratch_bytes);

    // Expand packed pair multiplicities into a small active-only contraction.
    // Index (input orbital, partner-left, partner-right, output orbital).
    ExactCtxPairMatrix weights = ExactCtxPairMatrix::Zero(n, n * n * n);
    const Eigen::Index pairs = cache.active_pair_first_indices.size();
    for (Eigen::Index p = 0; p < pairs; ++p) {
        const int a = cache.active_pair_first_indices[p];
        const int b = cache.active_pair_second_indices[p];
        for (Eigen::Index q = 0; q < pairs; ++q) {
            const int c = cache.active_pair_first_indices[q];
            const int d = cache.active_pair_second_indices[q];
            const double value = cache.active_pair_gradient_matrix(q, p);
            weights(c, (b * n + d) * n + a) += value;
            weights(d, (b * n + c) * n + a) += value;
            weights(c, (a * n + d) * n + b) += value;
            weights(d, (a * n + c) * n + b) += value;
        }
    }
    const ExactCtxPairMatrix mixed = direction * weights;
    const Eigen::Map<const ExactCtxPairMatrix> source(mixed.data(), n_bf * n * n, n);
    ExactCtxPairMatrix result(n_bf, n);
    ExactCtxPairMatrix panel(tile, row_values);
    // Only a shared output-row panel is expanded; the complete ordered image
    // never exists. Each worker writes disjoint rows, with no private big tensor.
#pragma omp parallel num_threads(threads)
    {
        for (Eigen::Index begin = 0; begin < n_bf; begin += tile) {
            const Eigen::Index count = std::min(tile, n_bf - begin);
#pragma omp for schedule(static)
            for (Eigen::Index row = 0; row < count; ++row) {
                const Eigen::Index mu = begin + row;
                for (Eigen::Index nu = 0; nu < n_bf; ++nu) {
                    Eigen::Map<ExactCtxPairMatrix> target(panel.row(row).data() + nu * n * n, n, n);
                    const Eigen::Map<const ExactCtxPairMatrix> block(
                        cache.exchange_integrals.row(ao_pair_index(mu, nu)).data(), n, n);
                    if (mu >= nu)
                        target = block;
                    else
                        target = block.transpose();
                }
            }
#pragma omp for schedule(static)
            for (Eigen::Index row = 0; row < count; ++row)
                result.row(begin + row).noalias() = panel.row(row) * source;
        }
    }
    return result;
}

} // namespace xmvb::integrals::detail

namespace xmvb::integrals {

Eigen::MatrixXd
project_exact_active_density_response(const ExactPackedActiveTwoElectronAdjointCache& cache,
                                      const Eigen::Ref<const Eigen::MatrixXd>& weighted_direction) {
    const Eigen::Index n_bf = cache.n_basis_functions;
    const Eigen::Index n = cache.n_active_orbitals;
    if (weighted_direction.rows() != n_bf || weighted_direction.cols() != n ||
        cache.accepted_pair_products == nullptr ||
        cache.exchange_integrals.rows() != n_bf * (n_bf + 1) / 2 ||
        cache.exchange_integrals.cols() != n * n)
        throw std::invalid_argument("active density projection dimensions mismatch");
    const ExactCtxPairMatrix weights = weighted_direction;
    ExactCtxPairMatrix result = ExactCtxPairMatrix::Zero(n_bf, n);
    const int threads = std::min<int>(effective_openmp_thread_count(), n_bf);
    // Output rows have exclusive owners. Integral images and weights are shared;
    // each worker needs only one active-pair block, never an AO-sized array.
#pragma omp parallel num_threads(threads)
    {
        ExactCtxPairMatrix pair_block(n, n);
#pragma omp for schedule(static)
        for (Eigen::Index mu = 0; mu < n_bf; ++mu) {
            for (Eigen::Index nu = 0; nu < n_bf; ++nu) {
                const auto pair = detail::ao_pair_index(mu, nu);
                const Eigen::Map<const ExactCtxPairMatrix> exchange(
                    cache.exchange_integrals.row(pair).data(), n, n);
                const auto products = cache.accepted_pair_products->row(pair);
                for (Eigen::Index p = 0; p < products.size(); ++p) {
                    const int a = cache.active_pair_first_indices[p];
                    const int b = cache.active_pair_second_indices[p];
                    pair_block(a, b) = products[p];
                    pair_block(b, a) = products[p];
                }
                // E(mu,nu)_(a,b)=(mu a|nu b). Reversed AO orientation
                // transposes its active indices. J(A)C=2 E W,
                // K(A)C=(mu nu|a b)W + E^T W, A=W C^T+C W^T.
                pair_block *= -1.0;
                if (mu >= nu)
                    pair_block += 4.0 * exchange.transpose() - exchange;
                else
                    pair_block += 4.0 * exchange - exchange.transpose();
                result.row(mu).noalias() += weights.row(nu) * pair_block;
            }
        }
    }
    return result;
}

Eigen::MatrixXd project_exact_active_h1e_response(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const ExactPackedActiveTwoElectronAdjointCache& cache) {
    const Eigen::Index n_bf = cache.n_basis_functions;
    const Eigen::Index n = cache.n_active_orbitals;
    const Eigen::Index ao_pairs = n_bf * (n_bf + 1) / 2;
    const Eigen::Index active_pairs = cache.active_pair_first_indices.size();
    if (density.rows() != n_bf || density.cols() != n_bf ||
        cache.accepted_pair_products == nullptr ||
        cache.accepted_pair_products->rows() != ao_pairs ||
        cache.accepted_pair_products->cols() != active_pairs ||
        cache.exchange_integrals.rows() != ao_pairs || cache.exchange_integrals.cols() != n * n)
        throw std::invalid_argument("active H1e response integral dimensions mismatch");

    const int threads =
        coarse_openmp_thread_count(effective_openmp_thread_count(), ao_pairs * n * n);
    Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(active_pairs, threads);
    Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(n * n, threads);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int part = 0; part < threads; ++part) {
        auto j = coulomb.col(part);
        auto k = exchange.col(part);
        for (Eigen::Index pair = ao_pairs * part / threads;
             pair < ao_pairs * (part + 1) / threads; ++pair) {
            const int mu = cache.ao_pair_first_indices[pair], nu = cache.ao_pair_second_indices[pair];
            const double weight = mu == nu ? density(mu, mu) : density(mu, nu) + density(nu, mu);
            j.noalias() += weight * cache.accepted_pair_products->row(pair).transpose();
            k.noalias() += weight * cache.exchange_integrals.row(pair).transpose();
        }
    }
    const Eigen::VectorXd j = coulomb.rowwise().sum();
    const Eigen::VectorXd k = exchange.rowwise().sum();
    const Eigen::Map<const ExactCtxPairMatrix> oriented_exchange(k.data(), n, n);
    // Reversing a triangular AO pair transposes the active exchange indices.
    Eigen::MatrixXd projected = -0.5 * (oriented_exchange + oriented_exchange.transpose());
    for (Eigen::Index pair = 0; pair < active_pairs; ++pair) {
        const int a = cache.active_pair_first_indices[pair];
        const int b = cache.active_pair_second_indices[pair];
        projected(a, b) += 2.0 * j[pair];
        if (a != b)
            projected(b, a) += 2.0 * j[pair];
    }
    return projected;
}

std::vector<double>
backpropagate_exact_active_h1e_gradient(const std::vector<double>& active_gradient,
                                        const ExactPackedActiveTwoElectronAdjointCache& cache) {
    const int n_bf = cache.n_basis_functions;
    const int n = cache.n_active_orbitals;
    if (n_bf <= 0 || n <= 0 || active_gradient.size() != static_cast<std::size_t>(n) * n ||
        cache.accepted_pair_products == nullptr ||
        cache.exchange_integrals.rows() != static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2 ||
        cache.exchange_integrals.cols() != static_cast<Eigen::Index>(n) * n)
        throw std::invalid_argument("active H1e adjoint integral dimensions mismatch");
    const Eigen::Map<const Eigen::MatrixXd> gradient(active_gradient.data(), n, n);
    const ExactCtxPairMatrix density = gradient + gradient.transpose();
    const Eigen::Index pairs = cache.active_pair_first_indices.size();
    Eigen::VectorXd packed(pairs);
    for (Eigen::Index p = 0; p < pairs; ++p) {
        const int a = cache.active_pair_first_indices[p];
        const int b = cache.active_pair_second_indices[p];
        packed[p] = density(a, b) * (a == b ? 1.0 : 2.0);
    }
    const Eigen::Map<const Eigen::VectorXd> full(density.data(), n * n);
    std::vector<double> result(static_cast<std::size_t>(n_bf) * n_bf);
    const int threads = coarse_openmp_thread_count(effective_openmp_thread_count(), result.size());
    // For D=C*(G+G^T)*C^T, the physical AO adjoint is 2*J(D)-K(D).
    // Half of this symmetric matrix has exactly the original pairing with all
    // admissible dD; the orbital pullback consumes R+R^T, not an AO gauge.
#pragma omp parallel for num_threads(threads) schedule(static)
    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(result.size()); ++row) {
        const int mu = row / n_bf;
        const int nu = row % n_bf;
        const double coulomb =
            cache.accepted_pair_products->row(detail::ao_pair_index(mu, nu)).dot(packed);
        // The active density is symmetric, so either AO orientation has the
        // same scalar contraction; no transpose buffer is required here.
        const double exchange =
            cache.exchange_integrals.row(detail::ao_pair_index(mu, nu)).dot(full);
        result[static_cast<std::size_t>(nu) * n_bf + mu] = coulomb - 0.5 * exchange;
    }
    return result;
}

} // namespace xmvb::integrals
