#include "integrals/response/internal.hpp"
#include "integrals/response/adjoint.hpp"

#include <algorithm>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "integrals/ao/input.hpp"

namespace xmvb::integrals::detail {

void build_exchange_integrals(ExactPackedActiveTwoElectronAdjointCache* cache) {
    const Eigen::Index n_bf = cache->n_basis_functions;
    const Eigen::Index n = cache->n_active_orbitals;
    const std::size_t budget = cache->ao_integral_input->exact_exchange_memory_budget_bytes;
    const std::size_t width = static_cast<std::size_t>(n) * n;
    const std::size_t rows = static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
    const auto& graph = cache->ao_integral_input->pair_graph;
    // Keep the large-space streamed representation. The additional integral
    // image has its own memory bound and needs resident accepted pair products.
    if (cache->accepted_pair_products == nullptr || !graph.has_row_graph() || rows == 0 ||
        width == 0 || width > budget / sizeof(double) / rows)
        return;
    // Include the shared active weights and mixed-direction workspace in the
    // admission bound, rather than bounding only the resident integral tensor.
    const std::size_t slice_values = static_cast<std::size_t>(n_bf) * n_bf * n;
    const std::size_t coefficients_values = static_cast<std::size_t>(n_bf) * n;
    const std::size_t action_values =
        static_cast<std::size_t>(n_bf + n) * n * width + 2 * coefficients_values;
    const std::size_t panel_row = static_cast<std::size_t>(n_bf) * width;
    const std::size_t workspace =
        std::max(slice_values + coefficients_values, action_values + panel_row);
    const std::size_t available = budget / sizeof(double) - rows * width;
    if (workspace > available)
        return;
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ExchangeBuild);
    const ExactCtxPairMatrix coefficients = *cache->accepted_active_coefficients;
    cache->exchange_integrals.resize(rows, n * n);
    const int threads = std::min<int>(effective_openmp_thread_count(), n_bf);
    const Eigen::Index tile =
        std::min<Eigen::Index>(n_bf, (available - coefficients_values) / slice_values);
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
                for (int edge = graph.row_offsets[pair]; edge < graph.row_offsets[pair + 1];
                     ++edge) {
                    const int right = graph.columns[edge];
                    const int k = graph.pair_first[right], l = graph.pair_second[right];
                    const double value = graph.values[edge];
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
                }
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

Eigen::MatrixXd contract_exchange_direction(const ExactPackedActiveTwoElectronAdjointCache& cache,
                                            const Eigen::Ref<const Eigen::MatrixXd>& direction) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ExchangeAction);
    const Eigen::Index n_bf = cache.n_basis_functions;
    const Eigen::Index n = cache.n_active_orbitals;
    if (direction.rows() != n_bf || direction.cols() != n ||
        cache.exchange_integrals.rows() != n_bf * (n_bf + 1) / 2 ||
        cache.exchange_integrals.cols() != n * n)
        throw std::invalid_argument("exchange integral direction dimensions mismatch");

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
    const int threads = std::min<int>(effective_openmp_thread_count(), n_bf);
    const std::size_t occupied =
        cache.exchange_integrals.size() + weights.size() + mixed.size() + 2 * result.size();
    const std::size_t capacity =
        cache.ao_integral_input->exact_exchange_memory_budget_bytes / sizeof(double);
    const std::size_t row_values = static_cast<std::size_t>(n_bf) * n * n;
    if (occupied > capacity || capacity - occupied < row_values)
        throw std::logic_error("exchange action exceeds its admitted memory budget");
    const Eigen::Index tile = std::min<std::size_t>(threads, (capacity - occupied) / row_values);
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
