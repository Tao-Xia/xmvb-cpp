#include "vbscf/optimization/preconditioners/density.hpp"

#include <algorithm>
#include <stdexcept>
#include <Eigen/QR>
#include "parallel/openmp.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "mcscf/optimization/trace.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri.hpp"
#include "core/index.hpp"
#include "core/pair_index.hpp"
#include "core/workspace.hpp"

namespace xmvb::vb {
namespace {
std::size_t tensor4_index(int p, int q, int r, int s, int dimension) noexcept {
    return (((static_cast<std::size_t>(p) * dimension + q) * dimension + r) * dimension + s);
}

/** @brief Project the borrowed (mu a|nu b) image against symmetric densities. */
Eigen::MatrixXd project_cached_exchange(const Eigen::MatrixXd& mo,
                                        const Eigen::MatrixXd& weights,
                                        const integrals::ExactCtxPairMatrix& exchange,
                                        const Eigen::MatrixXd* active_to_exchange) {
    const Eigen::Index n_bf = mo.rows(), width = weights.cols();
    const auto pairs = core::triangular_count(n_bf);
    if (exchange.rows() != static_cast<Eigen::Index>(pairs) || exchange.cols() != weights.rows())
        throw std::invalid_argument("invalid accepted exchange image for analytic diagonal");
    const int threads = bounded_openmp_thread_count(effective_openmp_thread_count(), mo.cols());
    const auto worker_bytes = sizeof(double) * threads * width;
    const auto weight_bytes = active_to_exchange ? sizeof(double) * weights.size() : 0;
    auto packed = workspace::make_matrix<integrals::ExactCtxPairMatrix>(
        pairs, width, worker_bytes + weight_bytes);
    if (!packed)
        return {};
    workspace::Temporary packed_temporary(sizeof(double) * pairs * width);
    {
        workspace::Reservation scratch;
        if (!scratch.try_resize(weight_bytes))
            return {};
        if (active_to_exchange) {
            workspace::Temporary weights_temporary(weight_bytes);
            Eigen::MatrixXd transformed(weights.rows(), width);
            const auto& r = *active_to_exchange;
            for (Eigen::Index c = 0; c < width; ++c) {
                const Eigen::Map<const Eigen::MatrixXd> w(weights.col(c).data(), r.cols(), r.cols());
                Eigen::Map<Eigen::MatrixXd> target(transformed.col(c).data(), r.rows(), r.rows());
                target.noalias() = (r * w) * r.transpose();
            }
            packed->noalias() = exchange * transformed;
        } else {
            packed->noalias() = exchange * weights;
        }
    }
    workspace::Reservation workers;
    if (!workers.try_resize(worker_bytes))
        return {};
    workspace::Temporary worker_temporary(worker_bytes);
    Eigen::MatrixXd result(mo.cols(), width);
#pragma omp parallel num_threads(threads)
    {
        Eigen::RowVectorXd diagonal(width);
#pragma omp for schedule(static)
        for (Eigen::Index p = 0; p < mo.cols(); ++p) {
            diagonal.setZero();
            Eigen::Index pair = 0;
            for (Eigen::Index mu = 0; mu < n_bf; ++mu)
                for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair)
                    diagonal.noalias() +=
                        (mo(mu, p) * mo(nu, p) * (mu == nu ? 1.0 : 2.0)) * packed->row(pair);
            result.row(p) = diagonal;
        }
    }
    return result;
}

/** @brief Shared density channels; active Coulomb images reuse accepted ERIs. */
DensityDiagonals density_diagonal_channels(const VbScfInput& input,
                                           const Eigen::MatrixXd& mo,
                                           int n_core,
                                           int na,
                                           const Eigen::MatrixXd& active_weights,
                                           const std::vector<double>& active_coulomb,
                                           const integrals::ExactCtxPairMatrix* exchange_image,
                                           const Eigen::MatrixXd* active_to_exchange) {
    const int n_bf = mo.rows();
    if (na * na != active_weights.rows())
        throw std::invalid_argument("invalid active density channels");
    const auto active = mo.middleCols(n_core, na);
    if (input.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity) {
        if (!input.ri_factorization)
            throw std::invalid_argument("RI diagonal requires accepted AO factors");
        const Eigen::Map<const integrals::OrbitalPairMatrix> coulomb(
            active_coulomb.data(), mo.cols(), na * na);
        return integrals::contract_ri_density_diagonals(
            *input.ri_factorization, mo, n_core, na, active_weights, coulomb);
    }
    const Eigen::Map<const integrals::OrbitalPairMatrix> j_active(
        active_coulomb.data(), mo.cols(), na * na);
    Eigen::MatrixXd cached_active;
    if (exchange_image) {
        cached_active = project_cached_exchange(mo, active_weights, *exchange_image, active_to_exchange);
        if (cached_active.size())
            cached_active = 2.0 * j_active * active_weights - cached_active;
    }
    const bool reuse = cached_active.size() != 0;
    const int channels = n_core + (reuse ? 0 : active_weights.cols());
    mcscf::diagnostics::ciah_trace("ciah_diagonal_exchange_reuse",
                                   "exchange_channels=%d,reused_channels=%d",
                                   channels,
                                   reuse ? static_cast<int>(active_weights.cols()) : 0);
    if (!channels)
        return {Eigen::MatrixXd(n_bf, 0), std::move(cached_active), Eigen::MatrixXd(n_bf, 0)};
    const auto& graph = input.ao_integral_input.pair_graph;
    // The half transform scans each left pair once. Use it for core-only dense
    // pair rows, where a dense one-index transform is cheaper than updating
    // both orientations of every stored quartet across all density channels.
    const long double half_terms =
        static_cast<long double>(graph.pair_first.size()) * n_bf * n_bf;
    if (reuse && n_core >= Eigen::internal::packet_traits<double>::size &&
        graph.has_rows() && half_terms < 4.0L * graph.directed_count()) {
        Eigen::MatrixXd exchange, coulomb;
        if (integrals::apply_ao_core_diagonals(input.ao_integral_input,
                                              mo,
                                              n_core,
                                              effective_openmp_thread_count(),
                                              &exchange,
                                              &coulomb)) {
            exchange *= -1.0;
            exchange.noalias() += 2.0 * coulomb;
            return {std::move(exchange), std::move(cached_active), std::move(coulomb)};
        }
    }
    Eigen::MatrixXd result(n_bf, channels);
    const auto active_result = [&]() -> Eigen::MatrixXd {
        if (reuse)
            return std::move(cached_active);
        return result.rightCols(active_weights.cols());
    };
    const std::size_t matrix_bytes = sizeof(double) * static_cast<std::size_t>(n_bf) * n_bf;
    const std::size_t pair_bytes = sizeof(double) * static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
    const std::size_t active_temp = sizeof(double) * static_cast<std::size_t>(n_bf) * na;
    const std::size_t budget = workspace::available();
    const std::size_t stream_minimum = 3 * matrix_bytes + pair_bytes + sizeof(double) + active_temp;
    if (budget < stream_minimum)
        throw std::runtime_error("diagonal density resource_limit: one channel exceeds workspace");
    const int stream_workers = static_cast<int>(
        std::min({static_cast<std::size_t>(effective_openmp_thread_count()),
                  static_cast<std::size_t>(n_bf),
                  (budget - 3 * matrix_bytes - active_temp) / (pair_bytes + sizeof(double))}));
    int lanes = 1;
#if defined(__GNUC__) && defined(__x86_64__)
    lanes = __builtin_cpu_supports("avx512f") ? 8 : __builtin_cpu_supports("avx2") ? 4 : 2;
#endif
    const std::size_t stream_channel =
        2 * matrix_bytes + stream_workers * (pair_bytes + sizeof(double));
    const int stream_width =
        static_cast<int>(std::min({static_cast<std::size_t>(channels),
                                   static_cast<std::size_t>(lanes),
                                   (budget - matrix_bytes - active_temp) / stream_channel}));
    const std::size_t output_bytes =
        sizeof(double) * static_cast<std::size_t>(n_bf) * (3 * channels + 2 * n_core);
    const std::size_t n_pairs = static_cast<std::size_t>(na) * (na + 1) / 2;
    const std::size_t span_bytes =
        sizeof(double) * (4 * n_pairs * na + n_pairs * active_weights.cols() +
                          2 * na * active_weights.cols() + active_weights.size());
    // Row workers own one K/J row, its MO projection, and a partial diagonal.
    const std::size_t worker_channel = 6 * sizeof(double) * static_cast<std::size_t>(n_bf);
    const std::size_t base = matrix_bytes + output_bytes + active_temp + span_bytes;
    if (graph.has_rows() && budget >= base + matrix_bytes + worker_channel) {
        const int workers =
            static_cast<int>(std::min({static_cast<std::size_t>(effective_openmp_thread_count()),
                                       static_cast<std::size_t>(n_bf),
                                       (budget - base - matrix_bytes) / worker_channel}));
        const std::size_t per_channel = matrix_bytes + workers * worker_channel;
        int width = std::min<std::size_t>(channels, (budget - base) / per_channel);
        // A final SIMD tail does not justify another complete ERI traversal.
        if (width < channels && width >= lanes)
            width -= width % lanes;
        // Choose the exact layout by counted integral-record visits, not a
        // molecular threshold. Ordered AO rows replay off-diagonal CSR rows;
        // canonical scatter reads unique ERIs. Do not use the saved core action
        // to justify extra main-contraction reads: its fused projection also has
        // work, and is deliberately not priced as a free benefit in admission.
        std::size_t row_records = 2 * graph.directed_count();
        for (int mu = 0; mu < n_bf; ++mu) {
            const auto diagonal_pair = core::PairIndex::packed_pair_index(mu, mu);
            graph.visit_row(diagonal_pair, [&](std::size_t, double) { --row_records; });
        }
        const long double row_reads =
            static_cast<long double>(row_records) * ((channels + width - 1) / width);
        const long double stream_reads = static_cast<long double>(graph.integral_count()) *
                                         ((channels + stream_width - 1) / stream_width);
        // Both layouts are exact. The row consumer no longer pays for an AO
        // response/staging image, so choose by actual ERI visits within its budget.
        const bool row_selected = row_reads < stream_reads;
        mcscf::diagnostics::ciah_trace("ciah_diagonal_layout_plan",
                                       "row_width=%d,stream_width=%d,row_records=%.0Lf,stream_"
                                       "records=%.0Lf,row_workspace_bytes=%zu,row_selected=%d",
                                       width,
                                       stream_width,
                                       row_reads,
                                       stream_reads,
                                       base + width * per_channel,
                                       static_cast<int>(row_selected));
        if (row_selected) {
            workspace::Reservation action_scratch;
            const std::size_t row_peak = base + width * per_channel;
            if (!action_scratch.try_resize(row_peak))
                throw std::runtime_error(
                    "diagonal density resource_limit: row action reservation failed");
            workspace::Temporary action_temporary(row_peak);
            xmvb::integrals::AoH1eBatchWorkspace::Channels sources;
            xmvb::integrals::AoDensityDiagonalWorkspace workspace;
            Eigen::MatrixXd scratch(n_bf, n_bf);
            Eigen::MatrixXd exchange, coulomb;
            Eigen::MatrixXd core_coulomb(n_bf, n_core);
            for (int begin = 0; begin < channels; begin += width) {
                const int count = std::min(width, channels - begin);
                const int core_count = std::max(0, std::min(count, n_core - begin));
                sources.resize(static_cast<Eigen::Index>(n_bf) * n_bf, count);
                for (int j = 0; j < count; ++j) {
                    const int channel = begin + j;
                    if (channel < n_core)
                        scratch.noalias() = mo.col(channel) * mo.col(channel).transpose();
                    else {
                        const Eigen::Map<const Eigen::MatrixXd> weights(
                            active_weights.col(channel - n_core).data(), na, na);
                        scratch.noalias() = active * weights * active.transpose();
                    }
                    for (Eigen::Index row = 0; row < scratch.size(); ++row)
                        sources(row, j) = scratch.data()[row];
                }
                xmvb::integrals::apply_ao_density_diagonals(input.ao_integral_input,
                                                            sources,
                                                            mo,
                                                            core_count,
                                                            workers,
                                                            &workspace,
                                                            &exchange,
                                                            &coulomb);
                if (core_count) {
                    core_coulomb.middleCols(begin, core_count) = coulomb;
                    result.middleCols(begin, core_count) =
                        2.0 * coulomb - exchange.leftCols(core_count);
                }
                const int active_count = count - core_count;
                if (active_count)
                    result.middleCols(begin + core_count, active_count).noalias() =
                        2.0 * j_active *
                            active_weights.middleCols(begin + core_count - n_core, active_count) -
                        exchange.rightCols(active_count);
            }
            mcscf::diagnostics::ciah_trace("ciah_diagonal_density_batch",
                                           "channels=%d,width=%d,workers=%d,workspace_bytes=%zu,"
                                           "budget_bytes=%zu,row_owned=1,active_exchange_only=1",
                                           channels,
                                           width,
                                           workers,
                                           base + width * per_channel,
                                           budget);
            return {result.leftCols(n_core),
                    active_result(),
                    std::move(core_coulomb)};
        }
    }
    // Canonical scatter is also the required large-address storage algorithm.
    // Budget all batch inputs, packed worker outputs, response and shared scratch.
    // For exceptionally large AO bases the unavoidable one-channel/one-worker
    // working set is the lower bound; the budget never enables further replicas.
    const auto channel_bytes = stream_channel;
    const int workers = stream_workers, width = stream_width;
    workspace::Reservation action_scratch;
    const std::size_t stream_peak = matrix_bytes + active_temp + width * channel_bytes;
    if (!action_scratch.try_resize(stream_peak))
        throw std::runtime_error(
            "diagonal density resource_limit: stream action reservation failed");
    workspace::Temporary action_temporary(stream_peak);
    xmvb::integrals::AoH1eBatchWorkspace workspace;
    Eigen::MatrixXd response, scratch(n_bf, n_bf);
    for (int begin = 0; begin < channels; begin += width) {
        const int count = std::min(width, channels - begin);
        workspace.sources.resize(static_cast<Eigen::Index>(n_bf) * n_bf, count);
        for (int j = 0; j < count; ++j) {
            const int channel = begin + j;
            if (channel < n_core)
                scratch.noalias() = mo.col(channel) * mo.col(channel).transpose();
            else {
                const Eigen::Map<const Eigen::MatrixXd> weights(
                    active_weights.col(channel - n_core).data(), na, na);
                scratch.noalias() = active * weights * active.transpose();
            }
            for (Eigen::Index row = 0; row < scratch.size(); ++row)
                workspace.sources(row, j) = scratch.data()[row];
        }
        xmvb::integrals::apply_symmetric_ao_h1e_channels(
            input.ao_integral_input, workers, &workspace, &response);
        for (int j = 0; j < count; ++j) {
            const Eigen::Map<const Eigen::MatrixXd> image(response.col(j).data(), n_bf, n_bf);
            scratch.noalias() = image * mo;
            result.col(begin + j) = (mo.array() * scratch.array()).colwise().sum().transpose();
        }
    }
    mcscf::diagnostics::ciah_trace(
        "ciah_diagonal_density_batch",
        "channels=%d,width=%d,workers=%d,workspace_bytes=%zu,budget_bytes=%zu",
        channels,
        width,
        workers,
        matrix_bytes + active_temp + width * channel_bytes,
        budget);
    return {result.leftCols(n_core), active_result(), {}};
}

} // namespace

/** @brief Evaluate only the density span consumed by the analytic diagonal.
 *
 * Untruncated Householder QR reduces the number of integral sweeps to
 * min(n_active_pairs,2*n_active). The total one-RDM source is their exact sum.
 * There is no numerical rank cutoff,
 * division by a small singular value, or approximate density compression.
 */
DensityDiagonals contract_diagonal_densities(const VbScfInput& input,
                                             const Eigen::MatrixXd& mo,
                                             int n_core,
                                             const Eigen::MatrixXd& dm1,
                                             const std::vector<double>& dm2,
                                             const std::vector<double>& active_coulomb,
                                             const integrals::ExactCtxPairMatrix* exchange,
                                             const Eigen::MatrixXd* active_to_exchange) {
    const int n = dm1.rows();
    const int n_pairs = n * (n + 1) / 2;
    const int n_sources = 2 * n;
    Eigen::MatrixXd sources = Eigen::MatrixXd::Zero(n_pairs, n_sources);
    for (int w = 0; w < n; ++w)
        for (int z = 0; z <= w; ++z) {
            const auto pair = xmvb::core::PairIndex::packed_pair_index(w, z);
            for (int a = 0; a < n; ++a) {
                sources(pair, a) = w == z ? (w == a ? dm1(a, w) : 0.0)
                                          : (w == a ? dm1(a, z) : 0.0) + (z == a ? dm1(a, w) : 0.0);
                const double wz =
                    dm2[tensor4_index(a, w, z, a, n)] + dm2[tensor4_index(w, a, z, a, n)];
                const double zw =
                    dm2[tensor4_index(a, z, w, a, n)] + dm2[tensor4_index(z, a, w, a, n)];
                sources(pair, n + a) = w == z ? wz : wz + zw;
            }
        }
    const int width = std::min(n_pairs, n_sources);
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(sources);
    const Eigen::MatrixXd basis = qr.householderQ() * Eigen::MatrixXd::Identity(n_pairs, width);
    const Eigen::MatrixXd reconstruction =
        qr.matrixQR().topRows(width).template triangularView<Eigen::Upper>();
    Eigen::MatrixXd weights(n * n, width);
    for (int j = 0; j < width; ++j) {
        Eigen::Map<Eigen::MatrixXd> matrix(weights.col(j).data(), n, n);
        for (int w = 0; w < n; ++w)
            for (int z = 0; z <= w; ++z) {
                const auto pair = xmvb::core::PairIndex::packed_pair_index(w, z);
                matrix(w, z) = matrix(z, w) = basis(pair, j) * (w == z ? 1.0 : 0.5);
            }
    }
    auto images = density_diagonal_channels(
        input, mo, n_core, n, weights, active_coulomb, exchange, active_to_exchange);
    Eigen::MatrixXd active_images(mo.cols(), 2 * n + 1);
    active_images.rightCols(2 * n).noalias() = images.active * reconstruction;
    // The total one-RDM source is exactly the sum of its n orbital sources.
    active_images.col(0) = active_images.middleCols(1, n).rowwise().sum();
    return {std::move(images.core), std::move(active_images), std::move(images.core_coulomb)};
}

} // namespace xmvb::vb
