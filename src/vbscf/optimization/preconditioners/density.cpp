#include "vbscf/optimization/preconditioners/density.hpp"

#include <algorithm>
#include <stdexcept>
#include <Eigen/QR>
#include "parallel/openmp.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "mcscf/optimization/trace.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri.hpp"
#include "core/pair_index.hpp"

namespace xmvb::vb {
namespace {
std::size_t tensor4_index(int p, int q, int r, int s, int dimension) noexcept {
    return (((static_cast<std::size_t>(p) * dimension + q) * dimension + r) * dimension + s);
}

/** @brief Shared density channels; active Coulomb images reuse accepted ERIs. */
DensityDiagonals density_diagonal_channels(const VbScfInput& input,
                                           const Eigen::MatrixXd& mo,
                                           int n_core,
                                           int na,
                                           const Eigen::MatrixXd& active_weights,
                                           const std::vector<double>& active_coulomb) {
    const int n_bf = mo.rows();
    if (na * na != active_weights.rows())
        throw std::invalid_argument("invalid active density channels");
    const int channels = n_core + active_weights.cols();
    Eigen::MatrixXd result(n_bf, channels);
    const auto active = mo.middleCols(n_core, na);
    if (input.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity) {
        if (!input.ri_factorization)
            throw std::invalid_argument("RI diagonal requires accepted AO factors");
        const Eigen::Map<const integrals::OrbitalPairMatrix> coulomb(
            active_coulomb.data(), mo.cols(), na * na);
        return integrals::contract_ri_density_diagonals(
            *input.ri_factorization, mo, n_core, na, active_weights, coulomb);
    }
    const std::size_t matrix_bytes = sizeof(double) * static_cast<std::size_t>(n_bf) * n_bf;
    const std::size_t pair_bytes = sizeof(double) * static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
    const std::size_t active_temp = sizeof(double) * static_cast<std::size_t>(n_bf) * na;
    const std::size_t stream_budget = std::max<std::size_t>(
        64 * 1024 * 1024, 3 * matrix_bytes + pair_bytes + sizeof(double) + active_temp);
    const int stream_workers =
        std::min(effective_openmp_thread_count(),
                 static_cast<int>((stream_budget - 3 * matrix_bytes - active_temp) /
                                  (pair_bytes + sizeof(double))));
    int lanes = 1;
#if defined(__GNUC__) && defined(__x86_64__)
    lanes = __builtin_cpu_supports("avx512f") ? 8 : __builtin_cpu_supports("avx2") ? 4 : 2;
#endif
    const std::size_t stream_channel =
        2 * matrix_bytes + stream_workers * (pair_bytes + sizeof(double));
    const int stream_width =
        std::min({channels,
                  lanes,
                  static_cast<int>((stream_budget - matrix_bytes - active_temp) / stream_channel)});
    const auto& graph = input.ao_integral_input.pair_graph;
    if (graph.has_row_graph()) {
        const std::size_t matrix_bytes = sizeof(double) * static_cast<std::size_t>(n_bf) * n_bf;
        const std::size_t output_bytes =
            sizeof(double) * static_cast<std::size_t>(n_bf) * (3 * channels + 2 * n_core);
        const std::size_t active_temp = sizeof(double) * static_cast<std::size_t>(n_bf) * na;
        const std::size_t n_pairs = static_cast<std::size_t>(na) * (na + 1) / 2;
        const std::size_t span_bytes =
            sizeof(double) * (4 * n_pairs * na + n_pairs * active_weights.cols() +
                              2 * na * active_weights.cols() + active_weights.size());
        // One shared source batch and one AO source-building scratch. Each worker
        // keeps a K/J row, its MO projection and its partial diagonal. There are
        // at most two row columns per source, never a worker AO response matrix.
        const std::size_t worker_channel = 6 * sizeof(double) * static_cast<std::size_t>(n_bf);
        const std::size_t base = matrix_bytes + output_bytes + active_temp + span_bytes;
        const std::size_t budget =
            std::max<std::size_t>(64 * 1024 * 1024, base + matrix_bytes + worker_channel);
        const int workers =
            std::min({effective_openmp_thread_count(),
                      n_bf,
                      static_cast<int>((budget - base - matrix_bytes) / worker_channel)});
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
        std::size_t row_records = 2 * graph.values.size();
        for (int mu = 0; mu < n_bf; ++mu) {
            const int diagonal_pair = mu * (mu + 1) / 2 + mu;
            row_records -= graph.row_offsets[diagonal_pair + 1] - graph.row_offsets[diagonal_pair];
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
            xmvb::integrals::AoH1eBatchWorkspace::Channels sources;
            xmvb::integrals::AoDensityDiagonalWorkspace workspace;
            Eigen::MatrixXd scratch(n_bf, n_bf);
            Eigen::MatrixXd exchange, coulomb;
            Eigen::MatrixXd core_coulomb(n_bf, n_core);
            // Only n_bf*n_active^2 accepted values enter the active Coulomb images.
            const Eigen::Map<
                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                j_active(active_coulomb.data(), n_bf, na * na);
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
                    result.rightCols(active_weights.cols()),
                    std::move(core_coulomb)};
        }
    }
    // Canonical scatter is also the required large-address storage algorithm.
    // Budget all batch inputs, packed worker outputs, response and shared scratch.
    // For exceptionally large AO bases the unavoidable one-channel/one-worker
    // working set is the lower bound; the budget never enables further replicas.
    const auto budget = stream_budget, channel_bytes = stream_channel;
    const int workers = stream_workers, width = stream_width;
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
    return {result.leftCols(n_core), result.rightCols(active_weights.cols()), {}};
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
                                             const std::vector<double>& active_coulomb) {
    const int n = dm1.rows();
    const int n_pairs = n * (n + 1) / 2;
    const int n_sources = 2 * n;
    Eigen::MatrixXd sources = Eigen::MatrixXd::Zero(n_pairs, n_sources);
    for (int w = 0; w < n; ++w)
        for (int z = 0; z <= w; ++z) {
            const int pair = xmvb::core::PairIndex::packed_pair_index(w, z);
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
                const int pair = xmvb::core::PairIndex::packed_pair_index(w, z);
                matrix(w, z) = matrix(z, w) = basis(pair, j) * (w == z ? 1.0 : 0.5);
            }
    }
    auto images = density_diagonal_channels(input, mo, n_core, n, weights, active_coulomb);
    Eigen::MatrixXd active_images(mo.cols(), 2 * n + 1);
    active_images.rightCols(2 * n).noalias() = images.active * reconstruction;
    // The total one-RDM source is exactly the sum of its n orbital sources.
    active_images.col(0) = active_images.middleCols(1, n).rowwise().sum();
    return {std::move(images.core), std::move(active_images), std::move(images.core_coulomb)};
}

} // namespace xmvb::vb
