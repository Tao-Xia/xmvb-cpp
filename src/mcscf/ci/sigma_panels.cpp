#include "mcscf/ci/sigma.hpp"

#include <algorithm>
#include <stdexcept>

#include "core/index.hpp"
#include "mcscf/ci/sigma_scatter.hpp"
#include "mcscf/ci/sigma_diagonal.hpp"
#include "mcscf/ci/workspace.hpp"
#include "mcscf/optimization/trace.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf {

void DirectCiSigmaAction::apply_sigma_panels(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                             Eigen::MatrixXd* sigma,
                                             int parity,
                                             bool project_exchange) const {
    if (parity < -1 || parity > 1 || (parity && distinct_beta_) ||
        coefficients.rows() != n_alpha_ || coefficients.cols() <= 0 ||
        coefficients.cols() % n_beta_)
        throw std::invalid_argument("invalid direct-CI panel input");
    const auto& channels =
        diagonal_channels_->get(hamiltonian_pair_kernel_, *alpha_, beta_connections());
    const auto off_pairs = channels.kernel.rows();
    const auto pairs = static_cast<std::size_t>(std::max<Eigen::Index>(1, off_pairs));
    const auto width = core::checked_index<int>(coefficients.cols() / n_beta_);
    const CiWorkspace operation(workspace::available(), false);
    const auto values = operation.bytes() / sizeof(double);
    if (values < 3 * pairs + 1)
        throw std::length_error("one CI panel row exceeds workspace");
    const auto team_capacity =
        (values - pairs) / (2 * static_cast<std::size_t>(n_beta_) + 2 * pairs + 1);
    const int workers = static_cast<int>(
        std::max<std::size_t>(1,
                              std::min({static_cast<std::size_t>(effective_openmp_thread_count()),
                                        static_cast<std::size_t>(n_beta_),
                                        team_capacity})));
    // Only output contributions are replicated in a shared bank. Each bank
    // covers one alpha panel, never a complete CI coefficient/image vector.
    const auto row_values =
        static_cast<std::size_t>(workers) * ((workers > 1 ? n_beta_ : 0) + 2 * pairs + 1);
    // One initialized prefix per beta column and worker, shared by all RHS.
    const auto prefix_values =
        workers > 1 ? (static_cast<std::size_t>(workers) * n_beta_ * sizeof(StringIndex) +
                       sizeof(double) - 1) /
                          sizeof(double)
                    : 0;
    const auto panel_values = values - prefix_values;
    const auto source_values = pairs;
    const int lanes = static_cast<int>(std::min<std::size_t>(
        width, std::max<std::size_t>(1, (panel_values - source_values) / row_values)));
    const auto per_row = row_values * lanes + source_values;
    const auto cache_values =
        static_cast<std::size_t>(std::max<std::ptrdiff_t>(sizeof(double), Eigen::l2CacheSize())) /
        sizeof(double);
    const auto kernel_values = pairs * pairs;
    // Only GEMM operands need L2 residency. Requiring the output contribution
    // bank to fit L2 would reduce arithmetic panels to one row at large CAS.
    const auto local_row = static_cast<std::size_t>(lanes) * (2 * pairs + 1);
    const auto cache_rows = std::max<std::size_t>(
        1, (cache_values - std::min(cache_values, kernel_values)) / local_row);
    const auto extent =
        static_cast<StringIndex>(std::min({static_cast<std::size_t>(n_alpha_),
                                           panel_values / per_row,
                                           cache_rows,
                                           static_cast<std::size_t>(detail::sigma_alpha_tile)}));
    if (!extent)
        throw std::length_error("one shared CI panel exceeds workspace");
    diagnostics::ciah_trace("ciah_ci_panel",
                            "workers=%d,alpha=%ld,rhs=%d,scratch_bytes=%zu",
                            workers,
                            static_cast<long>(extent),
                            lanes,
                            (static_cast<std::size_t>(extent) * per_row + prefix_values) *
                                sizeof(double));
    Eigen::MatrixXd partial;
    Eigen::Matrix<StringIndex, Eigen::Dynamic, Eigen::Dynamic> initialized;
    std::size_t written_values = 0;
    detail::AlphaChannelSources sources;
    sigma->resize(n_alpha_, coefficients.cols());
    int team = 1;
#pragma omp parallel if (workers > 1) num_threads(workers) reduction(+ : written_values)
    {
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        Eigen::MatrixXd input(extent * lanes, off_pairs), transformed(extent * lanes, off_pairs);
        Eigen::MatrixXd diagonal(extent, lanes);
#pragma omp single
        {
#ifdef _OPENMP
            team = omp_get_num_threads();
#endif
            partial.resize(extent, team > 1 ? team * n_beta_ * lanes : 0);
            initialized.resize(n_beta_, team > 1 ? team : 0);
        }
#pragma omp for schedule(static)
        for (Eigen::Index b = 0; b < sigma->cols(); ++b)
            sigma->col(b).setZero();
        for (int lane = 0; lane < width; lane += lanes) {
            const int count_lanes = std::min(lanes, width - lane);
            const auto coefficients_block =
                coefficients.middleCols(lane * n_beta_, count_lanes * n_beta_);
            auto bank =
                partial.middleCols(worker * n_beta_ * lanes, team > 1 ? n_beta_ * count_lanes : 0);
            for (StringIndex first = 0; first < n_alpha_; first += extent) {
                const auto count = std::min(extent, n_alpha_ - first);
#pragma omp single
                {
                    sources.reset(*alpha_, first, count, off_pairs, true);
                }
                if (team > 1)
                    initialized.col(worker).setZero();
#pragma omp for schedule(static, 1)
                for (StringIndex b = parity ? first + (parity < 0) : 0; b < n_beta_; ++b) {
                    const auto rows = parity ? std::min(count, b - first + (parity > 0)) : count;
                    auto scatter = input.topRows(rows * count_lanes);
                    auto image = transformed.topRows(rows * count_lanes);
                    sources.scatter(
                        beta_connections(), coefficients_block, first, rows, b, scatter);
                    const double factor = parity ? 2.0 : 1.0;
                    image.noalias() = factor * scatter * channels.kernel.transpose();
                    // n = n_alpha + n_beta. The diagonal channels have rank one
                    // per product row; retain BOTH off/diagonal cross terms.
                    const auto& beta_channels = channels.beta();
                    for (StringIndex a = 0; a < rows; ++a) {
                        double value = channels.alpha.self[first + a] + beta_channels.self[b];
                        for (int p : beta_connections().occupied(b))
                            value += 2.0 * channels.alpha.potential(first + a, p);
                        for (int v = 0; v < count_lanes; ++v)
                            diagonal(a, v) = value * coefficients_block(first + a, v * n_beta_ + b);
                    }
                    for (Eigen::Index p = 0; p < off_pairs; ++p) {
                        const double beta_value = beta_channels.off(b, p);
                        for (int v = 0; v < count_lanes; ++v) {
                            const auto* x = coefficients_block.col(v * n_beta_ + b).data() + first;
                            const auto* r = scatter.col(p).data() + v * rows;
                            auto* z = image.col(p).data() + v * rows;
                            auto* d = diagonal.col(v).data();
                            const auto* alpha_value = channels.alpha.off.col(p).data() + first;
#pragma omp simd
                            for (StringIndex a = 0; a < rows; ++a) {
                                const double cross = alpha_value[a] + beta_value;
                                z[a] += factor * cross * x[a];
                                d[a] += cross * r[a];
                            }
                        }
                    }
                    // Alpha and the collapsed number-channel return have one
                    // beta-column owner. Beta off-diagonal returns use the bank.
                    sources.gather(rows, image, *sigma, lane * n_beta_ + b, n_beta_, count_lanes);
                    for (int v = 0; v < count_lanes; ++v) {
                        auto* target = sigma->col((lane + v) * n_beta_ + b).data() + first;
#pragma omp simd
                        for (StringIndex a = 0; a < rows; ++a)
                            target[a] += (parity && first + a == b ? 1.0 : factor) * diagonal(a, v);
                    }
                    // Remove one diagonal orientation before the final exchange
                    // projection. P(2 Z) is the full H image; no diagonal doubling.
                    const auto beta_rows = rows - (parity > 0 && b < first + count);
                    const auto add_beta = [&](StringIndex source, int pair, double sign) {
                        if (team > 1) {
                            auto& end = initialized(source, worker);
                            // Every link writes a prefix. Extend it on first use;
                            // entries outside the union are exact structural zeros.
                            if (beta_rows > end) {
                                for (int v = 0; v < count_lanes; ++v)
                                    bank.col(v * n_beta_ + source)
                                        .segment(end, beta_rows - end)
                                        .setZero();
                                end = beta_rows;
                            }
                        }
                        for (int v = 0; v < count_lanes; ++v) {
                            double* target =
                                team > 1 ? bank.col(v * n_beta_ + source).data()
                                         : sigma->col((lane + v) * n_beta_ + source).data() + first;
                            const double* values = image.col(pair).data() + v * rows;
#pragma omp simd
                            for (StringIndex a = 0; a < beta_rows; ++a)
                                target[a] += sign * values[a];
                        }
                    };
                    const auto& beta = beta_connections();
                    for (const auto& link : beta.singles(b))
                        add_beta(link.source,
                                 link.pair - std::max(link.created, link.annihilated),
                                 link.sign);
                }
                if (team > 1) {
#pragma omp for schedule(static)
                    for (StringIndex b = 0; b < count_lanes * n_beta_; ++b) {
                        double* target = sigma->col(lane * n_beta_ + b).data() + first;
                        for (int part = 0; part < team; ++part) {
                            const auto end = initialized(b % n_beta_, part);
                            written_values += static_cast<std::size_t>(end);
                            const double* source = partial.col(part * n_beta_ * lanes + b).data();
#pragma omp simd
                            for (StringIndex a = 0; a < end; ++a)
                                target[a] += source[a];
                        }
                    }
                }
            }
        }
    }
    diagnostics::ciah_trace("ciah_ci_beta_bank",
                            "written=%zu,dense=%zu",
                            written_values,
                            team > 1 ? static_cast<std::size_t>(team) * n_alpha_ * n_beta_ * width
                                     : 0);
    if (parity && project_exchange) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (Eigen::Index column = 0; column < sigma->cols(); ++column) {
            const auto b = column % n_beta_, offset = column - b;
            for (StringIndex a = 0; a < b; ++a) {
                const double value = 0.5 * ((*sigma)(a, column) + parity * (*sigma)(b, offset + a));
                (*sigma)(a, column) = value;
                (*sigma)(b, offset + a) = parity * value;
            }
            if (parity < 0)
                (*sigma)(b, column) = 0.0;
        }
    }
}

} // namespace xmvb::mcscf
