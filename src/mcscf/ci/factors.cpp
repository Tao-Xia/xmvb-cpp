#include <cstdint>
#include "mcscf/ci/factors.hpp"
#include "mcscf/ci/factors_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"

namespace xmvb::mcscf {
namespace detail {

using Clock = std::chrono::steady_clock;

/** @brief Mix channel columns without replicating or coupling source labels. */
void mix_channels(const Eigen::Ref<const Eigen::MatrixXd>& channels,
                  const Eigen::MatrixXd& kernel,
                  int p,
                  int q,
                  Eigen::Index width,
                  Eigen::Ref<Eigen::MatrixXd> image) {
    const int rows = channels.rows() * width;
    const int np = image.cols() / width, nq = channels.cols() / width;
    Eigen::Map<const Eigen::MatrixXd> input(channels.data(), rows, nq);
    Eigen::Map<Eigen::MatrixXd> output(image.data(), rows, np);
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(rows) * np * nq);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (int part = 0; part < workers; ++part) {
        const Eigen::Index first = rows * part / workers, last = rows * (part + 1) / workers;
        if (last > first)
            output.middleRows(first, last - first).noalias() +=
                input.middleRows(first, last - first) * kernel.block(p, q, np, nq).transpose();
    }
}

/** @brief Sum two spin-channel Grams, with disjoint shared output row ownership. */
void add_channel_gram(const Eigen::Ref<const Eigen::MatrixXd>& ca,
                      const Eigen::Ref<const Eigen::MatrixXd>& cb,
                      const Eigen::Ref<const Eigen::MatrixXd>& qa,
                      const Eigen::Ref<const Eigen::MatrixXd>& qb,
                      Eigen::Index width,
                      Eigen::Ref<Eigen::MatrixXd> result) {
    const int np = result.rows(), nq = result.cols();
    const std::int64_t na = ca.rows() * width, nb = cb.rows() * width;
    const Eigen::Map<const Eigen::MatrixXd> a(ca.data(), na, np), b(cb.data(), nb, np);
    const Eigen::Map<const Eigen::MatrixXd> x(qa.data(), na, nq), y(qb.data(), nb, nq);
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(na + nb) * np * nq);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (int part = 0; part < workers; ++part) {
        const Eigen::Index first = np * part / workers, last = np * (part + 1) / workers;
        if (last == first)
            continue;
        result.middleRows(first, last - first).noalias() +=
            a.middleCols(first, last - first).transpose() * x;
        result.middleRows(first, last - first).noalias() +=
            b.middleCols(first, last - first).transpose() * y;
    }
}

void validate_source(const DirectCiSigmaAction& sigma, const SpinFactorSource& source) {
    if (source.n_alpha != sigma.n_alpha_strings() || source.n_beta != sigma.n_beta_strings() ||
        source.count < 0 || !source.load || source.load_work < 0 ||
        !std::isfinite(source.load_work) || source.contract_work < 0 ||
        !std::isfinite(source.contract_work))
        throw std::invalid_argument("invalid direct-CI spin-factor source");
}

/** @brief Accumulate one product by shared output rows, with single-thread BLAS. */
void add_product(const Eigen::Ref<const Eigen::MatrixXd>& left,
                 const Eigen::Ref<const Eigen::MatrixXd>& right,
                 Eigen::Ref<Eigen::MatrixXd> result) {
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(result.rows()) *
                                                       result.cols() * left.cols());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (int part = 0; part < workers; ++part) {
        const Eigen::Index first = result.rows() * part / workers,
                           last = result.rows() * (part + 1) / workers;
        if (last > first)
            result.middleRows(first, last - first).noalias() +=
                left.middleRows(first, last - first) * right;
    }
}

/** @brief Source/channel moments in one product, partitioned by output rows. */
void transpose_product(const Eigen::Ref<const Eigen::MatrixXd>& channels,
                       const Eigen::Ref<const Eigen::MatrixXd>& ket,
                       Eigen::Ref<Eigen::MatrixXd> result) {
    const bool by_rows = result.rows() >= result.cols();
    const int extent = by_rows ? result.rows() : result.cols();
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(channels.rows()) *
                                                       channels.cols() * ket.cols());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (int part = 0; part < workers; ++part) {
        const Eigen::Index first = extent * part / workers, last = extent * (part + 1) / workers;
        if (last == first)
            continue;
        if (by_rows)
            result.middleRows(first, last - first).noalias() =
                channels.middleCols(first, last - first).transpose() * ket;
        else
            result.middleCols(first, last - first).noalias() =
                channels.transpose() * ket.middleCols(first, last - first);
    }
}

} // namespace detail

using detail::add_channel_gram;
using detail::Clock;
using detail::mix_channels;
using detail::validate_source;

SpinFactorSource sum_factors(std::vector<SpinFactorSource> sources) {
    if (sources.empty())
        throw std::invalid_argument("empty factor sum");
    SpinFactorSource result{sources.front().n_alpha, sources.front().n_beta, 0, {}};
    for (const auto& source : sources) {
        if (source.n_alpha != result.n_alpha || source.n_beta != result.n_beta ||
            source.count < 0 || !source.load)
            throw std::invalid_argument("incompatible factor sum");
        result.count += source.count;
        result.workspace_rows =
            std::max<Eigen::Index>(result.workspace_rows, source.workspace_rows);
        result.load_work +=
            source.count *
            std::max(source.load_work, static_cast<long double>(source.n_alpha) + source.n_beta);
    }
    if (result.count)
        result.load_work /= result.count;
    result.load = [sources = std::move(sources)](Eigen::Index first,
                                                 Eigen::Index width,
                                                 Eigen::Ref<Eigen::MatrixXd> a,
                                                 Eigen::Ref<Eigen::MatrixXd> b) {
        std::int64_t offset = 0;
        for (const auto& source : sources) {
            if (first >= source.count) {
                first -= source.count;
                continue;
            }
            const Eigen::Index count = std::min<Eigen::Index>(width - offset, source.count - first);
            source.load(first, count, a.middleCols(offset, count), b.middleCols(offset, count));
            offset += count;
            first = 0;
            if (offset == width)
                break;
        }
    };
    return result;
}

SpinFactorSource generator_factors(const DirectCiSigmaAction& sigma,
                                   const SpinFactorSource& source,
                                   const Eigen::Ref<const Eigen::MatrixXd>& generator) {
    validate_source(sigma, source);
    return {source.n_alpha,
            source.n_beta,
            2 * source.count,
            [&sigma, source, generator](Eigen::Index first,
                                        Eigen::Index width,
                                        Eigen::Ref<Eigen::MatrixXd> a,
                                        Eigen::Ref<Eigen::MatrixXd> b) {
                std::int64_t offset = 0;
                while (offset < width) {
                    const bool beta = first >= source.count;
                    const int local = beta ? first - source.count : first;
                    const Eigen::Index count =
                        std::min<Eigen::Index>(width - offset, source.count - local);
                    auto aa = a.middleCols(offset, count), bb = b.middleCols(offset, count);
                    source.load(local, count, aa, bb);
                    Eigen::MatrixXd image(beta ? source.n_beta : source.n_alpha, count);
                    sigma.apply_spin_generator(beta, beta ? bb : aa, generator, image);
                    if (beta)
                        bb = image;
                    else
                        aa = image;
                    first += count;
                    offset += count;
                }
            },
            std::max<Eigen::Index>(
                source.workspace_rows,
                static_cast<std::size_t>(std::max<Eigen::Index>(source.n_alpha, source.n_beta))),
            std::max(source.load_work, static_cast<long double>(source.n_alpha) + source.n_beta) +
                sigma.spin_link_count() + 0.5L * (source.n_alpha + source.n_beta)};
}

FactorActionWorkspace::FactorActionWorkspace(const SpinFactorSource& source,
                                             int channels,
                                             std::size_t workspace_bytes,
                                             bool overlap_only,
                                             int reserved_channel_panels)
    : n_alpha_(source.n_alpha), n_beta_(source.n_beta), channels_(channels),
      load_rows_(source.workspace_rows), overlap_only_(overlap_only),
      reserved_channel_panels_(reserved_channel_panels) {
    if (n_alpha_ <= 0 || n_beta_ <= 0 || source.count < 0 || channels <= 0 ||
        reserved_channel_panels < 0)
        throw std::invalid_argument("invalid factor workspace dimensions");
    const std::size_t unit_bytes = (static_cast<std::size_t>(n_alpha_) + n_beta_) * sizeof(double);
    const std::size_t load_bytes = source.workspace_rows * sizeof(double);
    if (workspace_bytes <
        unit_bytes * ((overlap_only ? 2U : 9U) + reserved_channel_panels) + load_bytes)
        throw std::length_error("source workspace cannot hold one single-spin channel tile");
    const int columns_per_source =
        overlap_only ? 2 + reserved_channel_panels : (3 + reserved_channel_panels) * channels + 6;
    source_tile_ = static_cast<int>(std::max<std::size_t>(
        1,
        std::min<std::size_t>(source.count,
                              workspace_bytes / (unit_bytes * columns_per_source + load_bytes))));
    channel_tile_ = overlap_only
                        ? 0
                        : static_cast<int>(std::min<std::size_t>(
                              channels,
                              ((workspace_bytes / source_tile_ - load_bytes) / unit_bytes - 6) /
                                  (3 + reserved_channel_panels)));
    alpha_.resize(n_alpha_, source_tile_);
    beta_.resize(n_beta_, source_tile_);
    ha_.resize(n_alpha_, overlap_only ? 0 : source_tile_);
    hb_.resize(n_beta_, overlap_only ? 0 : source_tile_);
    ap_.resize(n_alpha_, source_tile_ * channel_tile_);
    bp_.resize(n_beta_, source_tile_ * channel_tile_);
    va_.resize(n_alpha_, source_tile_ * channel_tile_);
    vb_.resize(n_beta_, source_tile_ * channel_tile_);
    if (!overlap_only && channel_tile_ < channels) {
        aq_.resize(n_alpha_, source_tile_ * channel_tile_);
        bq_.resize(n_beta_, source_tile_ * channel_tile_);
    }
    temporary_.resize(sizeof(double) *
                      static_cast<std::size_t>(alpha_.size() + beta_.size() + ha_.size() +
                                               hb_.size() + ap_.size() + bp_.size() + va_.size() +
                                               vb_.size() + aq_.size() + bq_.size()));
}

FactorActionWork FactorActionWorkspace::apply(const DirectCiSigmaAction& sigma,
                                              const SpinFactorSource& source,
                                              const std::function<void(FactorCrossBlock)>& cross,
                                              const std::function<void(FactorImageBlock)>& finish) {
    validate_source(sigma, source);
    if (source.n_alpha != n_alpha_ || source.n_beta != n_beta_ ||
        source.workspace_rows > load_rows_ || sigma.channel_kernel().rows() != channels_ ||
        !finish || (!overlap_only_ && !cross))
        throw std::invalid_argument("factor action does not match its workspace or consumers");
    FactorActionWork work;
    work.source_tile = source_tile_;
    work.channel_tile = channel_tile_;
    const bool timing = profiling::phase_totals().enabled;
    const auto tick = [&] { return timing ? Clock::now() : Clock::time_point{}; };
    const auto seconds = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const auto& kernel = sigma.channel_kernel();
    for (Eigen::Index first = 0; first < source.count; first += source_tile_) {
        const Eigen::Index width = std::min<Eigen::Index>(source_tile_, source.count - first);
        ++work.panels;
        work.sources += width;
        work.workspace_bound = std::max<Eigen::Index>(
            work.workspace_bound,
            sizeof(double) *
                ((static_cast<std::size_t>(n_alpha_) + n_beta_) * width *
                     (overlap_only_ ? 2 + reserved_channel_panels_
                                    : (3 + reserved_channel_panels_) * channel_tile_ + 6) +
                 source.workspace_rows * width));
        auto a = alpha_.leftCols(width), b = beta_.leftCols(width);
        auto ha = ha_.leftCols(overlap_only_ ? 0 : width);
        auto hb = hb_.leftCols(overlap_only_ ? 0 : width);
        source.load(first, width, a, b);
        ha.setZero();
        hb.setZero();
        if (!overlap_only_) {
            for (int p = 0; p < channels_; p += channel_tile_) {
                const int np = std::min<Eigen::Index>(channel_tile_, channels_ - p);
                auto ap = ap_.leftCols(np * width), bp = bp_.leftCols(np * width);
                auto va = va_.leftCols(np * width), vb = vb_.leftCols(np * width);
                const auto scattered = tick();
                sigma.scatter_spin(false, a, p, np, ap);
                sigma.scatter_spin(true, b, p, np, bp);
                work.scatter_seconds += seconds(scattered, tick());
                va.setZero();
                vb.setZero();
                for (int q = 0; q < channels_; q += channel_tile_) {
                    ++work.channel_blocks;
                    const int nq = std::min<Eigen::Index>(channel_tile_, channels_ - q);
                    if (q == p) {
                        const auto start = tick();
                        mix_channels(ap, kernel, p, q, width, va);
                        mix_channels(bp, kernel, p, q, width, vb);
                        work.mix_seconds += seconds(start, tick());
                    } else {
                        const auto start = tick();
                        auto aq = aq_.leftCols(nq * width), bq = bq_.leftCols(nq * width);
                        sigma.scatter_spin(false, a, q, nq, aq);
                        sigma.scatter_spin(true, b, q, nq, bq);
                        const auto mixed = tick();
                        work.scatter_seconds += seconds(start, mixed);
                        mix_channels(aq, kernel, p, q, width, va);
                        mix_channels(bq, kernel, p, q, width, vb);
                        work.mix_seconds += seconds(mixed, tick());
                    }
                }
                const auto gathered = tick();
                sigma.gather_spin(false, va, p, np, ha);
                sigma.gather_spin(true, vb, p, np, hb);
                work.gather_seconds += seconds(gathered, tick());
                cross({first, width, p, np, ap, bp, va, vb});
            }
        }
        finish({first, width, a, b, ha, hb});
    }
    return work;
}

DirectCiIntegralAdjoint factor_integral_adjoint(const DirectCiSigmaAction& sigma,
                                                const SpinFactorSource& left,
                                                const SpinFactorSource& right,
                                                std::size_t workspace_bytes) {
    profiling::ScopedPhase phase(profiling::Phase::CiIntegralAdjoint);
    validate_source(sigma, left);
    validate_source(sigma, right);
    const int d = sigma.channel_kernel().rows();
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(d, d);
    if (!left.count || !right.count)
        return sigma.pullback_channel_kernel(gradient);
    // Self-adjointness permits either bilinear order. Include the independent
    // support-map multiplication cost, not only the number of source factors.
    const bool first = detail::integral_bra_is_left(sigma, left, right, workspace_bytes);
    const auto& bra = first ? left : right;
    const auto& ket = first ? right : left;
    const auto plan = detail::plan_factor_panels(
        sigma, bra, ket, workspace_bytes, detail::FactorContraction::Integral);
    const int nl = plan.left, nr = plan.right, p = plan.channels;
    const bool timing = profiling::phase_totals().enabled;
    const auto tick = [&] { return timing ? Clock::now() : Clock::time_point{}; };
    const auto seconds = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    double load_time = 0, scatter_time = 0, covector_time = 0, gram_time = 0;
    SpinAmplitudeWorkspace moments(bra, nl, nr * p);
    Eigen::MatrixXd ra(left.n_alpha, nr), rb(left.n_beta, nr);
    Eigen::MatrixXd wa(left.n_alpha, nr), wb(left.n_beta, nr);
    Eigen::MatrixXd rp_a(left.n_alpha, nr * p), rp_b(left.n_beta, nr * p);
    Eigen::MatrixXd rq_a(left.n_alpha, p < d ? nr * p : 0);
    Eigen::MatrixXd rq_b(left.n_beta, p < d ? nr * p : 0);
    Eigen::MatrixXd ca(left.n_alpha, nr * p), cb(left.n_beta, nr * p);
    workspace::Temporary temporary(
        sizeof(double) *
        static_cast<std::size_t>(ra.size() + rb.size() + wa.size() + wb.size() + rp_a.size() +
                                 rp_b.size() + rq_a.size() + rq_b.size() + ca.size() + cb.size()));
    for (int r = 0; r < ket.count; r += nr) {
        const int wr = std::min<Eigen::Index>(nr, ket.count - r);
        auto a = ra.leftCols(wr), b = rb.leftCols(wr);
        const auto load_start = tick();
        ket.load(r, wr, a, b);
        load_time += seconds(load_start, tick());
        auto weighted_a = wa.leftCols(wr), weighted_b = wb.leftCols(wr);
        weighted_a.setZero();
        weighted_b.setZero();
        // Move self-adjoint generators through the spectator contraction:
        // C_alpha,P = E_alpha,P(Y_left F_right) + Y_left(E_beta,P F_right).
        // The weighted bases are formed once, with no bra-channel scatter.
        const auto weighted_start = tick();
        moments.add(a, b, weighted_a, weighted_b);
        covector_time += seconds(weighted_start, tick());
        for (Eigen::Index first = 0; first < d; first += p) {
            const int np = std::min<Eigen::Index>(p, d - first);
            auto bar_a = ca.leftCols(wr * np), bar_b = cb.leftCols(wr * np);
            auto pa = rp_a.leftCols(wr * np), pb = rp_b.leftCols(wr * np);
            const auto scattered = tick();
            sigma.scatter_spin(false, weighted_a, first, np, bar_a);
            sigma.scatter_spin(true, weighted_b, first, np, bar_b);
            sigma.scatter_spin(false, a, first, np, pa);
            sigma.scatter_spin(true, b, first, np, pb);
            scatter_time += seconds(scattered, tick());
            // Accumulate the entire bra before the Q-channel Gram: splitting bra
            // sources must not multiply the leading d^2 contraction by its tile count.
            const auto covectors = tick();
            moments.add(pa, pb, bar_a, bar_b);
            covector_time += seconds(covectors, tick());
            for (int second = 0; second < d; second += p) {
                const int nq = std::min<Eigen::Index>(p, d - second);
                if (second == first) {
                    const auto start = tick();
                    add_channel_gram(
                        bar_a, bar_b, pa, pb, wr, gradient.block(first, second, np, nq));
                    gram_time += seconds(start, tick());
                } else {
                    auto qa = rq_a.leftCols(wr * nq), qb = rq_b.leftCols(wr * nq);
                    const auto scatter_start = tick();
                    sigma.scatter_spin(false, a, second, nq, qa);
                    sigma.scatter_spin(true, b, second, nq, qb);
                    const auto gram_start = tick();
                    scatter_time += seconds(scatter_start, gram_start);
                    add_channel_gram(
                        bar_a, bar_b, qa, qb, wr, gradient.block(first, second, np, nq));
                    gram_time += seconds(gram_start, tick());
                }
            }
        }
    }
    if (timing)
        std::fprintf(stderr,
                     "XMVB_FACTOR_ADJOINT,bra=%d,ket=%d,left_tile=%d,right_tile=%d,channel_tile=%d,"
                     "workspace_bytes=%zu,source_loads=%.0Lf,load=%.9f,scatter=%.9f,covectors=%.9f,"
                     "gram=%.9f\n",
                     bra.count,
                     ket.count,
                     nl,
                     nr,
                     p,
                     plan.workspace_bytes,
                     plan.source_columns,
                     load_time,
                     scatter_time,
                     covector_time,
                     gram_time);
    return sigma.pullback_channel_kernel(gradient);
}

} // namespace xmvb::mcscf
