#include "mcscf/ci/factors_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"

namespace xmvb::mcscf {
SpinAmplitudeWorkspace::SpinAmplitudeWorkspace(const SpinFactorSource& source,
                                               int width,
                                               int columns)
    : source_(source), width_(width), mapped_(detail::use_amplitude_map(source)),
      a_(source.n_alpha, mapped_ ? 0 : width), b_(source.n_beta, mapped_ ? 0 : width),
      moment_(mapped_ ? 0 : width, mapped_ ? 0 : columns) {
    if (width <= 0 || columns <= 0)
        throw std::invalid_argument("invalid amplitude panel dimensions");
    temporary_.resize(sizeof(double) *
                      static_cast<std::size_t>(a_.size() + b_.size() + moment_.size()));
    if (mapped_)
        return;
    if (width_ == source_.count) {
        source_.load(0, width_, a_, b_);
        loads_ = 1;
        columns_ = width_;
    }
}

void SpinAmplitudeWorkspace::add(const Eigen::Ref<const Eigen::MatrixXd>& alpha,
                                 const Eigen::Ref<const Eigen::MatrixXd>& beta,
                                 Eigen::Ref<Eigen::MatrixXd> result_alpha,
                                 Eigen::Ref<Eigen::MatrixXd> result_beta,
                                 double scale) {
    if (mapped_) {
        source_.contract(false, beta, result_alpha, scale);
        source_.contract(true, alpha, result_beta, scale);
        return;
    }
    for (int first = 0; first < source_.count; first += width_) {
        const int count = std::min(width_, source_.count - first);
        auto a = a_.leftCols(count), b = b_.leftCols(count);
        if (width_ != source_.count) {
            source_.load(first, count, a, b);
            ++loads_;
            columns_ += count;
        }
        auto moment = moment_.topLeftCorner(count, alpha.cols());
        detail::transpose_product(b, beta, moment);
        if (scale != 1.0)
            moment *= scale;
        detail::add_product(a, moment, result_alpha);
        detail::transpose_product(a, alpha, moment);
        if (scale != 1.0)
            moment *= scale;
        detail::add_product(b, moment, result_beta);
    }
}

namespace {

using Clock = std::chrono::steady_clock;

Eigen::MatrixXd generator_adjoint(const DirectCiSigmaAction& sigma,
                                  const SpinFactorSource& left,
                                  const SpinFactorSource& right,
                                  bool hamiltonian,
                                  double energy,
                                  std::size_t workspace_bytes) {
    profiling::ScopedPhase phase(profiling::Phase::CiGeneratorAdjoint);
    detail::validate_source(sigma, left);
    detail::validate_source(sigma, right);
    if (!std::isfinite(energy))
        throw std::invalid_argument("nonfinite residual energy");
    const int n = sigma.n_orbitals(), d = sigma.channel_kernel().rows();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
    if (!left.count || !right.count)
        return result;
    // Only an identity bilinear permits exchanging sources followed by a
    // transpose. H on the left cannot be moved through an oriented generator.
    const auto moment_work = [&](const SpinFactorSource& a, const SpinFactorSource& b) {
        const long double rows = static_cast<long double>(a.n_alpha) + a.n_beta;
        return (detail::use_amplitude_map(a) ? a.contract_work : 4 * rows * a.count) * b.count +
               static_cast<long double>(sigma.spin_link_count()) * b.count;
    };
    const bool transpose = !hamiltonian && moment_work(right, left) < moment_work(left, right);
    const auto& bra = transpose ? right : left;
    const auto& ket = transpose ? left : right;
    const std::size_t rows = static_cast<std::size_t>(left.n_alpha) + left.n_beta;
    const auto plan = detail::plan_factor_panels(
        sigma,
        bra,
        ket,
        workspace_bytes,
        hamiltonian ? detail::FactorContraction::Residual : detail::FactorContraction::Generator);
    const int nl = plan.left, nr = plan.right, p = plan.channels;
    const bool timing = profiling::phase_totals().enabled;
    const auto start = timing ? Clock::now() : Clock::time_point{};
    SpinAmplitudeWorkspace moments(bra, nl, nr * std::max(1, p));
    Eigen::MatrixXd a(left.n_alpha, nr), b(left.n_beta, nr);
    Eigen::MatrixXd wa(left.n_alpha, nr), wb(left.n_beta, nr);
    Eigen::MatrixXd ca(left.n_alpha, hamiltonian ? nr : 0), cb(left.n_beta, hamiltonian ? nr : 0);
    workspace::Temporary temporary(sizeof(double) *
                                   static_cast<std::size_t>(a.size() + b.size() + wa.size() +
                                                            wb.size() + ca.size() + cb.size()));
    const SpinFactorSource shape{left.n_alpha, left.n_beta, nr, {}};
    std::unique_ptr<FactorActionWorkspace> sweep;
    if (hamiltonian)
        sweep = std::make_unique<FactorActionWorkspace>(
            shape, d, rows * nr * (3 * p + 6) * sizeof(double));
    for (int first = 0; first < ket.count; first += nr) {
        const int width = std::min(nr, ket.count - first);
        auto aa = a.leftCols(width), bb = b.leftCols(width);
        ket.load(first, width, aa, bb);
        auto weighted_a = wa.leftCols(width), weighted_b = wb.leftCols(width);
        weighted_a.setZero();
        weighted_b.setZero();
        moments.add(aa, bb, weighted_a, weighted_b);
        if (!hamiltonian) {
            result += sigma.spin_generator_adjoint(false, weighted_a, aa);
            result += sigma.spin_generator_adjoint(true, weighted_b, bb);
            continue;
        }
        auto out_a = ca.leftCols(width), out_b = cb.leftCols(width);
        out_a = -energy * weighted_a;
        out_b = -energy * weighted_b;
        const SpinFactorSource weighted{left.n_alpha,
                                        left.n_beta,
                                        width,
                                        [&](int offset,
                                            int count,
                                            Eigen::Ref<Eigen::MatrixXd> x,
                                            Eigen::Ref<Eigen::MatrixXd> y) {
                                            x = weighted_a.middleCols(offset, count);
                                            y = weighted_b.middleCols(offset, count);
                                        }};
        sweep->apply(
            sigma,
            weighted,
            [](FactorCrossBlock) {},
            [&](FactorImageBlock block) {
                out_a.middleCols(block.source, block.width) += block.same_alpha;
                out_b.middleCols(block.source, block.width) += block.same_beta;
            });
        const SpinFactorSource factors{left.n_alpha,
                                       left.n_beta,
                                       width,
                                       [&](int offset,
                                           int count,
                                           Eigen::Ref<Eigen::MatrixXd> x,
                                           Eigen::Ref<Eigen::MatrixXd> y) {
                                           x = aa.middleCols(offset, count);
                                           y = bb.middleCols(offset, count);
                                       }};
        sweep->apply(
            sigma,
            factors,
            [&](FactorCrossBlock block) {
                // Same-spin gather is complete: reuse raw channel slabs for X V_sigma.
                block.alpha_channels.setZero();
                block.beta_channels.setZero();
                moments.add(block.mixed_alpha,
                            block.mixed_beta,
                            block.alpha_channels,
                            block.beta_channels,
                            2.0);
                sigma.gather_spin(false,
                                  block.alpha_channels,
                                  block.channel,
                                  block.channels,
                                  out_a.middleCols(block.source, block.width));
                sigma.gather_spin(true,
                                  block.beta_channels,
                                  block.channel,
                                  block.channels,
                                  out_b.middleCols(block.source, block.width));
            },
            [&](FactorImageBlock block) {
                moments.add(block.same_alpha,
                            block.same_beta,
                            out_a.middleCols(block.source, block.width),
                            out_b.middleCols(block.source, block.width));
            });
        result += sigma.spin_generator_adjoint(false, out_a, aa);
        result += sigma.spin_generator_adjoint(true, out_b, bb);
    }
    if (timing)
        std::fprintf(stderr,
                     "XMVB_FACTOR_GENERATOR,hamiltonian=%d,bra=%d,ket=%d,left_tile=%d,right_tile=%"
                     "d,channels=%d,workspace_bytes=%zu,planned_source_columns=%.0Lf,source_"
                     "columns=%zu,source_calls=%zu,seconds=%.9f\n",
                     hamiltonian,
                     bra.count,
                     ket.count,
                     nl,
                     nr,
                     p,
                     plan.workspace_bytes,
                     plan.source_columns,
                     moments.columns(),
                     moments.loads(),
                     std::chrono::duration<double>(Clock::now() - start).count());
    if (transpose)
        return result.transpose();
    return result;
}

} // namespace

Eigen::MatrixXd factor_generator_adjoint(const DirectCiSigmaAction& sigma,
                                         const SpinFactorSource& left,
                                         const SpinFactorSource& right,
                                         std::size_t workspace_bytes) {
    return generator_adjoint(sigma, left, right, false, 0.0, workspace_bytes);
}

Eigen::MatrixXd factor_residual_generator_adjoint(const DirectCiSigmaAction& sigma,
                                                  const SpinFactorSource& left,
                                                  const SpinFactorSource& right,
                                                  double energy,
                                                  std::size_t workspace_bytes) {
    return generator_adjoint(sigma, left, right, true, energy, workspace_bytes);
}

} // namespace xmvb::mcscf
