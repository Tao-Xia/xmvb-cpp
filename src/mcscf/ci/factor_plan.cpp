#include "mcscf/ci/factors_internal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "parallel/openmp.hpp"

namespace xmvb::mcscf::detail {

bool use_amplitude_map(const SpinFactorSource& source) {
    return source.contract &&
           source.contract_work <
               4.0L * (static_cast<long double>(source.n_alpha) + source.n_beta) * source.count;
}

std::size_t amplitude_workspace(const SpinFactorSource& source, int tile, int columns) {
    if (use_amplitude_map(source))
        return source.contract_rows * columns;
    return (static_cast<std::size_t>(source.n_alpha) + source.n_beta + source.workspace_rows +
            columns) *
           tile;
}

FactorPanels plan_factor_panels(const DirectCiSigmaAction& sigma,
                                const SpinFactorSource& left,
                                const SpinFactorSource& right,
                                std::size_t workspace_bytes,
                                FactorContraction contraction) {
    const bool integral = contraction == FactorContraction::Integral;
    const bool residual = contraction == FactorContraction::Residual;
    const bool mapped = use_amplitude_map(left);
    const int d = sigma.channel_kernel().rows();
    const long double rows = static_cast<long double>(left.n_alpha) + left.n_beta;
    const long double capacity = workspace_bytes / sizeof(double);
    const long double reductions =
        integral ? 0
                 : static_cast<long double>(effective_openmp_thread_count() + 1) *
                       sigma.n_orbitals() * sigma.n_orbitals();
    const auto payload = [&](int l, int r, int p) {
        const long double loading =
            std::max(mapped ? 0.0L : static_cast<long double>(left.workspace_rows) * l,
                     static_cast<long double>(right.workspace_rows) * r);
        const long double amplitude =
            mapped ? static_cast<long double>(left.contract_rows) * r * std::max(1, p)
                   : rows * l + static_cast<long double>(l) * r * std::max(1, p);
        if (integral)
            return amplitude + rows * (2.0L * r + (p < d ? 3.0L : 2.0L) * r * p) + loading;
        const long double sweep = residual ? rows * r * (3.0L * p + 6) : 0;
        return amplitude + rows * (residual ? 3.0L : 2.0L) * r + loading + sweep + reductions;
    };
    FactorPanels best;
    long double best_calls = std::numeric_limits<long double>::infinity();
    const long double load_work = std::max(left.load_work, rows);
    const long double links = sigma.spin_link_count();
    // Leading channel GEMMs and source-pair moment arithmetic are invariant
    // across these schedules. Compare only work that actually repeats.
    for (int p = integral || residual ? 1 : 0; p <= (integral || residual ? d : 0); ++p) {
        if (payload(1, 1, p) > capacity)
            continue;
        int low = 1, high = residual && p < d ? 1 : right.count;
        // The shared forward sweep can stream partial channels at width one;
        // otherwise it independently chooses a larger channel width. Keep its
        // actual schedule identical to this parent's moment/slab bound.
        while (low < high) {
            const int middle = low + (high - low + 1) / 2;
            if (payload(1, middle, p) <= capacity)
                low = middle;
            else
                high = middle - 1;
        }
        const int maximum_right = low;
        for (int r = 1; r <= maximum_right; ++r) {
            low = 1;
            high = mapped ? 1 : left.count;
            while (low < high) {
                const int middle = low + (high - low + 1) / 2;
                if (payload(middle, r, p) <= capacity)
                    low = middle;
                else
                    high = middle - 1;
            }
            const int l = low;
            const long double nr = (right.count + r - 1) / r,
                              nl = mapped ? 1 : (left.count + l - 1) / l;
            const long double np = p ? (d + p - 1) / p : 0;
            const long double moment_passes = integral ? np + 1 : residual ? np + 2 : 1;
            const long double loads =
                mapped ? 0 : (left.count == l ? left.count : nr * moment_passes * left.count);
            // Sorted channel ranges visit only matching links. Binary range lookup
            // comparisons still repeat per scatter/gather, but rejected links do not.
            // Count zero fills and repeated output read+writes separately from FLOPs.
            const long double visits = integral ? np + 1 : residual ? 2 * np + 3 : 0;
            const long double searches = integral   ? np * np + np
                                         : residual ? 2 * np * np + 3 * np
                                                    : 0;
            const long double comparisons =
                np > 1 ? searches * sigma.channel_search_work() * right.count : 0;
            const long double valid_updates = integral ? 2 * (np + 1) : residual ? 4 * np + 6 : 0;
            const long double zeroes = integral   ? rows * right.count * d * (np + 1)
                                       : residual ? 2 * rows * right.count * d * np
                                                  : 0;
            const long double output_columns = integral ? d + 1 : residual ? d + 2 : 1;
            const long double writes = 2 * rows * right.count * output_columns * nl;
            const long double work = loads * load_work +
                                     right.count * (visits + valid_updates) * links + zeroes +
                                     writes + comparisons;
            const long double calls = nr * moment_passes * nl;
            if (best.left && (work > best.work || (work == best.work && calls >= best_calls)))
                continue;
            best = {
                l, r, p, static_cast<std::size_t>(sizeof(double) * payload(l, r, p)), work, loads};
            best_calls = calls;
        }
    }
    if (!best.left)
        throw std::length_error("factor workspace cannot hold one source/channel panel");
    return best;
}

namespace {
long double ordered_integral_work(const DirectCiSigmaAction& sigma,
                                  const SpinFactorSource& bra,
                                  const SpinFactorSource& ket,
                                  std::size_t workspace_bytes) {
    const auto plan =
        plan_factor_panels(sigma, bra, ket, workspace_bytes, FactorContraction::Integral);
    const long double rows = static_cast<long double>(bra.n_alpha) + bra.n_beta;
    const long double d = sigma.channel_kernel().rows();
    // These leading terms do not depend on tile widths, and were deliberately
    // omitted from plan_factor_panels(). They do depend on the bilinear order.
    const long double moment = use_amplitude_map(bra) ? bra.contract_work : 4 * rows * bra.count;
    return plan.work + 2 * rows * ket.count * d * d + moment * ket.count * (d + 1);
}
} // namespace

bool integral_bra_is_left(const DirectCiSigmaAction& sigma,
                          const SpinFactorSource& left,
                          const SpinFactorSource& right,
                          std::size_t bytes) {
    if (!use_amplitude_map(left) && !use_amplitude_map(right))
        return left.count >= right.count;
    return ordered_integral_work(sigma, left, right, bytes) <=
           ordered_integral_work(sigma, right, left, bytes);
}

long double factor_integral_work(const DirectCiSigmaAction& sigma,
                                 const SpinFactorSource& left,
                                 const SpinFactorSource& right,
                                 std::size_t bytes) {
    if (!left.count || !right.count)
        return 0;
    const bool first = integral_bra_is_left(sigma, left, right, bytes);
    return ordered_integral_work(sigma, first ? left : right, first ? right : left, bytes);
}

} // namespace xmvb::mcscf::detail
