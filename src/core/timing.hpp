#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace xmvb::profiling {

enum class Phase {
    Chart,
    Projection,
    Curvature,
    BlockFactor,
    BlockEigen,
    TrialForward,
    TrialGradient,
    ArmijoAccepted,
    ArmijoRejected,
    NeoSolve,
    HvpApply,
    HvpSetup,
    AoResponse,
    ActiveDirection,
    SelectedDirection,
    ActiveBackward,
    OrbitalPullback,
    TwoElectronResponse,
    FixedPullback,
    StructureSolve,
    StructureAdjoint,
    NeoProjectedSolve,
    NeoPreconditioner,
    NeoBasis,
    NeoMetric,
    MetricAction,
    MetricSolve,
    MetricBlock,
    AoPairAction,
    AoH1eFused,
    AoH1eTranspose,
    AoH1eForward,
    CiConnections,
    CiSigma,
    CiGenerator,
    CiIntegralDirection,
    CiTransformLeft,
    CiTransformRight,
    CiDirectionalTransform,
    CiIntegralAdjoint,
    CiGeneratorAdjoint,
    CiResponse,
    NeoProjectionFactor,
    NeoProjectionResidual,
    TrustSecular,
    NeoMetricFactor,
    NeoStructureFactor,
    NeoSpectrum,
    ChartConstraintSvd,
    ChartGaugeSvd,
    ExchangeBuild,
    ExchangeAction,
    NeoBoundarySpectrum,
    NeoIndefiniteSpectrum,
    NeoCurvatureSpectrum,
    CiahMacro,
    CiahNext,
    CiahImages,
    CiahKeyframe,
    CiahRefresh,
    CiahTrial,
    CiahCompletion,
    CiahSetup,
    CiahStructure,
    CiahConnection,
    CiahObserve,
    ChartSpan,
    ChartQuotient,
    ChartWhitening,
    ChartGram,
    ChartCurvature,
    CiProductProjection,
    CiTransformBuild,
    CiOverlapApply,
    CiOverlapDirection,
    CiOverlapAdjoint,
    CiOverlapAdjointDirection,
    CiTransportRetract,
    CiTransportGradient,
    CiOverlapTransportGradient,
    CiProductPattern,
    CiProductGenerate,
    CiProductPack,
    CiSupportedValidation,
    CiExteriorGeneratorPullback,
    CiExteriorMomentPullback,
    StructureNormalization,
    CiTilePaths,
    CiTileLayout,
    CiTileStorage,
    CiTileIndices,
    CiTileLinks,
    CiCacheSchedule,
    CiCacheGenerate,
    CiCacheCopy,
    CiCacheInsert,
    RiOneIndex,
    RiAoH1e,
    RiOrbitalPairBuild,
    OrbitalPairPullback,
    RiInactiveBuild,
    RiAoDirection,
    RiAoAccepted,
    RiAoExchange,
    RiAoCoulomb,
    RiDensityDiagonal,
    ActiveFrame,
    ActiveFrameDirection,
    ActiveFrameAdjoint,
    ActiveFrameAdjointDirection,
    ActivePreparation,
    SpinTopology,
    CiOperatorSetup,
    CiPreconditioner,
    Count
};

/** @brief Opt-in wall/process-CPU totals. Nested scopes are not additive.
 * Process CPU includes all worker threads, including spin/wait time. Scopes
 * belong at calling-thread phase boundaries, not inside parallel worker loops.
 */
class PhaseTotals {
  public:
    PhaseTotals() {
        const char* value = std::getenv("XMVB_PHASE_TIMING");
        enabled = value != nullptr && value[0] == '1';
        for (auto& value : matrix_min_rows)
            value.store(std::numeric_limits<unsigned long long>::max());
        for (auto& value : matrix_min_cols)
            value.store(std::numeric_limits<unsigned long long>::max());
    }

    /** @brief Stable stage labels shared by totals and bounded live events. */
    static const char* name(Phase phase) {
        constexpr const char* names[] = {"chart",
                                         "projection",
                                         "curvature",
                                         "block_factor",
                                         "block_eigen",
                                         "trial_forward",
                                         "trial_gradient",
                                         "armijo_accepted",
                                         "armijo_rejected",
                                         "neo_solve",
                                         "hvp_apply",
                                         "hvp_setup",
                                         "ao_response",
                                         "active_direction",
                                         "selected_direction",
                                         "active_backward",
                                         "orbital_pullback",
                                         "two_electron_response",
                                         "fixed_pullback",
                                         "structure_solve",
                                         "structure_adjoint",
                                         "neo_projected_solve",
                                         "neo_preconditioner",
                                         "neo_basis",
                                         "neo_metric",
                                         "metric_action",
                                         "metric_solve",
                                         "metric_block",
                                         "ao_pair_action",
                                         "ao_h1e_fused",
                                         "ao_h1e_transpose",
                                         "ao_h1e_forward",
                                         "ci_connections",
                                         "ci_sigma",
                                         "ci_generator",
                                         "ci_integral_direction",
                                         "ci_transform_left",
                                         "ci_transform_right",
                                         "ci_directional_transform",
                                         "ci_integral_adjoint",
                                         "ci_generator_adjoint",
                                         "ci_response",
                                         "neo_projection_factor",
                                         "neo_projection_residual",
                                         "trust_secular",
                                         "neo_metric_factor",
                                         "neo_structure_factor",
                                         "neo_spectrum",
                                         "chart_constraint_svd",
                                         "chart_gauge_svd",
                                         "exchange_build",
                                         "exchange_action",
                                         "neo_boundary_spectrum",
                                         "neo_indefinite_spectrum",
                                         "neo_curvature_spectrum",
                                         "ciah_macro",
                                         "ciah_next",
                                         "ciah_images",
                                         "ciah_keyframe",
                                         "ciah_refresh",
                                         "ciah_trial",
                                         "ciah_completion",
                                         "ciah_setup",
                                         "ciah_structure",
                                         "ciah_connection",
                                         "ciah_observe",
                                         "chart_span",
                                         "chart_quotient",
                                         "chart_whitening",
                                         "chart_gram",
                                         "chart_curvature",
                                         "ci_product_projection",
                                         "ci_transform_build",
                                         "ci_overlap_apply",
                                         "ci_overlap_direction",
                                         "ci_overlap_adjoint",
                                         "ci_overlap_adjoint_direction",
                                         "ci_transport_retract",
                                         "ci_transport_gradient",
                                         "ci_overlap_transport_gradient",
                                         "ci_product_pattern",
                                         "ci_product_generate",
                                         "ci_product_pack",
                                         "ci_supported_validation",
                                         "ci_exterior_generator_pullback",
                                         "ci_exterior_moment_pullback",
                                         "structure_normalization",
                                         "ci_tile_paths",
                                         "ci_tile_layout",
                                         "ci_tile_storage",
                                         "ci_tile_indices",
                                         "ci_tile_links",
                                         "ci_cache_schedule",
                                         "ci_cache_generate",
                                         "ci_cache_copy",
                                         "ci_cache_insert",
                                         "ri_one_index",
                                         "ri_ao_h1e",
                                         "ri_orbital_pair_build",
                                         "orbital_pair_pullback",
                                         "ri_inactive_build",
                                         "ri_ao_direction",
                                         "ri_ao_accepted",
                                         "ri_ao_exchange",
                                         "ri_ao_coulomb",
                                         "ri_density_diagonal",
                                         "active_frame",
                                         "active_frame_direction",
                                         "active_frame_adjoint",
                                         "active_frame_adjoint_direction",
                                         "active_preparation",
                                         "spin_topology",
                                         "ci_operator_setup",
                                         "ci_preconditioner"};
        static_assert(sizeof(names) / sizeof(names[0]) == static_cast<std::size_t>(Phase::Count));
        return names[static_cast<std::size_t>(phase)];
    }

    ~PhaseTotals() {
        if (!enabled)
            return;
        std::fprintf(stderr, "XMVB_PHASE,stage,calls,wall_seconds\n");
        for (std::size_t i = 0; i < calls.size(); ++i)
            std::fprintf(stderr,
                         "XMVB_PHASE,%s,%llu,%.9f\n",
                         name(static_cast<Phase>(i)),
                         calls[i].load(std::memory_order_relaxed),
                         nanoseconds[i].load(std::memory_order_relaxed) * 1e-9);
        std::fprintf(stderr, "XMVB_CPU,stage,cpu_seconds\n");
        for (std::size_t i = 0; i < calls.size(); ++i)
            std::fprintf(stderr,
                         "XMVB_CPU,%s,%.9f\n",
                         name(static_cast<Phase>(i)),
                         cpu_ticks[i].load(std::memory_order_relaxed) /
                             static_cast<double>(CLOCKS_PER_SEC));
        std::fprintf(stderr,
                     "XMVB_WORKSPACE,projector_peak_payload_bytes,%llu\n",
                     projector_bytes.load(std::memory_order_relaxed));
        std::fprintf(stderr, "XMVB_MATRIX,stage,calls,min_rows,max_rows,min_cols,max_cols\n");
        for (std::size_t i = 0; i < calls.size(); ++i) {
            if (matrix_calls[i].load(std::memory_order_relaxed) == 0)
                continue;
            std::fprintf(stderr,
                         "XMVB_MATRIX,%s,%llu,%llu,%llu,%llu,%llu\n",
                         name(static_cast<Phase>(i)),
                         matrix_calls[i].load(std::memory_order_relaxed),
                         matrix_min_rows[i].load(std::memory_order_relaxed),
                         matrix_max_rows[i].load(std::memory_order_relaxed),
                         matrix_min_cols[i].load(std::memory_order_relaxed),
                         matrix_max_cols[i].load(std::memory_order_relaxed));
        }
        std::fprintf(stderr,
                     "XMVB_BALANCE,stage,calls,min_loop_seconds,max_loop_seconds,sum_loop_seconds,"
                     "capacity_seconds,start_skew_seconds,critical_seconds\n");
        for (std::size_t i = 0; i < calls.size(); ++i) {
            if (worker_calls[i].load(std::memory_order_relaxed) == 0)
                continue;
            std::fprintf(stderr,
                         "XMVB_BALANCE,%s,%llu,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                         name(static_cast<Phase>(i)),
                         worker_calls[i].load(std::memory_order_relaxed),
                         worker_min[i].load(std::memory_order_relaxed) * 1e-9,
                         worker_max[i].load(std::memory_order_relaxed) * 1e-9,
                         worker_sum[i].load(std::memory_order_relaxed) * 1e-9,
                         worker_capacity[i].load(std::memory_order_relaxed) * 1e-9,
                         worker_skew[i].load(std::memory_order_relaxed) * 1e-9,
                         worker_critical[i].load(std::memory_order_relaxed) * 1e-9);
        }
    }

    bool enabled = false;
    std::atomic<unsigned long long> projector_bytes{0};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)> calls{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        nanoseconds{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)> cpu_ticks{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        matrix_calls{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        matrix_min_rows{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        matrix_max_rows{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        matrix_min_cols{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        matrix_max_cols{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_calls{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_min{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_max{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_sum{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_capacity{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_skew{};
    std::array<std::atomic<unsigned long long>, static_cast<std::size_t>(Phase::Count)>
        worker_critical{};
};

inline PhaseTotals& phase_totals() {
    static PhaseTotals totals;
    return totals;
}

/** @brief Records an event count without changing the optimization path. */
inline void count_phase(Phase phase) {
    auto& totals = phase_totals();
    if (totals.enabled)
        totals.calls[static_cast<std::size_t>(phase)].fetch_add(1, std::memory_order_relaxed);
}

/** @brief Opt-in matrix-shape ranges; stores scalars only, never matrix contents. */
inline void record_matrix_shape(Phase phase, unsigned long long rows, unsigned long long cols) {
    auto& totals = phase_totals();
    if (!totals.enabled)
        return;
    const auto i = static_cast<std::size_t>(phase);
    const auto update =
        [](std::atomic<unsigned long long>& slot, unsigned long long value, bool minimum) {
            auto previous = slot.load(std::memory_order_relaxed);
            while ((minimum ? value < previous : value > previous) &&
                   !slot.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {
            }
        };
    totals.matrix_calls[i].fetch_add(1, std::memory_order_relaxed);
    update(totals.matrix_min_rows[i], rows, true);
    update(totals.matrix_max_rows[i], rows, false);
    update(totals.matrix_min_cols[i], cols, true);
    update(totals.matrix_max_cols[i], cols, false);
}

/** @brief Largest projector-owned matrix payload, excluding referenced inputs. */
inline void record_projector_bytes(unsigned long long bytes) {
    auto& totals = phase_totals();
    if (!totals.enabled)
        return;
    auto peak = totals.projector_bytes.load(std::memory_order_relaxed);
    while (peak < bytes &&
           !totals.projector_bytes.compare_exchange_weak(peak, bytes, std::memory_order_relaxed)) {
    }
}

/** @brief Times a scope only when XMVB_PHASE_TIMING=1. */
class ScopedPhase {
  public:
    explicit ScopedPhase(Phase phase, bool live_event = false)
        : phase_(phase), enabled_(phase_totals().enabled), live_event_(live_event) {
        if (enabled_) {
            if (live_event_)
                std::fprintf(stderr, "XMVB_PHASE_EVENT,%s,begin\n", PhaseTotals::name(phase_));
            start_ = std::chrono::steady_clock::now();
            cpu_start_ = std::clock();
        }
    }
    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;
    ~ScopedPhase() {
        stop();
    }

    void stop() {
        if (!enabled_)
            return;
        const auto cpu_end = std::clock();
        const auto duration = std::chrono::steady_clock::now() - start_;
        const auto i = static_cast<std::size_t>(phase_);
        auto& totals = phase_totals();
        totals.calls[i].fetch_add(1, std::memory_order_relaxed);
        totals.nanoseconds[i].fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count(),
            std::memory_order_relaxed);
        if (live_event_)
            std::fprintf(stderr,
                         "XMVB_PHASE_EVENT,%s,end,%.9f\n",
                         PhaseTotals::name(phase_),
                         std::chrono::duration<double>(duration).count());
        if (cpu_start_ != static_cast<std::clock_t>(-1) && cpu_end >= cpu_start_)
            totals.cpu_ticks[i].fetch_add(cpu_end - cpu_start_, std::memory_order_relaxed);
        enabled_ = false;
    }

  private:
    Phase phase_;
    bool enabled_;
    bool live_event_;
    std::chrono::steady_clock::time_point start_;
    std::clock_t cpu_start_{};
};

} // namespace xmvb::profiling
