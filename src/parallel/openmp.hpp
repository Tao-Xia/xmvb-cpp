#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb {

/**
 * @brief Returns the effective worker count for thread-local buffer sizing.
 *
 * Many kernels allocate `partial_*` workspaces sized by the number of OpenMP
 * workers they plan to launch. When such a kernel is called from inside an
 * already-active parallel region, nested OpenMP usually executes the inner
 * region serially even though `omp_get_max_threads()` still reports the global
 * thread count. Returning `1` in that case prevents quadratic buffer growth in
 * projected-TNHVP style nested parallel execution.
 */
inline int effective_openmp_thread_count() {
#ifdef _OPENMP
    if (omp_in_parallel()) {
        return 1;
    }
    const int n_threads = omp_get_max_threads();
    return n_threads > 0 ? n_threads : 1;
#else
    return 1;
#endif
}

/** @brief Available workers for shared-buffer tasks, including an existing team. */
inline int cooperative_openmp_thread_count() {
#ifdef _OPENMP
    if (omp_in_parallel())
        return omp_get_num_threads();
#endif
    return effective_openmp_thread_count();
}

/** @brief Independent work sharing caller-owned buffers, without a nested team.
 * The callback must not throw and all tasks finish before returning. Its
 * lifetime is borrowed: no coefficient matrix is firstprivate-copied.
 */
template <class Function>
void cooperative_openmp_for(std::ptrdiff_t count, int workers, Function&& action) {
    if (workers <= 1) {
        for (std::ptrdiff_t i = 0; i < count; ++i)
            action(i);
        return;
    }
#ifdef _OPENMP
    if (omp_in_parallel()) {
#pragma omp taskloop num_tasks(workers) shared(action)
        for (std::ptrdiff_t i = 0; i < count; ++i)
            action(i);
        return;
    }
#endif
#pragma omp parallel for schedule(static) num_threads(workers)
    for (std::ptrdiff_t i = 0; i < count; ++i)
        action(i);
}

/**
 * @brief Bounds a requested worker count without narrowing the work size.
 *
 * The returned value is always representable as `int` because it cannot
 * exceed `requested`.  Comparing in `size_t` first is essential for kernels
 * whose work count can exceed `INT_MAX`, such as exact AO integral streams.
 */
inline int bounded_openmp_thread_count(int requested, std::size_t work_items) {
    if (requested <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
    if (work_items == 0) {
        return 1;
    }
    return static_cast<int>(std::min(static_cast<std::size_t>(requested), work_items));
}

/**
 * @brief Keeps short loops serial unless each worker has a team-width of work.
 *
 * Unlike heavy AO rows, CI links and carrier entries are fine-grained work.
 * Requiring workers squared items avoids a team for only a few items per
 * worker. Use either the existing full team or serial execution: repeatedly
 * changing team sizes between CI and AO kernels incurs worker-pool churn.
 * This is a granularity policy, not a claim of optimal throughput; no work
 * or precision is discarded.
 */
inline int coarse_openmp_thread_count(int requested, std::size_t work_items) {
    const auto workers =
        static_cast<std::size_t>(bounded_openmp_thread_count(requested, work_items));
    return workers <= work_items / workers ? static_cast<int>(workers) : 1;
}

/**
 * @brief Number of shared work pieces for dynamic loop balancing.
 *
 * Up to workers squared pieces give each worker workers average pieces, so
 * the nominal trailing-piece size is one worker share divided by workers.
 * This is a resource-derived granularity, not a molecule-specific threshold.
 * Actual benefit depends on loop cost and is verified by native benchmarks.
 */
inline std::size_t openmp_work_pieces(std::size_t work_items, int workers) {
    if (workers <= 0)
        throw std::invalid_argument("OpenMP worker count must be positive");
    if (work_items == 0)
        return 1;
    const std::size_t p = workers;
    return p > work_items / p ? work_items : std::min(work_items, p * p);
}

/** @brief Positive, overflow-safe chunk size for a dynamically balanced loop. */
inline std::size_t openmp_chunk_size(std::size_t work_items, int workers) {
    const auto pieces = openmp_work_pieces(work_items, workers);
    return std::max<std::size_t>(1, work_items / pieces + (work_items % pieces != 0));
}

} // namespace xmvb
