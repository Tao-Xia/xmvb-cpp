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

/**
 * @brief Bounds a requested worker count without narrowing the work size.
 *
 * The returned value is always representable as `int` because it cannot
 * exceed `requested`.  Comparing in `size_t` first is essential for kernels
 * whose work count can exceed `INT_MAX`, such as exact AO integral streams.
 */
inline int bounded_openmp_thread_count(
    int requested,
    std::size_t work_items) {
  if (requested <= 0) {
    throw std::invalid_argument("OpenMP thread count must be positive");
  }
  if (work_items == 0) {
    return 1;
  }
  return static_cast<int>(std::min(
      static_cast<std::size_t>(requested),
      work_items));
}

}  // namespace xmvb
