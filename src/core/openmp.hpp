#pragma once

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

}  // namespace xmvb
