#pragma once

namespace xmvb::parallel {

struct ThreadConfiguration {
    int openmp_threads = 1;
    int blas_threads = 1;
};

/** Process-entry policy. Explicit user settings win; otherwise use the Slurm
 * task allocation within the available OpenMP CPUs, and a single BLAS worker.
 * Numerical libraries and solvers do not alter the process threading policy.
 */
ThreadConfiguration configure_runtime_threads();
ThreadConfiguration runtime_threads();

} // namespace xmvb::parallel
