#include "parallel/runtime.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <cblas.h>
#include <omp.h>

namespace xmvb::parallel {
namespace {
const char* setting(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

int allocated_threads(int available) {
    const char* value = setting("SLURM_CPUS_PER_TASK");
    if (!value)
        return available;
    const std::string_view text(value);
    int count = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    return error == std::errc{} && end == text.data() + text.size() && count > 0
               ? std::min(available, count)
               : available;
}
} // namespace

ThreadConfiguration runtime_threads() {
    return {omp_get_max_threads(), openblas_get_num_threads()};
}

ThreadConfiguration configure_runtime_threads() {
    const int workers = setting("OMP_NUM_THREADS")
                            ? omp_get_max_threads()
                            : allocated_threads(std::max(1, omp_get_num_procs()));
    if (!setting("OPENBLAS_NUM_THREADS") && !setting("GOTO_NUM_THREADS"))
        openblas_set_num_threads(1);
    // Preserve the application team even if the BLAS backend shares OpenMP.
    omp_set_num_threads(workers);
    if (!setting("OMP_MAX_ACTIVE_LEVELS"))
        omp_set_max_active_levels(1);
    if (!setting("OMP_DYNAMIC"))
        omp_set_dynamic(0);
    return runtime_threads();
}

} // namespace xmvb::parallel
