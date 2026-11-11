#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <omp.h>

#include "parallel/runtime.hpp"

int main(int argc, char** argv) {
    if (argc != 2)
        throw std::invalid_argument("runtime test needs slurm, explicit, or affinity");
    const int available = omp_get_num_procs();
    const std::string_view mode(argv[1]);
    int expected = mode == "explicit" ? 3 : mode == "slurm" ? std::min(available, 2) : available;
    if (mode == "slurm-allocation") {
        const char* allocation = std::getenv("SLURM_CPUS_PER_TASK");
        if (!allocation)
            throw std::runtime_error("launcher lost the Slurm CPU allocation");
        expected = std::min(available, std::stoi(allocation));
    }
    const int blas = mode == "explicit" ? 2 : 1;
    for (int pass = 0; pass < 2; ++pass) {
        const auto result = xmvb::parallel::configure_runtime_threads();
        if (result.openmp_threads != expected || result.blas_threads != blas)
            throw std::runtime_error("runtime did not honor allocation/user thread settings");
        int team = 0;
#pragma omp parallel
        {
#pragma omp single
            team = omp_get_num_threads();
        }
        if (team != expected || omp_get_max_active_levels() != 1)
            throw std::runtime_error("actual OpenMP team does not match the configured policy");
    }
    std::cout << "OpenMP=" << expected << " BLAS=" << blas << '\n';
}
