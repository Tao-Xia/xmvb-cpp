#include "parallel/openmp.hpp"

#include <array>
#include <stdexcept>

int main() {
    if (xmvb::bounded_openmp_thread_count(4, 2) != 2 ||
        xmvb::bounded_openmp_thread_count(4, 0) != 1) {
        throw std::runtime_error("OpenMP work bound changed");
    }
    std::array<int, 64> values{};
    int nested_workers = 0;
#pragma omp parallel num_threads(2) shared(values, nested_workers)
    {
#pragma omp single
        {
            nested_workers = xmvb::effective_openmp_thread_count();
            xmvb::cooperative_openmp_for(
                values.size(), xmvb::cooperative_openmp_thread_count(), [&](std::ptrdiff_t i) {
                    values[i] = static_cast<int>(i) + 1;
                });
        }
    }
    if (nested_workers != 1) {
        throw std::runtime_error("nested work requested replicated worker buffers");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] != static_cast<int>(i) + 1) {
            throw std::runtime_error("cooperative tasks did not finish their shared work");
        }
    }
}
