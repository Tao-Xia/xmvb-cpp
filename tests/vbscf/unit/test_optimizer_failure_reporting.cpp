#include <stdexcept>
#include <string>
#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/optimizer.hpp"

int main(int argc, char** argv) {
    if (argc != 2)
        throw std::invalid_argument("test requires a nonstationary input fixture");
    xmvb::memory::configure({});
    auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
    xmvb::vb::VbScfOptimizerOptions options;
    options.backend = xmvb::vb::VbScfOptimizerBackend::Ciah;
    options.verbose = false;
    options.accepted_iteration_callback = [](const auto& point) {
        if (point.accepted_iteration_index == 1)
            throw std::runtime_error("injected observer failure");
    };
    const auto result = xmvb::vb::VbScfOptimizer(options).optimize(std::move(loaded.input),
                                                                   loaded.nuclear_repulsion_energy);
    if (result.n_iterations != 1 || result.total_energy_history.size() != 2 ||
        result.termination_reason != "injected observer failure")
        throw std::runtime_error("backend exception discarded its committed iteration count");
}
