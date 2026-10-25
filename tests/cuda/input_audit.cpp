#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "input/loading/loader.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: cuvb_input_audit INPUT.xmi");
        xmvb::vb::VbScfInputLoadOptions options;
        options.ao_pair_layout = xmvb::integrals::AoPairLayout::Chunked;
        const auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1], options);
        std::cout << std::setprecision(16) << "input_seconds=" << loaded.total_seconds << '\n'
                  << "provider_seconds=" << loaded.ao_integral_provider_seconds << '\n'
                  << "setup_seconds=" << loaded.ao_integral_setup_seconds << '\n'
                  << "generation_seconds=" << loaded.ao_integral_generation_seconds << '\n'
                  << "merge_seconds=" << loaded.ao_integral_merge_seconds << '\n'
                  << "buffered_bytes=" << loaded.ao_integral_buffered_bytes << '\n'
                  << "unique_integrals="
                  << loaded.input.ao_integral_input.pair_graph.integral_count() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
