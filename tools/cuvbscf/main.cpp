#include <exception>
#include <iostream>
#include <stdexcept>

#include "cli/cuvbscf.hpp"
#include "core/memory.hpp"
#include "input/deck/metadata.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: cuvbscf INPUT.xmi");
        xmvb::cli::Options options;
        options.input_path = argv[1];
        xmvb::memory::configure(options.ci_workspace_bytes);
        return xmvb::cli::run_cuvbscf(options,
                                      xmvb::vb::parse_input_deck_metadata(options.input_path));
    } catch (const std::exception& error) {
        std::cerr << "cuvbscf: " << error.what() << '\n';
        return 1;
    }
}
