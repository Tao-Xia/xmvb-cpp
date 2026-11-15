#include <chrono>
#include <exception>
#include <iostream>
#include <utility>

#include "cli/options.hpp"
#include "cli/run.hpp"

int main(int argc, char** argv) {
    const auto program_entry = std::chrono::steady_clock::now();
    try {
        auto options = xmvb::cli::parse_options(argc, argv);
        if (!options.has_value()) {
            return 1;
        }
        return xmvb::cli::run(std::move(*options), program_entry);
    } catch (const std::exception& error) {
        std::cerr << "XMVB-CPP error: " << error.what() << '\n';
        return 2;
    }
}
