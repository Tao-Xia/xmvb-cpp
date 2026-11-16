#pragma once

#include <chrono>

#include "cli/options.hpp"

namespace xmvb::cli {

int run(Options options, std::chrono::steady_clock::time_point program_entry);

} // namespace xmvb::cli
