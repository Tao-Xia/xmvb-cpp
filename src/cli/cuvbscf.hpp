#pragma once

#include "cli/options.hpp"
#include "input/deck/metadata.hpp"

namespace xmvb::cli {
/** @brief Shared GPU SCF entry for the input keyword and standalone executable. */
int run_cuvbscf(const Options& options, const vb::InputDeckMetadata& metadata);
} // namespace xmvb::cli
