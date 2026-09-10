#pragma once

#include <filesystem>

namespace xmvb::vb {
struct VbScfOptimizerResult;
}

namespace xmvb::output {

/** Write the lightweight accepted-step TNHVP diagnostics as tab-separated data. */
void write_tnhvp_trace(
    const std::filesystem::path& output_path,
    const vb::VbScfOptimizerResult& result);

}  // namespace xmvb::output
