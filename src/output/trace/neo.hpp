#pragma once

#include <filesystem>

namespace xmvb::vb {
struct VbScfOptimizerResult;
}

namespace xmvb::output {

/** Write the lightweight accepted-step NEO diagnostics as tab-separated data. */
void write_neo_trace(
    const std::filesystem::path& output_path,
    const vb::VbScfOptimizerResult& result);

}  // namespace xmvb::output
