#pragma once

#include <filesystem>
#include <fstream>

namespace xmvb::vb {
struct CiahIterationRecord;
}

namespace xmvb::output {

/** Write accepted CIAH iteration diagnostics as tab-separated rows. */
class CiahTraceWriter {
  public:
    explicit CiahTraceWriter(const std::filesystem::path& output_path);

    void write(const vb::CiahIterationRecord& record);

  private:
    std::ofstream stream_;
    std::filesystem::path output_path_;
};

} // namespace xmvb::output
