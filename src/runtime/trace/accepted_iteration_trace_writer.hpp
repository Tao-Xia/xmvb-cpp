#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace xmvb::vb {
struct VbScfAcceptedIterationSnapshot;
struct VbScfInputLoadResult;
struct VbScfOptimizerResult;
}  // namespace xmvb::vb

namespace xmvb::runtime {

class AcceptedIterationTraceWriter {
public:
  AcceptedIterationTraceWriter(
      const std::filesystem::path& dataset_root,
      const std::string& input_file_path,
      const vb::VbScfInputLoadResult& load_result,
      const std::string& optimizer_backend_name);
  ~AcceptedIterationTraceWriter();

  AcceptedIterationTraceWriter(const AcceptedIterationTraceWriter&) = delete;
  AcceptedIterationTraceWriter& operator=(const AcceptedIterationTraceWriter&) = delete;

  void write_accepted_iteration(
      const vb::VbScfAcceptedIterationSnapshot& snapshot);
  void finalize(const vb::VbScfOptimizerResult& result);

  const std::filesystem::path& sample_directory() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xmvb::runtime
