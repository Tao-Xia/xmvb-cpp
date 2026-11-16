#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

void print_header(const std::string& input_path,
                  const vb::VbScfOptimizerOptions& options,
                  const vb::VbScfInputLoadResult& load_result,
                  const std::chrono::system_clock::time_point& start_time);

std::function<void(const vb::VbScfAcceptedIterationSnapshot&)>
combine_callbacks(std::function<void(const vb::VbScfAcceptedIterationSnapshot&)> first,
                  std::function<void(const vb::VbScfAcceptedIterationSnapshot&)> second);

std::function<void(const vb::VbScfAcceptedIterationSnapshot&)> iteration_logger();

/** @brief Emit and flush a machine-readable boundary without changing stream formatting. */
void print_scf_convergence_marker(std::ostream& output, double program_wall_seconds);

void print_summary(const vb::VbScfOptimizerOptions& options,
                   const vb::VbScfInputLoadResult& load_result,
                   const vb::VbScfOptimizerResult& result,
                   const std::optional<std::filesystem::path>& trace_sample_directory,
                   const std::optional<std::filesystem::path>& molden_output_path,
                   const std::chrono::system_clock::time_point& command_start_time,
                   const std::chrono::steady_clock::time_point& program_entry,
                   const std::optional<std::chrono::steady_clock::time_point>& scf_converged_at);

} // namespace xmvb::output
