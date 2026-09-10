#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>

#include "cli/options.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::app::vbscf {

void print_header(
    const Options& command,
    const vb::VbScfInputLoadResult& load_result,
    const std::chrono::system_clock::time_point& start_time);

std::function<void(const vb::VbScfAcceptedIterationSnapshot&)>
combine_callbacks(
    std::function<void(const vb::VbScfAcceptedIterationSnapshot&)> first,
    std::function<void(const vb::VbScfAcceptedIterationSnapshot&)> second);

std::function<void(const vb::VbScfAcceptedIterationSnapshot&)>
iteration_logger();

void print_summary(
    const Options& command,
    const vb::VbScfInputLoadResult& load_result,
    const vb::VbScfOptimizerResult& result,
    const std::optional<std::filesystem::path>& trace_sample_directory,
    const std::optional<std::filesystem::path>& molden_output_path,
    const std::chrono::system_clock::time_point& command_start_time,
    const std::chrono::steady_clock::time_point& command_start_steady_time);

}  // namespace xmvb::app::vbscf
