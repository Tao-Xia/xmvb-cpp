#pragma once

#include <chrono>
#include <iosfwd>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

void print_program_preamble(
    std::ostream& output,
    const std::string& input_path,
    const std::chrono::system_clock::time_point& start_time,
    int n_threads);

void print_input_sections(
    std::ostream& output,
    const std::string& input_path,
    const vb::VbScfInputLoadResult& load_result,
    const char* optimizer_name,
    int max_iterations);

void print_final_state_sections(
    std::ostream& output,
    const vb::VbScfInputLoadResult& load_result,
    const vb::VbScfOptimizerResult& result);

}  // namespace xmvb::output
