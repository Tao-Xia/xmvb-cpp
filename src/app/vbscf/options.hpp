#pragma once

#include <string>

#include "runtime/vbscf_input_loader.hpp"
#include "vbscf/optimization/vbscf_optimizer.hpp"

namespace xmvb::app::vbscf {

struct Options {
  std::string input_path;
  vb::VbScfInputLoadOptions load;
  vb::VbScfOptimizerOptions optimizer;
  std::string trace_directory;
  std::string final_orbitals_path;
  bool max_iterations_explicit = false;
};

bool parse_options(int argc, char** argv, Options* parsed_options);
void print_usage();

}  // namespace xmvb::app::vbscf
