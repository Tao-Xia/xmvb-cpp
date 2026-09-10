#pragma once

#include <optional>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/optimizer.hpp"

namespace xmvb::app::vbscf {

struct Options {
  std::string input_path;
  vb::VbScfInputLoadOptions load;
  vb::VbScfOptimizerOptions optimizer;
  std::string trace_directory;
  std::string final_orbitals_path;
  bool max_iterations_explicit = false;
};

std::optional<Options> parse_options(int argc, char** argv);

}  // namespace xmvb::app::vbscf
