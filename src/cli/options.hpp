#pragma once

#include <optional>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/optimizer.hpp"

namespace xmvb::cli {

struct Options {
  std::string input_path;
  vb::VbScfInputLoadOptions load;
  vb::VbScfOptimizerOptions optimizer;
  std::string trace_directory;
  std::string tnhvp_trace_path;
  std::string ciah_trace_path;
  std::string final_orbitals_path;
  std::size_t exact_exchange_memory_budget_bytes = 64 * 1024 * 1024;
  std::optional<std::size_t> ci_workspace_bytes;
  bool optimizer_backend_explicit = false;
  bool structure_eigensolver_explicit = false;
  bool max_iterations_explicit = false;
};

std::optional<Options> parse_options(int argc, char** argv);

}  // namespace xmvb::cli
