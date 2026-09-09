#pragma once

#include <string>

#include "runtime/vbscf_input_loader.hpp"
#include "vb/scf/deepvbh_onnx_direct_final_optimizer.hpp"
#include "vb/scf/deepvbh_onnx_hybrid_optimizer.hpp"
#include "vbscf/adaptive/structure_space_optimizer.hpp"
#include "vbscf/optimization/vbscf_optimizer.hpp"

namespace xmvb::app::vbscf {

enum class StructureMode {
  Standard,
  AdaptiveMvp,
};

enum class Backend {
  Core,
  DeepVBHOnnx,
  DeepVBHOnnxDirectFinal,
};

struct Options {
  std::string input_path;
  StructureMode structure_mode = StructureMode::Standard;
  vb::VbScfInputLoadOptions load;
  vb::VbScfOptimizerOptions optimizer;
  Backend backend = Backend::Core;
  vb::AdaptiveStructureSpaceOptimizerOptions adaptive;
  vb::DeepVBHOnnxHybridOptimizerOptions deepvbh_hybrid;
  vb::DeepVBHOnnxDirectFinalOptimizerOptions deepvbh_direct;
  std::string trace_directory;
  std::string final_orbitals_path;
  bool max_iterations_explicit = false;
};

bool deepvbh_backend_supported();
bool uses_deepvbh_backend(Backend backend);
const char* backend_name(
    Backend backend,
    vb::VbScfOptimizerBackend core_backend);
const char* structure_mode_name(StructureMode mode);

bool parse_options(int argc, char** argv, Options* parsed_options);
void print_usage();

}  // namespace xmvb::app::vbscf

