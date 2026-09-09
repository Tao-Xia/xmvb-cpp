#include "app/vbscf/options.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace xmvb::app::vbscf {
namespace fs = std::filesystem;

bool deepvbh_backend_supported() {
#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME
  return true;
#else
  return false;
#endif
}

bool uses_deepvbh_backend(Backend backend) {
  return backend != Backend::Core;
}

const char* backend_name(
    Backend backend,
    vb::VbScfOptimizerBackend core_backend) {
  switch (backend) {
    case Backend::Core:
      return vb::vbscf_optimizer_backend_name(core_backend);
    case Backend::DeepVBHOnnx:
      return "deepvbh_onnx";
    case Backend::DeepVBHOnnxDirectFinal:
      return "deepvbh_onnx_direct_final";
  }
  return "unknown";
}

void apply_optimizer_backend_argument(
    const std::string& backend_name,
    xmvb::vb::VbScfOptimizerOptions* options,
    Backend* run_backend) {
  if (options == nullptr || run_backend == nullptr) {
    throw std::invalid_argument("optimizer backend outputs must not be null");
  }
  if (backend_name == "lbfgspp") {
    *run_backend = Backend::Core;
    options->backend = xmvb::vb::VbScfOptimizerBackend::Lbfgspp;
    return;
  }
  if (backend_name == "nonredundant_projected_gradient") {
    *run_backend = Backend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantProjectedGradient;
    return;
  }
  if (backend_name == "nonredundant_lbfgspp") {
    *run_backend = Backend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp;
    return;
  }
  if (backend_name == "nonredundant_truncated_newton") {
    *run_backend = Backend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton;
    return;
  }
  if (backend_name == "deepvbh_onnx") {
    if (!deepvbh_backend_supported()) {
      throw std::invalid_argument(
          "deepvbh_onnx backend is not enabled in this build");
    }
    *run_backend = Backend::DeepVBHOnnx;
    return;
  }
  if (backend_name == "deepvbh_onnx_direct_final") {
    if (!deepvbh_backend_supported()) {
      throw std::invalid_argument(
          "deepvbh_onnx_direct_final backend is not enabled in this build");
    }
    *run_backend = Backend::DeepVBHOnnxDirectFinal;
    return;
  }
  throw std::invalid_argument("invalid optimizer backend: " + backend_name);
}

void apply_nonredundant_truncated_newton_hvp_mode_argument(
    const std::string& mode_name,
    xmvb::vb::VbScfOptimizerOptions* options) {
  if (mode_name == "full_fd") {
    options->nonredundant_truncated_newton_hvp_mode =
        xmvb::vb::NonredundantTruncatedNewtonHvpMode::FullFiniteDifference;
    return;
  }
  if (mode_name == "exact_ctx" || mode_name == "exact_context") {
    options->nonredundant_truncated_newton_hvp_mode =
        xmvb::vb::NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction;
    return;
  }
  throw std::invalid_argument(
      "invalid nonredundant truncated-Newton HVP mode: " + mode_name);
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  }
  if (value == "false" || value == "0" || value == "no") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

void apply_raw_structure_selection_argument(
    const std::string& selection_name,
    xmvb::vb::VbScfInputLoadOptions* options) {
  if (selection_name == "full") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (selection_name == "covalent") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid raw structure selection: " + selection_name);
}

void apply_ao_integral_source_argument(
    const std::string& source_name,
    xmvb::vb::VbScfInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (source_name == "auto") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
    return;
  }
  if (source_name == "libcint_cpp") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    return;
  }
  if (source_name == "runtime_hcore") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
    return;
  }
  throw std::invalid_argument("invalid AO integral source: " + source_name);
}

void apply_standard_two_electron_mode_argument(
    const std::string& mode_name,
    xmvb::vb::VbScfInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (mode_name == "auto") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
    return;
  }
  if (mode_name == "exact") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    return;
  }
  if (mode_name == "ri") {
    options->standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    return;
  }
  throw std::invalid_argument("invalid standard two-electron mode: " + mode_name);
}

void print_usage() {
  std::cerr << "usage: xmvb-cpp.exe <input.xmi> "
               "[--optimizer-backend lbfgspp|nonredundant_projected_gradient|nonredundant_lbfgspp|nonredundant_truncated_newton";
  if (deepvbh_backend_supported()) {
    std::cerr << "|deepvbh_onnx";
    std::cerr << "|deepvbh_onnx_direct_final";
  }
  std::cerr << "] [--max-iterations <count>]"
               " [--verbose true|false]"
               " [--gradient-tolerance <value>]"
               " [--energy-tolerance <value>]"
               "\n"
               " [--nonredundant-truncated-newton-max-cg-iterations <count|0=32>]"
               " [--nonredundant-truncated-newton-hvp-mode full_fd|exact_ctx]"
               " [--nonredundant-truncated-newton-hvp-step-size <value>]"
               " [--nonredundant-truncated-newton-transport-history-size <count>]"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--ao-integral-source auto|libcint_cpp|runtime_hcore]"
               " [--skip-orbital-guess true|false]"
               " [--raw-structure-selection full|covalent]"
               " [--dump-trace-dir <dataset_root>]"
               " [--dump-final-orbital-value-table-bin <path>]"
               " [--onnx-model <path>]"
               " [--ml-initial-step-scale <value>]"
               " [--ml-minimum-step-scale <value>]"
               " [--ml-step-shrink-factor <value>]"
               " [--ml-max-backtracks <count>]"
               " [--ml-fallback-max-iterations <count>]"
               " [--ml-keep-work-dir true|false]"
               " [--ml-work-dir <path>]\n"
               "default optimizer backend: nonredundant_lbfgspp\n";
}

bool parse_options(int argc, char** argv, Options* parsed_options) {
  if (parsed_options == nullptr) {
    throw std::invalid_argument("parsed_options must not be null");
  }
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return false;
  }

  const std::string input_path = argv[1];
  xmvb::vb::VbScfInputLoadOptions load_options;
  load_options.ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
  load_options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
  xmvb::vb::VbScfOptimizerOptions options;
  Backend run_backend = Backend::Core;
  // The CLI path only needs accepted-iterate snapshots when trace dumping is
  // explicitly requested. Keep the default optimizer API behavior unchanged,
  // but disable trace retention here so routine runs avoid the extra
  // reference-gradient work performed per accepted iterate.
  options.retain_accepted_iteration_trace = false;
  options.accepted_iteration_callback_requires_full_snapshot = false;
  options.max_iterations = 2000;
  options.gradient_tolerance = 1.0e-3;
  options.energy_tolerance = 1.0e-7;
  options.initial_step_size = 1.0e20;
  options.minimum_step_size = 1.0e-20;
  options.history_size = 100;
  xmvb::vb::DeepVBHOnnxHybridOptimizerOptions deepvbh_options;
  xmvb::vb::DeepVBHOnnxDirectFinalOptimizerOptions deepvbh_direct_options;
  deepvbh_options.optimizer_options = options;
  deepvbh_options.inference_options.repo_root = fs::current_path();
  deepvbh_options.inference_options.backend = "onnx_runtime";
  deepvbh_direct_options.optimizer_options = options;
  deepvbh_direct_options.inference_options.repo_root = fs::current_path();
  deepvbh_direct_options.inference_options.backend = "onnx_runtime";

  std::string dump_trace_dir;
  std::string dump_final_orbital_value_table_bin;
  bool user_specified_ao_integral_source = false;
  bool user_specified_max_iterations = false;
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    try {
      if (argument_name == "--optimizer-backend") {
        apply_optimizer_backend_argument(argument_value, &options, &run_backend);
      } else if (argument_name == "--max-iterations") {
        user_specified_max_iterations = true;
        options.max_iterations = std::stoi(argument_value);
      } else if (argument_name == "--verbose") {
        options.verbose = parse_bool_argument(argument_value);
      } else if (argument_name == "--gradient-tolerance") {
        options.gradient_tolerance = std::stod(argument_value);
      } else if (argument_name == "--energy-tolerance") {
        options.energy_tolerance = std::stod(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-max-cg-iterations") {
        options.nonredundant_truncated_newton_max_cg_iterations =
            std::stoi(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-hvp-step-size") {
        options.nonredundant_truncated_newton_hvp_step_size =
            std::stod(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-hvp-mode") {
        apply_nonredundant_truncated_newton_hvp_mode_argument(
            argument_value,
            &options);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-transport-history-size") {
        options.nonredundant_truncated_newton_transport_history_size =
            std::stoi(argument_value);
      } else if (argument_name == "--ao-integral-source") {
        user_specified_ao_integral_source = true;
        apply_ao_integral_source_argument(argument_value, &load_options);
      } else if (argument_name == "--standard-two-electron-mode") {
        apply_standard_two_electron_mode_argument(argument_value, &load_options);
      } else if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else if (argument_name == "--raw-structure-selection") {
        apply_raw_structure_selection_argument(argument_value, &load_options);
      } else if (argument_name == "--dump-trace-dir") {
        dump_trace_dir = argument_value;
      } else if (argument_name == "--dump-final-orbital-value-table-bin") {
        dump_final_orbital_value_table_bin = argument_value;
      } else if (argument_name == "--onnx-model") {
        deepvbh_options.inference_options.onnx_model_path = argument_value;
        deepvbh_direct_options.inference_options.onnx_model_path = argument_value;
      } else if (argument_name == "--ml-initial-step-scale") {
        deepvbh_options.initial_step_scale = std::stod(argument_value);
        deepvbh_direct_options.initial_step_scale = std::stod(argument_value);
      } else if (argument_name == "--ml-minimum-step-scale") {
        deepvbh_options.minimum_step_scale = std::stod(argument_value);
        deepvbh_direct_options.minimum_step_scale = std::stod(argument_value);
      } else if (argument_name == "--ml-step-shrink-factor") {
        deepvbh_options.step_shrink_factor = std::stod(argument_value);
        deepvbh_direct_options.step_shrink_factor = std::stod(argument_value);
      } else if (argument_name == "--ml-max-backtracks") {
        deepvbh_options.max_backtracks = std::stoi(argument_value);
        deepvbh_direct_options.max_backtracks = std::stoi(argument_value);
      } else if (argument_name == "--ml-fallback-max-iterations") {
        deepvbh_options.exact_fallback_max_iterations = std::stoi(argument_value);
        deepvbh_direct_options.exact_fallback_max_iterations = std::stoi(argument_value);
      } else if (argument_name == "--ml-keep-work-dir") {
        deepvbh_options.inference_options.keep_work_directory =
            parse_bool_argument(argument_value);
        deepvbh_direct_options.inference_options.keep_work_directory =
            parse_bool_argument(argument_value);
      } else if (argument_name == "--ml-work-dir") {
        deepvbh_options.inference_options.work_directory = argument_value;
        deepvbh_direct_options.inference_options.work_directory = argument_value;
      } else {
        std::cerr << "unknown argument: " << argument_name << '\n';
        print_usage();
        return false;
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return false;
    }
  }
  if (!user_specified_ao_integral_source &&
      uses_deepvbh_backend(run_backend)) {
    load_options.ao_integral_source =
        xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
  }
  load_options.build_ao_effective_one_electron_graph =
      options.backend ==
          xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton &&
      options.nonredundant_truncated_newton_hvp_mode ==
          xmvb::vb::NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction;

  Options parsed;
  parsed.input_path = input_path;
  parsed.load = std::move(load_options);
  parsed.optimizer = std::move(options);
  parsed.backend = run_backend;
  parsed.deepvbh_hybrid = std::move(deepvbh_options);
  parsed.deepvbh_direct = std::move(deepvbh_direct_options);
  parsed.trace_directory = std::move(dump_trace_dir);
  parsed.final_orbitals_path = std::move(dump_final_orbital_value_table_bin);
  parsed.max_iterations_explicit = user_specified_max_iterations;
  *parsed_options = std::move(parsed);
  return true;
}

}  // namespace xmvb::app::vbscf
