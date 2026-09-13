#include "cli/options.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace xmvb::cli {
namespace {

void apply_optimizer_backend_argument(
    const std::string& backend_name,
    xmvb::vb::VbScfOptimizerOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("optimizer options must not be null");
  }
  if (backend_name == "lbfgspp") {
    options->backend = xmvb::vb::VbScfOptimizerBackend::Lbfgspp;
    return;
  }
  if (backend_name == "nonredundant_projected_gradient") {
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantProjectedGradient;
    return;
  }
  if (backend_name == "nonredundant_lbfgspp") {
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp;
    return;
  }
  if (backend_name == "nonredundant_truncated_newton") {
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton;
    return;
  }
  throw std::invalid_argument("invalid optimizer backend: " + backend_name);
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
               "[--optimizer-backend lbfgspp|nonredundant_projected_gradient|nonredundant_lbfgspp|nonredundant_truncated_newton]"
               " [--max-iterations <count>]"
               " [--verbose true|false]"
               " [--gradient-tolerance <value>]"
               " [--energy-tolerance <value>]"
               "\n"
               " [--nonredundant-truncated-newton-max-cg-iterations <count|0=32>]"
               " [--nonredundant-truncated-newton-transport-history-size <count>]"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--skip-orbital-guess true|false]"
               " [--raw-structure-selection full|covalent]"
               " [--dump-trace-dir <dataset_root>]"
               " [--tnhvp-trace <path.tsv>]"
               " [--dump-final-orbital-value-table-bin <path>]\n"
               "default optimizer backend: nonredundant_lbfgspp\n";
}

}  // namespace

std::optional<Options> parse_options(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return std::nullopt;
  }

  const std::string input_path = argv[1];
  xmvb::vb::VbScfInputLoadOptions load_options;
  load_options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
  xmvb::vb::VbScfOptimizerOptions options;
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
  std::string dump_trace_dir;
  std::string tnhvp_trace_path;
  std::string dump_final_orbital_value_table_bin;
  bool user_specified_max_iterations = false;
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    try {
      if (argument_name == "--optimizer-backend") {
        apply_optimizer_backend_argument(argument_value, &options);
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
          "--nonredundant-truncated-newton-transport-history-size") {
        options.nonredundant_truncated_newton_transport_history_size =
            std::stoi(argument_value);
      } else if (argument_name == "--standard-two-electron-mode") {
        apply_standard_two_electron_mode_argument(argument_value, &load_options);
      } else if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else if (argument_name == "--raw-structure-selection") {
        apply_raw_structure_selection_argument(argument_value, &load_options);
      } else if (argument_name == "--dump-trace-dir") {
        dump_trace_dir = argument_value;
      } else if (argument_name == "--tnhvp-trace") {
        tnhvp_trace_path = argument_value;
      } else if (argument_name == "--dump-final-orbital-value-table-bin") {
        dump_final_orbital_value_table_bin = argument_value;
      } else {
        std::cerr << "unknown argument: " << argument_name << '\n';
        print_usage();
        return std::nullopt;
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return std::nullopt;
    }
  }
  load_options.build_ao_effective_one_electron_graph =
      options.backend ==
          xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton;

  Options parsed;
  parsed.input_path = input_path;
  parsed.load = std::move(load_options);
  parsed.optimizer = std::move(options);
  parsed.trace_directory = std::move(dump_trace_dir);
  parsed.tnhvp_trace_path = std::move(tnhvp_trace_path);
  parsed.final_orbitals_path = std::move(dump_final_orbital_value_table_bin);
  parsed.max_iterations_explicit = user_specified_max_iterations;
  return parsed;
}

}  // namespace xmvb::cli
