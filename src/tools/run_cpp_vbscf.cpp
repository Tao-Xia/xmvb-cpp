#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/vbscf_input_loader.hpp"
#include "runtime/molden_file_writer.hpp"
#include "runtime/io/binary_file.hpp"
#include "runtime/trace/accepted_iteration_trace_writer.hpp"
#include "vb/scf/deepvbh_onnx_direct_final_optimizer.hpp"
#include "vb/scf/deepvbh_onnx_hybrid_optimizer.hpp"
#include "vbscf/adaptive/structure_space_optimizer.hpp"
#include "vbscf/optimization/vbscf_optimizer.hpp"

namespace {

namespace fs = std::filesystem;

enum class StructureSpaceMode {
  Standard,
  AdaptiveMvp,
};

enum class RunOptimizerBackend {
  Core,
  DeepVBHOnnx,
  DeepVBHOnnxDirectFinal,
};

bool deepvbh_onnx_backend_supported() {
#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME
  return true;
#else
  return false;
#endif
}

bool uses_deepvbh_backend(RunOptimizerBackend backend) {
  return backend != RunOptimizerBackend::Core;
}

bool core_backend_reports_projected_gradient(
    xmvb::vb::VbScfOptimizerBackend backend) {
  switch (backend) {
    case xmvb::vb::VbScfOptimizerBackend::NonredundantProjectedGradient:
    case xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp:
    case xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton:
      return true;
    case xmvb::vb::VbScfOptimizerBackend::Lbfgspp:
      return false;
  }
  return false;
}

const char* run_optimizer_backend_name(
    RunOptimizerBackend run_backend,
    xmvb::vb::VbScfOptimizerBackend core_backend) {
  switch (run_backend) {
    case RunOptimizerBackend::Core:
      return xmvb::vb::vbscf_optimizer_backend_name(core_backend);
    case RunOptimizerBackend::DeepVBHOnnx:
      return "deepvbh_onnx";
    case RunOptimizerBackend::DeepVBHOnnxDirectFinal:
      return "deepvbh_onnx_direct_final";
  }
  return "unknown";
}

const char* structure_space_mode_name(StructureSpaceMode mode) {
  switch (mode) {
    case StructureSpaceMode::Standard:
      return "standard";
    case StructureSpaceMode::AdaptiveMvp:
      return "adaptive_mvp";
  }
  return "unknown";
}

const char* gradient_tolerance_metric_name(
    xmvb::vb::VbScfOptimizerBackend backend) {
  switch (backend) {
    case xmvb::vb::VbScfOptimizerBackend::Lbfgspp:
      return "full_gradient_l2_norm";
    case xmvb::vb::VbScfOptimizerBackend::NonredundantProjectedGradient:
    case xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp:
    case xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton:
      return "projected_gradient_inf_norm";
  }
  return "unknown";
}

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

bool parse_env_flag_with_default(
    const char* variable_name,
    bool default_value);
int parse_env_int_with_default(
    const char* variable_name,
    int default_value);
double parse_env_double_with_default(
    const char* variable_name,
    double default_value);

constexpr int kLogRuleWidth = 88;
constexpr int kLogLabelWidth = 34;

std::optional<bool> parse_env_optional_flag(
    const char* variable_name) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

bool parse_env_flag_with_default(
    const char* variable_name,
    bool default_value) {
  const auto value = parse_env_optional_flag(variable_name);
  return value.has_value() ? *value : default_value;
}

int parse_env_int_with_default(
    const char* variable_name,
    int default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

double parse_env_double_with_default(
    const char* variable_name,
    double default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      !std::isfinite(parsed)) {
    return default_value;
  }
  return parsed;
}

const char* env_value_or_unset(const char* variable_name) {
  const char* value = std::getenv(variable_name);
  return (value == nullptr || value[0] == '\0') ? "<unset>" : value;
}

// Centralize terminal formatting so the standalone driver prints a stable
// run log instead of a long unstructured key/value dump.
void print_log_rule(char fill = '=') {
  std::cout << std::string(kLogRuleWidth, fill) << '\n';
}

void print_log_section_title(const std::string& title) {
  print_log_rule('=');
  std::cout << title << '\n';
  print_log_rule('=');
}

void print_log_subsection_title(const std::string& title) {
  std::cout << '\n' << title << '\n'
            << std::string(title.size(), '-') << '\n';
}

void print_log_field(
    const std::string& label,
    const std::string& value) {
  std::ostringstream stream;
  stream << std::left << std::setw(kLogLabelWidth) << label
         << " : " << value;
  std::cout << stream.str() << '\n';
}

std::string format_fixed_double(
    double value,
    int precision = 12) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string format_scientific_double(
    double value,
    int precision = 8) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(precision) << value;
  return stream.str();
}

std::string format_compact_double(
    double value,
    int precision = 6) {
  std::ostringstream stream;
  stream << std::defaultfloat << std::setprecision(precision) << value;
  return stream.str();
}

std::string format_convergence_threshold_summary(
    double energy_tolerance,
    double gradient_tolerance) {
  return format_scientific_double(energy_tolerance, 0) +
      " for energy and " +
      format_compact_double(gradient_tolerance, 6) +
      " for gradient.";
}

std::string format_seconds(
    double seconds,
    int precision = 6) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << seconds << " s";
  return stream.str();
}

std::string format_core_hours(
    double wall_time_seconds,
    int n_threads) {
  const int effective_threads = std::max(1, n_threads);
  const double core_hours =
      wall_time_seconds * static_cast<double>(effective_threads) / 3600.0;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << core_hours << " core-h";
  return stream.str();
}

std::string format_timestamp(
    const std::chrono::system_clock::time_point& time_point) {
  const std::time_t raw_time = std::chrono::system_clock::to_time_t(time_point);
  std::tm local_time{};
  localtime_r(&raw_time, &local_time);
  std::array<char, 32> buffer{};
  if (std::strftime(
          buffer.data(),
          buffer.size(),
          "%Y-%m-%d %H:%M:%S",
          &local_time) == 0) {
    return "<unavailable>";
  }
  return buffer.data();
}

std::string simplify_basis_name(const std::string& basis_name) {
  if (basis_name.empty()) {
    return "<unknown>";
  }
  fs::path basis_path(basis_name);
  std::string leaf_name = basis_path.filename().string();
  if (leaf_name.empty()) {
    leaf_name = basis_name;
  }
  if (leaf_name.size() > 4 &&
      leaf_name.compare(leaf_name.size() - 4, 4, ".gbs") == 0) {
    leaf_name.erase(leaf_name.size() - 4);
  }
  return leaf_name;
}

std::string get_host_name() {
  std::array<char, 256> buffer{};
  if (gethostname(buffer.data(), buffer.size()) != 0) {
    return "<unknown>";
  }
  buffer.back() = '\0';
  return buffer.data();
}

int openmp_max_thread_count() {
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  return n_threads;
}

int allocated_cpu_thread_count() {
  const char* slurm_value = std::getenv("SLURM_CPUS_PER_TASK");
  if (slurm_value != nullptr && slurm_value[0] != '\0') {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(slurm_value, &end, 10);
    if (errno == 0 && end != slurm_value && (end == nullptr || end[0] == '\0') &&
        parsed > 0 && parsed <= static_cast<long>(std::numeric_limits<int>::max())) {
      return static_cast<int>(parsed);
    }
  }
  return openmp_max_thread_count();
}

std::string convergence_status_name(bool converged) {
  return converged ? "converged" : "not converged";
}

std::string exact_ctx_initial_outer_response_policy_name(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  (void)orbital_preparation_input;
  return "full_exact_hessian";
}

std::string exact_ctx_physical_chart_name(
    const xmvb::vb::OrbitalPreparationInput& orbital_input) {
  for (const int basis_count : orbital_input.orbital_basis_counts) {
    if (basis_count > 0 && basis_count < orbital_input.n_basis_functions) {
      return "strict_sparse_U_p";
    }
  }
  return "full_ao_U_p";
}

void print_exact_ctx_policy_summary(
    const xmvb::vb::VbScfOptimizerOptions& options,
    const xmvb::vb::VbScfInput& input) {
  if (options.backend !=
          xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton ||
      options.nonredundant_truncated_newton_hvp_mode !=
          xmvb::vb::NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction) {
    return;
  }

  const auto& orbital_input = input.orbital_preparation_input;
  print_log_subsection_title("Exact-CTX Matrix-Free Newton");
  print_log_field(
      "Hessian model",
      exact_ctx_initial_outer_response_policy_name(orbital_input));
  print_log_field(
      "Inexact Newton forcing",
      "adaptive sqrt(projected gradient 2-norm)");
  print_log_field(
      "Trust-radius update",
      "Ritz spectrum + observed model remainder");
  print_log_field(
      "Krylov safety limit",
      options.nonredundant_truncated_newton_max_cg_iterations > 0
          ? std::to_string(
                options.nonredundant_truncated_newton_max_cg_iterations)
          : "32");
  print_log_field(
      "Transport history",
      std::to_string(
          options.nonredundant_truncated_newton_transport_history_size));
  print_log_field(
      "Physical chart",
      exact_ctx_physical_chart_name(orbital_input));
}

void print_run_header(
    const std::string& input_path,
    const xmvb::vb::VbScfInputLoadResult& load_result,
    const xmvb::vb::VbScfOptimizerOptions& options,
    RunOptimizerBackend run_backend,
    StructureSpaceMode structure_space_mode,
    const std::chrono::system_clock::time_point& start_time) {
  const fs::path absolute_input_path = fs::absolute(fs::path(input_path));
  const auto& orbital_input = load_result.input.orbital_preparation_input;
  const std::string basis_set_name = simplify_basis_name(load_result.basis_name);
  const int expanded_determinant_count =
      static_cast<int>(load_result.input.structure_data.alpha_det.size());

  print_log_section_title("XMVB-CPP VBSCF Run");

  print_log_subsection_title("Run Setup");
  print_log_field("Input file", absolute_input_path.string());
  print_log_field("Start time", format_timestamp(start_time));
  print_log_field("SCF algorithm", "VBSCF");
  print_log_field(
      "Optimizer backend",
      run_optimizer_backend_name(run_backend, options.backend));
  if (run_backend == RunOptimizerBackend::Core &&
      options.backend ==
      xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton) {
    print_log_field(
        "HVP mode",
        xmvb::vb::nonredundant_truncated_newton_hvp_mode_name(
            options.nonredundant_truncated_newton_hvp_mode));
  }
  print_log_field("Structure space", structure_space_mode_name(structure_space_mode));
  print_log_field("Basis set", basis_set_name);
  if (!load_result.basis_name.empty() && load_result.basis_name != basis_set_name) {
    print_log_field("Basis file", load_result.basis_name);
  }
  print_log_field(
      "AO integral source",
      xmvb::vb::ao_integral_source_name(load_result.ao_integral_source));
  print_log_field(
      "Two-electron mode",
      xmvb::vb::standard_two_electron_mode_name(load_result.standard_two_electron_mode));
  print_log_field("Orbital guess", "standalone C++");
  print_log_field(
      "Molden output",
      load_result.request_molden_output ? "requested" : "not requested");

  print_log_subsection_title("System Summary");
  print_log_field("Atoms", std::to_string(load_result.static_molecule_metadata.n_atoms));
  print_log_field("Shells", std::to_string(load_result.static_molecule_metadata.n_shells));
  print_log_field("Basis functions", std::to_string(orbital_input.n_basis_functions));
  print_log_field("Orbitals", std::to_string(orbital_input.n_orbitals));
  print_log_field("Active orbitals", std::to_string(orbital_input.n_active_orbitals));
  print_log_field(
      "Electrons (total/active)",
      std::to_string(orbital_input.n_total_electrons) + " / " +
          std::to_string(orbital_input.n_active_electrons));
  print_log_field(
      "Spin multiplicity",
      std::to_string(load_result.raw_structure_data.spin_multiplicity));
  print_log_field(
      "Raw structures (source/selected)",
      std::to_string(load_result.source_raw_structure_count) + " / " +
          std::to_string(load_result.raw_structure_data.n_structures));
  print_log_field(
      "Expanded determinants",
      expanded_determinant_count > 0 ? std::to_string(expanded_determinant_count) : "deferred");

  print_log_subsection_title("Execution Resources");
  print_log_field("Host", get_host_name());
  print_log_field("Allocated CPU threads", std::to_string(allocated_cpu_thread_count()));
  print_log_field("OpenMP max threads", std::to_string(openmp_max_thread_count()));
  print_log_field("SLURM_CPUS_PER_TASK", env_value_or_unset("SLURM_CPUS_PER_TASK"));
  print_log_field("OMP_NUM_THREADS", env_value_or_unset("OMP_NUM_THREADS"));
  print_log_field("OPENBLAS_NUM_THREADS", env_value_or_unset("OPENBLAS_NUM_THREADS"));
  print_log_field("MKL_NUM_THREADS", env_value_or_unset("MKL_NUM_THREADS"));

  print_log_subsection_title("Convergence Targets");
  print_log_field(
      "Gradient metric",
      run_backend == RunOptimizerBackend::Core
          ? gradient_tolerance_metric_name(options.backend)
          : "full_gradient_inf_norm");
  print_log_field(
      "Convergence threshold",
      format_convergence_threshold_summary(
          options.energy_tolerance,
          options.gradient_tolerance));
  print_log_field("Max iterations", std::to_string(options.max_iterations));
  if (run_backend == RunOptimizerBackend::Core &&
      options.backend ==
      xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton) {
    print_log_field(
        "Max CG iterations",
        options.nonredundant_truncated_newton_max_cg_iterations > 0
            ? std::to_string(options.nonredundant_truncated_newton_max_cg_iterations)
            : "32 (dimension bounded)");
  }

  if (run_backend == RunOptimizerBackend::Core) {
    print_exact_ctx_policy_summary(options, load_result.input);
  }
}

std::function<void(const xmvb::vb::VbScfAcceptedIterationSnapshot&)>
compose_accepted_iteration_callbacks(
    std::function<void(const xmvb::vb::VbScfAcceptedIterationSnapshot&)> first,
    std::function<void(const xmvb::vb::VbScfAcceptedIterationSnapshot&)> second) {
  if (!first) {
    return second;
  }
  if (!second) {
    return first;
  }
  return [first = std::move(first), second = std::move(second)](
             const xmvb::vb::VbScfAcceptedIterationSnapshot& snapshot) {
    first(snapshot);
    second(snapshot);
  };
}

std::function<void(const xmvb::vb::VbScfAcceptedIterationSnapshot&)>
build_terminal_iteration_logger() {
  // Mirror the legacy CLI monitor: keep the initial accepted snapshot only as
  // the energy reference for `DE`, then print one fixed-width row per accepted
  // optimization step.
  struct LoggerState {
    bool has_reference_energy = false;
    double previous_total_energy = 0.0;
    std::chrono::steady_clock::time_point previous_iteration_time =
        std::chrono::steady_clock::now();
  };

  auto state = std::make_shared<LoggerState>();
  std::cout << '\n';
  print_log_subsection_title("SCF Iteration History");
  std::cout << "  "
            << std::setw(6) << "Iter"
            << std::setw(23) << "Total Energy"
            << std::setw(18) << "Delta E"
            << std::setw(18) << "Grad_inf"
            << std::setw(18) << "Grad_l2"
            << std::setw(14) << "Accepted(s)"
            << '\n';
  print_log_rule('-');
  std::cout.flush();

  return [state](const xmvb::vb::VbScfAcceptedIterationSnapshot& snapshot) {
    const auto iteration_time = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(iteration_time - state->previous_iteration_time)
            .count();
    state->previous_iteration_time = iteration_time;

    if (snapshot.accepted_iteration_index == 0) {
      state->previous_total_energy = snapshot.total_energy;
      state->has_reference_energy = true;
      return;
    }

    if (!state->has_reference_energy) {
      state->previous_total_energy = snapshot.total_energy;
      state->has_reference_energy = true;
    }
    const double delta_energy = snapshot.total_energy - state->previous_total_energy;
    std::ostringstream stream;
    stream << "  "
           << std::setw(4) << snapshot.accepted_iteration_index
           << std::setw(23) << std::fixed << std::setprecision(12)
           << snapshot.total_energy
           << std::setw(18) << std::scientific << std::setprecision(8)
           << delta_energy
           << std::setw(18) << std::scientific << std::setprecision(8)
           << snapshot.sparse_orbital_energy_gradient_inf_norm
           << std::setw(18) << std::scientific << std::setprecision(8)
           << snapshot.sparse_orbital_energy_gradient_l2_norm
           << std::setw(14) << std::fixed << std::setprecision(6)
           << elapsed_seconds
           << '\n';
    std::cout << stream.str();
    std::cout.flush();
    state->previous_total_energy = snapshot.total_energy;
    state->has_reference_energy = true;
  };
}

void apply_optimizer_backend_argument(
    const std::string& backend_name,
    xmvb::vb::VbScfOptimizerOptions* options,
    RunOptimizerBackend* run_backend) {
  if (options == nullptr || run_backend == nullptr) {
    throw std::invalid_argument("optimizer backend outputs must not be null");
  }
  if (backend_name == "lbfgspp") {
    *run_backend = RunOptimizerBackend::Core;
    options->backend = xmvb::vb::VbScfOptimizerBackend::Lbfgspp;
    return;
  }
  if (backend_name == "nonredundant_projected_gradient") {
    *run_backend = RunOptimizerBackend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantProjectedGradient;
    return;
  }
  if (backend_name == "nonredundant_lbfgspp") {
    *run_backend = RunOptimizerBackend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp;
    return;
  }
  if (backend_name == "nonredundant_truncated_newton") {
    *run_backend = RunOptimizerBackend::Core;
    options->backend =
        xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton;
    return;
  }
  if (backend_name == "deepvbh_onnx") {
    if (!deepvbh_onnx_backend_supported()) {
      throw std::invalid_argument(
          "deepvbh_onnx backend is not enabled in this build");
    }
    *run_backend = RunOptimizerBackend::DeepVBHOnnx;
    return;
  }
  if (backend_name == "deepvbh_onnx_direct_final") {
    if (!deepvbh_onnx_backend_supported()) {
      throw std::invalid_argument(
          "deepvbh_onnx_direct_final backend is not enabled in this build");
    }
    *run_backend = RunOptimizerBackend::DeepVBHOnnxDirectFinal;
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

void apply_structure_space_mode_argument(
    const std::string& mode_name,
    StructureSpaceMode* mode) {
  if (mode == nullptr) {
    throw std::invalid_argument("mode must not be null");
  }
  if (mode_name == "standard") {
    *mode = StructureSpaceMode::Standard;
    return;
  }
  if (mode_name == "adaptive_mvp") {
    *mode = StructureSpaceMode::AdaptiveMvp;
    return;
  }
  throw std::invalid_argument("invalid structure space mode: " + mode_name);
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

void apply_adaptive_seed_selection_argument(
    const std::string& selection_name,
    xmvb::vb::AdaptiveStructureSpaceOptimizerOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("adaptive options must not be null");
  }
  if (selection_name == "full") {
    options->seed_selection = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (selection_name == "covalent") {
    options->seed_selection = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid adaptive seed selection: " + selection_name);
}

void apply_adaptive_determinant_score_mode_argument(
    const std::string& mode_name,
    xmvb::vb::AdaptiveStructureSpaceOptimizerOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("adaptive options must not be null");
  }
  if (mode_name == "proposal_all") {
    options->determinant_score_mode =
        xmvb::vb::AdaptiveDeterminantScoreMode::ProposalAll;
    return;
  }
  if (mode_name == "outside_only") {
    options->determinant_score_mode =
        xmvb::vb::AdaptiveDeterminantScoreMode::OutsideOnly;
    return;
  }
  throw std::invalid_argument("invalid adaptive determinant score mode: " + mode_name);
}

void print_usage() {
  std::cerr << "usage: run_cpp_vbscf <input.xmi> "
               "[--optimizer-backend lbfgspp|nonredundant_projected_gradient|nonredundant_lbfgspp|nonredundant_truncated_newton";
  if (deepvbh_onnx_backend_supported()) {
    std::cerr << "|deepvbh_onnx";
    std::cerr << "|deepvbh_onnx_direct_final";
  }
  std::cerr << "] [--structure-space-mode standard|adaptive_mvp]"
               " [--max-iterations <count>]"
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
               " [--adaptive-seed-selection full|covalent]"
               " [--adaptive-determinant-score-mode proposal_all|outside_only]"
               " [--adaptive-max-outer-iterations <count>]"
               " [--adaptive-max-topology-distance <count>]"
               " [--adaptive-max-neighbors-per-structure <count>]"
               " [--adaptive-max-candidate-pool-size <count>]"
               " [--adaptive-batch-size <count>]"
               " [--adaptive-max-total-structures <count>]"
               " [--adaptive-minimum-candidate-score <value>]"
               " [--adaptive-verbose true|false]"
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return 1;
  }

  const std::string input_path = argv[1];
  StructureSpaceMode structure_space_mode = StructureSpaceMode::Standard;
  xmvb::vb::VbScfInputLoadOptions load_options;
  load_options.ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
  load_options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
  xmvb::vb::VbScfOptimizerOptions options;
  RunOptimizerBackend run_backend = RunOptimizerBackend::Core;
  xmvb::vb::AdaptiveStructureSpaceOptimizerOptions adaptive_options;
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
  bool user_specified_raw_structure_selection = false;
  bool user_specified_max_iterations = false;
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    try {
      if (argument_name == "--optimizer-backend") {
        apply_optimizer_backend_argument(argument_value, &options, &run_backend);
      } else if (argument_name == "--structure-space-mode") {
        apply_structure_space_mode_argument(argument_value, &structure_space_mode);
      } else if (argument_name == "--algorithm") {
        throw std::invalid_argument(
            "--algorithm is no longer supported; use the standard VBSCF input deck");
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
        user_specified_raw_structure_selection = true;
        apply_raw_structure_selection_argument(argument_value, &load_options);
      } else if (argument_name == "--adaptive-seed-selection") {
        apply_adaptive_seed_selection_argument(argument_value, &adaptive_options);
      } else if (argument_name == "--adaptive-determinant-score-mode") {
        apply_adaptive_determinant_score_mode_argument(argument_value, &adaptive_options);
      } else if (argument_name == "--adaptive-max-outer-iterations") {
        adaptive_options.max_outer_iterations = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-topology-distance") {
        adaptive_options.max_topology_distance = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-neighbors-per-structure") {
        adaptive_options.max_neighbors_per_structure = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-candidate-pool-size") {
        adaptive_options.max_candidate_pool_size = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-batch-size") {
        adaptive_options.batch_size = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-total-structures") {
        adaptive_options.max_total_structures = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-minimum-candidate-score") {
        adaptive_options.minimum_candidate_score = std::stod(argument_value);
      } else if (argument_name == "--adaptive-verbose") {
        adaptive_options.verbose = parse_bool_argument(argument_value);
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
        return 1;
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
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

  if (structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    if (uses_deepvbh_backend(run_backend)) {
      throw std::invalid_argument(
          "adaptive_mvp currently supports only standard C++ VBSCF optimizer backends");
    }
    if (!dump_trace_dir.empty()) {
      throw std::invalid_argument(
          "--dump-trace-dir is not yet supported with --structure-space-mode adaptive_mvp");
    }
    if (user_specified_raw_structure_selection &&
        load_options.raw_structure_selection != xmvb::vb::RawStructureSelectionMode::Full) {
      throw std::invalid_argument(
          "--raw-structure-selection is only for standard mode; "
          "use --adaptive-seed-selection in adaptive_mvp mode");
    }
    load_options.raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Full;
    load_options.expand_selected_raw_structures = false;
  }

  const auto command_start_time = std::chrono::system_clock::now();
  const auto command_start_steady_time = std::chrono::steady_clock::now();
  const auto load_result = xmvb::vb::load_vbscf_input_with_timings(input_path, load_options);
  const auto& input = load_result.input;
  if (!user_specified_max_iterations) {
    // Keep the standalone SCF loop aligned with the legacy deck semantics:
    // `.xmi` `itmax` controls the maximum iteration count, and omitted `itmax`
    // falls back to the project default of 2000.
    options.max_iterations = load_result.requested_scf_max_iterations;
  }

  print_run_header(
      input_path,
      load_result,
      options,
      run_backend,
      structure_space_mode,
      command_start_time);

  std::shared_ptr<xmvb::runtime::AcceptedIterationTraceWriter> trace_writer;
  if (options.verbose) {
    options.accepted_iteration_callback = compose_accepted_iteration_callbacks(
        std::move(options.accepted_iteration_callback),
        build_terminal_iteration_logger());
  }
  if (!dump_trace_dir.empty()) {
    trace_writer = std::make_shared<xmvb::runtime::AcceptedIterationTraceWriter>(
        dump_trace_dir,
        input_path,
        load_result,
        run_optimizer_backend_name(run_backend, options.backend));
    options.retain_accepted_iteration_trace = false;
    options.accepted_iteration_callback_requires_reference_gradient = true;
    options.accepted_iteration_callback_requires_full_snapshot = true;
    options.accepted_iteration_callback = compose_accepted_iteration_callbacks(
        std::move(options.accepted_iteration_callback),
        [trace_writer](const xmvb::vb::VbScfAcceptedIterationSnapshot& snapshot) {
          trace_writer->write_accepted_iteration(snapshot);
        });
  }

  deepvbh_options.optimizer_options = options;
  deepvbh_direct_options.optimizer_options = options;
  xmvb::vb::VbScfOptimizerResult result;
  std::optional<xmvb::vb::AdaptiveStructureSpaceOptimizerResult> adaptive_result;
  if (structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    xmvb::vb::AdaptiveStructureSpaceOptimizer optimizer(options, adaptive_options);
    adaptive_result = optimizer.optimize({
        load_result.input,
        load_result.raw_structure_data,
        load_result.nuclear_repulsion_energy});
    result = adaptive_result->inner_result;
  } else if (run_backend == RunOptimizerBackend::DeepVBHOnnx) {
    if (deepvbh_options.inference_options.onnx_model_path.empty()) {
      throw std::invalid_argument(
          "--onnx-model is required for --optimizer-backend deepvbh_onnx");
    }
    xmvb::vb::DeepVBHOnnxHybridOptimizer optimizer(deepvbh_options);
    result = optimizer.optimize(
        input,
        load_result.raw_structure_data,
        load_result.static_molecule_metadata,
        load_result.nuclear_repulsion_energy);
  } else if (
      run_backend == RunOptimizerBackend::DeepVBHOnnxDirectFinal) {
    if (deepvbh_direct_options.inference_options.onnx_model_path.empty()) {
      throw std::invalid_argument(
          "--onnx-model is required for --optimizer-backend deepvbh_onnx_direct_final");
    }
    xmvb::vb::DeepVBHOnnxDirectFinalOptimizer optimizer(deepvbh_direct_options);
    result = optimizer.optimize(
        input,
        load_result.raw_structure_data,
        load_result.static_molecule_metadata,
        load_result.nuclear_repulsion_energy);
  } else {
    xmvb::vb::VbScfOptimizer optimizer(options);
    result = optimizer.optimize(input, load_result.nuclear_repulsion_energy);
  }
  if (trace_writer != nullptr) {
    trace_writer->finalize(result);
  }
  if (!dump_final_orbital_value_table_bin.empty()) {
    // The final orbital table is the minimal state needed for reduced-chart
    // finite-difference diagnostics.  Keep this separate from trace dumping so
    // production convergence tests do not pay for per-iteration matrix dumps.
    xmvb::runtime::write_binary_container(
        fs::path(dump_final_orbital_value_table_bin),
        result.optimized_input.orbital_preparation_input.orbital_value_table);
  }
  const bool command_converged =
      adaptive_result.has_value() ? adaptive_result->converged : result.converged;
  std::optional<fs::path> molden_output_path;
  if (load_result.request_molden_output) {
    molden_output_path =
        xmvb::vb::write_molden_file(fs::path(input_path), result.optimized_input);
  }

  const double initial_electronic_energy =
      result.initial_total_energy - load_result.nuclear_repulsion_energy;
  const double final_electronic_energy =
      result.final_total_energy - load_result.nuclear_repulsion_energy;
  const double initial_valence_structure_eigenvalue =
      result.initial_total_energy -
      load_result.nuclear_repulsion_energy -
      result.initial_one_electron_reference_energy;
  const double final_valence_structure_eigenvalue =
      result.final_total_energy -
      load_result.nuclear_repulsion_energy -
      result.final_one_electron_reference_energy;
  const auto command_finish_time = std::chrono::system_clock::now();
  const double total_job_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - command_start_steady_time)
          .count();
  const double optimizer_wall_time_seconds =
      adaptive_result.has_value() ? adaptive_result->total_wall_time_seconds
                                  : result.total_wall_time_seconds;
  const int effective_thread_count = allocated_cpu_thread_count();

  print_log_subsection_title("SCF Summary");
  print_log_field("Status", convergence_status_name(command_converged));
  print_log_field(
      "Termination reason",
      adaptive_result.has_value() ? adaptive_result->termination_reason
                                  : result.termination_reason);
  print_log_field("Iterations", std::to_string(result.n_iterations));
  print_log_field("Start time", format_timestamp(command_start_time));
  print_log_field("Finish time", format_timestamp(command_finish_time));
  print_log_field(
      "Input preparation wall time",
      format_seconds(load_result.total_seconds));
  print_log_field(
      "SCF iteration wall time",
      format_seconds(optimizer_wall_time_seconds));
  print_log_field(
      "End-to-end wall time",
      format_seconds(total_job_wall_time_seconds));
  print_log_field(
      "Estimated OpenMP core-hours",
      format_core_hours(total_job_wall_time_seconds, effective_thread_count));
  print_log_field(
      "Raw-structure selection",
      xmvb::vb::raw_structure_selection_mode_name(load_result.raw_structure_selection));
  if (trace_writer != nullptr) {
    print_log_field("Trace sample directory", trace_writer->sample_directory().string());
  }
  if (molden_output_path.has_value()) {
    print_log_field("Molden output", molden_output_path->string());
  }
  if (run_backend == RunOptimizerBackend::DeepVBHOnnx) {
    print_log_field(
        "ONNX model",
        fs::absolute(deepvbh_options.inference_options.onnx_model_path).string());
  } else if (
      run_backend == RunOptimizerBackend::DeepVBHOnnxDirectFinal) {
    print_log_field(
        "ONNX model",
        fs::absolute(deepvbh_direct_options.inference_options.onnx_model_path).string());
  }

  if (adaptive_result.has_value()) {
    double adaptive_total_scoring_wall_time_seconds = 0.0;
    std::uint64_t adaptive_total_boundary_pair_evaluation_count = 0;
    for (const auto& iteration_summary : adaptive_result->iteration_summaries) {
      adaptive_total_scoring_wall_time_seconds +=
          iteration_summary.scoring_wall_time_seconds;
      adaptive_total_boundary_pair_evaluation_count +=
          iteration_summary.boundary_pair_evaluation_count;
    }
    const auto& last_iteration_summary = adaptive_result->iteration_summaries.back();
    print_log_field(
        "Adaptive seed selection",
        xmvb::vb::adaptive_structure_space_seed_selection_name(
            adaptive_options.seed_selection));
    print_log_field(
        "Adaptive determinant score",
        xmvb::vb::adaptive_determinant_score_mode_name(
            adaptive_options.determinant_score_mode));
    print_log_field(
        "Selected raw structures",
        std::to_string(adaptive_result->selected_raw_structure_indices.size()));
    print_log_field(
        "Expanded determinants",
        std::to_string(result.optimized_input.structure_data.alpha_det.size()));
    print_log_field(
        "Adaptive outer iterations",
        std::to_string(adaptive_result->iteration_summaries.size()));
    print_log_field(
        "Last proposal determinants",
        std::to_string(last_iteration_summary.unique_proposal_determinant_count));
    print_log_field(
        "Last outside determinants",
        std::to_string(last_iteration_summary.unique_outside_determinant_count));
    print_log_field(
        "Last shared determinants",
        std::to_string(last_iteration_summary.unique_shared_determinant_count));
    print_log_field(
        "Boundary pair evaluations",
        std::to_string(adaptive_total_boundary_pair_evaluation_count));
    print_log_field(
        "Adaptive scoring wall time",
        format_seconds(adaptive_total_scoring_wall_time_seconds));
  } else {
    print_log_field(
        "Selected raw structures",
        std::to_string(load_result.raw_structure_data.n_structures));
    print_log_field(
        "Expanded determinants",
        std::to_string(input.structure_data.alpha_det.size()));
  }

  print_log_subsection_title("Energy and Gradient");
  print_log_field(
      "Nuclear repulsion energy",
      format_fixed_double(load_result.nuclear_repulsion_energy, 12));
  print_log_field(
      "Initial total energy",
      format_fixed_double(result.initial_total_energy, 12));
  print_log_field(
      "Final total energy",
      format_fixed_double(result.final_total_energy, 12));
  print_log_field(
      "Total energy change",
      format_scientific_double(result.final_total_energy - result.initial_total_energy, 8));
  print_log_field(
      "Initial electronic energy",
      format_fixed_double(initial_electronic_energy, 12));
  print_log_field(
      "Final electronic energy",
      format_fixed_double(final_electronic_energy, 12));
  print_log_field(
      "Initial reference energy",
      format_fixed_double(result.initial_one_electron_reference_energy, 12));
  print_log_field(
      "Final reference energy",
      format_fixed_double(result.final_one_electron_reference_energy, 12));
  print_log_field(
      "Initial VB eigenvalue",
      format_fixed_double(initial_valence_structure_eigenvalue, 12));
  print_log_field(
      "Final VB eigenvalue",
      format_fixed_double(final_valence_structure_eigenvalue, 12));
  print_log_field(
      "Average structure overlap",
      format_fixed_double(result.scf_result.average_structure_overlap, 12));
  print_log_field(
      "Final gradient |g|_inf",
      format_scientific_double(result.final_gradient_inf_norm, 8));
  print_log_field(
      "Final gradient |g|_2",
      format_scientific_double(result.final_gradient_l2_norm, 8));
  if (run_backend == RunOptimizerBackend::Core &&
      core_backend_reports_projected_gradient(options.backend)) {
    print_log_field(
        "Final projected |g|_inf",
        format_scientific_double(result.final_projected_gradient_inf_norm, 8));
    print_log_field(
        "Final projected |g|_2",
        format_scientific_double(result.final_projected_gradient_l2_norm, 8));
  }
  if (run_backend == RunOptimizerBackend::Core &&
      options.backend ==
      xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton) {
    print_log_field(
        "Matrix-free HVP directions",
        std::to_string(result.matrix_free_hvp_direction_count));
    print_log_field(
        "Block-HVP calls",
        std::to_string(result.matrix_free_hvp_batch_count));
    print_log_field(
        "Inner solves meeting KKT target",
        std::to_string(result.matrix_free_residual_converged_count) + " / " +
            std::to_string(result.matrix_free_subproblem_count));
    print_log_field(
        "Matrix-free HVP wall time",
        format_seconds(result.matrix_free_hvp_wall_time_seconds));
  }
  print_log_subsection_title("Timing Breakdown (Wall Time)");
  print_log_field(
      "AO integral provider wall time",
      format_seconds(load_result.ao_integral_provider_seconds));
  print_log_field(
      "AO integral input wall time",
      format_seconds(load_result.ao_integral_input_build_seconds));
  print_log_field(
      "Orbital guess build wall time",
      format_seconds(load_result.orbital_guess_seconds));
  print_log_field(
      "Raw-structure selection wall",
      format_seconds(load_result.raw_structure_selection_seconds));
  print_log_field(
      "Structure expansion wall time",
      format_seconds(load_result.structure_expansion_seconds));
  print_log_field(
      "Input preparation wall time",
      format_seconds(load_result.total_seconds));
  print_log_field(
      "SCF iteration wall time",
      format_seconds(optimizer_wall_time_seconds));
  print_log_field(
      "End-to-end wall time",
      format_seconds(total_job_wall_time_seconds));
  print_log_rule('=');

  return command_converged ? 0 : 2;
}
