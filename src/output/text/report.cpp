#include "output/text/report.hpp"
#include "output/text/sections.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::output {

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
  throw std::invalid_argument("invalid VBSCF optimizer backend");
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
  throw std::invalid_argument("invalid VBSCF optimizer backend");
}

constexpr int kLogLabelWidth = 34;

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

std::string physical_chart_name(
    const xmvb::vb::OrbitalPreparationInput& orbital_input) {
  for (const int basis_count : orbital_input.orbital_basis_counts) {
    if (basis_count > 0 && basis_count < orbital_input.n_basis_functions) {
      return "strict_sparse_U_p";
    }
  }
  return "full_ao_U_p";
}

void print_tnhvp_summary(
    const xmvb::vb::VbScfOptimizerOptions& options,
    const xmvb::vb::VbScfInput& input) {
  if (options.backend !=
      xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton) {
    return;
  }

  const auto& orbital_input = input.orbital_preparation_input;
  print_log_subsection_title("TNHVP Matrix-Free Newton");
  print_log_field(
      "Hessian model",
      "full_exact_hessian");
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
      physical_chart_name(orbital_input));
}

void print_header(
    const std::string& input_path,
    const xmvb::vb::VbScfOptimizerOptions& options,
    const xmvb::vb::VbScfInputLoadResult& load_result,
    const std::chrono::system_clock::time_point& start_time) {
  print_program_preamble(
      std::cout,
      input_path,
      start_time,
      allocated_cpu_thread_count());
  print_input_sections(
      std::cout,
      input_path,
      load_result,
      xmvb::vb::vbscf_optimizer_backend_name(options.backend),
      options.max_iterations);
}

std::function<void(const xmvb::vb::VbScfAcceptedIterationSnapshot&)>
combine_callbacks(
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
iteration_logger() {
  struct LoggerState {
    bool has_reference_energy = false;
    double previous_total_energy = 0.0;
  };

  auto state = std::make_shared<LoggerState>();
  std::cout << "\n                ITER           ENERGY               DE"
               "              GNORM\n";
  std::cout.flush();

  return [state](const xmvb::vb::VbScfAcceptedIterationSnapshot& snapshot) {
    const double delta_energy = state->has_reference_energy
        ? snapshot.total_energy - state->previous_total_energy
        : snapshot.total_energy;
    std::ostringstream stream;
    stream << std::setw(19) << snapshot.accepted_iteration_index
           << std::setw(22) << std::fixed << std::setprecision(10)
           << snapshot.total_energy
           << std::setw(18) << std::fixed << std::setprecision(10)
           << delta_energy
           << std::setw(18) << std::fixed << std::setprecision(10)
           << snapshot.sparse_orbital_energy_gradient_l2_norm
           << '\n';
    std::cout << stream.str();
    std::cout.flush();
    state->previous_total_energy = snapshot.total_energy;
    state->has_reference_energy = true;
  };
}


void print_summary(
    const vb::VbScfOptimizerOptions& options,
    const vb::VbScfInputLoadResult& load_result,
    const vb::VbScfOptimizerResult& result,
    const std::optional<std::filesystem::path>& trace_sample_directory,
    const std::optional<std::filesystem::path>& molden_output_path,
    const std::chrono::system_clock::time_point& command_start_time,
    const std::chrono::steady_clock::time_point& command_start_steady_time) {
  const auto& input = result.optimized_input;
  const bool command_converged = result.converged;
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
  const double optimizer_wall_time_seconds = result.total_wall_time_seconds;
  const int effective_thread_count = allocated_cpu_thread_count();

  std::cout << "\n"
            << (command_converged ? "                        VBSCF converged in "
                                  : "                    VBSCF did not converge in ")
            << std::setw(5) << result.n_iterations << " iterations\n\n"
            << "                  Total Energy:   "
            << std::fixed << std::setprecision(10) << result.final_total_energy
            << '\n';
  print_final_state_sections(std::cout, load_result, result);

  std::cout << "\n\n                 ===============================================\n"
            << "                       XMVB-CPP RUN DIAGNOSTICS\n"
            << "                 ===============================================\n";
  print_log_field("Termination reason", result.termination_reason);
  print_log_field("Start time", format_timestamp(command_start_time));
  print_log_field("Finish time", format_timestamp(command_finish_time));
  print_log_field("CPU threads", std::to_string(effective_thread_count));
  if (trace_sample_directory.has_value()) {
    print_log_field("Trace sample directory", trace_sample_directory->string());
  }
  if (molden_output_path.has_value()) {
    print_log_field("Molden output", molden_output_path->string());
  }
  print_log_field("Gradient metric", gradient_tolerance_metric_name(options.backend));
  print_log_field(
      "Convergence threshold",
      format_convergence_threshold_summary(
          options.energy_tolerance,
          options.gradient_tolerance));
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
  if (core_backend_reports_projected_gradient(options.backend)) {
    print_log_field(
        "Final projected |g|_inf",
        format_scientific_double(result.final_projected_gradient_inf_norm, 8));
    print_log_field(
        "Final projected |g|_2",
        format_scientific_double(result.final_projected_gradient_l2_norm, 8));
  }
  if (options.backend ==
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
    print_log_field(
        "Trial structure-energy solves",
        std::to_string(result.energy_only_evaluation_count));
    print_log_field(
        "Trial structure-energy wall time",
        format_seconds(result.energy_only_wall_time_seconds));
  }
  print_tnhvp_summary(options, input);
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
  std::cout << "\n        Cpu time for the job: "
            << std::fixed << std::setprecision(3)
            << total_job_wall_time_seconds << " seconds.\n";

}

}  // namespace xmvb::output
