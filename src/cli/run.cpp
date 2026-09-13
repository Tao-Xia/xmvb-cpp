#include "cli/run.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "output/text/report.hpp"
#include "output/trace/binary.hpp"
#include "output/molden/writer.hpp"
#include "output/trace/accepted_iteration.hpp"
#include "output/trace/tnhvp.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/optimizer.hpp"

namespace xmvb::cli {

namespace fs = std::filesystem;

int run(Options command_line) {
  const std::string& input_path = command_line.input_path;
  auto& load_options = command_line.load;
  auto& options = command_line.optimizer;
  const std::string& dump_trace_dir = command_line.trace_directory;
  const std::string& tnhvp_trace_path = command_line.tnhvp_trace_path;
  const std::string& dump_final_orbital_value_table_bin =
      command_line.final_orbitals_path;
  const bool user_specified_max_iterations =
      command_line.max_iterations_explicit;

  const auto command_start_time = std::chrono::system_clock::now();
  const auto command_start_steady_time = std::chrono::steady_clock::now();
  auto load_result =
      xmvb::vb::load_vbscf_input_with_timings(input_path, load_options);
  if (!command_line.optimizer_backend_explicit) {
    switch (load_result.scf_optimizer) {
      case xmvb::vb::InputScfOptimizer::Lbfgs:
        options.backend =
            xmvb::vb::VbScfOptimizerBackend::NonredundantLbfgspp;
        break;
      case xmvb::vb::InputScfOptimizer::Tnhvp:
        options.backend =
            xmvb::vb::VbScfOptimizerBackend::NonredundantTruncatedNewton;
        break;
      case xmvb::vb::InputScfOptimizer::Unspecified:
        break;
    }
  }
  if (!user_specified_max_iterations) {
    // Keep the standalone SCF loop aligned with the input deck semantics:
    // `.xmi` `itmax` controls the maximum iteration count, and omitted `itmax`
    // falls back to the project default of 2000.
    options.max_iterations = load_result.requested_scf_max_iterations;
  }

  xmvb::output::print_header(
      input_path,
      options,
      load_result,
      command_start_time);

  std::shared_ptr<xmvb::output::AcceptedIterationTraceWriter> trace_writer;
  if (options.verbose) {
    options.accepted_iteration_callback = xmvb::output::combine_callbacks(
        std::move(options.accepted_iteration_callback),
        xmvb::output::iteration_logger());
  }
  if (!dump_trace_dir.empty()) {
    trace_writer = std::make_shared<xmvb::output::AcceptedIterationTraceWriter>(
        dump_trace_dir,
        input_path,
        load_result,
        xmvb::vb::vbscf_optimizer_backend_name(options.backend));
    options.retain_accepted_iteration_trace = false;
    options.accepted_iteration_callback_requires_reference_gradient = true;
    options.accepted_iteration_callback_requires_full_snapshot = true;
    options.accepted_iteration_callback = xmvb::output::combine_callbacks(
        std::move(options.accepted_iteration_callback),
        [trace_writer](const xmvb::vb::VbScfAcceptedIterationSnapshot& snapshot) {
          trace_writer->write_accepted_iteration(snapshot);
        });
  }

  xmvb::vb::VbScfOptimizer optimizer(options);
  const xmvb::vb::VbScfOptimizerResult result =
      optimizer.optimize(
          std::move(load_result.input),
          load_result.nuclear_repulsion_energy);
  if (!tnhvp_trace_path.empty()) {
    xmvb::output::write_tnhvp_trace(tnhvp_trace_path, result);
  }
  if (trace_writer != nullptr) {
    trace_writer->finalize(result);
  }
  if (!dump_final_orbital_value_table_bin.empty()) {
    // The final orbital table is the minimal state needed for reduced-chart
    // finite-difference diagnostics.  Keep this separate from trace dumping so
    // production convergence tests do not pay for per-iteration matrix dumps.
    xmvb::output::write_binary_container(
        fs::path(dump_final_orbital_value_table_bin),
        result.optimized_input.orbital_preparation_input.orbital_value_table);
  }
  const bool command_converged = result.converged;
  std::optional<fs::path> molden_output_path;
  if (load_result.request_molden_output) {
    molden_output_path =
        xmvb::vb::write_molden_file(fs::path(input_path), result.optimized_input);
  }

  std::optional<fs::path> trace_sample_directory;
  if (trace_writer != nullptr) {
    trace_sample_directory = trace_writer->sample_directory();
  }
  xmvb::output::print_summary(
      options,
      load_result,
      result,
      trace_sample_directory,
      molden_output_path,
      command_start_time,
      command_start_steady_time);

  return command_converged ? 0 : 2;
}

}  // namespace xmvb::cli
