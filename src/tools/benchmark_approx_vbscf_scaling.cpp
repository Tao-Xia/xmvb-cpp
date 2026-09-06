#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/approx/approx_vbscf_evaluator.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<std::string> input_paths;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int pair_scf_iterations = 200;
  double pair_scf_step_size = 0.05;
  bool skip_exact = false;
};

struct SampleTimingResult {
  std::string input_path;
  int n_structures = 0;
  int n_active_orbitals = 0;
  int n_active_electrons = 0;
  int n_active_pairs = 0;
  int pair_interaction_terms = 0;
  int resonance_pair_terms = 0;
  double occupied_pair_count = 0.0;
  double load_seconds = 0.0;
  double active_prepare_seconds = 0.0;
  double approx_model_seconds = 0.0;
  double approx_label_eval_seconds = 0.0;
  double pair_scf_seconds = 0.0;
  double exact_matrix_seconds = 0.0;
  double exact_diagonalization_seconds = 0.0;
  double exact_total_seconds = 0.0;
  double exact_total_energy = 0.0;
  double approx_label_total_energy = 0.0;
  double approx_label_diagnostic_metric_quotient_total_energy = 0.0;
  double approx_label_cluster_quotient_total_energy = 0.0;
  double approx_label_resonance_functional_total_energy = 0.0;
  double pair_scf_total_energy = 0.0;
  double pair_scf_diagnostic_metric_quotient_total_energy = 0.0;
  double pair_scf_cluster_quotient_total_energy = 0.0;
  double pair_scf_resonance_functional_total_energy = 0.0;
  double approx_label_total_error = 0.0;
  double approx_label_diagnostic_metric_quotient_total_error = 0.0;
  double approx_label_cluster_quotient_total_error = 0.0;
  double approx_label_resonance_functional_total_error = 0.0;
  double pair_scf_total_error = 0.0;
  double pair_scf_diagnostic_metric_quotient_total_error = 0.0;
  double pair_scf_cluster_quotient_total_error = 0.0;
  double pair_scf_resonance_functional_total_error = 0.0;
  double pair_scf_total_metric_log = 0.0;
  double pair_scf_metric_denominator = 1.0;
  double mapped_weight_sum = 0.0;
  int pair_scf_iterations = 0;
  bool pair_scf_converged = false;
  bool exact_evaluated = false;
};

void print_usage() {
  std::cerr << "usage: benchmark_approx_vbscf_scaling <input.xmi> "
               "[input2.xmi ...] "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--pair-scf-iterations N] "
               "[--pair-scf-step-size X] "
               "[--skip-exact]\n";
}

double elapsed_seconds(Clock::time_point start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

xmvb::vb::StandardTwoElectronMode parse_standard_two_electron_mode(
    const std::string& value) {
  if (value == "auto") {
    return xmvb::vb::StandardTwoElectronMode::Auto;
  }
  if (value == "exact") {
    return xmvb::vb::StandardTwoElectronMode::Exact;
  }
  if (value == "ri") {
    return xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
  }
  throw std::invalid_argument(
      "invalid standard two-electron mode: " + value);
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("missing input path");
  }

  Options options;
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    if (argument == "--skip-exact") {
      options.skip_exact = true;
      continue;
    }
    if (argument == "--standard-two-electron-mode") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(
            "--standard-two-electron-mode requires a value");
      }
      options.standard_two_electron_mode =
          parse_standard_two_electron_mode(argv[++argument_index]);
      continue;
    }
    if (argument == "--pair-scf-iterations") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--pair-scf-iterations requires a value");
      }
      options.pair_scf_iterations = std::stoi(argv[++argument_index]);
      continue;
    }
    if (argument == "--pair-scf-step-size") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--pair-scf-step-size requires a value");
      }
      options.pair_scf_step_size = std::stod(argv[++argument_index]);
      continue;
    }
    if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("unknown argument: " + argument);
    }
    options.input_paths.push_back(argument);
  }

  if (options.input_paths.empty()) {
    throw std::invalid_argument("missing input path");
  }
  return options;
}

double occupied_active_pair_count(
    int n_active_electrons,
    int spin_multiplicity) {
  const int n_open_shell_electrons = spin_multiplicity - 1;
  return 0.5 * static_cast<double>(
      n_active_electrons - n_open_shell_electrons);
}

xmvb::vb::ApproxVbScfModelInput make_model_input(
    int n_active_orbitals,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    double nuclear_repulsion_energy) {
  xmvb::vb::ApproxVbScfModelInput model_input;
  model_input.n_active_orbitals = n_active_orbitals;
  model_input.one_electron_reference_energy =
      prepared_active_space.one_electron_reference_energy;
  model_input.nuclear_repulsion_energy = nuclear_repulsion_energy;
  model_input.active_one_electron_integrals =
      &prepared_active_space.active_space_one_electron_result.h1e_act;
  model_input.active_orbital_overlap_matrix =
      &prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  model_input.active_two_electron_result =
      &prepared_active_space.active_space_two_electron_result;
  return model_input;
}

SampleTimingResult benchmark_sample(
    const std::string& input_path,
    const Options& options) {
  SampleTimingResult result;
  result.input_path = input_path;

  xmvb::vb::CppVbInputLoadOptions load_options;
  load_options.standard_two_electron_mode =
      options.standard_two_electron_mode;
  const auto load_start = Clock::now();
  const auto load_result =
      xmvb::vb::load_cpp_vb_input_with_timings(
          input_path,
          load_options);
  result.load_seconds = elapsed_seconds(load_start);

  result.n_structures = load_result.raw_structure_data.n_structures;
  result.n_active_orbitals =
      load_result.input.orbital_preparation_input.n_active_orbitals;
  result.n_active_electrons =
      load_result.input.orbital_preparation_input.n_active_electrons;
  result.occupied_pair_count =
      occupied_active_pair_count(
          result.n_active_electrons,
          load_result.raw_structure_data.spin_multiplicity);

  xmvb::vb::StructureMatrixEvaluator matrix_evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto active_prepare_start = Clock::now();
  const auto prepared_active_space =
      matrix_evaluator.prepare_active_space(load_result.input);
  result.active_prepare_seconds =
      elapsed_seconds(active_prepare_start);

  const auto model_input =
      make_model_input(
          result.n_active_orbitals,
          prepared_active_space,
          load_result.nuclear_repulsion_energy);
  const auto approx_model_start = Clock::now();
  const auto model =
      xmvb::vb::build_approx_vbscf_model(model_input);
  result.approx_model_seconds =
      elapsed_seconds(approx_model_start);
  result.n_active_pairs = model.n_active_pairs;
  result.pair_interaction_terms =
      model.n_active_pairs * (model.n_active_pairs - 1) / 2;
  result.resonance_pair_terms =
      model.resonance_functional_model.compatible_pair_pair_term_count;

  xmvb::vb::ApproxVbScfPairScfOptions pair_scf_options;
  pair_scf_options.max_iterations = options.pair_scf_iterations;
  pair_scf_options.step_size = options.pair_scf_step_size;
  const auto pair_scf_start = Clock::now();
  const auto pair_scf_result =
      xmvb::vb::optimize_approx_vbscf_pair_occupations(
          model,
          result.occupied_pair_count,
          pair_scf_options);
  result.pair_scf_seconds = elapsed_seconds(pair_scf_start);
  result.pair_scf_total_energy =
      pair_scf_result.result.approximate_total_energy;
  result.pair_scf_diagnostic_metric_quotient_total_energy =
      pair_scf_result.result.diagnostic_metric_quotient_total_energy;
  result.pair_scf_cluster_quotient_total_energy =
      pair_scf_result.result.cluster_quotient_total_energy;
  result.pair_scf_resonance_functional_total_energy =
      pair_scf_result.result.resonance_functional_total_energy;
  result.pair_scf_total_metric_log =
      pair_scf_result.result.total_metric_log;
  result.pair_scf_metric_denominator =
      pair_scf_result.result.metric_denominator;
  result.pair_scf_iterations = pair_scf_result.iterations;
  result.pair_scf_converged = pair_scf_result.converged;

  if (!options.skip_exact) {
    const auto exact_matrix_start = Clock::now();
    const auto structure_matrices =
        matrix_evaluator.evaluate(
            load_result.input,
            prepared_active_space);
    result.exact_matrix_seconds = elapsed_seconds(exact_matrix_start);

    const auto diagonalization_start = Clock::now();
    xmvb::core::GeneralizedEigensolver eigensolver;
    const auto eigen_result =
        eigensolver.solve(
            structure_matrices.hamiltonian_matrix,
            structure_matrices.overlap_matrix,
            structure_matrices.n_structures);
    result.exact_diagonalization_seconds =
        elapsed_seconds(diagonalization_start);
    result.exact_total_seconds =
        result.exact_matrix_seconds +
        result.exact_diagonalization_seconds;
    result.exact_total_energy =
        prepared_active_space.one_electron_reference_energy +
        eigen_result.eigenvalues.front() +
        load_result.nuclear_repulsion_energy;

    const auto approx_label_eval_start = Clock::now();
    xmvb::vb::ApproxVbScfStateMappingInput mapping_input;
    mapping_input.n_active_orbitals = result.n_active_orbitals;
    mapping_input.raw_structure_data = &load_result.raw_structure_data;
    mapping_input.structure_overlap_matrix =
        &structure_matrices.overlap_matrix;
    mapping_input.eigenvector_matrix = &eigen_result.eigenvector_matrix;
    const auto mapped_state =
        xmvb::vb::map_selected_state_to_pair_state(mapping_input);
    const auto approx_label_result =
        xmvb::vb::evaluate_approx_vbscf(
            model,
            mapped_state.pair_occupations,
            mapped_state.pair_pair_occupations);
    result.approx_label_eval_seconds =
        elapsed_seconds(approx_label_eval_start);
    result.approx_label_total_energy =
        approx_label_result.approximate_total_energy;
    result.approx_label_diagnostic_metric_quotient_total_energy =
        approx_label_result.diagnostic_metric_quotient_total_energy;
    result.approx_label_cluster_quotient_total_energy =
        approx_label_result.cluster_quotient_total_energy;
    result.approx_label_resonance_functional_total_energy =
        approx_label_result.resonance_functional_total_energy;
    result.approx_label_total_error =
        result.approx_label_total_energy -
        result.exact_total_energy;
    result.approx_label_diagnostic_metric_quotient_total_error =
        result.approx_label_diagnostic_metric_quotient_total_energy -
        result.exact_total_energy;
    result.approx_label_cluster_quotient_total_error =
        result.approx_label_cluster_quotient_total_energy -
        result.exact_total_energy;
    result.approx_label_resonance_functional_total_error =
        result.approx_label_resonance_functional_total_energy -
        result.exact_total_energy;
    result.pair_scf_total_error =
        result.pair_scf_total_energy -
        result.exact_total_energy;
    result.pair_scf_diagnostic_metric_quotient_total_error =
        result.pair_scf_diagnostic_metric_quotient_total_energy -
        result.exact_total_energy;
    result.pair_scf_cluster_quotient_total_error =
        result.pair_scf_cluster_quotient_total_energy -
        result.exact_total_energy;
    result.pair_scf_resonance_functional_total_error =
        result.pair_scf_resonance_functional_total_energy -
        result.exact_total_energy;
    result.mapped_weight_sum = mapped_state.mapped_weight_sum;
    result.exact_evaluated = true;
  }

  return result;
}

void print_header() {
  std::cout
      << "input"
      << '\t' << "n_structures"
      << '\t' << "n_active_orbitals"
      << '\t' << "n_active_electrons"
      << '\t' << "n_active_pairs"
      << '\t' << "pair_terms"
      << '\t' << "resonance_pair_terms"
      << '\t' << "occupied_pair_count"
      << '\t' << "load_s"
      << '\t' << "active_prepare_s"
      << '\t' << "approx_model_s"
      << '\t' << "approx_label_eval_s"
      << '\t' << "pair_scf_s"
      << '\t' << "exact_matrix_s"
      << '\t' << "exact_diag_s"
      << '\t' << "exact_total_s"
      << '\t' << "exact_total"
      << '\t' << "approx_label_total"
      << '\t' << "approx_label_diag_metric_total"
      << '\t' << "approx_label_cluster_quotient_total"
      << '\t' << "approx_label_resonance_functional_total"
      << '\t' << "pair_scf_total"
      << '\t' << "pair_scf_diag_metric_total"
      << '\t' << "pair_scf_cluster_quotient_total"
      << '\t' << "pair_scf_resonance_functional_total"
      << '\t' << "approx_label_error"
      << '\t' << "approx_label_diag_metric_error"
      << '\t' << "approx_label_cluster_quotient_error"
      << '\t' << "approx_label_resonance_functional_error"
      << '\t' << "pair_scf_error"
      << '\t' << "pair_scf_diag_metric_error"
      << '\t' << "pair_scf_cluster_quotient_error"
      << '\t' << "pair_scf_resonance_functional_error"
      << '\t' << "pair_scf_metric_log"
      << '\t' << "pair_scf_metric_denominator"
      << '\t' << "mapped_weight_sum"
      << '\t' << "pair_scf_iterations"
      << '\t' << "pair_scf_converged"
      << '\n';
}

void print_sample(const SampleTimingResult& sample) {
  std::cout
      << sample.input_path
      << '\t' << sample.n_structures
      << '\t' << sample.n_active_orbitals
      << '\t' << sample.n_active_electrons
      << '\t' << sample.n_active_pairs
      << '\t' << sample.pair_interaction_terms
      << '\t' << sample.resonance_pair_terms
      << '\t' << sample.occupied_pair_count
      << '\t' << sample.load_seconds
      << '\t' << sample.active_prepare_seconds
      << '\t' << sample.approx_model_seconds
      << '\t' << sample.approx_label_eval_seconds
      << '\t' << sample.pair_scf_seconds
      << '\t' << sample.exact_matrix_seconds
      << '\t' << sample.exact_diagonalization_seconds
      << '\t' << sample.exact_total_seconds
      << '\t' << sample.exact_total_energy
      << '\t' << sample.approx_label_total_energy
      << '\t' << sample.approx_label_diagnostic_metric_quotient_total_energy
      << '\t' << sample.approx_label_cluster_quotient_total_energy
      << '\t' << sample.approx_label_resonance_functional_total_energy
      << '\t' << sample.pair_scf_total_energy
      << '\t' << sample.pair_scf_diagnostic_metric_quotient_total_energy
      << '\t' << sample.pair_scf_cluster_quotient_total_energy
      << '\t' << sample.pair_scf_resonance_functional_total_energy
      << '\t' << sample.approx_label_total_error
      << '\t' << sample.approx_label_diagnostic_metric_quotient_total_error
      << '\t' << sample.approx_label_cluster_quotient_total_error
      << '\t' << sample.approx_label_resonance_functional_total_error
      << '\t' << sample.pair_scf_total_error
      << '\t' << sample.pair_scf_diagnostic_metric_quotient_total_error
      << '\t' << sample.pair_scf_cluster_quotient_total_error
      << '\t' << sample.pair_scf_resonance_functional_total_error
      << '\t' << sample.pair_scf_total_metric_log
      << '\t' << sample.pair_scf_metric_denominator
      << '\t' << sample.mapped_weight_sum
      << '\t' << sample.pair_scf_iterations
      << '\t' << (sample.pair_scf_converged ? "true" : "false")
      << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    std::cout << std::setprecision(12);
    print_header();
    for (const std::string& input_path : options.input_paths) {
      print_sample(benchmark_sample(input_path, options));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
