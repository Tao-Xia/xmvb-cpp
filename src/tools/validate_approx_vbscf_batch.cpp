#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/approx/approx_vbscf_evaluator.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<std::string> input_paths;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
};

struct TimedSampleResult {
  std::string input_path;
  int n_structures = 0;
  int n_active_orbitals = 0;
  int n_active_electrons = 0;
  int n_active_pairs = 0;
  double exact_total_energy = 0.0;
  double approximate_total_energy = 0.0;
  double diagnostic_metric_quotient_total_energy = 0.0;
  double cluster_quotient_total_energy = 0.0;
  double resonance_functional_total_energy = 0.0;
  double total_energy_error = 0.0;
  double diagnostic_metric_quotient_total_error = 0.0;
  double cluster_quotient_total_error = 0.0;
  double resonance_functional_total_error = 0.0;
  double total_metric_log = 0.0;
  double metric_denominator = 1.0;
  double mapped_weight_sum = 0.0;
  int local_term_count = 0;
  int pair_interaction_term_count = 0;
  int resonance_functional_pair_pair_term_count = 0;
  double load_seconds = 0.0;
  double exact_seconds = 0.0;
  double approx_seconds = 0.0;
};

void print_usage() {
  std::cerr << "usage: validate_approx_vbscf_batch <input.xmi> "
               "[input2.xmi ...] "
               "[--standard-two-electron-mode auto|exact|ri]\n";
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
    if (argument == "--standard-two-electron-mode") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(
            "--standard-two-electron-mode requires a value");
      }
      options.standard_two_electron_mode =
          parse_standard_two_electron_mode(argv[++argument_index]);
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

TimedSampleResult evaluate_sample(
    const std::string& input_path,
    xmvb::vb::StandardTwoElectronMode standard_two_electron_mode) {
  TimedSampleResult sample;
  sample.input_path = input_path;

  xmvb::vb::CppVbInputLoadOptions load_options;
  load_options.standard_two_electron_mode = standard_two_electron_mode;
  const auto load_start = Clock::now();
  const auto load_result =
      xmvb::vb::load_cpp_vb_input_with_timings(
          input_path,
          load_options);
  sample.load_seconds = elapsed_seconds(load_start);

  sample.n_structures = load_result.raw_structure_data.n_structures;
  sample.n_active_orbitals =
      load_result.input.orbital_preparation_input.n_active_orbitals;
  sample.n_active_electrons =
      load_result.input.orbital_preparation_input.n_active_electrons;

  xmvb::vb::StructureMatrixEvaluator matrix_evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto approx_start = Clock::now();
  const auto prepared_active_space =
      matrix_evaluator.prepare_active_space(load_result.input);
  xmvb::vb::ApproxVbScfModelInput model_input;
  model_input.n_active_orbitals = sample.n_active_orbitals;
  model_input.one_electron_reference_energy =
      prepared_active_space.one_electron_reference_energy;
  model_input.nuclear_repulsion_energy =
      load_result.nuclear_repulsion_energy;
  model_input.active_one_electron_integrals =
      &prepared_active_space.active_space_one_electron_result.h1e_act;
  model_input.active_orbital_overlap_matrix =
      &prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  model_input.active_two_electron_result =
      &prepared_active_space.active_space_two_electron_result;
  const auto model = xmvb::vb::build_approx_vbscf_model(model_input);
  sample.approx_seconds = elapsed_seconds(approx_start);

  const auto exact_start = Clock::now();
  xmvb::vb::CppVbScfEvaluator scf_evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto scf_result =
      scf_evaluator.evaluate(
          load_result.input,
          load_result.nuclear_repulsion_energy);
  xmvb::vb::ApproxVbScfStateMappingInput mapping_input;
  mapping_input.n_active_orbitals = sample.n_active_orbitals;
  mapping_input.raw_structure_data = &load_result.raw_structure_data;
  mapping_input.structure_overlap_matrix =
      &scf_result.structure_matrices.overlap_matrix;
  mapping_input.eigenvector_matrix = &scf_result.eigenvector_matrix;
  const auto mapped_state =
      xmvb::vb::map_selected_state_to_pair_state(mapping_input);
  sample.exact_seconds = elapsed_seconds(exact_start);

  const auto eval_start = Clock::now();
  const auto approx_result =
      xmvb::vb::evaluate_approx_vbscf(
          model,
          mapped_state.pair_occupations,
          mapped_state.pair_pair_occupations);
  sample.approx_seconds += elapsed_seconds(eval_start);

  sample.n_active_pairs = approx_result.n_active_pairs;
  sample.exact_total_energy = scf_result.total_energy;
  sample.approximate_total_energy = approx_result.approximate_total_energy;
  sample.diagnostic_metric_quotient_total_energy =
      approx_result.diagnostic_metric_quotient_total_energy;
  sample.cluster_quotient_total_energy =
      approx_result.cluster_quotient_total_energy;
  sample.resonance_functional_total_energy =
      approx_result.resonance_functional_total_energy;
  sample.total_energy_error =
      sample.approximate_total_energy - sample.exact_total_energy;
  sample.diagnostic_metric_quotient_total_error =
      sample.diagnostic_metric_quotient_total_energy -
      sample.exact_total_energy;
  sample.cluster_quotient_total_error =
      sample.cluster_quotient_total_energy -
      sample.exact_total_energy;
  sample.resonance_functional_total_error =
      sample.resonance_functional_total_energy -
      sample.exact_total_energy;
  sample.total_metric_log = approx_result.total_metric_log;
  sample.metric_denominator = approx_result.metric_denominator;
  sample.mapped_weight_sum = mapped_state.mapped_weight_sum;
  sample.local_term_count = approx_result.local_term_count;
  sample.pair_interaction_term_count =
      approx_result.pair_interaction_term_count;
  sample.resonance_functional_pair_pair_term_count =
      approx_result.resonance_functional_pair_pair_term_count;
  return sample;
}

void print_header() {
  std::cout
      << "input"
      << '\t' << "n_structures"
      << '\t' << "n_active_orbitals"
      << '\t' << "n_active_electrons"
      << '\t' << "n_active_pairs"
      << '\t' << "exact_total"
      << '\t' << "approximate_total"
      << '\t' << "diagnostic_metric_quotient_total"
      << '\t' << "cluster_quotient_total"
      << '\t' << "resonance_functional_total"
      << '\t' << "total_error"
      << '\t' << "diagnostic_metric_quotient_error"
      << '\t' << "cluster_quotient_error"
      << '\t' << "resonance_functional_error"
      << '\t' << "total_metric_log"
      << '\t' << "metric_denominator"
      << '\t' << "mapped_weight_sum"
      << '\t' << "local_terms"
      << '\t' << "pair_terms"
      << '\t' << "resonance_pair_terms"
      << '\t' << "load_s"
      << '\t' << "exact_s"
      << '\t' << "approx_s"
      << '\n';
}

void print_sample(const TimedSampleResult& sample) {
  std::cout
      << sample.input_path
      << '\t' << sample.n_structures
      << '\t' << sample.n_active_orbitals
      << '\t' << sample.n_active_electrons
      << '\t' << sample.n_active_pairs
      << '\t' << sample.exact_total_energy
      << '\t' << sample.approximate_total_energy
      << '\t' << sample.diagnostic_metric_quotient_total_energy
      << '\t' << sample.cluster_quotient_total_energy
      << '\t' << sample.resonance_functional_total_energy
      << '\t' << sample.total_energy_error
      << '\t' << sample.diagnostic_metric_quotient_total_error
      << '\t' << sample.cluster_quotient_total_error
      << '\t' << sample.resonance_functional_total_error
      << '\t' << sample.total_metric_log
      << '\t' << sample.metric_denominator
      << '\t' << sample.mapped_weight_sum
      << '\t' << sample.local_term_count
      << '\t' << sample.pair_interaction_term_count
      << '\t' << sample.resonance_functional_pair_pair_term_count
      << '\t' << sample.load_seconds
      << '\t' << sample.exact_seconds
      << '\t' << sample.approx_seconds
      << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    std::cout << std::setprecision(12);
    print_header();
    for (const std::string& input_path : options.input_paths) {
      print_sample(
          evaluate_sample(
              input_path,
              options.standard_two_electron_mode));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
