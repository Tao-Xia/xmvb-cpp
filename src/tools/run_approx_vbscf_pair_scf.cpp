#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/approx/approx_vbscf_evaluator.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int max_iterations = 200;
  double step_size = 0.05;
  double gradient_tolerance = 1.0e-8;
  int top_pairs = 12;
  bool compare_exact = false;
};

void print_usage() {
  std::cerr << "usage: run_approx_vbscf_pair_scf <input.xmi> "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--max-iterations N] "
               "[--step-size X] "
               "[--gradient-tolerance X] "
               "[--top-pairs N] "
               "[--compare-exact]\n";
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
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    if (argument == "--compare-exact") {
      options.compare_exact = true;
      continue;
    }
    if (argument_index + 1 >= argc) {
      throw std::invalid_argument(argument + " requires a value");
    }
    const std::string value = argv[++argument_index];
    if (argument == "--standard-two-electron-mode") {
      options.standard_two_electron_mode =
          parse_standard_two_electron_mode(value);
    } else if (argument == "--max-iterations") {
      options.max_iterations = std::stoi(value);
    } else if (argument == "--step-size") {
      options.step_size = std::stod(value);
    } else if (argument == "--gradient-tolerance") {
      options.gradient_tolerance = std::stod(value);
    } else if (argument == "--top-pairs") {
      options.top_pairs = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }

  if (options.max_iterations < 0) {
    throw std::invalid_argument("--max-iterations must be non-negative");
  }
  if (options.top_pairs < 0) {
    throw std::invalid_argument("--top-pairs must be non-negative");
  }
  return options;
}

std::pair<int, int> unpack_pair(int pair_index) {
  int second = 0;
  while ((second + 1) * (second + 2) / 2 <= pair_index) {
    ++second;
  }
  const int first = pair_index - second * (second + 1) / 2;
  return {first, second};
}

double occupied_active_pair_count(
    int n_active_electrons,
    int spin_multiplicity) {
  const int n_open_shell_electrons = spin_multiplicity - 1;
  return 0.5 * static_cast<double>(
      n_active_electrons - n_open_shell_electrons);
}

std::vector<int> rank_pairs_by_occupation(
    const std::vector<double>& pair_occupations) {
  std::vector<int> order(pair_occupations.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(
      order.begin(),
      order.end(),
      [&pair_occupations](int left, int right) {
        const double left_abs = std::abs(pair_occupations[left]);
        const double right_abs = std::abs(pair_occupations[right]);
        if (left_abs != right_abs) {
          return left_abs > right_abs;
        }
        return left < right;
      });
  return order;
}

void print_pair_occupations(
    const xmvb::vb::ApproxVbScfPairScfResult& result,
    int top_pairs) {
  const std::vector<int> order =
      rank_pairs_by_occupation(result.pair_occupations);
  const int count =
      std::min(top_pairs, static_cast<int>(order.size()));
  std::cout << "top_pair_occupation_count = " << count << '\n';
  for (int rank = 0; rank < count; ++rank) {
    const int pair_index = order[rank];
    const auto [first, second] = unpack_pair(pair_index);
    std::cout << "pair[" << rank << "]"
              << " index=" << pair_index
              << " orbitals=(" << first << "," << second << ")"
              << " occupation=" << result.pair_occupations[pair_index]
              << " local_energy="
              << result.result.pair_local_energy_values[pair_index]
              << '\n';
  }
}

void print_exact_comparison(
    const xmvb::vb::CppVbInputLoadResult& load_result,
    const xmvb::vb::ApproxVbScfModel& model,
    const xmvb::vb::ApproxVbScfPairScfResult& pair_scf_result) {
  xmvb::vb::CppVbScfEvaluator scf_evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto scf_result =
      scf_evaluator.evaluate(
          load_result.input,
          load_result.nuclear_repulsion_energy);
  xmvb::vb::ApproxVbScfStateMappingInput mapping_input;
  mapping_input.n_active_orbitals = model.n_active_orbitals;
  mapping_input.raw_structure_data = &load_result.raw_structure_data;
  mapping_input.structure_overlap_matrix =
      &scf_result.structure_matrices.overlap_matrix;
  mapping_input.eigenvector_matrix = &scf_result.eigenvector_matrix;
  const auto mapped_state =
      xmvb::vb::map_selected_state_to_pair_state(mapping_input);
  const auto label_result =
      xmvb::vb::evaluate_approx_vbscf(
          model,
          mapped_state.pair_occupations,
          mapped_state.pair_pair_occupations);

  std::cout << "exact_total_energy = "
            << scf_result.total_energy << '\n';
  std::cout << "exact_label_approx_total_energy = "
            << label_result.approximate_total_energy << '\n';
  std::cout << "exact_label_diagnostic_metric_quotient_total_energy = "
            << label_result.diagnostic_metric_quotient_total_energy << '\n';
  std::cout << "exact_label_diagnostic_metric_quotient_total_error = "
            << label_result.diagnostic_metric_quotient_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "exact_label_cluster_quotient_total_energy = "
            << label_result.cluster_quotient_total_energy << '\n';
  std::cout << "exact_label_cluster_quotient_total_error = "
            << label_result.cluster_quotient_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "exact_label_resonance_functional_total_energy = "
            << label_result.resonance_functional_total_energy << '\n';
  std::cout << "exact_label_resonance_functional_total_error = "
            << label_result.resonance_functional_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "pair_scf_total_error_vs_exact = "
            << pair_scf_result.result.approximate_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "pair_scf_diagnostic_metric_quotient_total_error_vs_exact = "
            << pair_scf_result.result.diagnostic_metric_quotient_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "pair_scf_cluster_quotient_total_error_vs_exact = "
            << pair_scf_result.result.cluster_quotient_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "pair_scf_resonance_functional_total_error_vs_exact = "
            << pair_scf_result.result.resonance_functional_total_energy -
                   scf_result.total_energy
            << '\n';
  std::cout << "mapped_weight_sum = "
            << mapped_state.mapped_weight_sum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(
            options.input_path,
            load_options);

    xmvb::vb::StructureMatrixEvaluator matrix_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto prepared_active_space =
        matrix_evaluator.prepare_active_space(load_result.input);

    xmvb::vb::ApproxVbScfModelInput model_input;
    model_input.n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
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

    xmvb::vb::ApproxVbScfPairScfOptions pair_scf_options;
    pair_scf_options.max_iterations = options.max_iterations;
    pair_scf_options.step_size = options.step_size;
    pair_scf_options.gradient_tolerance = options.gradient_tolerance;
    const double occupied_pair_count =
        occupied_active_pair_count(
            load_result.input.orbital_preparation_input.n_active_electrons,
            load_result.raw_structure_data.spin_multiplicity);
    const auto pair_scf_result =
        xmvb::vb::optimize_approx_vbscf_pair_occupations(
            model,
            occupied_pair_count,
            pair_scf_options);

    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "production_structure_dependency = none\n";
    std::cout << "optimizer = xi_only_mirror_descent\n";
    std::cout << "n_structures_available = "
              << load_result.raw_structure_data.n_structures << '\n';
    std::cout << "n_active_orbitals = "
              << model.n_active_orbitals << '\n';
    std::cout << "n_active_electrons = "
              << load_result.input.orbital_preparation_input.n_active_electrons
              << '\n';
    std::cout << "n_active_pairs = " << model.n_active_pairs << '\n';
    std::cout << "resonance_functional_one_pair_terms = "
              << pair_scf_result.result
                     .resonance_functional_one_pair_term_count
              << '\n';
    std::cout << "resonance_functional_dense_pair_pair_terms = "
              << pair_scf_result.result
                     .resonance_functional_dense_pair_pair_term_count
              << '\n';
    std::cout << "resonance_functional_pair_pair_terms = "
              << pair_scf_result.result
                     .resonance_functional_pair_pair_term_count
              << '\n';
    std::cout << "occupied_pair_count = "
              << occupied_pair_count << '\n';
    std::cout << "converged = "
              << (pair_scf_result.converged ? "true" : "false") << '\n';
    std::cout << "iterations = "
              << pair_scf_result.iterations << '\n';
    std::cout << "initial_approximate_total_energy = "
              << pair_scf_result.initial_total_energy << '\n';
    std::cout << "final_approximate_total_energy = "
              << pair_scf_result.result.approximate_total_energy << '\n';
    std::cout << "final_projected_gradient_norm = "
              << pair_scf_result.final_projected_gradient_norm << '\n';
    std::cout << "diagonal_pair_energy = "
              << pair_scf_result.result.diagonal_pair_energy << '\n';
    std::cout << "offdiagonal_pair_energy = "
              << pair_scf_result.result.offdiagonal_pair_energy << '\n';
    std::cout << "pair_interaction_energy = "
              << pair_scf_result.result.pair_interaction_energy << '\n';
    std::cout << "pair_energy_kernel = "
              << pair_scf_result.result.pair_energy_kernel << '\n';
    std::cout << "cluster_quotient_pair_energy = "
              << pair_scf_result.result.cluster_quotient_pair_energy << '\n';
    std::cout << "cluster_quotient_pair_pair_energy = "
              << pair_scf_result.result.cluster_quotient_pair_pair_energy
              << '\n';
    std::cout << "cluster_quotient_total_energy = "
              << pair_scf_result.result.cluster_quotient_total_energy << '\n';
    std::cout << "resonance_functional_pair_energy = "
              << pair_scf_result.result.resonance_functional_pair_energy
              << '\n';
    std::cout << "resonance_functional_pair_pair_energy = "
              << pair_scf_result.result
                     .resonance_functional_pair_pair_energy
              << '\n';
    std::cout << "resonance_functional_total_energy = "
              << pair_scf_result.result.resonance_functional_total_energy
              << '\n';
    std::cout << "overlap_response_energy = "
              << pair_scf_result.result.overlap_response_energy << '\n';
    std::cout << "one_pair_metric_log = "
              << pair_scf_result.result.one_pair_metric_log << '\n';
    std::cout << "pair_pair_metric_log = "
              << pair_scf_result.result.pair_pair_metric_log << '\n';
    std::cout << "total_metric_log = "
              << pair_scf_result.result.total_metric_log << '\n';
    std::cout << "metric_denominator = "
              << pair_scf_result.result.metric_denominator << '\n';
    std::cout << "diagnostic_metric_quotient_total_energy = "
              << pair_scf_result.result.diagnostic_metric_quotient_total_energy
              << '\n';
    if (options.compare_exact) {
      print_exact_comparison(load_result, model, pair_scf_result);
    }
    print_pair_occupations(pair_scf_result, options.top_pairs);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
