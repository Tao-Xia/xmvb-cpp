#include <algorithm>
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

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int top_pairs = 12;
};

void print_usage() {
  std::cerr << "usage: dump_approx_vbscf_features <input.xmi> "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--top-pairs N]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--standard-two-electron-mode") {
      if (argument_value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (argument_value == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (argument_value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "invalid standard two-electron mode: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--top-pairs") {
      options.top_pairs = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
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

std::vector<int> rank_pairs_by_occupation(
    const std::vector<double>& pair_occupations) {
  std::vector<int> order(pair_occupations.size(), 0);
  for (int index = 0; index < static_cast<int>(order.size()); ++index) {
    order[index] = index;
  }
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
    const xmvb::vb::ApproxVbScfResult& result,
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
              << '\n';
  }
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
    const auto structure_matrices =
        matrix_evaluator.evaluate(
            load_result.input,
            prepared_active_space);

    xmvb::vb::CppVbScfEvaluator scf_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto scf_result =
        scf_evaluator.evaluate(
            load_result.input,
            load_result.nuclear_repulsion_energy);

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

    xmvb::vb::ApproxVbScfStateMappingInput mapping_input;
    mapping_input.n_active_orbitals = model_input.n_active_orbitals;
    mapping_input.raw_structure_data = &load_result.raw_structure_data;
    mapping_input.structure_overlap_matrix =
        &structure_matrices.overlap_matrix;
    mapping_input.eigenvector_matrix =
        &scf_result.eigenvector_matrix;

    const auto model = xmvb::vb::build_approx_vbscf_model(model_input);
    const auto mapped_state =
        xmvb::vb::map_selected_state_to_pair_state(mapping_input);
    auto approx_result =
        xmvb::vb::evaluate_approx_vbscf(
            model,
            mapped_state.pair_occupations,
            mapped_state.pair_pair_occupations);
    approx_result.exact_active_energy = scf_result.electronic_energy;
    approx_result.exact_total_energy = scf_result.total_energy;

    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "production_structure_dependency = none\n";
    std::cout << "label_mapping_dependency = exact_vbscf_state\n";
    std::cout << "n_structures = " << mapped_state.n_structures << '\n';
    std::cout << "n_active_orbitals = "
              << model_input.n_active_orbitals << '\n';
    std::cout << "n_active_electrons = "
              << load_result.input.orbital_preparation_input.n_active_electrons
              << '\n';
    std::cout << "n_active_pairs = " << approx_result.n_active_pairs << '\n';
    std::cout << "mapped_weight_sum = "
              << mapped_state.mapped_weight_sum << '\n';
    std::cout << "local_term_count = "
              << approx_result.local_term_count << '\n';
    std::cout << "pair_interaction_term_count = "
              << approx_result.pair_interaction_term_count << '\n';
    std::cout << "resonance_functional_one_pair_terms = "
              << approx_result.resonance_functional_one_pair_term_count
              << '\n';
    std::cout << "resonance_functional_dense_pair_pair_terms = "
              << approx_result
                     .resonance_functional_dense_pair_pair_term_count
              << '\n';
    std::cout << "resonance_functional_pair_pair_terms = "
              << approx_result.resonance_functional_pair_pair_term_count
              << '\n';
    std::cout << "one_electron_reference_energy = "
              << prepared_active_space.one_electron_reference_energy << '\n';
    std::cout << "exact_active_energy = "
              << approx_result.exact_active_energy << '\n';
    std::cout << "exact_total_energy = "
              << approx_result.exact_total_energy << '\n';
    std::cout << "approximate_active_energy = "
              << approx_result.approximate_active_energy << '\n';
    std::cout << "approximate_total_energy = "
              << approx_result.approximate_total_energy << '\n';
    std::cout << "active_energy_error = "
              << approx_result.approximate_active_energy -
                     approx_result.exact_active_energy << '\n';
    std::cout << "total_energy_error = "
              << approx_result.approximate_total_energy -
                     approx_result.exact_total_energy << '\n';
    std::cout << "diagonal_pair_energy = "
              << approx_result.diagonal_pair_energy << '\n';
    std::cout << "offdiagonal_pair_energy = "
              << approx_result.offdiagonal_pair_energy << '\n';
    std::cout << "pair_interaction_energy = "
              << approx_result.pair_interaction_energy << '\n';
    std::cout << "pair_energy_kernel = "
              << approx_result.pair_energy_kernel << '\n';
    std::cout << "cluster_quotient_pair_energy = "
              << approx_result.cluster_quotient_pair_energy << '\n';
    std::cout << "cluster_quotient_pair_pair_energy = "
              << approx_result.cluster_quotient_pair_pair_energy << '\n';
    std::cout << "cluster_quotient_active_energy = "
              << approx_result.cluster_quotient_active_energy << '\n';
    std::cout << "cluster_quotient_total_energy = "
              << approx_result.cluster_quotient_total_energy << '\n';
    std::cout << "cluster_quotient_total_error = "
              << approx_result.cluster_quotient_total_energy -
                     approx_result.exact_total_energy << '\n';
    std::cout << "resonance_functional_pair_energy = "
              << approx_result.resonance_functional_pair_energy << '\n';
    std::cout << "resonance_functional_pair_pair_energy = "
              << approx_result.resonance_functional_pair_pair_energy
              << '\n';
    std::cout << "resonance_functional_active_energy = "
              << approx_result.resonance_functional_active_energy << '\n';
    std::cout << "resonance_functional_total_energy = "
              << approx_result.resonance_functional_total_energy << '\n';
    std::cout << "resonance_functional_total_error = "
              << approx_result.resonance_functional_total_energy -
                     approx_result.exact_total_energy << '\n';
    std::cout << "overlap_response_energy = "
              << approx_result.overlap_response_energy << '\n';
    std::cout << "one_pair_metric_log = "
              << approx_result.one_pair_metric_log << '\n';
    std::cout << "pair_pair_metric_log = "
              << approx_result.pair_pair_metric_log << '\n';
    std::cout << "total_metric_log = "
              << approx_result.total_metric_log << '\n';
    std::cout << "metric_denominator = "
              << approx_result.metric_denominator << '\n';
    std::cout << "diagnostic_metric_quotient_active_energy = "
              << approx_result.diagnostic_metric_quotient_active_energy
              << '\n';
    std::cout << "diagnostic_metric_quotient_total_energy = "
              << approx_result.diagnostic_metric_quotient_total_energy
              << '\n';
    std::cout << "diagnostic_metric_quotient_total_error = "
              << approx_result.diagnostic_metric_quotient_total_energy -
                     approx_result.exact_total_energy << '\n';
    print_pair_occupations(approx_result, options.top_pairs);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
