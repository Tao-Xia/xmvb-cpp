#include <algorithm>
#include <cmath>
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

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int top_structures = 20;
  int pair_scf_iterations = 200;
  double pair_scf_step_size = 0.05;
  int projection_iterations = 5000;
  double projection_step_size = 0.1;
};

struct RankedStructure {
  int structure_index = 0;
  int exact_rank = 0;
  int projected_rank = 0;
  double exact_weight = 0.0;
  double projected_weight = 0.0;
};

void print_usage() {
  std::cerr << "usage: compare_approx_structure_importance <input.xmi> "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--top-structures N] "
               "[--pair-scf-iterations N] "
               "[--pair-scf-step-size X] "
               "[--projection-iterations N] "
               "[--projection-step-size X]\n";
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
    if (argument_index + 1 >= argc) {
      throw std::invalid_argument(argument + " requires a value");
    }
    const std::string value = argv[++argument_index];
    if (argument == "--standard-two-electron-mode") {
      options.standard_two_electron_mode =
          parse_standard_two_electron_mode(value);
    } else if (argument == "--top-structures") {
      options.top_structures = std::stoi(value);
    } else if (argument == "--pair-scf-iterations") {
      options.pair_scf_iterations = std::stoi(value);
    } else if (argument == "--pair-scf-step-size") {
      options.pair_scf_step_size = std::stod(value);
    } else if (argument == "--projection-iterations") {
      options.projection_iterations = std::stoi(value);
    } else if (argument == "--projection-step-size") {
      options.projection_step_size = std::stod(value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }

  if (options.top_structures < 0) {
    throw std::invalid_argument("--top-structures must be non-negative");
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

std::vector<int> rank_by_abs_value(const std::vector<double>& values) {
  std::vector<int> order(values.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(
      order.begin(),
      order.end(),
      [&values](int left, int right) {
        const double left_abs = std::abs(values[left]);
        const double right_abs = std::abs(values[right]);
        if (left_abs != right_abs) {
          return left_abs > right_abs;
        }
        return left < right;
      });
  return order;
}

std::vector<int> invert_rank(const std::vector<int>& order) {
  std::vector<int> rank(order.size(), 0);
  for (int position = 0; position < static_cast<int>(order.size()); ++position) {
    rank[order[position]] = position + 1;
  }
  return rank;
}

int top_overlap_count(
    const std::vector<int>& exact_order,
    const std::vector<int>& projected_order,
    int top_count) {
  std::vector<char> selected(exact_order.size(), 0);
  for (int index = 0; index < top_count; ++index) {
    selected[exact_order[index]] = 1;
  }
  int overlap = 0;
  for (int index = 0; index < top_count; ++index) {
    if (selected[projected_order[index]]) {
      ++overlap;
    }
  }
  return overlap;
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

void print_rank_table(
    const std::vector<double>& exact_weights,
    const std::vector<double>& projected_weights,
    const std::vector<int>& exact_order,
    int top_structures) {
  const std::vector<int> exact_rank = invert_rank(exact_order);
  const std::vector<int> projected_order = rank_by_abs_value(projected_weights);
  const std::vector<int> projected_rank = invert_rank(projected_order);
  const int count =
      std::min(top_structures, static_cast<int>(exact_order.size()));

  std::cout << "rank"
            << '\t' << "structure"
            << '\t' << "exact_rank"
            << '\t' << "projected_rank"
            << '\t' << "rank_delta"
            << '\t' << "exact_weight"
            << '\t' << "projected_weight"
            << '\n';
  for (int rank = 0; rank < count; ++rank) {
    const int structure_index = exact_order[rank];
    std::cout << rank + 1
              << '\t' << structure_index
              << '\t' << exact_rank[structure_index]
              << '\t' << projected_rank[structure_index]
              << '\t' << projected_rank[structure_index] -
                     exact_rank[structure_index]
              << '\t' << exact_weights[structure_index]
              << '\t' << projected_weights[structure_index]
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

    xmvb::core::GeneralizedEigensolver eigensolver;
    const auto eigen_result =
        eigensolver.solve(
            structure_matrices.hamiltonian_matrix,
            structure_matrices.overlap_matrix,
            structure_matrices.n_structures);

    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    const int n_active_electrons =
        load_result.input.orbital_preparation_input.n_active_electrons;
    xmvb::vb::ApproxVbScfStateMappingInput exact_mapping_input;
    exact_mapping_input.n_active_orbitals = n_active_orbitals;
    exact_mapping_input.raw_structure_data = &load_result.raw_structure_data;
    exact_mapping_input.structure_overlap_matrix =
        &structure_matrices.overlap_matrix;
    exact_mapping_input.eigenvector_matrix = &eigen_result.eigenvector_matrix;
    const auto exact_state =
        xmvb::vb::map_selected_state_to_pair_state(exact_mapping_input);

    const auto model_input =
        make_model_input(
            n_active_orbitals,
            prepared_active_space,
            load_result.nuclear_repulsion_energy);
    const auto model =
        xmvb::vb::build_approx_vbscf_model(model_input);

    xmvb::vb::ApproxVbScfPairScfOptions pair_scf_options;
    pair_scf_options.max_iterations = options.pair_scf_iterations;
    pair_scf_options.step_size = options.pair_scf_step_size;
    const double occupied_pair_count =
        occupied_active_pair_count(
            n_active_electrons,
            load_result.raw_structure_data.spin_multiplicity);
    const auto pair_scf_result =
        xmvb::vb::optimize_approx_vbscf_pair_occupations(
            model,
            occupied_pair_count,
            pair_scf_options);

    xmvb::vb::ApproxVbScfStructureProjectionOptions projection_options;
    projection_options.max_iterations = options.projection_iterations;
    projection_options.step_size = options.projection_step_size;
    const auto projection_result =
        xmvb::vb::project_pair_occupations_to_structure_weights(
            load_result.raw_structure_data,
            n_active_orbitals,
            pair_scf_result.pair_occupations,
            projection_options);

    const std::vector<int> exact_order =
        rank_by_abs_value(exact_state.structure_weights);
    const std::vector<int> projected_order =
        rank_by_abs_value(projection_result.structure_weights);
    const int top_count =
        std::min(
            options.top_structures,
            load_result.raw_structure_data.n_structures);
    const int overlap =
        top_overlap_count(exact_order, projected_order, top_count);

    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "importance_definition = simplex_projection_from_pair_occupations\n";
    std::cout << "n_structures = "
              << load_result.raw_structure_data.n_structures << '\n';
    std::cout << "n_active_orbitals = "
              << n_active_orbitals << '\n';
    std::cout << "n_active_electrons = "
              << n_active_electrons << '\n';
    std::cout << "n_active_pairs = "
              << model.n_active_pairs << '\n';
    std::cout << "occupied_pair_count = "
              << occupied_pair_count << '\n';
    std::cout << "exact_mapped_weight_sum = "
              << exact_state.mapped_weight_sum << '\n';
    std::cout << "projection_residual_norm = "
              << projection_result.residual_norm << '\n';
    std::cout << "projection_projected_gradient_norm = "
              << projection_result.projected_gradient_norm << '\n';
    std::cout << "projection_iterations = "
              << projection_result.iterations << '\n';
    std::cout << "projection_converged = "
              << (projection_result.converged ? "true" : "false") << '\n';
    std::cout << "pair_scf_iterations = "
              << pair_scf_result.iterations << '\n';
    std::cout << "pair_scf_converged = "
              << (pair_scf_result.converged ? "true" : "false") << '\n';
    std::cout << "top_overlap_count = "
              << overlap << '\n';
    std::cout << "top_overlap_fraction = "
              << static_cast<double>(overlap) /
                     static_cast<double>(std::max(1, top_count))
              << '\n';
    print_rank_table(
        exact_state.structure_weights,
        projection_result.structure_weights,
        exact_order,
        options.top_structures);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
