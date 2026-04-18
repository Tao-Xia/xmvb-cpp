#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_gradient.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_two_electron_transform_backward.hpp"
#include "vb/matrices/determinant_pair_storage_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"

namespace {

constexpr double kContributionTolerance = 1.0e-15;
constexpr double kFiniteDifferenceStep = 1.0e-6;

struct Options {
  std::string input_path;
  int subspace_size = 0;
  double tolerance = 1.0e-10;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_two_electron_transform_backward <input.xmi>"
         " [--subspace-size N]"
         " [--tolerance X]\n";
}

Options parse_options(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid arguments");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--subspace-size") {
      options.subspace_size = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.subspace_size < 0) {
    throw std::invalid_argument("--subspace-size must be non-negative");
  }
  if (options.tolerance < 0.0) {
    throw std::invalid_argument("--tolerance must be non-negative");
  }
  return options;
}

std::vector<int> build_selected_structure_indices(
    int n_structures,
    int subspace_size) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  const int selected_count =
      subspace_size == 0 ? n_structures : std::min(n_structures, subspace_size);
  std::vector<int> selected_structure_indices;
  selected_structure_indices.reserve(xmvb::to_size(selected_count));
  for (int structure_index = 0; structure_index < selected_count; ++structure_index) {
    selected_structure_indices.push_back(structure_index);
  }
  return selected_structure_indices;
}

std::string format_indices(const std::vector<int>& indices) {
  std::string result = "{";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index > 0) {
      result += ",";
    }
    result += std::to_string(indices[index]);
  }
  result += "}";
  return result;
}

double max_abs_vector_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector sizes do not match");
  }
  double max_abs = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_abs = std::max(max_abs, std::abs(left[index] - right[index]));
  }
  return max_abs;
}

double evaluate_two_electron_transform_objective(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronResult& right_right_two_electron_result,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights) {
  if (pair_weights.n_determinants != static_cast<int>(input.structure_data.alpha_det.size()) ||
      pair_weights.n_determinants != static_cast<int>(input.structure_data.beta_det.size())) {
    throw std::invalid_argument("pair_weights determinant count does not match input");
  }

  double objective = 0.0;
  for (int determinant_index_left = 0;
       determinant_index_left < pair_weights.n_determinants;
       ++determinant_index_left) {
    xmvb::vb::biorthogonal_vbscf::BiorthogonalDeterminant left_determinant;
    left_determinant.alpha_occupied_orbitals =
        input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)];
    left_determinant.beta_occupied_orbitals =
        input.structure_data.beta_det[xmvb::to_size(determinant_index_left)];
    for (int determinant_index_right = 0;
         determinant_index_right < pair_weights.n_determinants;
         ++determinant_index_right) {
      const double weight =
          pair_weights.ordered_hamiltonian_weights[
              xmvb::to_size(determinant_index_left) *
                  pair_weights.n_determinants +
              xmvb::to_size(determinant_index_right)];
      if (std::abs(weight) <= kContributionTolerance) {
        continue;
      }

      xmvb::vb::biorthogonal_vbscf::BiorthogonalDeterminant right_determinant;
      right_determinant.alpha_occupied_orbitals =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      right_determinant.beta_occupied_orbitals =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      const auto entry =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_determinant_hamiltonian(
              left_determinant,
              right_determinant,
              orbital_integrals,
              right_right_two_electron_result);
      objective += weight * (entry.total_hamiltonian - entry.one_electron_hamiltonian);
    }
  }
  return objective;
}

double evaluate_unordered_two_electron_transform_objective(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalIntegrals& orbital_integrals,
    const xmvb::vb::ActiveSpaceTwoElectronResult& right_right_two_electron_result,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights) {
  if (pair_weights.n_determinants != static_cast<int>(input.structure_data.alpha_det.size()) ||
      pair_weights.n_determinants != static_cast<int>(input.structure_data.beta_det.size())) {
    throw std::invalid_argument("pair_weights determinant count does not match input");
  }

  double objective = 0.0;
  for (int determinant_index_left = 0;
       determinant_index_left < pair_weights.n_determinants;
       ++determinant_index_left) {
    xmvb::vb::biorthogonal_vbscf::BiorthogonalDeterminant left_determinant;
    left_determinant.alpha_occupied_orbitals =
        input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)];
    left_determinant.beta_occupied_orbitals =
        input.structure_data.beta_det[xmvb::to_size(determinant_index_left)];
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          xmvb::vb::canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const double weight =
          pair_weights.unordered_combined_hamiltonian_weights[unordered_index];
      if (std::abs(weight) <= kContributionTolerance) {
        continue;
      }

      xmvb::vb::biorthogonal_vbscf::BiorthogonalDeterminant right_determinant;
      right_determinant.alpha_occupied_orbitals =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      right_determinant.beta_occupied_orbitals =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      const auto entry =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_determinant_hamiltonian(
              left_determinant,
              right_determinant,
              orbital_integrals,
              right_right_two_electron_result);
      objective += weight * (entry.total_hamiltonian - entry.one_electron_hamiltonian);
    }
  }
  return objective;
}

std::vector<double> build_symmetric_overlap_finite_difference_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& right_right_one_electron,
    const xmvb::vb::ActiveSpaceTwoElectronResult& right_right_two_electron_result,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  std::vector<double> finite_difference_gradient(
      xmvb::to_size(n_active_orbitals) * n_active_orbitals,
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = column; row < n_active_orbitals; ++row) {
      std::vector<double> plus_overlap = active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = active_orbital_overlap_matrix;
      const std::size_t lower_index =
          xmvb::to_size(column) * n_active_orbitals + row;
      const std::size_t upper_index =
          xmvb::to_size(row) * n_active_orbitals + column;
      plus_overlap[lower_index] += kFiniteDifferenceStep;
      minus_overlap[lower_index] -= kFiniteDifferenceStep;
      if (row != column) {
        plus_overlap[upper_index] += kFiniteDifferenceStep;
        minus_overlap[upper_index] -= kFiniteDifferenceStep;
      }

      const auto plus_integrals =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
              n_active_orbitals,
              plus_overlap,
              right_right_one_electron);
      const auto minus_integrals =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
              n_active_orbitals,
              minus_overlap,
              right_right_one_electron);
      const double plus_objective =
          evaluate_two_electron_transform_objective(
              input,
              plus_integrals,
              right_right_two_electron_result,
              pair_weights);
      const double minus_objective =
          evaluate_two_electron_transform_objective(
              input,
              minus_integrals,
              right_right_two_electron_result,
              pair_weights);
      const double directional_derivative =
          (plus_objective - minus_objective) / (2.0 * kFiniteDifferenceStep);
      if (row == column) {
        finite_difference_gradient[lower_index] = directional_derivative;
      } else {
        const double symmetric_entry = 0.5 * directional_derivative;
        finite_difference_gradient[lower_index] = symmetric_entry;
        finite_difference_gradient[upper_index] = symmetric_entry;
      }
    }
  }
  return finite_difference_gradient;
}

std::vector<double> build_symmetric_unordered_overlap_finite_difference_gradient(
    const xmvb::vb::CppVbInput& input,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& right_right_one_electron,
    const xmvb::vb::ActiveSpaceTwoElectronResult& right_right_two_electron_result,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  std::vector<double> finite_difference_gradient(
      xmvb::to_size(n_active_orbitals) * n_active_orbitals,
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = column; row < n_active_orbitals; ++row) {
      std::vector<double> plus_overlap = active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = active_orbital_overlap_matrix;
      const std::size_t lower_index =
          xmvb::to_size(column) * n_active_orbitals + row;
      const std::size_t upper_index =
          xmvb::to_size(row) * n_active_orbitals + column;
      plus_overlap[lower_index] += kFiniteDifferenceStep;
      minus_overlap[lower_index] -= kFiniteDifferenceStep;
      if (row != column) {
        plus_overlap[upper_index] += kFiniteDifferenceStep;
        minus_overlap[upper_index] -= kFiniteDifferenceStep;
      }

      const auto plus_integrals =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
              n_active_orbitals,
              plus_overlap,
              right_right_one_electron);
      const auto minus_integrals =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
              n_active_orbitals,
              minus_overlap,
              right_right_one_electron);
      const double plus_objective =
          evaluate_unordered_two_electron_transform_objective(
              input,
              plus_integrals,
              right_right_two_electron_result,
              pair_weights);
      const double minus_objective =
          evaluate_unordered_two_electron_transform_objective(
              input,
              minus_integrals,
              right_right_two_electron_result,
              pair_weights);
      const double directional_derivative =
          (plus_objective - minus_objective) / (2.0 * kFiniteDifferenceStep);
      if (row == column) {
        finite_difference_gradient[lower_index] = directional_derivative;
      } else {
        const double symmetric_entry = 0.5 * directional_derivative;
        finite_difference_gradient[lower_index] = symmetric_entry;
        finite_difference_gradient[upper_index] = symmetric_entry;
      }
    }
  }
  return finite_difference_gradient;
}

void print_square_matrix(
    const std::string& label,
    const std::vector<double>& matrix_data) {
  const std::size_t element_count = matrix_data.size();
  std::size_t dimension = 0;
  while (dimension * dimension < element_count) {
    ++dimension;
  }
  if (dimension * dimension != element_count) {
    throw std::invalid_argument("matrix_data size is not a perfect square");
  }
  std::cout << label << " =";
  for (std::size_t row = 0; row < dimension; ++row) {
    std::cout << '\n';
    for (std::size_t column = 0; column < dimension; ++column) {
      if (column > 0) {
        std::cout << ' ';
      }
      std::cout << matrix_data[column * dimension + row];
    }
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::cout << std::setprecision(15);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const std::vector<int> selected_structure_indices =
        build_selected_structure_indices(
            load_result.input.structure_data.n_structures,
            options.subspace_size);

    xmvb::vb::CppActiveSpaceGradientEvaluator gradient_evaluator(options.algorithm);
    const auto full_gradient_result =
        gradient_evaluator.evaluate(
            load_result.input,
            load_result.nuclear_repulsion_energy);
    if (full_gradient_result.second_order_context == nullptr) {
      throw std::runtime_error("full_gradient_result.second_order_context is null");
    }
    const auto& second_order_context = *full_gradient_result.second_order_context;

    const auto evaluation_result =
        xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_subspace(
            load_result.input,
            second_order_context.prepared_active_space,
            selected_structure_indices);
    const auto selected_state_matrices =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_selected_state_matrices(
            evaluation_result,
            second_order_context.same_spin_pair_cache,
            {0},
            {1.0});
    const auto orbital_integrals =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
            load_result.input.orbital_preparation_input.n_active_orbitals,
            second_order_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            second_order_context.prepared_active_space.active_space_one_electron_result
                .h1e_act);
    const auto transform_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_two_electron_transform_backward_contribution(
                load_result.input,
                selected_state_matrices,
                orbital_integrals,
                second_order_context.prepared_active_space.active_space_two_electron_result);
    const auto pair_weights =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
                selected_state_matrices);
    const std::vector<double> finite_difference_overlap_gradient =
        build_symmetric_overlap_finite_difference_gradient(
            load_result.input,
            second_order_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            second_order_context.prepared_active_space.active_space_one_electron_result
                .h1e_act,
            second_order_context.prepared_active_space.active_space_two_electron_result,
            pair_weights);
    const std::vector<double> unordered_finite_difference_overlap_gradient =
        build_symmetric_unordered_overlap_finite_difference_gradient(
            load_result.input,
            second_order_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            second_order_context.prepared_active_space.active_space_one_electron_result
                .h1e_act,
            second_order_context.prepared_active_space.active_space_two_electron_result,
            pair_weights);

    const auto wrapper_result =
        xmvb::vb::biorthogonal_vbscf::
            evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
                load_result.input,
                selected_structure_indices,
                options.algorithm,
                load_result.nuclear_repulsion_energy,
                0.0);
    const double two_electron_max_abs =
        max_abs_vector_difference(
            transform_contribution.packed_active_two_electron_gradient,
            wrapper_result.packed_active_two_electron_gradient);
    const double overlap_finite_difference_max_abs =
        max_abs_vector_difference(
            transform_contribution.active_orbital_overlap_gradient,
            finite_difference_overlap_gradient);

    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "n_determinants = " << evaluation_result.n_determinants << '\n';
    std::cout << "n_unique_alpha = " << selected_state_matrices.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_state_matrices.n_unique_beta << '\n';
    std::cout << "transform_overlap_finite_difference_max_abs = "
              << overlap_finite_difference_max_abs << '\n';
    std::cout << "transform_two_electron_gradient_max_abs = "
              << two_electron_max_abs << '\n';
    print_square_matrix(
        "analytic_overlap_gradient_matrix",
        transform_contribution.active_orbital_overlap_gradient);
    print_square_matrix(
        "finite_difference_overlap_gradient_matrix",
        finite_difference_overlap_gradient);
    print_square_matrix(
        "unordered_finite_difference_overlap_gradient_matrix",
        unordered_finite_difference_overlap_gradient);

    if (std::max(overlap_finite_difference_max_abs, two_electron_max_abs) >
        options.tolerance) {
      std::cerr << "biorthogonal two-electron transform backward check failed: worst_max_abs = "
                << std::max(overlap_finite_difference_max_abs, two_electron_max_abs)
                << " > tolerance = " << options.tolerance << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_two_electron_transform_backward failed: "
              << exception.what() << '\n';
    return 1;
  }
}
