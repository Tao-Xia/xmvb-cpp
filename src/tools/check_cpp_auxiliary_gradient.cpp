#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  int count = 8;
  double step = 1.0e-6;
};

struct ActiveSpaceMatrices {
  std::vector<double> active_orbital_overlap_matrix;
  std::vector<double> h1e_act;
  std::vector<double> packed_active_two_electron_integrals;
};

struct FiniteDifferenceChainBreakdown {
  double overlap = 0.0;
  double one_electron = 0.0;
  double two_electron = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_cpp_auxiliary_gradient <input.xmi> "
               "[--algorithm original|biorthogonal] "
               "[--count N] [--step h]\n";
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
    if (argument_name == "--algorithm") {
      if (argument_value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else if (argument_value == "biorthogonal") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Biorthogonal;
      } else {
        throw std::invalid_argument("invalid algorithm: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--count") {
      options.count = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--step") {
      options.step = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (options.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  return options;
}

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  double one_electron_reference_energy = 0.0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          static_cast<std::size_t>(column) * n_basis_functions + row;
      one_electron_reference_energy +=
          inactive_density_matrix[index] *
          (ao_effective_h1e[index] + ao_core_hamiltonian_matrix[index]);
    }
  }
  return one_electron_reference_energy;
}

double evaluate_total_energy_from_auxiliary(
    const xmvb::vb::CppVbInput& input,
    const std::vector<double>& auxiliary_orbital_matrix,
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    double nuclear_repulsion_energy,
    xmvb::vb::VBSCFAlgorithm algorithm) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  const Eigen::Map<const Matrix> active_orbital_overlap_matrix(
      input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Matrix active_orbital_overlap_matrix =
      active_auxiliary_orbitals.transpose() * active_orbital_overlap_matrix * active_auxiliary_orbitals;

  xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
  xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
  const auto active_space_one_electron_result =
      active_space_one_electron_builder.build(
          ao_effective_h1e,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto active_space_two_electron_result =
      active_space_two_electron_builder.build(
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(algorithm);
  const auto structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      std::vector<double>(
          active_orbital_overlap_matrix.data(),
          active_orbital_overlap_matrix.data() + active_orbital_overlap_matrix.size()),
      active_space_one_electron_result.h1e_act,
      n_active_orbitals,
      active_space_two_electron_result.packed_active_two_electron_integrals,
      input.structure_data.n_structures);

  xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  const auto eigen_result = generalized_eigensolver.solve(
      structure_matrices.hamiltonian_matrix,
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  const double one_electron_reference_energy = compute_one_electron_reference_energy(
      inactive_density_matrix,
      ao_effective_h1e,
      input.ao_integral_input.ao_core_hamiltonian_matrix,
      n_basis_functions);
  return one_electron_reference_energy + eigen_result.eigenvalues.front() + nuclear_repulsion_energy;
}

ActiveSpaceMatrices build_active_space_matrices(
    const xmvb::vb::CppVbInput& input,
    const std::vector<double>& auxiliary_orbital_matrix,
    const std::vector<double>& ao_effective_h1e) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  const Eigen::Map<const Matrix> active_orbital_overlap_matrix(
      input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Matrix> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Matrix active_orbital_overlap_matrix =
      active_auxiliary_orbitals.transpose() * active_orbital_overlap_matrix * active_auxiliary_orbitals;

  xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
  xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
  const auto active_space_one_electron_result =
      active_space_one_electron_builder.build(
          ao_effective_h1e,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto active_space_two_electron_result =
      active_space_two_electron_builder.build(
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  ActiveSpaceMatrices result;
  result.active_orbital_overlap_matrix.assign(
      active_orbital_overlap_matrix.data(),
      active_orbital_overlap_matrix.data() + active_orbital_overlap_matrix.size());
  result.h1e_act = active_space_one_electron_result.h1e_act;
  result.packed_active_two_electron_integrals =
      active_space_two_electron_result.packed_active_two_electron_integrals;
  return result;
}

FiniteDifferenceChainBreakdown finite_difference_chain_breakdown(
    const ActiveSpaceMatrices& plus_matrices,
    const ActiveSpaceMatrices& minus_matrices,
    const xmvb::vb::CppActiveSpaceGradientResult& active_space_gradient_result,
    double step) {
  if (plus_matrices.active_orbital_overlap_matrix.size() !=
          minus_matrices.active_orbital_overlap_matrix.size() ||
      plus_matrices.active_orbital_overlap_matrix.size() !=
          active_space_gradient_result.active_orbital_overlap_gradient.size()) {
    throw std::runtime_error("active overlap finite-difference size mismatch");
  }
  if (plus_matrices.h1e_act.size() !=
          minus_matrices.h1e_act.size() ||
      plus_matrices.h1e_act.size() !=
          active_space_gradient_result.active_one_electron_gradient.size()) {
    throw std::runtime_error("active one-electron finite-difference size mismatch");
  }
  if (plus_matrices.packed_active_two_electron_integrals.size() !=
          minus_matrices.packed_active_two_electron_integrals.size() ||
      plus_matrices.packed_active_two_electron_integrals.size() !=
          active_space_gradient_result.packed_active_two_electron_gradient.size()) {
    throw std::runtime_error("active two-electron finite-difference size mismatch");
  }

  const double inverse_two_step = 1.0 / (2.0 * step);
  FiniteDifferenceChainBreakdown result;
  for (std::size_t index = 0;
       index < plus_matrices.active_orbital_overlap_matrix.size();
       ++index) {
    result.overlap +=
        (plus_matrices.active_orbital_overlap_matrix[index] -
         minus_matrices.active_orbital_overlap_matrix[index]) *
        inverse_two_step * active_space_gradient_result.active_orbital_overlap_gradient[index];
  }
  for (std::size_t index = 0;
       index < plus_matrices.h1e_act.size();
       ++index) {
    result.one_electron +=
        (plus_matrices.h1e_act[index] -
         minus_matrices.h1e_act[index]) *
        inverse_two_step * active_space_gradient_result.active_one_electron_gradient[index];
  }
  for (std::size_t index = 0;
       index < plus_matrices.packed_active_two_electron_integrals.size();
       ++index) {
    result.two_electron +=
        (plus_matrices.packed_active_two_electron_integrals[index] -
         minus_matrices.packed_active_two_electron_integrals[index]) *
        inverse_two_step * active_space_gradient_result.packed_active_two_electron_gradient[index];
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;

    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;

    xmvb::vb::CppActiveSpaceGradientEvaluator active_space_gradient_evaluator(options.algorithm);
    const auto active_space_gradient_result =
        active_space_gradient_evaluator.evaluate(input, load_result.nuclear_repulsion_energy);
    xmvb::vb::ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator;
    xmvb::vb::ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
    const auto active_space_matrix_backpropagation_result =
        active_space_matrix_backpropagator.backpropagate(
            active_space_gradient_result.active_orbital_overlap_gradient,
            active_space_gradient_result.active_one_electron_gradient,
            input.orbital_preparation_input.active_orbital_overlap_matrix,
            active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
            active_space_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const auto active_space_two_electron_backpropagation_result =
        active_space_two_electron_backpropagator.backpropagate(
            active_space_gradient_result.packed_active_two_electron_gradient,
            input.ao_integral_input.ao_two_electron_integral_values,
            input.ao_integral_input.ao_two_electron_integral_indices,
            active_space_gradient_result.orbital_preparation_result,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);

    std::vector<double> total_auxiliary_gradient =
        active_space_matrix_backpropagation_result.auxiliary_orbital_gradient;
    if (total_auxiliary_gradient.size() !=
        active_space_two_electron_backpropagation_result.auxiliary_orbital_gradient.size()) {
      throw std::runtime_error("auxiliary gradient size mismatch");
    }
    for (std::size_t index = 0; index < total_auxiliary_gradient.size(); ++index) {
      total_auxiliary_gradient[index] +=
          active_space_two_electron_backpropagation_result.auxiliary_orbital_gradient[index];
    }

    std::vector<std::pair<double, int>> ranked_entries;
    for (int active_orbital_index = 0; active_orbital_index < n_active_orbitals; ++active_orbital_index) {
      const int column_index = n_inactive_doubly_occupied_orbitals + active_orbital_index;
      for (int basis_function_index = 0; basis_function_index < n_basis_functions; ++basis_function_index) {
        const int storage_index = column_index * n_basis_functions + basis_function_index;
        ranked_entries.emplace_back(
            std::abs(total_auxiliary_gradient[static_cast<std::size_t>(storage_index)]),
            storage_index);
      }
    }
    std::sort(
        ranked_entries.begin(),
        ranked_entries.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });

    const int n_to_report =
        std::min(options.count, static_cast<int>(ranked_entries.size()));
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "initial_total_energy = " << active_space_gradient_result.scf_result.total_energy
              << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int storage_index = ranked_entries[static_cast<std::size_t>(report_index)].second;
      const int basis_function_index = storage_index % n_basis_functions;
      const int column_index = storage_index / n_basis_functions;
      const int active_orbital_index = column_index - n_inactive_doubly_occupied_orbitals;
      const double analytic_matrix =
          active_space_matrix_backpropagation_result.auxiliary_orbital_gradient
              [static_cast<std::size_t>(storage_index)];
      const double analytic_two_electron =
          active_space_two_electron_backpropagation_result.auxiliary_orbital_gradient
              [static_cast<std::size_t>(storage_index)];

      std::vector<double> plus_auxiliary =
          active_space_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix;
      std::vector<double> minus_auxiliary = plus_auxiliary;
      plus_auxiliary[static_cast<std::size_t>(storage_index)] += options.step;
      minus_auxiliary[static_cast<std::size_t>(storage_index)] -= options.step;

      const double plus_energy = evaluate_total_energy_from_auxiliary(
          input,
          plus_auxiliary,
          active_space_gradient_result.orbital_preparation_result.inactive_density_matrix,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
          load_result.nuclear_repulsion_energy,
          options.algorithm);
      const double minus_energy = evaluate_total_energy_from_auxiliary(
          input,
          minus_auxiliary,
          active_space_gradient_result.orbital_preparation_result.inactive_density_matrix,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
          load_result.nuclear_repulsion_energy,
          options.algorithm);
      const ActiveSpaceMatrices plus_matrices = build_active_space_matrices(
          input,
          plus_auxiliary,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e);
      const ActiveSpaceMatrices minus_matrices = build_active_space_matrices(
          input,
          minus_auxiliary,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e);
      const FiniteDifferenceChainBreakdown finite_difference_chain =
          finite_difference_chain_breakdown(
          plus_matrices,
          minus_matrices,
          active_space_gradient_result,
          options.step);
      const double finite_difference_chain_total =
          finite_difference_chain.overlap +
          finite_difference_chain.one_electron +
          finite_difference_chain.two_electron;
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);
      const double analytic = total_auxiliary_gradient[static_cast<std::size_t>(storage_index)];
      const double chain_absolute_error = std::abs(analytic - finite_difference_chain_total);
      const double chain_relative_error =
          chain_absolute_error / std::max(1.0, std::abs(finite_difference_chain_total));
      const double energy_absolute_error = std::abs(analytic - finite_difference);
      const double energy_relative_error =
          energy_absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "entry[" << report_index << "]"
                << " active_orbital=" << active_orbital_index
                << " basis_function=" << basis_function_index
                << " analytic=" << analytic
                << " analytic_matrix=" << analytic_matrix
                << " analytic_two_electron=" << analytic_two_electron
                << " fd_chain_total=" << finite_difference_chain_total
                << " fd_chain_overlap=" << finite_difference_chain.overlap
                << " fd_chain_one_electron=" << finite_difference_chain.one_electron
                << " fd_chain_two_electron=" << finite_difference_chain.two_electron
                << " fd_energy=" << finite_difference
                << " chain_abs_error=" << chain_absolute_error
                << " chain_rel_error=" << chain_relative_error
                << " energy_abs_error=" << energy_absolute_error
                << " energy_rel_error=" << energy_relative_error
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
