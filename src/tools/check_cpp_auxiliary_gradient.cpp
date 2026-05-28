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
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"
#include "vb/orbital/ri_active_space_two_electron_builder.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

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

bool use_standard_ri_active_space_path(
    const xmvb::vb::CppVbInput& input) {
  return input.standard_two_electron_mode ==
      xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
}

bool has_materialized_ao_two_electron_integrals(
    const xmvb::vb::CppVbInput& input) {
  return !input.ao_integral_input.ao_two_electron_integral_values.empty();
}

const char* active_space_representation_name(
    xmvb::vb::ActiveSpaceTwoElectronRepresentation representation) {
  switch (representation) {
    case xmvb::vb::ActiveSpaceTwoElectronRepresentation::PackedExact:
      return "packed_exact";
    case xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity:
      return "ri";
  }
  return "unknown";
}

double compute_matrix_inner_product(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("matrix inner-product size mismatch");
  }

  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

void print_usage() {
  std::cerr << "usage: check_cpp_auxiliary_gradient <input.xmi> "
               "[--algorithm original] "
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
          column * n_basis_functions + row;
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

  const Eigen::Map<const Eigen::MatrixXd> active_orbital_overlap_input(
      input.orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd active_orbital_overlap_matrix =
      active_auxiliary_orbitals.transpose() *
      active_orbital_overlap_input *
      active_auxiliary_orbitals;

  xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
  const auto active_space_one_electron_result =
      active_space_one_electron_builder.build(
          ao_effective_h1e,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  xmvb::vb::ActiveSpaceTwoElectronResult active_space_two_electron_result;
  if (use_standard_ri_active_space_path(input)) {
    xmvb::vb::OrbitalPreparationResult active_only_orbital_result;
    active_only_orbital_result.active_sparse_row_offsets.resize(
        n_basis_functions + 1,
        0);
    int sparse_count = 0;
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      active_only_orbital_result.active_sparse_row_offsets[
          basis_function_index] = sparse_count;
      for (int active_orbital_index = 0;
           active_orbital_index < n_active_orbitals;
           ++active_orbital_index) {
        const double coefficient =
            active_auxiliary_orbitals(basis_function_index, active_orbital_index);
        if (coefficient == 0.0) {
          continue;
        }
        active_only_orbital_result.active_sparse_orbital_indices.push_back(
            active_orbital_index);
        active_only_orbital_result.active_sparse_values.push_back(coefficient);
        ++sparse_count;
      }
    }
    active_only_orbital_result.active_sparse_row_offsets[
        n_basis_functions] = sparse_count;

    xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_active_space_two_electron_builder;
    active_space_two_electron_result =
        ri_active_space_two_electron_builder.build(
            xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
            active_only_orbital_result,
            n_basis_functions,
            n_active_orbitals,
            {.reconstruct_packed_integrals = true});
  } else {
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    active_space_two_electron_result =
        active_space_two_electron_builder.build(
            input.ao_integral_input.ao_two_electron_integral_values,
            input.ao_integral_input.ao_two_electron_integral_indices,
            auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
  }

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

  const Eigen::Map<const Eigen::MatrixXd> active_orbital_overlap_input(
      input.orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
      auxiliary_orbital_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd active_orbital_overlap_matrix =
      active_auxiliary_orbitals.transpose() *
      active_orbital_overlap_input *
      active_auxiliary_orbitals;

  xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
  const auto active_space_one_electron_result =
      active_space_one_electron_builder.build(
          ao_effective_h1e,
          auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  xmvb::vb::ActiveSpaceTwoElectronResult active_space_two_electron_result;
  if (use_standard_ri_active_space_path(input)) {
    xmvb::vb::OrbitalPreparationResult active_only_orbital_result;
    active_only_orbital_result.active_sparse_row_offsets.resize(
        n_basis_functions + 1,
        0);
    int sparse_count = 0;
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      active_only_orbital_result.active_sparse_row_offsets[
          basis_function_index] = sparse_count;
      for (int active_orbital_index = 0;
           active_orbital_index < n_active_orbitals;
           ++active_orbital_index) {
        const double coefficient =
            active_auxiliary_orbitals(basis_function_index, active_orbital_index);
        if (coefficient == 0.0) {
          continue;
        }
        active_only_orbital_result.active_sparse_orbital_indices.push_back(
            active_orbital_index);
        active_only_orbital_result.active_sparse_values.push_back(coefficient);
        ++sparse_count;
      }
    }
    active_only_orbital_result.active_sparse_row_offsets[
        n_basis_functions] = sparse_count;

    xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_active_space_two_electron_builder;
    active_space_two_electron_result =
        ri_active_space_two_electron_builder.build(
            xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
            active_only_orbital_result,
            n_basis_functions,
            n_active_orbitals,
            {.reconstruct_packed_integrals = true});
  } else {
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    active_space_two_electron_result =
        active_space_two_electron_builder.build(
            input.ao_integral_input.ao_two_electron_integral_values,
            input.ao_integral_input.ao_two_electron_integral_indices,
            auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
  }

  ActiveSpaceMatrices result;
  result.active_orbital_overlap_matrix.assign(
      active_orbital_overlap_matrix.data(),
      active_orbital_overlap_matrix.data() + active_orbital_overlap_matrix.size());
  result.h1e_act = active_space_one_electron_result.h1e_act;
  result.packed_active_two_electron_integrals =
      active_space_two_electron_result.packed_active_two_electron_integrals;
  return result;
}

Eigen::MatrixXd build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    int n_active_orbitals) {
  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  Eigen::MatrixXd active_pair_gradient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));

  for (int row_first = 0; row_first < n_active_orbitals; ++row_first) {
    for (int row_second = 0; row_second <= row_first; ++row_second) {
      const std::size_t row_pair_index =
          row_first * (row_first + 1) / 2 + row_second;
      for (int column_first = 0; column_first < n_active_orbitals; ++column_first) {
        for (int column_second = 0; column_second <= column_first; ++column_second) {
          const std::size_t column_pair_index =
              column_first * (column_first + 1) / 2 + column_second;
          const int packed_index =
              (row_pair_index >= column_pair_index)
                  ? xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                        row_first,
                        row_second,
                        column_first,
                        column_second)
                  : xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                        column_first,
                        column_second,
                        row_first,
                        row_second);
          double value =
              packed_active_two_electron_gradient[packed_index];
          if (row_pair_index == column_pair_index) {
            value *= 2.0;
          }
          active_pair_gradient_matrix(
              static_cast<Eigen::Index>(row_pair_index),
              static_cast<Eigen::Index>(column_pair_index)) = value;
        }
      }
    }
  }

  return active_pair_gradient_matrix;
}

std::vector<double> build_ri_active_pair_factor_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const xmvb::vb::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals) {
  if (active_space_two_electron_result.representation !=
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument("RI active-pair-factor gradient requires an RI forward result");
  }
  const std::size_t n_active_pairs =
      n_active_orbitals * (n_active_orbitals + 1) / 2;
  const std::size_t expected_factor_size =
      active_space_two_electron_result.n_auxiliary_functions *
      n_active_pairs;
  if (active_space_two_electron_result.ri_active_pair_factors.size() !=
          expected_factor_size ||
      active_space_two_electron_result.ri_active_pair_factors.rows() !=
          active_space_two_electron_result.n_auxiliary_functions ||
      active_space_two_electron_result.ri_active_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("RI active-pair-factor buffer size mismatch");
  }

  const auto active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          n_active_orbitals);
  const Eigen::MatrixXd ri_active_pair_factor_gradient =
      active_space_two_electron_result.ri_active_pair_factors *
      active_pair_gradient_matrix;
  return xmvb::vb::flatten_matrix_column_major(ri_active_pair_factor_gradient);
}

std::vector<double> build_total_ao_effective_one_electron_gradient(
    const xmvb::vb::CppActiveSpaceGradientResult& active_space_gradient_result,
    const xmvb::vb::ActiveSpaceMatrixBackpropagationResult& matrix_backpropagation_result) {
  std::vector<double> total_ao_effective_one_electron_gradient =
      matrix_backpropagation_result.ao_effective_one_electron_gradient;
  if (total_ao_effective_one_electron_gradient.size() !=
      active_space_gradient_result.orbital_preparation_result.inactive_density_matrix.size()) {
    throw std::runtime_error("ao effective one-electron gradient size mismatch");
  }
  for (std::size_t index = 0;
       index < total_ao_effective_one_electron_gradient.size();
       ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        active_space_gradient_result.orbital_preparation_result
            .inactive_density_matrix.data()[index];
  }
  return total_ao_effective_one_electron_gradient;
}

double evaluate_ao_effective_one_electron_objective(
    const xmvb::vb::CppVbInput& input,
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_one_electron_gradient) {
  xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
  const auto ao_effective_one_electron_result =
      has_materialized_ao_two_electron_integrals(input)
          ? ao_effective_one_electron_builder.build(
                inactive_density_matrix,
                input.ao_integral_input)
          : ao_effective_one_electron_builder.build(
                inactive_density_matrix,
                input.ao_integral_input.ao_core_hamiltonian_matrix,
                xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
                input.ao_integral_input.n_basis_functions);
  return compute_matrix_inner_product(
      ao_effective_one_electron_gradient,
      ao_effective_one_electron_result.ao_effective_h1e);
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
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(
            options.input_path,
            load_options);
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
    xmvb::vb::AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
    const auto active_space_matrix_backpropagation_result =
        active_space_matrix_backpropagator.backpropagate(
            active_space_gradient_result.active_orbital_overlap_gradient,
            active_space_gradient_result.active_one_electron_gradient,
            input.orbital_preparation_input.ao_overlap_matrix,
            active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
            active_space_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix,
            n_basis_functions,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const auto active_space_two_electron_backpropagation_result =
        active_space_gradient_result.active_space_two_electron_result.representation ==
                xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
            ? active_space_two_electron_backpropagator.backpropagate(
                  build_ri_active_pair_factor_gradient(
                      active_space_gradient_result.packed_active_two_electron_gradient,
                      active_space_gradient_result.active_space_two_electron_result,
                      n_active_orbitals),
                  input,
                  active_space_gradient_result.orbital_preparation_result,
                  active_space_gradient_result.active_space_two_electron_result,
                  n_basis_functions,
                  n_inactive_doubly_occupied_orbitals,
                  n_active_orbitals)
            : active_space_two_electron_backpropagator.backpropagate(
                  active_space_gradient_result.packed_active_two_electron_gradient,
                  input.ao_integral_input.ao_two_electron_integral_values,
                  input.ao_integral_input.ao_two_electron_integral_indices,
                  active_space_gradient_result.orbital_preparation_result,
                  n_basis_functions,
                  n_inactive_doubly_occupied_orbitals,
                  n_active_orbitals);

    Eigen::MatrixXd total_active_auxiliary_gradient =
        active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
    if (total_active_auxiliary_gradient.rows() != n_basis_functions ||
        total_active_auxiliary_gradient.cols() != n_active_orbitals ||
        active_space_two_electron_backpropagation_result
                .active_auxiliary_orbital_gradient.rows() !=
            n_basis_functions ||
        active_space_two_electron_backpropagation_result
                .active_auxiliary_orbital_gradient.cols() !=
            n_active_orbitals) {
      throw std::runtime_error("auxiliary gradient size mismatch");
    }
    total_active_auxiliary_gradient.noalias() +=
        active_space_two_electron_backpropagation_result
            .active_auxiliary_orbital_gradient;
    const std::vector<double> total_ao_effective_one_electron_gradient =
        build_total_ao_effective_one_electron_gradient(
            active_space_gradient_result,
            active_space_matrix_backpropagation_result);
    const auto ao_effective_one_electron_backpropagation_result =
        has_materialized_ao_two_electron_integrals(input)
            ? ao_effective_one_electron_backpropagator.backpropagate(
                  total_ao_effective_one_electron_gradient,
                  input.ao_integral_input)
            : ao_effective_one_electron_backpropagator.backpropagate(
                  total_ao_effective_one_electron_gradient,
                  xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
                  n_basis_functions);

    std::vector<std::pair<double, int>> ranked_entries;
    for (int active_orbital_index = 0; active_orbital_index < n_active_orbitals; ++active_orbital_index) {
      for (int basis_function_index = 0; basis_function_index < n_basis_functions; ++basis_function_index) {
        const int full_storage_index =
            (n_inactive_doubly_occupied_orbitals + active_orbital_index) *
                n_basis_functions +
            basis_function_index;
        ranked_entries.emplace_back(
            std::abs(
                total_active_auxiliary_gradient(
                    basis_function_index,
                    active_orbital_index)),
            full_storage_index);
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
    std::vector<std::pair<double, int>> ranked_inactive_density_entries;
    ranked_inactive_density_entries.reserve(
        ao_effective_one_electron_backpropagation_result.inactive_density_gradient.size());
    for (std::size_t storage_index = 0;
         storage_index <
         ao_effective_one_electron_backpropagation_result.inactive_density_gradient.size();
         ++storage_index) {
      ranked_inactive_density_entries.emplace_back(
          std::abs(
              ao_effective_one_electron_backpropagation_result.inactive_density_gradient[storage_index]),
          static_cast<int>(storage_index));
    }
    std::sort(
        ranked_inactive_density_entries.begin(),
        ranked_inactive_density_entries.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });
    const int n_inactive_density_to_report =
        std::min(options.count, static_cast<int>(ranked_inactive_density_entries.size()));

    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "initial_total_energy = " << active_space_gradient_result.scf_result.total_energy
              << '\n';
    std::cout << "active_space_representation = "
              << active_space_representation_name(
                     active_space_gradient_result.active_space_two_electron_result.representation)
              << '\n';
    std::cout << "ao_effective_one_electron_mode = "
              << (has_materialized_ao_two_electron_integrals(input) ? "packed_exact" : "ri")
              << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_auxiliary_entries = " << n_to_report << '\n';
    std::cout << "reported_inactive_density_entries = "
              << n_inactive_density_to_report << '\n';

    const std::vector<double> baseline_auxiliary_matrix(
        active_space_gradient_result.orbital_preparation_result
            .auxiliary_orbital_matrix.data(),
        active_space_gradient_result.orbital_preparation_result
                .auxiliary_orbital_matrix.data() +
            active_space_gradient_result.orbital_preparation_result
                .auxiliary_orbital_matrix.size());
    const std::vector<double> baseline_inactive_density(
        active_space_gradient_result.orbital_preparation_result
            .inactive_density_matrix.data(),
        active_space_gradient_result.orbital_preparation_result
                .inactive_density_matrix.data() +
            active_space_gradient_result.orbital_preparation_result
                .inactive_density_matrix.size());

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int full_storage_index =
          ranked_entries[report_index].second;
      const int basis_function_index = full_storage_index % n_basis_functions;
      const int column_index = full_storage_index / n_basis_functions;
      const int active_orbital_index =
          column_index - n_inactive_doubly_occupied_orbitals;
      const double analytic_matrix =
          active_space_matrix_backpropagation_result
              .active_auxiliary_orbital_gradient(
                  basis_function_index,
                  active_orbital_index);
      const double analytic_two_electron =
          active_space_two_electron_backpropagation_result
              .active_auxiliary_orbital_gradient(
                  basis_function_index,
                  active_orbital_index);

      std::vector<double> plus_auxiliary =
          baseline_auxiliary_matrix;
      std::vector<double> minus_auxiliary = plus_auxiliary;
      plus_auxiliary[full_storage_index] += options.step;
      minus_auxiliary[full_storage_index] -= options.step;

      const double plus_energy = evaluate_total_energy_from_auxiliary(
          input,
          plus_auxiliary,
          baseline_inactive_density,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
          load_result.nuclear_repulsion_energy,
          options.algorithm);
      const double minus_energy = evaluate_total_energy_from_auxiliary(
          input,
          minus_auxiliary,
          baseline_inactive_density,
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
      const double analytic =
          total_active_auxiliary_gradient(
              basis_function_index,
              active_orbital_index);
      const double chain_absolute_error = std::abs(analytic - finite_difference_chain_total);
      const double chain_relative_error =
          chain_absolute_error / std::max(1.0, std::abs(finite_difference_chain_total));
      const double energy_absolute_error = std::abs(analytic - finite_difference);
      const double energy_relative_error =
          energy_absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "auxiliary_entry[" << report_index << "]"
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

    for (int report_index = 0;
         report_index < n_inactive_density_to_report;
         ++report_index) {
      const int storage_index =
          ranked_inactive_density_entries[report_index].second;
      const int row_index = storage_index % n_basis_functions;
      const int column_index = storage_index / n_basis_functions;
      const double analytic =
          ao_effective_one_electron_backpropagation_result.inactive_density_gradient
              [storage_index];

      std::vector<double> plus_inactive_density = baseline_inactive_density;
      std::vector<double> minus_inactive_density = plus_inactive_density;
      plus_inactive_density[storage_index] += options.step;
      minus_inactive_density[storage_index] -= options.step;

      const double plus_objective =
          evaluate_ao_effective_one_electron_objective(
              input,
              plus_inactive_density,
              total_ao_effective_one_electron_gradient);
      const double minus_objective =
          evaluate_ao_effective_one_electron_objective(
              input,
              minus_inactive_density,
              total_ao_effective_one_electron_gradient);
      const double finite_difference =
          (plus_objective - minus_objective) / (2.0 * options.step);
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "ao_h1e_entry[" << report_index << "]"
                << " row=" << row_index
                << " column=" << column_index
                << " analytic=" << analytic
                << " fd=" << finite_difference
                << " abs_error=" << absolute_error
                << " rel_error=" << relative_error
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
