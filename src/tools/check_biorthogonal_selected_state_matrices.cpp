#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace {

struct Options {
  std::string input_path;
  int subspace_size = 0;
  double tolerance = 1.0e-12;
};

void print_usage() {
  std::cerr << "usage: check_biorthogonal_selected_state_matrices <input.xmi>"
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
    } else if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument_name);
    }
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

std::vector<double> dense_column_to_vector(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    int column_index) {
  std::vector<double> column_values(xmvb::to_size(matrix.rows()), 0.0);
  for (int row_index = 0; row_index < matrix.rows(); ++row_index) {
    column_values[xmvb::to_size(row_index)] = matrix(row_index, column_index);
  }
  return column_values;
}

Eigen::MatrixXd reconstruct_from_local_support(
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalSelectedStateDeterminantCoefficients& state,
    const xmvb::vb::SelectedStateLocalCoefficientMatrix& local_matrix,
    int n_unique_alpha,
    int n_unique_beta) {
  Eigen::MatrixXd reconstructed =
      Eigen::MatrixXd::Zero(n_unique_alpha, n_unique_beta);
  for (int alpha_local = 0;
       alpha_local < static_cast<int>(state.alpha_support.size());
       ++alpha_local) {
    const int alpha_global = state.alpha_support[xmvb::to_size(alpha_local)];
    for (int beta_local = 0;
         beta_local < static_cast<int>(state.beta_support.size());
         ++beta_local) {
      const int beta_global = state.beta_support[xmvb::to_size(beta_local)];
      reconstructed(alpha_global, beta_global) = local_matrix(alpha_local, beta_local);
    }
  }
  return reconstructed;
}

Eigen::MatrixXd build_dense_unique_pair_matrix(
    const std::vector<double>& determinant_coefficients,
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalSelectedStateMatrices& selected_state_matrices) {
  Eigen::MatrixXd dense_matrix =
      Eigen::MatrixXd::Zero(
          selected_state_matrices.n_unique_alpha,
          selected_state_matrices.n_unique_beta);
  for (int determinant_index = 0;
       determinant_index < selected_state_matrices.n_determinants;
       ++determinant_index) {
    const int unique_alpha_id =
        selected_state_matrices.determinant_to_unique_alpha_id[xmvb::to_size(
            determinant_index)];
    const int unique_beta_id =
        selected_state_matrices.determinant_to_unique_beta_id[xmvb::to_size(
            determinant_index)];
    dense_matrix(unique_alpha_id, unique_beta_id) =
        determinant_coefficients[xmvb::to_size(determinant_index)];
  }
  return dense_matrix;
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

    const auto prepared_input =
        xmvb::vb::biorthogonal_vbscf::prepare_biorthogonal_input(load_result.input);
    const std::vector<int> selected_structure_indices =
        build_selected_structure_indices(
            load_result.input.structure_data.n_structures,
            options.subspace_size);
    const auto evaluation_result =
        xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_subspace(
            load_result.input,
            prepared_input.prepared_active_space,
            selected_structure_indices);

    const xmvb::vb::FullDeterminantPairEvaluator pair_evaluator;
    const xmvb::vb::SameSpinPairCacheContext same_spin_pair_cache =
        xmvb::vb::build_same_spin_pair_cache_context(
            load_result.input.structure_data.alpha_det,
            load_result.input.structure_data.beta_det,
            pair_evaluator,
            prepared_input.prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_input.prepared_active_space.active_space_one_electron_result.h1e_act,
            load_result.input.orbital_preparation_input.n_active_orbitals,
            prepared_input.prepared_active_space.active_space_two_electron_result);

    const auto selected_state_matrices =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_selected_state_matrices(
            evaluation_result,
            same_spin_pair_cache,
            {0},
            {1.0});
    const auto pair_weights =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
                selected_state_matrices);

    const Eigen::MatrixXd hamiltonian_determinant_coefficients =
        evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns *
        evaluation_result.structure_coefficient_matrix;
    const Eigen::MatrixXd right_determinant_coefficients =
        evaluation_result.selected_structure_to_determinant *
        evaluation_result.structure_coefficient_matrix;
    const Eigen::MatrixXd left_determinant_coefficients =
        evaluation_result.overlap_action_on_selected_columns *
        evaluation_result.structure_coefficient_matrix;

    const auto& state = selected_state_matrices.states.front();
    const std::vector<double> expected_right =
        dense_column_to_vector(right_determinant_coefficients, 0);
    const std::vector<double> expected_left =
        dense_column_to_vector(left_determinant_coefficients, 0);
    std::vector<double> expected_residual(
        xmvb::to_size(evaluation_result.n_determinants),
        0.0);
    for (int determinant_index = 0;
         determinant_index < evaluation_result.n_determinants;
         ++determinant_index) {
      expected_residual[xmvb::to_size(determinant_index)] =
          hamiltonian_determinant_coefficients(determinant_index, 0) -
          evaluation_result.eigenvalues[0] * expected_right[xmvb::to_size(determinant_index)];
    }

    const Eigen::MatrixXd right_matrix =
        build_dense_unique_pair_matrix(
            state.right_determinant_coefficients,
            selected_state_matrices);
    const Eigen::MatrixXd left_matrix =
        build_dense_unique_pair_matrix(
            state.left_determinant_coefficients,
            selected_state_matrices);
    const Eigen::MatrixXd residual_matrix =
        build_dense_unique_pair_matrix(
            state.residual_determinant_coefficients,
            selected_state_matrices);

    const Eigen::MatrixXd reconstructed_right =
        reconstruct_from_local_support(
            state,
            state.local_right_coefficient_matrix,
            selected_state_matrices.n_unique_alpha,
            selected_state_matrices.n_unique_beta);
    const Eigen::MatrixXd reconstructed_left =
        reconstruct_from_local_support(
            state,
            state.local_left_coefficient_matrix,
            selected_state_matrices.n_unique_alpha,
            selected_state_matrices.n_unique_beta);
    const Eigen::MatrixXd reconstructed_residual =
        reconstruct_from_local_support(
            state,
            state.local_residual_coefficient_matrix,
            selected_state_matrices.n_unique_alpha,
            selected_state_matrices.n_unique_beta);

    std::vector<double> reference_ordered_hamiltonian_weights(
        xmvb::product_size(evaluation_result.n_determinants, evaluation_result.n_determinants),
        0.0);
    std::vector<double> reference_ordered_overlap_weights(
        xmvb::product_size(evaluation_result.n_determinants, evaluation_result.n_determinants),
        0.0);
    for (int determinant_index_left = 0;
         determinant_index_left < evaluation_result.n_determinants;
         ++determinant_index_left) {
      const std::size_t row_offset =
          xmvb::to_size(determinant_index_left) * evaluation_result.n_determinants;
      for (int determinant_index_right = 0;
           determinant_index_right < evaluation_result.n_determinants;
           ++determinant_index_right) {
        reference_ordered_hamiltonian_weights[row_offset +
                                              xmvb::to_size(determinant_index_right)] =
            expected_left[xmvb::to_size(determinant_index_left)] *
            expected_right[xmvb::to_size(determinant_index_right)];
        reference_ordered_overlap_weights[row_offset +
                                          xmvb::to_size(determinant_index_right)] =
            expected_residual[xmvb::to_size(determinant_index_left)] *
            expected_right[xmvb::to_size(determinant_index_right)];
      }
    }

    const double right_vector_max_abs =
        max_abs_vector_difference(state.right_determinant_coefficients, expected_right);
    const double left_vector_max_abs =
        max_abs_vector_difference(state.left_determinant_coefficients, expected_left);
    const double residual_vector_max_abs =
        max_abs_vector_difference(state.residual_determinant_coefficients, expected_residual);
    const double right_local_reconstruction_max_abs =
        (right_matrix - reconstructed_right).cwiseAbs().maxCoeff();
    const double left_local_reconstruction_max_abs =
        (left_matrix - reconstructed_left).cwiseAbs().maxCoeff();
    const double residual_local_reconstruction_max_abs =
        (residual_matrix - reconstructed_residual).cwiseAbs().maxCoeff();
    const double hamiltonian_weight_max_abs =
        max_abs_vector_difference(
            pair_weights.ordered_hamiltonian_weights,
            reference_ordered_hamiltonian_weights);
    const double overlap_weight_max_abs =
        max_abs_vector_difference(
            pair_weights.ordered_overlap_weights,
            reference_ordered_overlap_weights);

    const double worst_max_abs = std::max({
        right_vector_max_abs,
        left_vector_max_abs,
        residual_vector_max_abs,
        right_local_reconstruction_max_abs,
        left_local_reconstruction_max_abs,
        residual_local_reconstruction_max_abs,
        hamiltonian_weight_max_abs,
        overlap_weight_max_abs,
    });

    std::cout << "selected_structure_indices = {";
    for (std::size_t index = 0; index < selected_structure_indices.size(); ++index) {
      if (index > 0) {
        std::cout << ",";
      }
      std::cout << selected_structure_indices[index];
    }
    std::cout << "}\n";
    std::cout << "n_determinants = " << selected_state_matrices.n_determinants << '\n';
    std::cout << "n_unique_alpha = " << selected_state_matrices.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_state_matrices.n_unique_beta << '\n';
    std::cout << "alpha_support_size = " << state.alpha_support.size() << '\n';
    std::cout << "beta_support_size = " << state.beta_support.size() << '\n';
    std::cout << "nonzero_right_coefficient_count = "
              << state.nonzero_right_coefficient_count << '\n';
    std::cout << "nonzero_left_coefficient_count = "
              << state.nonzero_left_coefficient_count << '\n';
    std::cout << "nonzero_residual_coefficient_count = "
              << state.nonzero_residual_coefficient_count << '\n';
    std::cout << "right_vector_max_abs = " << right_vector_max_abs << '\n';
    std::cout << "left_vector_max_abs = " << left_vector_max_abs << '\n';
    std::cout << "residual_vector_max_abs = " << residual_vector_max_abs << '\n';
    std::cout << "right_local_reconstruction_max_abs = "
              << right_local_reconstruction_max_abs << '\n';
    std::cout << "left_local_reconstruction_max_abs = "
              << left_local_reconstruction_max_abs << '\n';
    std::cout << "residual_local_reconstruction_max_abs = "
              << residual_local_reconstruction_max_abs << '\n';
    std::cout << "hamiltonian_weight_max_abs = "
              << hamiltonian_weight_max_abs << '\n';
    std::cout << "overlap_weight_max_abs = "
              << overlap_weight_max_abs << '\n';
    std::cout << "worst_max_abs = " << worst_max_abs << '\n';

    if (worst_max_abs > options.tolerance) {
      std::cerr << "worst_max_abs exceeds tolerance\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_selected_state_matrices failed: "
              << exception.what() << '\n';
    return 1;
  }
}
