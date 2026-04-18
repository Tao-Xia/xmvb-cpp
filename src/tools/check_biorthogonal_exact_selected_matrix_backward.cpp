#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_gradient.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_one_electron_transform_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_two_electron_transform_backward.hpp"
#include "vb/matrices/structure_subspace_builder.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace {

struct Options {
  std::string input_path;
  int subspace_size = 0;
  double tolerance = 1.0e-10;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

constexpr double kFiniteDifferenceStep = 1.0e-6;

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_exact_selected_matrix_backward <input.xmi>"
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

std::vector<double> subtract_vectors(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector sizes do not match");
  }
  std::vector<double> result(left.size(), 0.0);
  for (std::size_t index = 0; index < left.size(); ++index) {
    result[index] = left[index] - right[index];
  }
  return result;
}

std::vector<double> negate_vector(
    const std::vector<double>& values) {
  std::vector<double> result(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] = -values[index];
  }
  return result;
}

std::vector<double> transpose_square_vector(
    const std::vector<double>& matrix_data) {
  const std::size_t element_count = matrix_data.size();
  std::size_t dimension = 0;
  while (dimension * dimension < element_count) {
    ++dimension;
  }
  if (dimension * dimension != element_count) {
    throw std::invalid_argument("matrix_data size is not a perfect square");
  }
  std::vector<double> transposed(element_count, 0.0);
  for (std::size_t column = 0; column < dimension; ++column) {
    for (std::size_t row = 0; row < dimension; ++row) {
      transposed[row * dimension + column] =
          matrix_data[column * dimension + row];
    }
  }
  return transposed;
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

void accumulate_additive_vector(
    const std::vector<double>& partial,
    std::vector<double>* total) {
  if (total == nullptr) {
    throw std::invalid_argument("total must not be null");
  }
  if (partial.size() != total->size()) {
    throw std::invalid_argument("vector sizes do not match for accumulation");
  }
  for (std::size_t index = 0; index < total->size(); ++index) {
    (*total)[index] += partial[index];
  }
}

std::vector<double> build_symmetric_exact_overlap_finite_difference_gradient(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    int selected_state_index) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  if (selected_state_index < 0) {
    throw std::invalid_argument("selected_state_index must be non-negative");
  }

  std::vector<double> finite_difference_gradient(
      xmvb::to_size(n_active_orbitals) * n_active_orbitals,
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = column; row < n_active_orbitals; ++row) {
      auto plus_prepared_active_space = prepared_active_space;
      auto minus_prepared_active_space = prepared_active_space;
      const std::size_t lower_index =
          xmvb::to_size(column) * n_active_orbitals + row;
      const std::size_t upper_index =
          xmvb::to_size(row) * n_active_orbitals + column;
      plus_prepared_active_space.orbital_result.active_orbital_overlap_matrix[lower_index] +=
          kFiniteDifferenceStep;
      minus_prepared_active_space.orbital_result.active_orbital_overlap_matrix[lower_index] -=
          kFiniteDifferenceStep;
      if (row != column) {
        plus_prepared_active_space.orbital_result.active_orbital_overlap_matrix[upper_index] +=
            kFiniteDifferenceStep;
        minus_prepared_active_space.orbital_result.active_orbital_overlap_matrix[upper_index] -=
            kFiniteDifferenceStep;
      }

      const auto plus_result =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_scf(
              input,
              plus_prepared_active_space,
              selected_structure_indices,
              {selected_state_index},
              {1.0},
              0.0);
      const auto minus_result =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_scf(
              input,
              minus_prepared_active_space,
              selected_structure_indices,
              {selected_state_index},
              {1.0},
              0.0);
      const double directional_derivative =
          (plus_result.electronic_energy - minus_result.electronic_energy) /
          (2.0 * kFiniteDifferenceStep);
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

std::vector<int> build_selected_determinant_indices(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices) {
  std::vector<unsigned char> selected_structure_mask(
      xmvb::to_size(full_structure_data.n_structures),
      0u);
  for (const int structure_index : selected_structure_indices) {
    if (structure_index < 0 || structure_index >= full_structure_data.n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    selected_structure_mask[xmvb::to_size(structure_index)] = 1u;
  }

  std::vector<int> selected_determinant_indices;
  for (int determinant_index = 0;
       determinant_index < static_cast<int>(
           full_structure_data.determinant_to_structure_terms.size());
       ++determinant_index) {
    bool determinant_selected = false;
    for (const auto& term :
         full_structure_data.determinant_to_structure_terms[xmvb::to_size(
             determinant_index)]) {
      if (selected_structure_mask[xmvb::to_size(term.structure_index)] != 0u) {
        determinant_selected = true;
        break;
      }
    }
    if (determinant_selected) {
      selected_determinant_indices.push_back(determinant_index);
    }
  }
  return selected_determinant_indices;
}

std::vector<double> restrict_ordered_weight_table(
    const std::vector<double>& ordered_weights,
    int n_full_determinants,
    const std::vector<int>& selected_determinant_indices) {
  if (ordered_weights.size() != xmvb::product_size(n_full_determinants, n_full_determinants)) {
    throw std::invalid_argument("ordered_weights size does not match n_full_determinants");
  }
  const int n_selected_determinants =
      static_cast<int>(selected_determinant_indices.size());
  std::vector<double> restricted_weights(
      xmvb::product_size(n_selected_determinants, n_selected_determinants),
      0.0);
  for (int left_local = 0; left_local < n_selected_determinants; ++left_local) {
    const int left_full = selected_determinant_indices[xmvb::to_size(left_local)];
    const std::size_t restricted_row_offset =
        xmvb::to_size(left_local) * n_selected_determinants;
    const std::size_t full_row_offset =
        xmvb::to_size(left_full) * n_full_determinants;
    for (int right_local = 0; right_local < n_selected_determinants; ++right_local) {
      const int right_full = selected_determinant_indices[xmvb::to_size(right_local)];
      restricted_weights[restricted_row_offset + xmvb::to_size(right_local)] =
          ordered_weights[full_row_offset + xmvb::to_size(right_full)];
    }
  }
  return restricted_weights;
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
    const auto exact_pair_weights =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
                selected_state_matrices);
    const auto orbital_integrals =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
            load_result.input.orbital_preparation_input.n_active_orbitals,
            second_order_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            second_order_context.prepared_active_space.active_space_one_electron_result
                .h1e_act);
    const auto right_right_two_electron_view =
        xmvb::vb::make_active_space_two_electron_view(
            second_order_context.prepared_active_space.active_space_two_electron_result);
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalForwardSpinPairTileProvider
        alpha_biorthogonal_provider(
            second_order_context.same_spin_pair_cache.alpha_reuse_table.unique_determinants,
            true,
            orbital_integrals,
            right_right_two_electron_view);
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalForwardSpinPairTileProvider
        beta_biorthogonal_provider(
            second_order_context.same_spin_pair_cache.beta_reuse_table.unique_determinants,
            false,
            orbital_integrals,
            right_right_two_electron_view);

    const auto same_spin_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_same_spin_matrix_backward_contribution(
                second_order_context.same_spin_pair_cache,
                alpha_biorthogonal_provider,
                beta_biorthogonal_provider,
                selected_state_matrices,
                load_result.input.orbital_preparation_input.n_active_orbitals);
    const auto opposite_spin_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_opposite_spin_matrix_backward_contribution(
                second_order_context.same_spin_pair_cache,
                selected_state_matrices,
                xmvb::vb::make_active_space_two_electron_view(
                    second_order_context.prepared_active_space
                        .active_space_two_electron_result),
                load_result.input.orbital_preparation_input.n_active_orbitals);
    const auto one_electron_transform_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_one_electron_transform_backward_contribution(
                second_order_context.same_spin_pair_cache,
                selected_state_matrices,
                orbital_integrals,
                second_order_context.prepared_active_space.active_space_one_electron_result
                    .h1e_act);
    const auto two_electron_transform_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_two_electron_transform_backward_contribution(
                load_result.input,
                selected_state_matrices,
                orbital_integrals,
                second_order_context.prepared_active_space.active_space_two_electron_result);

    std::vector<double> total_overlap =
        same_spin_contribution.active_orbital_overlap_gradient;
    std::vector<double> total_one_electron =
        same_spin_contribution.active_one_electron_gradient;
    std::vector<double> total_two_electron =
        same_spin_contribution.packed_active_two_electron_gradient;
    accumulate_additive_vector(
        opposite_spin_contribution.active_orbital_overlap_gradient,
        &total_overlap);
    accumulate_additive_vector(
        opposite_spin_contribution.packed_active_two_electron_gradient,
        &total_two_electron);
    std::vector<double> augmented_overlap = total_overlap;
    std::vector<double> augmented_one_electron = total_one_electron;
    std::vector<double> augmented_two_electron = total_two_electron;
    accumulate_additive_vector(
        one_electron_transform_contribution.active_orbital_overlap_gradient,
        &augmented_overlap);
    accumulate_additive_vector(
        one_electron_transform_contribution.active_one_electron_gradient,
        &augmented_one_electron);
    std::vector<double> hybrid_overlap = total_overlap;
    std::vector<double> hybrid_one_electron =
        one_electron_transform_contribution.active_one_electron_gradient;
    std::vector<double> hybrid_two_electron = total_two_electron;
    accumulate_additive_vector(
        one_electron_transform_contribution.active_orbital_overlap_gradient,
        &hybrid_overlap);
    std::vector<double> transform_closed_overlap = total_overlap;
    std::vector<double> transform_closed_one_electron =
        one_electron_transform_contribution.active_one_electron_gradient;
    std::vector<double> transform_closed_two_electron =
        two_electron_transform_contribution.packed_active_two_electron_gradient;
    accumulate_additive_vector(
        one_electron_transform_contribution.active_orbital_overlap_gradient,
        &transform_closed_overlap);
    accumulate_additive_vector(
        two_electron_transform_contribution.active_orbital_overlap_gradient,
        &transform_closed_overlap);

    const auto wrapper_result =
        xmvb::vb::biorthogonal_vbscf::
            evaluate_biorthogonal_exact_selected_structure_active_space_gradient(
                load_result.input,
                selected_structure_indices,
                options.algorithm,
                load_result.nuclear_repulsion_energy,
                0.0);
    if (wrapper_result.second_order_context == nullptr) {
      throw std::runtime_error("wrapper_result.second_order_context is null");
    }
    const xmvb::vb::StructureSubspaceBuilder subspace_builder;
    const xmvb::vb::CppVbInput selected_input =
        subspace_builder.build(load_result.input, selected_structure_indices);
    const auto wrapper_selected_state_matrices =
        xmvb::vb::build_selected_state_determinant_matrices(
            selected_input.structure_data,
            wrapper_result.scf_result.eigenvector_matrix,
            {0},
            {1.0},
            wrapper_result.second_order_context->same_spin_pair_cache);
    const auto wrapper_pair_weights =
        xmvb::vb::build_exact_determinant_pair_weight_tables_from_eigenvalues(
            wrapper_selected_state_matrices,
            wrapper_result.scf_result.electronic_state_energies);
    const std::vector<int> selected_determinant_indices =
        build_selected_determinant_indices(
            load_result.input.structure_data,
            selected_structure_indices);
    const std::vector<double> restricted_exact_hamiltonian_weights =
        restrict_ordered_weight_table(
            exact_pair_weights.ordered_hamiltonian_weights,
            evaluation_result.n_determinants,
            selected_determinant_indices);
    const std::vector<double> restricted_exact_overlap_weights =
        restrict_ordered_weight_table(
            exact_pair_weights.ordered_overlap_weights,
            evaluation_result.n_determinants,
            selected_determinant_indices);
    const std::vector<double> exact_overlap_finite_difference =
        build_symmetric_exact_overlap_finite_difference_gradient(
            load_result.input,
            second_order_context.prepared_active_space,
            selected_structure_indices,
            0);

    const double overlap_max_abs =
        max_abs_vector_difference(
            total_overlap,
            wrapper_result.active_orbital_overlap_gradient);
    const double one_electron_max_abs =
        max_abs_vector_difference(
            total_one_electron,
            wrapper_result.active_one_electron_gradient);
    const double two_electron_max_abs =
        max_abs_vector_difference(
            total_two_electron,
            wrapper_result.packed_active_two_electron_gradient);
    const double worst_max_abs =
        std::max(overlap_max_abs, std::max(one_electron_max_abs, two_electron_max_abs));
    const double augmented_overlap_max_abs =
        max_abs_vector_difference(
            augmented_overlap,
            wrapper_result.active_orbital_overlap_gradient);
    const double augmented_one_electron_max_abs =
        max_abs_vector_difference(
            augmented_one_electron,
            wrapper_result.active_one_electron_gradient);
    const double augmented_two_electron_max_abs =
        max_abs_vector_difference(
            augmented_two_electron,
            wrapper_result.packed_active_two_electron_gradient);
    const double augmented_worst_max_abs =
        std::max(
            augmented_overlap_max_abs,
            std::max(augmented_one_electron_max_abs, augmented_two_electron_max_abs));
    const double hybrid_overlap_max_abs =
        max_abs_vector_difference(
            hybrid_overlap,
            wrapper_result.active_orbital_overlap_gradient);
    const double hybrid_one_electron_max_abs =
        max_abs_vector_difference(
            hybrid_one_electron,
            wrapper_result.active_one_electron_gradient);
    const double hybrid_two_electron_max_abs =
        max_abs_vector_difference(
            hybrid_two_electron,
            wrapper_result.packed_active_two_electron_gradient);
    const double hybrid_worst_max_abs =
        std::max(
            hybrid_overlap_max_abs,
            std::max(hybrid_one_electron_max_abs, hybrid_two_electron_max_abs));
    const double transform_closed_overlap_max_abs =
        max_abs_vector_difference(
            transform_closed_overlap,
            wrapper_result.active_orbital_overlap_gradient);
    const double transform_closed_one_electron_max_abs =
        max_abs_vector_difference(
            transform_closed_one_electron,
            wrapper_result.active_one_electron_gradient);
    const double transform_closed_two_electron_max_abs =
        max_abs_vector_difference(
            transform_closed_two_electron,
            wrapper_result.packed_active_two_electron_gradient);
    const double transform_closed_worst_max_abs =
        std::max(
            transform_closed_overlap_max_abs,
            std::max(
                transform_closed_one_electron_max_abs,
                transform_closed_two_electron_max_abs));
    const std::vector<double> hybrid_overlap_residual =
        subtract_vectors(
            wrapper_result.active_orbital_overlap_gradient,
            hybrid_overlap);
    const double exact_overlap_finite_difference_wrapper_match_max_abs =
        max_abs_vector_difference(
            exact_overlap_finite_difference,
            wrapper_result.active_orbital_overlap_gradient);
    const double exact_overlap_finite_difference_transform_closed_match_max_abs =
        max_abs_vector_difference(
            exact_overlap_finite_difference,
            transform_closed_overlap);
    const double restricted_hamiltonian_weight_match_max_abs =
        max_abs_vector_difference(
            restricted_exact_hamiltonian_weights,
            wrapper_pair_weights.ordered_hamiltonian_weights);
    const double restricted_overlap_weight_match_max_abs =
        max_abs_vector_difference(
            restricted_exact_overlap_weights,
            wrapper_pair_weights.ordered_overlap_weights);
    const double two_electron_transform_overlap_match_max_abs =
        max_abs_vector_difference(
            two_electron_transform_contribution.active_orbital_overlap_gradient,
            hybrid_overlap_residual);
    const double two_electron_transform_overlap_neg_match_max_abs =
        max_abs_vector_difference(
            negate_vector(
                two_electron_transform_contribution.active_orbital_overlap_gradient),
            hybrid_overlap_residual);
    const double two_electron_transform_overlap_transpose_match_max_abs =
        max_abs_vector_difference(
            transpose_square_vector(
                two_electron_transform_contribution.active_orbital_overlap_gradient),
            hybrid_overlap_residual);
    const double two_electron_transform_overlap_neg_transpose_match_max_abs =
        max_abs_vector_difference(
            negate_vector(
                transpose_square_vector(
                    two_electron_transform_contribution.active_orbital_overlap_gradient)),
            hybrid_overlap_residual);

    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "n_determinants = " << evaluation_result.n_determinants << '\n';
    std::cout << "n_unique_alpha = " << selected_state_matrices.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_state_matrices.n_unique_beta << '\n';
    std::cout << "total_overlap_gradient_max_abs = " << overlap_max_abs << '\n';
    std::cout << "total_one_electron_gradient_max_abs = " << one_electron_max_abs << '\n';
    std::cout << "total_two_electron_gradient_max_abs = " << two_electron_max_abs << '\n';
    std::cout << "worst_max_abs = " << worst_max_abs << '\n';
    std::cout << "augmented_overlap_gradient_max_abs = "
              << augmented_overlap_max_abs << '\n';
    std::cout << "augmented_one_electron_gradient_max_abs = "
              << augmented_one_electron_max_abs << '\n';
    std::cout << "augmented_two_electron_gradient_max_abs = "
              << augmented_two_electron_max_abs << '\n';
    std::cout << "augmented_worst_max_abs = "
              << augmented_worst_max_abs << '\n';
    std::cout << "hybrid_overlap_gradient_max_abs = "
              << hybrid_overlap_max_abs << '\n';
    std::cout << "hybrid_one_electron_gradient_max_abs = "
              << hybrid_one_electron_max_abs << '\n';
    std::cout << "hybrid_two_electron_gradient_max_abs = "
              << hybrid_two_electron_max_abs << '\n';
    std::cout << "hybrid_worst_max_abs = "
              << hybrid_worst_max_abs << '\n';
    std::cout << "transform_closed_overlap_gradient_max_abs = "
              << transform_closed_overlap_max_abs << '\n';
    std::cout << "transform_closed_one_electron_gradient_max_abs = "
              << transform_closed_one_electron_max_abs << '\n';
    std::cout << "transform_closed_two_electron_gradient_max_abs = "
              << transform_closed_two_electron_max_abs << '\n';
    std::cout << "transform_closed_worst_max_abs = "
              << transform_closed_worst_max_abs << '\n';
    std::cout << "exact_overlap_finite_difference_wrapper_match_max_abs = "
              << exact_overlap_finite_difference_wrapper_match_max_abs << '\n';
    std::cout << "exact_overlap_finite_difference_transform_closed_match_max_abs = "
              << exact_overlap_finite_difference_transform_closed_match_max_abs << '\n';
    std::cout << "restricted_hamiltonian_weight_match_max_abs = "
              << restricted_hamiltonian_weight_match_max_abs << '\n';
    std::cout << "restricted_overlap_weight_match_max_abs = "
              << restricted_overlap_weight_match_max_abs << '\n';
    std::cout << "two_electron_transform_overlap_match_max_abs = "
              << two_electron_transform_overlap_match_max_abs << '\n';
    std::cout << "two_electron_transform_overlap_neg_match_max_abs = "
              << two_electron_transform_overlap_neg_match_max_abs << '\n';
    std::cout << "two_electron_transform_overlap_transpose_match_max_abs = "
              << two_electron_transform_overlap_transpose_match_max_abs << '\n';
    std::cout << "two_electron_transform_overlap_neg_transpose_match_max_abs = "
              << two_electron_transform_overlap_neg_transpose_match_max_abs << '\n';
    print_square_matrix(
        "exact_overlap_finite_difference_matrix",
        exact_overlap_finite_difference);
    print_square_matrix(
        "same_spin_overlap_matrix",
        same_spin_contribution.active_orbital_overlap_gradient);
    print_square_matrix(
        "opposite_spin_overlap_matrix",
        opposite_spin_contribution.active_orbital_overlap_gradient);
    print_square_matrix(
        "one_electron_transform_overlap_matrix",
        one_electron_transform_contribution.active_orbital_overlap_gradient);
    print_square_matrix(
        "restricted_exact_hamiltonian_weight_matrix",
        restricted_exact_hamiltonian_weights);
    print_square_matrix(
        "wrapper_hamiltonian_weight_matrix",
        wrapper_pair_weights.ordered_hamiltonian_weights);
    print_square_matrix(
        "restricted_exact_overlap_weight_matrix",
        restricted_exact_overlap_weights);
    print_square_matrix(
        "wrapper_overlap_weight_matrix",
        wrapper_pair_weights.ordered_overlap_weights);
    print_square_matrix(
        "hybrid_overlap_residual_matrix",
        hybrid_overlap_residual);
    print_square_matrix(
        "two_electron_transform_overlap_matrix",
        two_electron_transform_contribution.active_orbital_overlap_gradient);

    if (transform_closed_worst_max_abs > options.tolerance) {
    std::cerr << "biorthogonal exact-selected matrix backward check failed: worst_max_abs = "
              << transform_closed_worst_max_abs
              << " > tolerance = " << options.tolerance
              << " (this indicates missing biorthogonal kernel-response terms)"
              << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_exact_selected_matrix_backward failed: "
              << exception.what() << '\n';
    return 1;
  }
}
