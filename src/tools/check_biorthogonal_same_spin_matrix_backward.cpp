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
#include "vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

constexpr double kContributionTolerance = 1.0e-15;

struct Options {
  std::string input_path;
  int subspace_size = 0;
  double tolerance = 1.0e-8;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_same_spin_matrix_backward <input.xmi>"
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

void accumulate_one_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DenseMatrix& cofactor_1st,
    double weight,
    DenseMatrix* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }
  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[xmvb::to_size(right_row)];
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_same_spin_two_electron_gradient_contribution(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DenseMatrix& cofactor_1st,
    double overlap_determinant,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance || overlap_determinant == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double cofactor_scale = weight / overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[xmvb::to_size(right_first)];
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[xmvb::to_size(left_second)];
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[xmvb::to_size(right_second)];
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double second_order_cofactor =
              cofactor_scale * (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21);
          const int direct_index = xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          (*packed_active_two_electron_gradient)[xmvb::to_size(direct_index)] +=
              second_order_cofactor;
          (*packed_active_two_electron_gradient)[xmvb::to_size(exchange_index)] -=
              second_order_cofactor;
        }
      }
    }
  }
}

xmvb::vb::SameSpinMatrixBackwardContribution
build_reference_same_spin_contribution(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights,
    int n_active_orbitals) {
  if (pair_weights.n_determinants !=
      static_cast<int>(input.structure_data.alpha_det.size())) {
    throw std::invalid_argument("pair_weights determinant count does not match input");
  }

  DenseMatrix active_one_electron_gradient =
      DenseMatrix::Zero(n_active_orbitals, n_active_orbitals);
  xmvb::vb::SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      xmvb::product_size(n_active_orbitals, n_active_orbitals),
      0.0);
  result.active_one_electron_gradient.assign(
      xmvb::product_size(n_active_orbitals, n_active_orbitals),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      xmvb::vb::packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  const int n_determinants = pair_weights.n_determinants;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const auto& alpha_occ_L =
        input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)];
    const auto& beta_occ_L =
        input.structure_data.beta_det[xmvb::to_size(determinant_index_left)];
    const int alpha_unique_left =
        same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id[xmvb::to_size(
            determinant_index_left)];
    const int beta_unique_left =
        same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id[xmvb::to_size(
            determinant_index_left)];
    for (int determinant_index_right = 0;
         determinant_index_right < n_determinants;
         ++determinant_index_right) {
      const std::size_t ordered_index =
          xmvb::to_size(determinant_index_left) * n_determinants +
          xmvb::to_size(determinant_index_right);
      const double hamiltonian_weight =
          pair_weights.ordered_hamiltonian_weights[ordered_index];
      const double overlap_weight =
          pair_weights.ordered_overlap_weights[ordered_index];
      if (std::abs(hamiltonian_weight) <= kContributionTolerance &&
          std::abs(overlap_weight) <= kContributionTolerance) {
        continue;
      }

      const auto& alpha_occ_R =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      const auto& beta_occ_R =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      const int alpha_unique_right =
          same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id[xmvb::to_size(
              determinant_index_right)];
      const int beta_unique_right =
          same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id[xmvb::to_size(
              determinant_index_right)];

      const auto& alpha_pair =
          same_spin_pair_cache.alpha_pair_cache_ref()[xmvb::vb::ordered_spin_pair_storage_index(
              alpha_unique_left,
              alpha_unique_right,
              same_spin_pair_cache.n_unique_alpha)];
      const auto& beta_pair =
          same_spin_pair_cache.beta_pair_cache_ref()[xmvb::vb::ordered_spin_pair_storage_index(
              beta_unique_left,
              beta_unique_right,
              same_spin_pair_cache.n_unique_beta)];
      if (alpha_pair.overlap_result.nullity != 0 ||
          beta_pair.overlap_result.nullity != 0 ||
          alpha_pair.overlap_result.overlap_determinant == 0.0 ||
          beta_pair.overlap_result.overlap_determinant == 0.0) {
        throw std::runtime_error(
            "reference same-spin check requires nullity == 0 on both spin sectors");
      }
      if (!alpha_pair.has_same_spin_phi_cache || !beta_pair.has_same_spin_phi_cache) {
        throw std::runtime_error(
            "reference same-spin check requires cached same-spin phi payloads");
      }

      const DenseMatrix alpha_cofactor_1st =
          xmvb::vb::calc_cofactor_1st(alpha_pair.overlap_result);
      const DenseMatrix beta_cofactor_1st =
          xmvb::vb::calc_cofactor_1st(beta_pair.overlap_result);
      const double alpha_weight =
          hamiltonian_weight * beta_pair.overlap_result.overlap_determinant;
      const double beta_weight =
          hamiltonian_weight * alpha_pair.overlap_result.overlap_determinant;
      accumulate_one_electron_gradient_contribution(
          alpha_occ_L,
          alpha_occ_R,
          alpha_cofactor_1st,
          alpha_weight,
          &active_one_electron_gradient);
      accumulate_one_electron_gradient_contribution(
          beta_occ_L,
          beta_occ_R,
          beta_cofactor_1st,
          beta_weight,
          &active_one_electron_gradient);

      const double phi_sum =
          alpha_pair.same_spin_total_phi + beta_pair.same_spin_total_phi;
      const double alpha_determinant_overlap_weight =
          overlap_weight * beta_pair.overlap_result.overlap_determinant +
          hamiltonian_weight * beta_pair.overlap_result.overlap_determinant * phi_sum;
      const double beta_determinant_overlap_weight =
          overlap_weight * alpha_pair.overlap_result.overlap_determinant +
          hamiltonian_weight * alpha_pair.overlap_result.overlap_determinant * phi_sum;
      xmvb::vb::accumulate_spin_overlap_gradient(
          alpha_occ_L,
          alpha_occ_R,
          alpha_pair.overlap_result,
          alpha_determinant_overlap_weight,
          hamiltonian_weight * beta_pair.overlap_result.overlap_determinant *
              alpha_pair.same_spin_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          beta_occ_L,
          beta_occ_R,
          beta_pair.overlap_result,
          beta_determinant_overlap_weight,
          hamiltonian_weight * alpha_pair.overlap_result.overlap_determinant *
              beta_pair.same_spin_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);

      accumulate_same_spin_two_electron_gradient_contribution(
          alpha_occ_L,
          alpha_occ_R,
          alpha_cofactor_1st,
          alpha_pair.overlap_result.overlap_determinant,
          alpha_weight,
          &result.packed_active_two_electron_gradient);
      accumulate_same_spin_two_electron_gradient_contribution(
          beta_occ_L,
          beta_occ_R,
          beta_cofactor_1st,
          beta_pair.overlap_result.overlap_determinant,
          beta_weight,
          &result.packed_active_two_electron_gradient);
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
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

    // Reuse the accepted-point context builder already validated by the active
    // gradient path so the exact same ordered same-spin cache and cached
    // `same_spin_phi` payloads are available to the biorthogonal diagnostic.
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
    const auto pair_weights =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
                selected_state_matrices);

    const auto matrix_form_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_same_spin_matrix_backward_contribution(
                second_order_context.same_spin_pair_cache,
                selected_state_matrices,
                load_result.input.orbital_preparation_input.n_active_orbitals);
    const auto reference_contribution =
        build_reference_same_spin_contribution(
            load_result.input,
            second_order_context.same_spin_pair_cache,
            pair_weights,
            load_result.input.orbital_preparation_input.n_active_orbitals);

    const double overlap_max_abs =
        max_abs_vector_difference(
            matrix_form_contribution.active_orbital_overlap_gradient,
            reference_contribution.active_orbital_overlap_gradient);
    const double one_electron_max_abs =
        max_abs_vector_difference(
            matrix_form_contribution.active_one_electron_gradient,
            reference_contribution.active_one_electron_gradient);
    const double two_electron_max_abs =
        max_abs_vector_difference(
            matrix_form_contribution.packed_active_two_electron_gradient,
            reference_contribution.packed_active_two_electron_gradient);
    const double worst_max_abs =
        std::max(overlap_max_abs, std::max(one_electron_max_abs, two_electron_max_abs));

    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "n_determinants = " << evaluation_result.n_determinants << '\n';
    std::cout << "n_unique_alpha = " << selected_state_matrices.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_state_matrices.n_unique_beta << '\n';
    std::cout << "ordered_hamiltonian_weight_max_abs = ";
    if (pair_weights.ordered_hamiltonian_weights.empty()) {
      std::cout << 0.0 << '\n';
    } else {
      double max_abs = 0.0;
      for (const double value : pair_weights.ordered_hamiltonian_weights) {
        max_abs = std::max(max_abs, std::abs(value));
      }
      std::cout << max_abs << '\n';
    }
    std::cout << "ordered_overlap_weight_max_abs = ";
    if (pair_weights.ordered_overlap_weights.empty()) {
      std::cout << 0.0 << '\n';
    } else {
      double max_abs = 0.0;
      for (const double value : pair_weights.ordered_overlap_weights) {
        max_abs = std::max(max_abs, std::abs(value));
      }
      std::cout << max_abs << '\n';
    }
    std::cout << "overlap_gradient_max_abs = " << overlap_max_abs << '\n';
    std::cout << "one_electron_gradient_max_abs = " << one_electron_max_abs << '\n';
    std::cout << "two_electron_gradient_max_abs = " << two_electron_max_abs << '\n';
    std::cout << "worst_max_abs = " << worst_max_abs << '\n';

    if (worst_max_abs > options.tolerance) {
      std::cerr << "biorthogonal same-spin matrix backward check failed: worst_max_abs = "
                << worst_max_abs
                << " > tolerance = " << options.tolerance << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_same_spin_matrix_backward failed: "
              << exception.what() << '\n';
    return 1;
  }
}
