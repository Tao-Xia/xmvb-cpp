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
#include "vb/biorthogonal_vbscf/biorthogonal_one_electron_transform_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

constexpr double kContributionTolerance = 1.0e-15;

struct Options {
  std::string input_path;
  int subspace_size = 0;
  double tolerance = 1.0e-10;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
};

struct SpinExcitation {
  std::vector<int> removed_orbitals;
  std::vector<int> added_orbitals;
  int rank = 0;
};

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_one_electron_transform_backward <input.xmi>"
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

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

double apply_annihilation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator == occupied_orbitals->end() || *iterator != orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->erase(iterator);
  return parity_sign(parity);
}

double apply_creation(
    int orbital,
    std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  const auto iterator =
      std::lower_bound(occupied_orbitals->begin(), occupied_orbitals->end(), orbital);
  if (iterator != occupied_orbitals->end() && *iterator == orbital) {
    return 0.0;
  }
  const int parity = static_cast<int>(std::distance(occupied_orbitals->begin(), iterator));
  occupied_orbitals->insert(iterator, orbital);
  return parity_sign(parity);
}

SpinExcitation build_spin_excitation(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals) {
  SpinExcitation excitation;
  std::set_difference(
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      std::back_inserter(excitation.removed_orbitals));
  std::set_difference(
      left_occupied_orbitals.begin(),
      left_occupied_orbitals.end(),
      right_occupied_orbitals.begin(),
      right_occupied_orbitals.end(),
      std::back_inserter(excitation.added_orbitals));
  if (excitation.removed_orbitals.size() != excitation.added_orbitals.size()) {
    throw std::runtime_error("left/right determinants carry inconsistent electron counts");
  }
  excitation.rank = static_cast<int>(excitation.removed_orbitals.size());
  return excitation;
}

double compute_excitation_phase(
    const std::vector<int>& left_occupied_orbitals,
    const std::vector<int>& right_occupied_orbitals,
    const SpinExcitation& excitation) {
  if (excitation.rank == 0) {
    return left_occupied_orbitals == right_occupied_orbitals ? 1.0 : 0.0;
  }

  std::vector<int> occupied_orbitals = right_occupied_orbitals;
  double phase = 1.0;
  for (int removed_orbital : excitation.removed_orbitals) {
    phase *= apply_annihilation(removed_orbital, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  for (auto added_iterator = excitation.added_orbitals.rbegin();
       added_iterator != excitation.added_orbitals.rend();
       ++added_iterator) {
    phase *= apply_creation(*added_iterator, &occupied_orbitals);
    if (phase == 0.0) {
      return 0.0;
    }
  }
  return occupied_orbitals == left_occupied_orbitals ? phase : 0.0;
}

xmvb::vb::biorthogonal_vbscf::BiorthogonalOneElectronTransformBackwardContribution
build_reference_one_electron_transform_contribution(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalIntegrals& orbital_integrals,
    const std::vector<double>& right_right_one_electron,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights) {
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  DenseMatrix left_right_one_electron_gradient =
      DenseMatrix::Zero(n_active_orbitals, n_active_orbitals);

  const int n_determinants = pair_weights.n_determinants;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const auto& alpha_occ_L =
        input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)];
    const auto& beta_occ_L =
        input.structure_data.beta_det[xmvb::to_size(determinant_index_left)];
    for (int determinant_index_right = 0;
         determinant_index_right < n_determinants;
         ++determinant_index_right) {
      const std::size_t ordered_index =
          xmvb::to_size(determinant_index_left) * n_determinants +
          xmvb::to_size(determinant_index_right);
      const double weight =
          pair_weights.ordered_hamiltonian_weights[ordered_index];
      if (std::abs(weight) <= kContributionTolerance) {
        continue;
      }

      const auto& alpha_occ_R =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      const auto& beta_occ_R =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      const SpinExcitation alpha_excitation =
          build_spin_excitation(alpha_occ_L, alpha_occ_R);
      const SpinExcitation beta_excitation =
          build_spin_excitation(beta_occ_L, beta_occ_R);

      if (alpha_excitation.rank + beta_excitation.rank > 1) {
        continue;
      }
      if (alpha_excitation.rank == 0 && beta_excitation.rank == 0) {
        for (const int occupied_orbital : alpha_occ_L) {
          left_right_one_electron_gradient(occupied_orbital, occupied_orbital) += weight;
        }
        for (const int occupied_orbital : beta_occ_L) {
          left_right_one_electron_gradient(occupied_orbital, occupied_orbital) += weight;
        }
        continue;
      }
      if (alpha_excitation.rank == 1 && beta_excitation.rank == 0) {
        const double phase =
            compute_excitation_phase(alpha_occ_L, alpha_occ_R, alpha_excitation);
        if (phase == 0.0) {
          throw std::runtime_error("failed to build alpha one-electron reference phase");
        }
        left_right_one_electron_gradient(
            alpha_excitation.added_orbitals.front(),
            alpha_excitation.removed_orbitals.front()) += weight * phase;
        continue;
      }
      if (alpha_excitation.rank == 0 && beta_excitation.rank == 1) {
        const double phase =
            compute_excitation_phase(beta_occ_L, beta_occ_R, beta_excitation);
        if (phase == 0.0) {
          throw std::runtime_error("failed to build beta one-electron reference phase");
        }
        left_right_one_electron_gradient(
            beta_excitation.added_orbitals.front(),
            beta_excitation.removed_orbitals.front()) += weight * phase;
      }
    }
  }

  const Eigen::Map<const DenseMatrix> right_right_one_electron_matrix(
      right_right_one_electron.data(),
      n_active_orbitals,
      n_active_orbitals);
  const DenseMatrix active_one_electron_gradient =
      orbital_integrals.left_dual_from_right_transform.transpose() *
      left_right_one_electron_gradient;
  const DenseMatrix left_dual_from_right_transform_gradient =
      left_right_one_electron_gradient *
      right_right_one_electron_matrix.transpose();
  const DenseMatrix active_orbital_overlap_gradient =
      -orbital_integrals.left_dual_from_right_transform.transpose() *
      left_dual_from_right_transform_gradient *
      orbital_integrals.left_dual_from_right_transform.transpose();

  xmvb::vb::biorthogonal_vbscf::BiorthogonalOneElectronTransformBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
      active_orbital_overlap_gradient.data(),
      active_orbital_overlap_gradient.data() +
          active_orbital_overlap_gradient.size());
  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() +
          active_one_electron_gradient.size());
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

    xmvb::vb::CppActiveSpaceGradientEvaluator gradient_evaluator(options.algorithm);
    const auto full_gradient_result =
        gradient_evaluator.evaluate(
            load_result.input,
            load_result.nuclear_repulsion_energy);
    if (full_gradient_result.second_order_context == nullptr) {
      throw std::runtime_error("full_gradient_result.second_order_context is null");
    }
    const auto& second_order_context = *full_gradient_result.second_order_context;
    const auto& prepared_active_space = second_order_context.prepared_active_space;

    const auto evaluation_result =
        xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_subspace(
            load_result.input,
            prepared_active_space,
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
    const auto orbital_integrals =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
            load_result.input.orbital_preparation_input.n_active_orbitals,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act);

    const auto matrix_form_contribution =
        xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_one_electron_transform_backward_contribution(
                second_order_context.same_spin_pair_cache,
                selected_state_matrices,
                orbital_integrals,
                prepared_active_space.active_space_one_electron_result.h1e_act);
    const auto reference_contribution =
        build_reference_one_electron_transform_contribution(
            load_result.input,
            orbital_integrals,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            pair_weights);

    const double overlap_max_abs =
        max_abs_vector_difference(
            matrix_form_contribution.active_orbital_overlap_gradient,
            reference_contribution.active_orbital_overlap_gradient);
    const double one_electron_max_abs =
        max_abs_vector_difference(
            matrix_form_contribution.active_one_electron_gradient,
            reference_contribution.active_one_electron_gradient);
    const double worst_max_abs =
        std::max(overlap_max_abs, one_electron_max_abs);

    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "n_determinants = " << evaluation_result.n_determinants << '\n';
    std::cout << "n_unique_alpha = " << selected_state_matrices.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_state_matrices.n_unique_beta << '\n';
    std::cout << "transform_overlap_gradient_max_abs = " << overlap_max_abs << '\n';
    std::cout << "transform_one_electron_gradient_max_abs = " << one_electron_max_abs << '\n';
    std::cout << "worst_max_abs = " << worst_max_abs << '\n';

    if (worst_max_abs > options.tolerance) {
      std::cerr << "biorthogonal one-electron transform backward check failed: worst_max_abs = "
                << worst_max_abs
                << " > tolerance = " << options.tolerance << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_one_electron_transform_backward failed: "
              << exception.what() << '\n';
    return 1;
  }
}
