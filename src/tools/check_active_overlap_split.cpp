#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  int count = 4;
  double step = 1.0e-6;
};

struct StructurePairAdjoints {
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_active_overlap_split <input.xmi> "
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

double structure_upper_weight(
    const std::vector<double>& eigenvector_matrix,
    int n_structures,
    int structure_row,
    int structure_column) {
  const double coefficient_row =
      eigenvector_matrix[structure_row];
  const double coefficient_column =
      eigenvector_matrix[structure_column];
  if (structure_row == structure_column) {
    return coefficient_row * coefficient_column;
  }
  return 2.0 * coefficient_row * coefficient_column;
}

double structure_upper_overlap_weight(
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures,
    int structure_row,
    int structure_column) {
  return -eigenvalues.front() * structure_upper_weight(
      eigenvector_matrix,
      n_structures,
      structure_row,
      structure_column);
}

StructurePairAdjoints determinant_pair_structure_adjoints(
    const std::vector<xmvb::vb::StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<xmvb::vb::StructureExpansionTerm>& determinant_to_structures_right,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures) {
  StructurePairAdjoints adjoints;
  for (const auto& left_term : determinant_to_structures_left) {
    for (const auto& right_term : determinant_to_structures_right) {
      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }
      const double coefficient_product = left_term.coefficient * right_term.coefficient;
      adjoints.hamiltonian_weight += coefficient_product * structure_upper_weight(
          eigenvector_matrix,
          n_structures,
          left_term.structure_index,
          right_term.structure_index);
      adjoints.overlap_weight += coefficient_product * structure_upper_overlap_weight(
          eigenvector_matrix,
          eigenvalues,
          n_structures,
          left_term.structure_index,
          right_term.structure_index);
    }
  }
  return adjoints;
}

double evaluate_ground_state_energy(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int n_structures) {
  xmvb::core::GeneralizedEigensolver eigensolver;
  return eigensolver.solve(hamiltonian_matrix, overlap_matrix, n_structures).eigenvalues.front();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    xmvb::vb::CppActiveSpaceGradientEvaluator evaluator(options.algorithm);
    const auto baseline =
        evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
    const int n_active_orbitals = load_result.input.orbital_preparation_input.n_active_orbitals;
    const int n_determinants = static_cast<int>(
        load_result.input.structure_data.alpha_det.size());

    std::vector<double> overlap_only_gradient(
        baseline.active_orbital_overlap_matrix.size(),
        0.0);
    std::vector<double> hamiltonian_only_gradient(
        baseline.active_orbital_overlap_matrix.size(),
        0.0);

    xmvb::vb::DeterminantOverlapResolver determinant_overlap_resolver;
    for (int determinant_index_left = 0; determinant_index_left < n_determinants; ++determinant_index_left) {
      for (int determinant_index_right = 0;
           determinant_index_right < n_determinants;
           ++determinant_index_right) {
        const auto pair_adjoints = determinant_pair_structure_adjoints(
            load_result.input.structure_data.determinant_to_structure_terms[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.determinant_to_structure_terms[xmvb::to_size(determinant_index_right)],
            baseline.scf_result.eigenvector_matrix,
            baseline.scf_result.electronic_state_energies,
            load_result.input.structure_data.n_structures);
        if (pair_adjoints.hamiltonian_weight == 0.0 &&
            pair_adjoints.overlap_weight == 0.0) {
          continue;
        }

        const auto alpha_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)],
            baseline.active_orbital_overlap_matrix,
            n_active_orbitals);
        const auto beta_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_right)],
            baseline.active_orbital_overlap_matrix,
            n_active_orbitals);
        const auto alpha_result = determinant_overlap_resolver.resolve(
            alpha_overlap_submatrix,
            static_cast<int>(
                load_result.input.structure_data.alpha_det
                    [xmvb::to_size(determinant_index_left)].size()));
        const auto beta_result = determinant_overlap_resolver.resolve(
            beta_overlap_submatrix,
            static_cast<int>(
                load_result.input.structure_data.beta_det
                    [xmvb::to_size(determinant_index_left)].size()));
        if (alpha_result.nullity != 0 || beta_result.nullity != 0) {
          throw std::runtime_error("analytic overlap split diagnostic requires nullity == 0");
        }

        xmvb::vb::Matrix alpha_same_spin_inverse_overlap_gradient;
        xmvb::vb::Matrix beta_same_spin_inverse_overlap_gradient;
        xmvb::vb::Matrix alpha_opposite_spin_inverse_overlap_gradient;
        xmvb::vb::Matrix beta_opposite_spin_inverse_overlap_gradient;
        xmvb::vb::SameSpinPhiResult alpha_phi_result =
            xmvb::vb::compute_same_spin_original_phi(
                load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)],
                load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)],
                baseline.active_space_one_electron_result.h1e_act,
                n_active_orbitals,
                baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
                alpha_result,
                &alpha_same_spin_inverse_overlap_gradient);
        xmvb::vb::SameSpinPhiResult beta_phi_result =
            xmvb::vb::compute_same_spin_original_phi(
                load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_left)],
                load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_right)],
                baseline.active_space_one_electron_result.h1e_act,
                n_active_orbitals,
                baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
                beta_result,
                &beta_same_spin_inverse_overlap_gradient);
        const double opposite_spin_phi =
            xmvb::vb::compute_opposite_spin_original_phi(
                load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)],
                load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)],
                alpha_result,
                load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_left)],
                load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_right)],
                beta_result,
                baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
                &alpha_opposite_spin_inverse_overlap_gradient,
                &beta_opposite_spin_inverse_overlap_gradient);
        const xmvb::vb::Matrix alpha_inverse_overlap_gradient =
            alpha_same_spin_inverse_overlap_gradient +
            alpha_opposite_spin_inverse_overlap_gradient;
        const xmvb::vb::Matrix beta_inverse_overlap_gradient =
            beta_same_spin_inverse_overlap_gradient +
            beta_opposite_spin_inverse_overlap_gradient;

        const double alpha_ham_det_weight =
            pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
            (alpha_phi_result.total_phi + beta_phi_result.total_phi + opposite_spin_phi);
        const double beta_ham_det_weight =
            pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
            (alpha_phi_result.total_phi + beta_phi_result.total_phi + opposite_spin_phi);
        const xmvb::vb::Matrix zero_alpha_inverse =
            Matrix::Zero(alpha_inverse_overlap_gradient.rows(), alpha_inverse_overlap_gradient.cols());
        const xmvb::vb::Matrix zero_beta_inverse =
            Matrix::Zero(beta_inverse_overlap_gradient.rows(), beta_inverse_overlap_gradient.cols());

        xmvb::vb::accumulate_spin_overlap_gradient(
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)],
            alpha_result,
            pair_adjoints.overlap_weight * beta_result.overlap_determinant,
            zero_alpha_inverse,
            n_active_orbitals,
            &overlap_only_gradient);
        xmvb::vb::accumulate_spin_overlap_gradient(
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_right)],
            beta_result,
            pair_adjoints.overlap_weight * alpha_result.overlap_determinant,
            zero_beta_inverse,
            n_active_orbitals,
            &overlap_only_gradient);
        xmvb::vb::accumulate_spin_overlap_gradient(
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)],
            alpha_result,
            alpha_ham_det_weight,
            pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
                alpha_inverse_overlap_gradient,
            n_active_orbitals,
            &hamiltonian_only_gradient);
        xmvb::vb::accumulate_spin_overlap_gradient(
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_left)],
            load_result.input.structure_data.beta_det[xmvb::to_size(determinant_index_right)],
            beta_result,
            beta_ham_det_weight,
            pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
                beta_inverse_overlap_gradient,
            n_active_orbitals,
            &hamiltonian_only_gradient);
      }
    }

    std::vector<double> total_gradient = overlap_only_gradient;
    for (std::size_t index = 0; index < total_gradient.size(); ++index) {
      total_gradient[index] += hamiltonian_only_gradient[index];
    }

    std::vector<std::pair<double, int>> ranked_entries;
    ranked_entries.reserve(total_gradient.size());
    for (std::size_t index = 0; index < total_gradient.size(); ++index) {
      ranked_entries.emplace_back(std::abs(total_gradient[index]), static_cast<int>(index));
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

    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(options.algorithm);
    const int n_to_report = std::min(options.count, static_cast<int>(ranked_entries.size()));
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int entry_index = ranked_entries[xmvb::to_size(report_index)].second;
      std::vector<double> plus_overlap = baseline.active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = baseline.active_orbital_overlap_matrix;
      plus_overlap[xmvb::to_size(entry_index)] += options.step;
      minus_overlap[xmvb::to_size(entry_index)] -= options.step;

      const auto plus_structure = structure_builder.build(
          load_result.input.structure_data.alpha_det,
          load_result.input.structure_data.beta_det,
          load_result.input.structure_data.determinant_to_structure_terms,
          plus_overlap,
          baseline.active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          load_result.input.structure_data.n_structures);
      const auto minus_structure = structure_builder.build(
          load_result.input.structure_data.alpha_det,
          load_result.input.structure_data.beta_det,
          load_result.input.structure_data.determinant_to_structure_terms,
          minus_overlap,
          baseline.active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          load_result.input.structure_data.n_structures);

      const double fd_overlap_only = (
          evaluate_ground_state_energy(
              baseline.scf_result.structure_matrices.hamiltonian_matrix,
              plus_structure.overlap_matrix,
              load_result.input.structure_data.n_structures) -
          evaluate_ground_state_energy(
              baseline.scf_result.structure_matrices.hamiltonian_matrix,
              minus_structure.overlap_matrix,
              load_result.input.structure_data.n_structures)) / (2.0 * options.step);
      const double fd_hamiltonian_only = (
          evaluate_ground_state_energy(
              plus_structure.hamiltonian_matrix,
              baseline.scf_result.structure_matrices.overlap_matrix,
              load_result.input.structure_data.n_structures) -
          evaluate_ground_state_energy(
              minus_structure.hamiltonian_matrix,
              baseline.scf_result.structure_matrices.overlap_matrix,
              load_result.input.structure_data.n_structures)) / (2.0 * options.step);
      const double fd_total = (
          evaluate_ground_state_energy(
              plus_structure.hamiltonian_matrix,
              plus_structure.overlap_matrix,
              load_result.input.structure_data.n_structures) -
          evaluate_ground_state_energy(
              minus_structure.hamiltonian_matrix,
              minus_structure.overlap_matrix,
              load_result.input.structure_data.n_structures)) / (2.0 * options.step);

      std::cout << "entry[" << report_index << "]"
                << " index=" << entry_index
                << " analytic_overlap_only=" << overlap_only_gradient[xmvb::to_size(entry_index)]
                << " fd_overlap_only=" << fd_overlap_only
                << " analytic_hamiltonian_only=" << hamiltonian_only_gradient[xmvb::to_size(entry_index)]
                << " fd_hamiltonian_only=" << fd_hamiltonian_only
                << " analytic_total=" << total_gradient[xmvb::to_size(entry_index)]
                << " fd_total=" << fd_total
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
