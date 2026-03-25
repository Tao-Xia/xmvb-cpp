#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

enum class Spin {
  Alpha,
  Beta,
};

enum class Mode {
  Same,
  Opposite,
  Full,
};

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  Mode mode = Mode::Same;
  Spin spin = Spin::Alpha;
  int determinant_index_left = 0;
  int determinant_index_right = 0;
  int count = 8;
  double step = 1.0e-6;
};

void print_usage() {
  std::cerr << "usage: check_same_spin_overlap_gradient <input.xmi> "
               "[--algorithm original] "
               "[--mode same|opposite|full] "
               "[--spin alpha|beta] "
               "[--left i] [--right j] "
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
    if (argument_name == "--mode") {
      if (argument_value == "same") {
        options.mode = Mode::Same;
      } else if (argument_value == "opposite") {
        options.mode = Mode::Opposite;
      } else if (argument_value == "full") {
        options.mode = Mode::Full;
      } else {
        throw std::invalid_argument("invalid mode: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--spin") {
      if (argument_value == "alpha") {
        options.spin = Spin::Alpha;
      } else if (argument_value == "beta") {
        options.spin = Spin::Beta;
      } else {
        throw std::invalid_argument("invalid spin: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--left") {
      options.determinant_index_left = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right") {
      options.determinant_index_right = std::stoi(argument_value);
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

double evaluate_same_spin_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    xmvb::vb::VBSCFAlgorithm algorithm) {
  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  xmvb::vb::DeterminantHamiltonianResolver hamiltonian_resolver(algorithm);
  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      occ_L,
      occ_R,
      active_orbital_overlap_matrix,
      n_active_orbitals);
  const auto det_ovlp_result =
      overlap_resolver.resolve(overlap_submatrix, static_cast<int>(occ_L.size()));
  return hamiltonian_resolver.resolve(
      occ_L,
      occ_R,
      overlap_submatrix,
      det_ovlp_result,
      h1e_act,
      n_active_orbitals,
      packed_active_two_electron_integrals).total_hamiltonian;
}

double evaluate_opposite_spin_hamiltonian(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& active_orbital_overlap_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals) {
  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const auto alpha_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      alpha_occ_L,
      alpha_occ_R,
      active_orbital_overlap_matrix,
      n_active_orbitals);
  const auto beta_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      beta_occ_L,
      beta_occ_R,
      active_orbital_overlap_matrix,
      n_active_orbitals);
  const auto alpha_overlap_result =
      overlap_resolver.resolve(alpha_overlap_submatrix, static_cast<int>(alpha_occ_L.size()));
  const auto beta_overlap_result =
      overlap_resolver.resolve(beta_overlap_submatrix, static_cast<int>(beta_occ_L.size()));

  if (alpha_overlap_result.nullity >= 2 || beta_overlap_result.nullity >= 2) {
    return 0.0;
  }
  const xmvb::vb::Matrix alpha_cofactor_1st =
      xmvb::vb::calc_cofactor_1st(alpha_overlap_result);
  const xmvb::vb::Matrix beta_cofactor_1st =
      xmvb::vb::calc_cofactor_1st(beta_overlap_result);
  double hamiltonian = 0.0;
  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[static_cast<std::size_t>(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[static_cast<std::size_t>(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          const int two_electron_index = xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          hamiltonian +=
              alpha_cofactor * beta_cofactor *
              packed_active_two_electron_integrals[static_cast<std::size_t>(two_electron_index)];
        }
      }
    }
  }
  return hamiltonian;
}

double evaluate_full_pair_hamiltonian(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& active_orbital_overlap_matrix,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    xmvb::vb::VBSCFAlgorithm algorithm) {
  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  xmvb::vb::DeterminantHamiltonianResolver hamiltonian_resolver(algorithm);
  const auto alpha_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      alpha_occ_L,
      alpha_occ_R,
      active_orbital_overlap_matrix,
      n_active_orbitals);
  const auto beta_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      beta_occ_L,
      beta_occ_R,
      active_orbital_overlap_matrix,
      n_active_orbitals);
  const auto alpha_overlap_result =
      overlap_resolver.resolve(alpha_overlap_submatrix, static_cast<int>(alpha_occ_L.size()));
  const auto beta_overlap_result =
      overlap_resolver.resolve(beta_overlap_submatrix, static_cast<int>(beta_occ_L.size()));
  const auto alpha_hamiltonian_result = hamiltonian_resolver.resolve(
      alpha_occ_L,
      alpha_occ_R,
      alpha_overlap_submatrix,
      alpha_overlap_result,
      h1e_act,
      n_active_orbitals,
      packed_active_two_electron_integrals);
  const auto beta_hamiltonian_result = hamiltonian_resolver.resolve(
      beta_occ_L,
      beta_occ_R,
      beta_overlap_submatrix,
      beta_overlap_result,
      h1e_act,
      n_active_orbitals,
      packed_active_two_electron_integrals);
  return alpha_hamiltonian_result.total_hamiltonian * beta_overlap_result.overlap_determinant +
      beta_hamiltonian_result.total_hamiltonian * alpha_overlap_result.overlap_determinant +
      evaluate_opposite_spin_hamiltonian(
          alpha_occ_L,
          alpha_occ_R,
          beta_occ_L,
          beta_occ_R,
          active_orbital_overlap_matrix,
          n_active_orbitals,
          packed_active_two_electron_integrals);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    xmvb::vb::CppActiveSpaceGradientEvaluator active_space_evaluator(options.algorithm);
    const auto baseline =
        active_space_evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
    const auto& alpha_occupied_by_determinant =
        load_result.input.structure_data.alpha_det;
    const auto& beta_occupied_by_determinant =
        load_result.input.structure_data.beta_det;
    const auto& occupied_by_determinant =
        (options.spin == Spin::Alpha)
            ? alpha_occupied_by_determinant
            : beta_occupied_by_determinant;
    if (options.determinant_index_left < 0 ||
        static_cast<std::size_t>(options.determinant_index_left) >= occupied_by_determinant.size() ||
        options.determinant_index_right < 0 ||
        static_cast<std::size_t>(options.determinant_index_right) >= occupied_by_determinant.size()) {
      throw std::out_of_range("determinant index out of range");
    }

    const auto& occ_L =
        occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
    const auto& occ_R =
        occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
    const int n_active_orbitals = load_result.input.orbital_preparation_input.n_active_orbitals;

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    std::vector<double> analytic_gradient(
        baseline.active_orbital_overlap_matrix.size(),
        0.0);
    double baseline_hamiltonian = 0.0;
    if (options.mode == Mode::Same) {
      const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
          occ_L,
          occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto det_ovlp_result =
          overlap_resolver.resolve(overlap_submatrix, static_cast<int>(occ_L.size()));

      xmvb::vb::Matrix inverse_overlap_gradient;
      xmvb::vb::SameSpinPhiResult phi_result =
          xmvb::vb::compute_same_spin_original_phi(
              occ_L,
              occ_R,
              baseline.active_space_one_electron_result.h1e_act,
              n_active_orbitals,
              baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
              det_ovlp_result,
              &inverse_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          occ_L,
          occ_R,
          det_ovlp_result,
          phi_result.total_phi,
          inverse_overlap_gradient,
          n_active_orbitals,
          &analytic_gradient);
      baseline_hamiltonian = evaluate_same_spin_hamiltonian(
          occ_L,
          occ_R,
          baseline.active_orbital_overlap_matrix,
          baseline.active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          options.algorithm);
    } else if (options.mode == Mode::Opposite) {
      const auto& alpha_occ_L =
          alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
      const auto& alpha_occ_R =
          alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
      const auto& beta_occ_L =
          beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
      const auto& beta_occ_R =
          beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
      const auto alpha_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
          alpha_occ_L,
          alpha_occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto beta_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
          beta_occ_L,
          beta_occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto alpha_overlap_result = overlap_resolver.resolve(
          alpha_overlap_submatrix,
          static_cast<int>(alpha_occ_L.size()));
      const auto beta_overlap_result = overlap_resolver.resolve(
          beta_overlap_submatrix,
          static_cast<int>(beta_occ_L.size()));
      xmvb::vb::Matrix alpha_inverse_overlap_gradient;
      xmvb::vb::Matrix beta_inverse_overlap_gradient;
      const double opposite_phi = xmvb::vb::compute_opposite_spin_original_phi(
          alpha_occ_L,
          alpha_occ_R,
          alpha_overlap_result,
          beta_occ_L,
          beta_occ_R,
          beta_overlap_result,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          &alpha_inverse_overlap_gradient,
          &beta_inverse_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          alpha_occ_L,
          alpha_occ_R,
          alpha_overlap_result,
          beta_overlap_result.overlap_determinant * opposite_phi,
          beta_overlap_result.overlap_determinant * alpha_inverse_overlap_gradient,
          n_active_orbitals,
          &analytic_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          beta_occ_L,
          beta_occ_R,
          beta_overlap_result,
          alpha_overlap_result.overlap_determinant * opposite_phi,
          alpha_overlap_result.overlap_determinant * beta_inverse_overlap_gradient,
          n_active_orbitals,
          &analytic_gradient);
      baseline_hamiltonian = evaluate_opposite_spin_hamiltonian(
          alpha_occ_L,
          alpha_occ_R,
          beta_occ_L,
          beta_occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals);
    } else {
      const auto& alpha_occ_L =
          alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
      const auto& alpha_occ_R =
          alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
      const auto& beta_occ_L =
          beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
      const auto& beta_occ_R =
          beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
      const auto alpha_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
          alpha_occ_L,
          alpha_occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto beta_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
          beta_occ_L,
          beta_occ_R,
          baseline.active_orbital_overlap_matrix,
          n_active_orbitals);
      const auto alpha_overlap_result = overlap_resolver.resolve(
          alpha_overlap_submatrix,
          static_cast<int>(alpha_occ_L.size()));
      const auto beta_overlap_result = overlap_resolver.resolve(
          beta_overlap_submatrix,
          static_cast<int>(beta_occ_L.size()));
      xmvb::vb::Matrix alpha_inverse_overlap_gradient;
      xmvb::vb::Matrix beta_inverse_overlap_gradient;
      xmvb::vb::SameSpinPhiResult alpha_phi_result =
          xmvb::vb::compute_same_spin_original_phi(
              alpha_occ_L,
              alpha_occ_R,
              baseline.active_space_one_electron_result.h1e_act,
              n_active_orbitals,
              baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
              alpha_overlap_result,
              &alpha_inverse_overlap_gradient);
      xmvb::vb::SameSpinPhiResult beta_phi_result =
          xmvb::vb::compute_same_spin_original_phi(
              beta_occ_L,
              beta_occ_R,
              baseline.active_space_one_electron_result.h1e_act,
              n_active_orbitals,
              baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
              beta_overlap_result,
              &beta_inverse_overlap_gradient);
      const double opposite_phi = xmvb::vb::compute_opposite_spin_original_phi(
          alpha_occ_L,
          alpha_occ_R,
          alpha_overlap_result,
          beta_occ_L,
          beta_occ_R,
          beta_overlap_result,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          &alpha_inverse_overlap_gradient,
          &beta_inverse_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          alpha_occ_L,
          alpha_occ_R,
          alpha_overlap_result,
          beta_overlap_result.overlap_determinant *
              (alpha_phi_result.total_phi + beta_phi_result.total_phi + opposite_phi),
          beta_overlap_result.overlap_determinant * alpha_inverse_overlap_gradient,
          n_active_orbitals,
          &analytic_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          beta_occ_L,
          beta_occ_R,
          beta_overlap_result,
          alpha_overlap_result.overlap_determinant *
              (alpha_phi_result.total_phi + beta_phi_result.total_phi + opposite_phi),
          alpha_overlap_result.overlap_determinant * beta_inverse_overlap_gradient,
          n_active_orbitals,
          &analytic_gradient);
      baseline_hamiltonian = evaluate_full_pair_hamiltonian(
          alpha_occ_L,
          alpha_occ_R,
          beta_occ_L,
          beta_occ_R,
          baseline.active_orbital_overlap_matrix,
          baseline.active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
          options.algorithm);
    }

    std::vector<std::pair<double, int>> ranked_entries;
    ranked_entries.reserve(analytic_gradient.size());
    for (std::size_t index = 0; index < analytic_gradient.size(); ++index) {
      ranked_entries.emplace_back(std::abs(analytic_gradient[index]), static_cast<int>(index));
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
    std::cout << "mode = "
              << (options.mode == Mode::Same ? "same" :
                  (options.mode == Mode::Opposite ? "opposite" : "full"))
              << '\n';
    std::cout << "spin = " << (options.spin == Spin::Alpha ? "alpha" : "beta") << '\n';
    std::cout << "determinant_index_left = " << options.determinant_index_left << '\n';
    std::cout << "determinant_index_right = " << options.determinant_index_right << '\n';
    std::cout << "baseline_same_spin_hamiltonian = " << baseline_hamiltonian << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int entry_index = ranked_entries[static_cast<std::size_t>(report_index)].second;
      std::vector<double> plus_overlap = baseline.active_orbital_overlap_matrix;
      std::vector<double> minus_overlap = baseline.active_orbital_overlap_matrix;
      plus_overlap[static_cast<std::size_t>(entry_index)] += options.step;
      minus_overlap[static_cast<std::size_t>(entry_index)] -= options.step;
      double plus_hamiltonian = 0.0;
      double minus_hamiltonian = 0.0;
      if (options.mode == Mode::Same) {
        plus_hamiltonian = evaluate_same_spin_hamiltonian(
            occ_L,
            occ_R,
            plus_overlap,
            baseline.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
            options.algorithm);
        minus_hamiltonian = evaluate_same_spin_hamiltonian(
            occ_L,
            occ_R,
            minus_overlap,
            baseline.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
            options.algorithm);
      } else if (options.mode == Mode::Opposite) {
        const auto& alpha_occ_L =
            alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
        const auto& alpha_occ_R =
            alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
        const auto& beta_occ_L =
            beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
        const auto& beta_occ_R =
            beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
        plus_hamiltonian = evaluate_opposite_spin_hamiltonian(
            alpha_occ_L,
            alpha_occ_R,
            beta_occ_L,
            beta_occ_R,
            plus_overlap,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals);
        minus_hamiltonian = evaluate_opposite_spin_hamiltonian(
            alpha_occ_L,
            alpha_occ_R,
            beta_occ_L,
            beta_occ_R,
            minus_overlap,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals);
      } else {
        const auto& alpha_occ_L =
            alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
        const auto& alpha_occ_R =
            alpha_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
        const auto& beta_occ_L =
            beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_left)];
        const auto& beta_occ_R =
            beta_occupied_by_determinant[static_cast<std::size_t>(options.determinant_index_right)];
        plus_hamiltonian = evaluate_full_pair_hamiltonian(
            alpha_occ_L,
            alpha_occ_R,
            beta_occ_L,
            beta_occ_R,
            plus_overlap,
            baseline.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
            options.algorithm);
        minus_hamiltonian = evaluate_full_pair_hamiltonian(
            alpha_occ_L,
            alpha_occ_R,
            beta_occ_L,
            beta_occ_R,
            minus_overlap,
            baseline.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            baseline.active_space_two_electron_result.packed_active_two_electron_integrals,
            options.algorithm);
      }
      const double finite_difference = (plus_hamiltonian - minus_hamiltonian) / (2.0 * options.step);
      const double analytic = analytic_gradient[static_cast<std::size_t>(entry_index)];
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "entry[" << report_index << "]"
                << " index=" << entry_index
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
