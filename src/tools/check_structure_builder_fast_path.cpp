#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
};

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: check_structure_builder_fast_path <input.xmi> "
        "[--standard-two-electron-mode auto|exact|ri]");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    if (argument == "--standard-two-electron-mode") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(
            "--standard-two-electron-mode requires a value");
      }
      const std::string mode = argv[++argument_index];
      if (mode == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (mode == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (mode == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "unknown --standard-two-electron-mode value: " + mode);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }

  return options;
}

double compute_max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  double max_abs_difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_abs_difference =
        std::max(max_abs_difference, std::abs(left[index] - right[index]));
  }
  return max_abs_difference;
}

void print_largest_differences(
    const std::string& label,
    const std::vector<double>& fast_values,
    const std::vector<double>& reference_values,
    int count) {
  if (fast_values.size() != reference_values.size()) {
    throw std::invalid_argument("vector size mismatch");
  }

  std::vector<std::pair<double, int>> ranked_entries;
  ranked_entries.reserve(fast_values.size());
  for (std::size_t index = 0; index < fast_values.size(); ++index) {
    ranked_entries.emplace_back(
        std::abs(fast_values[index] - reference_values[index]),
        static_cast<int>(index));
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
      std::min(count, static_cast<int>(ranked_entries.size()));
  for (int report_index = 0; report_index < n_to_report; ++report_index) {
    const int flat_index = ranked_entries[xmvb::to_size(report_index)].second;
    std::cout << label
              << "_diff[" << report_index << "] index=" << flat_index
              << " fast=" << fast_values[xmvb::to_size(flat_index)]
              << " reference=" << reference_values[xmvb::to_size(flat_index)]
              << " abs_diff=" << ranked_entries[xmvb::to_size(report_index)].first
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto prepared_active_space = xmvb::vb::prepare_active_space_context(
        load_result.input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);

    std::vector<double> packed_active_eri =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    if (packed_active_eri.empty()) {
      packed_active_eri = xmvb::vb::reconstruct_packed_active_two_electron_integrals(
          xmvb::vb::make_active_space_two_electron_view(
              prepared_active_space.active_space_two_electron_result),
          load_result.input.orbital_preparation_input.n_active_orbitals);
    }

    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto pair_evaluator = structure_builder.make_pair_evaluator();
    const auto same_spin_pair_cache = xmvb::vb::build_same_spin_pair_cache_context(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        pair_evaluator,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        prepared_active_space.active_space_two_electron_result);
    const auto fast_start = std::chrono::high_resolution_clock::now();
    const auto fast_result = structure_builder.build(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        load_result.input.structure_data.determinant_to_structure_terms,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        prepared_active_space.active_space_two_electron_result,
        load_result.input.structure_data.n_structures,
        same_spin_pair_cache);
    const auto fast_end = std::chrono::high_resolution_clock::now();
    const auto exact_same_spin_pair_cache = xmvb::vb::build_same_spin_pair_cache_context(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        pair_evaluator,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        packed_active_eri);
    const auto fast_exact_start = std::chrono::high_resolution_clock::now();
    const auto fast_exact_result = structure_builder.build(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        load_result.input.structure_data.determinant_to_structure_terms,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        packed_active_eri,
        load_result.input.structure_data.n_structures,
        exact_same_spin_pair_cache);
    const auto fast_exact_end = std::chrono::high_resolution_clock::now();
    const auto reference_start = std::chrono::high_resolution_clock::now();
    const auto reference_result = structure_builder.build_with_pair_evaluations(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        load_result.input.structure_data.determinant_to_structure_terms,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        packed_active_eri,
        load_result.input.structure_data.n_structures);
    const auto reference_end = std::chrono::high_resolution_clock::now();

    std::cout << std::setprecision(15);
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     options.standard_two_electron_mode)
              << '\n';
    std::cout << "cache_enabled = "
              << (same_spin_pair_cache.enabled() ? "true" : "false") << '\n';
    std::cout << "n_determinants = "
              << load_result.input.structure_data.alpha_det.size() << '\n';
    std::cout << "n_unique_alpha = "
              << same_spin_pair_cache.alpha_reuse_table.unique_determinants.size() << '\n';
    std::cout << "n_unique_beta = "
              << same_spin_pair_cache.beta_reuse_table.unique_determinants.size() << '\n';
    std::cout << "exact_cache_enabled = "
              << (exact_same_spin_pair_cache.enabled() ? "true" : "false") << '\n';
    std::cout << "fast_result_seconds = "
              << std::chrono::duration<double>(fast_end - fast_start).count() << '\n';
    std::cout << "fast_exact_result_seconds = "
              << std::chrono::duration<double>(fast_exact_end - fast_exact_start).count()
              << '\n';
    std::cout << "reference_result_seconds = "
              << std::chrono::duration<double>(reference_end - reference_start).count()
              << '\n';
    std::cout << "max_abs_overlap_diff = "
              << compute_max_abs_difference(
                     fast_result.overlap_matrix,
                     reference_result.structure_matrices.overlap_matrix)
              << '\n';
    std::cout << "max_abs_hamiltonian_diff = "
              << compute_max_abs_difference(
                     fast_result.hamiltonian_matrix,
                     reference_result.structure_matrices.hamiltonian_matrix)
              << '\n';
    std::cout << "max_abs_det_overlap_cache_diff = "
              << compute_max_abs_difference(
                     fast_result.determinant_overlap_cache,
                     reference_result.structure_matrices.determinant_overlap_cache)
              << '\n';
    std::cout << "max_abs_exact_overlap_diff = "
              << compute_max_abs_difference(
                     fast_exact_result.overlap_matrix,
                     reference_result.structure_matrices.overlap_matrix)
              << '\n';
    std::cout << "max_abs_exact_hamiltonian_diff = "
              << compute_max_abs_difference(
                     fast_exact_result.hamiltonian_matrix,
                     reference_result.structure_matrices.hamiltonian_matrix)
              << '\n';

    print_largest_differences(
        "overlap",
        fast_result.overlap_matrix,
        reference_result.structure_matrices.overlap_matrix,
        5);
    print_largest_differences(
        "hamiltonian",
        fast_result.hamiltonian_matrix,
        reference_result.structure_matrices.hamiltonian_matrix,
        5);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
