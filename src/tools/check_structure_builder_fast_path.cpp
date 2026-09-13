#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/eigensolver.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/integrals/active/preparation/space.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

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

std::size_t projected_pair_value_count(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& pair_cache) {
  std::size_t count = 0;
  for (const auto& pair : pair_cache) {
    count += pair.opposite_spin_pair_cache
                 .first_order_cofactor_projection
                 .projected_pair_values.size();
    count += pair.opposite_spin_pair_cache
                 .inverse_overlap_projection
                 .projected_pair_values.size();
  }
  return count;
}

std::size_t projected_pair_value_count(
    const xmvb::vb::SameSpinPairCacheContext& cache) {
  std::size_t count = projected_pair_value_count(cache.alpha_pair_cache_ref());
  if (!cache.shares_same_spin_pair_cache_between_spins()) {
    count += projected_pair_value_count(cache.beta_pair_cache_ref());
  }
  return count;
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
    const int flat_index = ranked_entries[report_index].second;
    std::cout << label
              << "_diff[" << report_index << "] index=" << flat_index
              << " fast=" << fast_values[flat_index]
              << " reference=" << reference_values[flat_index]
              << " abs_diff=" << ranked_entries[report_index].first
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::VbScfInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_vbscf_input_with_timings(options.input_path, load_options);

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto prepared_active_space =
        xmvb::vb::prepare_timed_active_space_context(
            load_result.input,
            orbital_preparer,
            ao_effective_one_electron_builder,
            active_space_one_electron_builder,
            active_space_two_electron_builder)
            .prepared_active_space;

    std::vector<double> packed_active_eri =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    if (packed_active_eri.empty()) {
      packed_active_eri = xmvb::vb::reconstruct_packed_active_two_electron_integrals(
          xmvb::vb::make_active_space_two_electron_view(
              prepared_active_space.active_space_two_electron_result),
          load_result.input.orbital_preparation_input.n_active_orbitals);
    }

    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
    const auto pair_evaluator = structure_builder.make_pair_evaluator();
    const auto same_spin_pair_cache = xmvb::vb::build_same_spin_pair_cache_context(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        pair_evaluator,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        prepared_active_space.active_space_two_electron_result,
        xmvb::vb::SameSpinPairCacheBuildOptions{});
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
    const int n_structures =
        load_result.input.structure_data.n_structures;
    Eigen::MatrixXd trial_vectors(n_structures, 3);
    for (int column = 0; column < trial_vectors.cols(); ++column) {
      for (int row = 0; row < trial_vectors.rows(); ++row) {
        trial_vectors(row, column) =
            std::sin(0.013 * static_cast<double>((row + 1) * (column + 2)));
      }
    }
    const xmvb::vb::StructureAction structure_action(
        load_result.input.structure_data.determinant_to_structure_terms,
        n_structures,
        same_spin_pair_cache,
        prepared_active_space.active_space_two_electron_result,
        load_result.input.orbital_preparation_input.n_active_orbitals);
    const auto action_start = std::chrono::high_resolution_clock::now();
    const auto action_result = structure_action.apply(trial_vectors);
    const auto action_end = std::chrono::high_resolution_clock::now();
    const auto action_diagonal = structure_action.diagonal();
    const Eigen::Map<const Eigen::MatrixXd> dense_hamiltonian(
        fast_result.hamiltonian_matrix.data(), n_structures, n_structures);
    const Eigen::Map<const Eigen::MatrixXd> dense_overlap(
        fast_result.overlap_matrix.data(), n_structures, n_structures);
    const Eigen::MatrixXd reference_hamiltonian_action =
        dense_hamiltonian * trial_vectors;
    const Eigen::MatrixXd reference_overlap_action =
        dense_overlap * trial_vectors;
    const auto compact_pair_cache =
        xmvb::vb::build_same_spin_pair_cache_context(
            load_result.input.structure_data.alpha_det,
            load_result.input.structure_data.beta_det,
            pair_evaluator,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            load_result.input.orbital_preparation_input.n_active_orbitals,
            prepared_active_space.active_space_two_electron_result,
            xmvb::vb::SameSpinPairCacheBuildOptions{
                xmvb::vb::PairProjectionCache::SmallerSpin});
    const xmvb::vb::StructureAction compact_structure_action(
        load_result.input.structure_data.determinant_to_structure_terms,
        n_structures,
        compact_pair_cache,
        prepared_active_space.active_space_two_electron_result,
        load_result.input.orbital_preparation_input.n_active_orbitals);
    const auto compact_action_start = std::chrono::high_resolution_clock::now();
    const auto compact_action_result =
        compact_structure_action.apply(trial_vectors);
    const auto compact_action_end = std::chrono::high_resolution_clock::now();
    const auto compact_diagonal = compact_structure_action.diagonal();
    xmvb::core::GeneralizedEigensolver eigensolver;
    const auto dense_eigensolve_start = std::chrono::high_resolution_clock::now();
    const auto dense_eigenpairs = eigensolver.solve_dense(
        fast_result.hamiltonian_matrix,
        fast_result.overlap_matrix,
        n_structures);
    const auto dense_eigensolve_end = std::chrono::high_resolution_clock::now();
    double davidson_action_seconds = 0.0;
    const xmvb::core::GeneralizedEigenAction davidson_action =
        [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
          const auto action_call_start =
              std::chrono::high_resolution_clock::now();
          auto images = compact_structure_action.apply(vectors);
          davidson_action_seconds += std::chrono::duration<double>(
              std::chrono::high_resolution_clock::now() - action_call_start)
                                         .count();
          return xmvb::core::GeneralizedEigenActionResult{
              std::move(images.hamiltonian),
              std::move(images.overlap)};
        };
    const xmvb::core::DavidsonOptions davidson_options{
        1,
        2 * n_structures,
        std::min(n_structures, 128),
        1.0e-10};
    const auto davidson_start = std::chrono::high_resolution_clock::now();
    const auto davidson = eigensolver.solve_davidson(
        davidson_action,
        compact_diagonal.hamiltonian,
        compact_diagonal.overlap,
        davidson_options);
    const auto davidson_end = std::chrono::high_resolution_clock::now();
    const auto exact_same_spin_pair_cache = xmvb::vb::build_same_spin_pair_cache_context(
        load_result.input.structure_data.alpha_det,
        load_result.input.structure_data.beta_det,
        pair_evaluator,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        load_result.input.orbital_preparation_input.n_active_orbitals,
        packed_active_eri,
        xmvb::vb::SameSpinPairCacheBuildOptions{});
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
    std::cout << "n_structures = " << n_structures << '\n';
    std::cout << "n_unique_alpha = "
              << same_spin_pair_cache.alpha_reuse_table.unique_determinants.size() << '\n';
    std::cout << "n_unique_beta = "
              << same_spin_pair_cache.beta_reuse_table.unique_determinants.size() << '\n';
    std::cout << "exact_cache_enabled = "
              << (exact_same_spin_pair_cache.enabled() ? "true" : "false") << '\n';
    std::cout << "projected_pair_cache_bytes = "
              << projected_pair_value_count(same_spin_pair_cache) * sizeof(double)
              << '\n';
    std::cout << "compact_projected_pair_cache_bytes = "
              << projected_pair_value_count(compact_pair_cache) * sizeof(double)
              << '\n';
    const auto action_storage = compact_structure_action.storage();
    std::cout << "factorized_action_bytes = "
              << action_storage.factor_bytes << '\n';
    std::cout << "factorized_dense_channels = "
              << action_storage.dense_channels << '\n';
    std::cout << "factorized_sparse_channels = "
              << action_storage.sparse_channels << '\n';
    std::cout << "factorized_channel_nonzeros = "
              << action_storage.channel_nonzeros << '\n';
    std::cout << "factorized_channel_dense_values = "
              << action_storage.channel_dense_values << '\n';
    std::cout << "fast_result_seconds = "
              << std::chrono::duration<double>(fast_end - fast_start).count() << '\n';
    std::cout << "matrix_free_action_seconds = "
              << std::chrono::duration<double>(action_end - action_start).count() << '\n';
    std::cout << "matrix_free_hamiltonian_action_max_abs_diff = "
              << (action_result.hamiltonian - reference_hamiltonian_action)
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "matrix_free_overlap_action_max_abs_diff = "
              << (action_result.overlap - reference_overlap_action)
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "matrix_free_hamiltonian_diagonal_max_abs_diff = "
              << (action_diagonal.hamiltonian - dense_hamiltonian.diagonal())
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "matrix_free_overlap_diagonal_max_abs_diff = "
              << (action_diagonal.overlap - dense_overlap.diagonal())
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "compact_matrix_free_action_seconds = "
              << std::chrono::duration<double>(
                     compact_action_end - compact_action_start)
                     .count()
              << '\n';
    std::cout << "compact_matrix_free_hamiltonian_action_max_abs_diff = "
              << (compact_action_result.hamiltonian -
                  reference_hamiltonian_action)
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "compact_matrix_free_overlap_action_max_abs_diff = "
              << (compact_action_result.overlap - reference_overlap_action)
                     .cwiseAbs()
                     .maxCoeff()
              << '\n';
    std::cout << "dense_eigensolve_seconds = "
              << std::chrono::duration<double>(
                     dense_eigensolve_end - dense_eigensolve_start)
                     .count()
              << '\n';
    std::cout << "davidson_seconds = "
              << std::chrono::duration<double>(davidson_end - davidson_start)
                     .count()
              << '\n';
    std::cout << "davidson_action_seconds = "
              << davidson_action_seconds << '\n';
    std::cout << "davidson_iterations = " << davidson.iterations << '\n';
    std::cout << "davidson_block_actions = " << davidson.block_actions << '\n';
    std::cout << "davidson_peak_subspace_dimension = "
              << davidson.peak_subspace_dimension << '\n';
    std::cout << "davidson_ground_state_energy_abs_diff = "
              << std::abs(
                     davidson.eigenpairs.eigenvalues.front() -
                     dense_eigenpairs.eigenvalues.front())
              << '\n';
    std::cout << "davidson_ground_state_relative_residual = "
              << davidson.relative_residual_norms.front() << '\n';
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
