#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/union_graph_rank_predictor.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int max_rank_cap = -1;
  int report_every = 0;
};

struct PerStructureCache {
  std::vector<xmvb::vb::OrbitalPair> active_pairs;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> determinant_terms_global;
};

struct MatrixDifferenceSummary {
  double max_abs = 0.0;
  double frobenius_error = 0.0;
  double relative_frobenius_error = 0.0;
};

struct SolverSummary {
  bool converged = false;
  std::string error_message;
  double electronic_ground_state_energy = 0.0;
  double total_ground_state_energy = 0.0;
  double wall_time_seconds = 0.0;
};

void print_usage() {
  std::cerr << "usage: benchmark_union_graph_single_step <input.xmi>"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--max-rank-cap R]"
               " [--report-every N]\n";
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
    if (argument_name == "--standard-two-electron-mode") {
      if (argument_value == "auto") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (argument_value == "exact") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (argument_value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument("invalid standard two-electron mode: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--max-rank-cap") {
      options.max_rank_cap = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--report-every") {
      options.report_every = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_rank_cap < -1) {
    throw std::invalid_argument("--max-rank-cap must be >= -1");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  return options;
}

template <typename Key>
std::string format_histogram(const std::map<Key, int>& histogram) {
  std::ostringstream stream;
  stream << "{";
  bool first = true;
  for (const auto& [key, count] : histogram) {
    if (!first) {
      stream << ", ";
    }
    first = false;
    stream << key << ": " << count;
  }
  stream << "}";
  return stream.str();
}

void set_symmetric_matrix_entry(
    std::vector<double>* matrix,
    int dimension,
    int row,
    int column,
    double value) {
  if (matrix == nullptr) {
    throw std::invalid_argument("matrix must not be null");
  }
  const std::size_t upper_index =
      column * dimension + row;
  const std::size_t lower_index =
      row * dimension + column;
  (*matrix)[upper_index] = value;
  (*matrix)[lower_index] = value;
}

MatrixDifferenceSummary summarize_matrix_difference(
    const std::vector<double>& reference,
    const std::vector<double>& candidate) {
  if (reference.size() != candidate.size()) {
    throw std::invalid_argument("matrix sizes must match");
  }

  MatrixDifferenceSummary summary;
  double reference_frobenius_squared = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double difference = candidate[index] - reference[index];
    summary.max_abs = std::max(summary.max_abs, std::abs(difference));
    summary.frobenius_error += difference * difference;
    reference_frobenius_squared += reference[index] * reference[index];
  }
  summary.frobenius_error = std::sqrt(summary.frobenius_error);
  const double reference_frobenius = std::sqrt(reference_frobenius_squared);
  summary.relative_frobenius_error =
      summary.frobenius_error / std::max(1.0, reference_frobenius);
  return summary;
}

SolverSummary solve_ground_state(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension,
    double one_electron_reference_energy,
    double nuclear_repulsion_energy) {
  SolverSummary summary;
  const auto started_at = std::chrono::steady_clock::now();
  try {
    xmvb::core::GeneralizedEigensolver generalized_eigensolver;
    const auto result = generalized_eigensolver.solve(
        hamiltonian_matrix,
        overlap_matrix,
        dimension);
    summary.electronic_ground_state_energy = result.eigenvalues.front();
    summary.total_ground_state_energy =
        one_electron_reference_energy +
        summary.electronic_ground_state_energy +
        nuclear_repulsion_energy;
    summary.converged = true;
  } catch (const std::exception& error) {
    summary.error_message = error.what();
  }
  summary.wall_time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
  return summary;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto benchmark_started_at = std::chrono::steady_clock::now();
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto& input = load_result.input;
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "benchmark_union_graph_single_step currently supports only singlet closed-shell structures");
    }
    if (input.structure_data.n_structures != raw_structure_data.n_structures) {
      throw std::runtime_error(
          "raw structure count does not match expanded structure matrix dimension");
    }

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto timed_active_space = xmvb::vb::prepare_timed_active_space_context(
        input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);
    const auto& prepared_active_space = timed_active_space.prepared_active_space;
    const auto& active_overlap_storage =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
    const auto exact_cpp_build_started_at = std::chrono::steady_clock::now();
    const auto exact_cpp_structure_matrices = structure_builder.build(
        input.structure_data.alpha_det,
        input.structure_data.beta_det,
        input.structure_data.determinant_to_structure_terms,
        prepared_active_space.orbital_result.active_orbital_overlap_matrix,
        prepared_active_space.active_space_one_electron_result.h1e_act,
        n_active_orbitals,
        prepared_active_space.active_space_two_electron_result,
        input.structure_data.n_structures);
    const double exact_cpp_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - exact_cpp_build_started_at).count();

    const auto exact_cpp_solver_summary = solve_ground_state(
        exact_cpp_structure_matrices.hamiltonian_matrix,
        exact_cpp_structure_matrices.overlap_matrix,
        input.structure_data.n_structures,
        prepared_active_space.one_electron_reference_energy,
        load_result.nuclear_repulsion_energy);

    std::vector<PerStructureCache> structure_cache(
        raw_structure_data.n_structures);
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[structure_index];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.determinant_terms_global =
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs);
    }

    xmvb::vb::UnionGraphRankPredictorOptions predictor_options;
    predictor_options.max_predicted_rank = options.max_rank_cap;
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    const int n_structures = raw_structure_data.n_structures;
    const std::size_t matrix_size =
        n_structures * n_structures;
    std::vector<double> legacy_exact_overlap_matrix(matrix_size, 0.0);
    std::vector<double> legacy_predicted_overlap_matrix(matrix_size, 0.0);
    std::map<int, int> predicted_rank_histogram;
    std::map<std::string, int> prediction_reason_histogram;
    double common_preprocess_seconds = 0.0;
    double exact_overlap_kernel_seconds = 0.0;
    double predicted_overlap_kernel_seconds = 0.0;
    int processed_pairs = 0;
    const int total_pairs = n_structures * (n_structures + 1) / 2;

    for (int left_structure = 0; left_structure < n_structures; ++left_structure) {
      const auto& left_cache = structure_cache[left_structure];
      for (int right_structure = 0; right_structure <= left_structure; ++right_structure) {
        const auto pair_started_at = std::chrono::steady_clock::now();
        const auto& right_cache = structure_cache[right_structure];

        const auto support_orbitals =
            xmvb::vb::build_support_orbitals(
                left_cache.active_pairs,
                right_cache.active_pairs);
        const auto support_index =
            xmvb::vb::build_support_index(support_orbitals);
        const auto left_pairs_local =
            xmvb::vb::remap_pairs_to_support(left_cache.active_pairs, support_index);
        const auto right_pairs_local =
            xmvb::vb::remap_pairs_to_support(right_cache.active_pairs, support_index);
        const auto left_terms_local =
            xmvb::vb::remap_legacy_determinant_terms(
                left_cache.determinant_terms_global,
                support_index);
        const auto right_terms_local =
            xmvb::vb::remap_legacy_determinant_terms(
                right_cache.determinant_terms_global,
                support_index);
        const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
            support_orbitals,
            active_overlap_storage,
            n_active_orbitals);
        const auto components = xmvb::vb::build_union_graph_components(
            left_pairs_local,
            right_pairs_local,
            support_orbitals);
        const auto offblock_overlap = xmvb::vb::build_offblock_support_overlap(
            support_overlap,
            components);
        const auto block_diagonal_overlap =
            xmvb::vb::build_block_diagonalized_support_overlap(
                support_overlap,
                components);
        const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
            support_overlap,
            components,
            1.0e-8);
        const auto screening_summary = xmvb::vb::summarize_union_graph_screening(
            support_overlap,
            offblock_overlap,
            components,
            cross_blocks);
        const auto rank_prediction = xmvb::vb::predict_union_graph_rank_cap(
            screening_summary,
            predictor_options);
        common_preprocess_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pair_started_at).count();

        const auto exact_overlap_started_at = std::chrono::steady_clock::now();
        const double exact_overlap = xmvb::vb::legacy_structure_overlap(
            left_terms_local,
            right_terms_local,
            support_overlap,
            overlap_resolver);
        exact_overlap_kernel_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - exact_overlap_started_at).count();

        const auto predicted_overlap_started_at = std::chrono::steady_clock::now();
        const auto truncated_offblock = xmvb::vb::build_blockwise_truncated_offblock(
            support_overlap,
            components,
            rank_prediction.predicted_rank_cap);
        const auto predicted_support_overlap =
            block_diagonal_overlap + truncated_offblock;
        const double predicted_overlap = xmvb::vb::legacy_structure_overlap(
            left_terms_local,
            right_terms_local,
            predicted_support_overlap,
            overlap_resolver);
        predicted_overlap_kernel_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - predicted_overlap_started_at).count();

        set_symmetric_matrix_entry(
            &legacy_exact_overlap_matrix,
            n_structures,
            right_structure,
            left_structure,
            exact_overlap);
        set_symmetric_matrix_entry(
            &legacy_predicted_overlap_matrix,
            n_structures,
            right_structure,
            left_structure,
            predicted_overlap);
        ++predicted_rank_histogram[rank_prediction.predicted_rank_cap];
        ++prediction_reason_histogram[xmvb::vb::union_graph_rank_prediction_reason_name(
            rank_prediction.reason)];
        ++processed_pairs;

        if (options.report_every > 0 &&
            processed_pairs % options.report_every == 0) {
          const double elapsed_seconds = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - benchmark_started_at).count();
          std::cerr << "progress = " << processed_pairs << "/" << total_pairs
                    << " elapsed_s = " << std::fixed << std::setprecision(2)
                    << elapsed_seconds << '\n';
        }
      }
    }

    const auto legacy_exact_vs_cpp = summarize_matrix_difference(
        exact_cpp_structure_matrices.overlap_matrix,
        legacy_exact_overlap_matrix);
    const auto legacy_predicted_vs_exact = summarize_matrix_difference(
        legacy_exact_overlap_matrix,
        legacy_predicted_overlap_matrix);
    const auto mixed_exact_overlap_solver_summary = solve_ground_state(
        exact_cpp_structure_matrices.hamiltonian_matrix,
        legacy_exact_overlap_matrix,
        n_structures,
        prepared_active_space.one_electron_reference_energy,
        load_result.nuclear_repulsion_energy);
    const auto mixed_predicted_overlap_solver_summary = solve_ground_state(
        exact_cpp_structure_matrices.hamiltonian_matrix,
        legacy_predicted_overlap_matrix,
        n_structures,
        prepared_active_space.one_electron_reference_energy,
        load_result.nuclear_repulsion_energy);

    const auto total_elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - benchmark_started_at).count();
    const auto& prep_timings = timed_active_space.timings;
    const double prepared_active_space_total_seconds =
        prep_timings.orbital_preparation_wall_time_seconds +
        prep_timings.ao_effective_one_electron_wall_time_seconds +
        prep_timings.active_one_electron_wall_time_seconds +
        prep_timings.active_two_electron_wall_time_seconds;
    const double legacy_exact_overlap_total_estimated_seconds =
        common_preprocess_seconds + exact_overlap_kernel_seconds;
    const double legacy_predicted_overlap_total_estimated_seconds =
        common_preprocess_seconds + predicted_overlap_kernel_seconds;

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     load_result.standard_two_electron_mode)
              << '\n';
    std::cout << "experimental_screening_scope = structure_overlap_only\n";
    std::cout << "structure_count = " << n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << input.structure_data.alpha_det.size() << '\n';
    std::cout << "pair_count_with_diagonal = " << total_pairs << '\n';
    std::cout << "load_input_wall_time_seconds = "
              << load_result.total_seconds << '\n';
    std::cout << "prepare_active_space_wall_time_seconds = "
              << prepared_active_space_total_seconds << '\n';
    std::cout << "prepare_active_space_orbital_wall_time_seconds = "
              << prep_timings.orbital_preparation_wall_time_seconds << '\n';
    std::cout << "prepare_active_space_ao_h1e_wall_time_seconds = "
              << prep_timings.ao_effective_one_electron_wall_time_seconds << '\n';
    std::cout << "prepare_active_space_active_h1e_wall_time_seconds = "
              << prep_timings.active_one_electron_wall_time_seconds << '\n';
    std::cout << "prepare_active_space_active_2e_wall_time_seconds = "
              << prep_timings.active_two_electron_wall_time_seconds << '\n';
    std::cout << "cpp_exact_structure_build_wall_time_seconds = "
              << exact_cpp_build_seconds << '\n';
    std::cout << "cpp_exact_eigensolve_wall_time_seconds = "
              << exact_cpp_solver_summary.wall_time_seconds << '\n';
    std::cout << "cpp_exact_single_step_wall_time_seconds = "
              << (prepared_active_space_total_seconds +
                  exact_cpp_build_seconds +
                  exact_cpp_solver_summary.wall_time_seconds)
              << '\n';
    std::cout << "legacy_overlap_common_preprocess_wall_time_seconds = "
              << common_preprocess_seconds << '\n';
    std::cout << "legacy_overlap_exact_kernel_wall_time_seconds = "
              << exact_overlap_kernel_seconds << '\n';
    std::cout << "legacy_overlap_predicted_kernel_wall_time_seconds = "
              << predicted_overlap_kernel_seconds << '\n';
    std::cout << "legacy_overlap_exact_total_estimated_wall_time_seconds = "
              << legacy_exact_overlap_total_estimated_seconds << '\n';
    std::cout << "legacy_overlap_predicted_total_estimated_wall_time_seconds = "
              << legacy_predicted_overlap_total_estimated_seconds << '\n';
    std::cout << "predicted_rank_histogram = "
              << format_histogram(predicted_rank_histogram) << '\n';
    std::cout << "prediction_reason_histogram = "
              << format_histogram(prediction_reason_histogram) << '\n';
    std::cout << "legacy_exact_vs_cpp_overlap_max_abs = "
              << legacy_exact_vs_cpp.max_abs << '\n';
    std::cout << "legacy_exact_vs_cpp_overlap_fro_error = "
              << legacy_exact_vs_cpp.frobenius_error << '\n';
    std::cout << "legacy_exact_vs_cpp_overlap_relative_fro_error = "
              << legacy_exact_vs_cpp.relative_frobenius_error << '\n';
    std::cout << "legacy_predicted_vs_exact_overlap_max_abs = "
              << legacy_predicted_vs_exact.max_abs << '\n';
    std::cout << "legacy_predicted_vs_exact_overlap_fro_error = "
              << legacy_predicted_vs_exact.frobenius_error << '\n';
    std::cout << "legacy_predicted_vs_exact_overlap_relative_fro_error = "
              << legacy_predicted_vs_exact.relative_frobenius_error << '\n';
    std::cout << "cpp_exact_ground_state_total_energy = "
              << exact_cpp_solver_summary.total_ground_state_energy << '\n';
    if (mixed_exact_overlap_solver_summary.converged) {
      std::cout << "mixed_exact_overlap_ground_state_total_energy = "
                << mixed_exact_overlap_solver_summary.total_ground_state_energy
                << '\n';
      std::cout << "mixed_exact_overlap_ground_state_delta = "
                << (mixed_exact_overlap_solver_summary.total_ground_state_energy -
                    exact_cpp_solver_summary.total_ground_state_energy)
                << '\n';
    } else {
      std::cout << "mixed_exact_overlap_ground_state_error = "
                << mixed_exact_overlap_solver_summary.error_message << '\n';
    }
    if (mixed_predicted_overlap_solver_summary.converged) {
      std::cout << "mixed_predicted_overlap_ground_state_total_energy = "
                << mixed_predicted_overlap_solver_summary.total_ground_state_energy
                << '\n';
      std::cout << "mixed_predicted_overlap_ground_state_delta = "
                << (mixed_predicted_overlap_solver_summary.total_ground_state_energy -
                    exact_cpp_solver_summary.total_ground_state_energy)
                << '\n';
    } else {
      std::cout << "mixed_predicted_overlap_ground_state_error = "
                << mixed_predicted_overlap_solver_summary.error_message << '\n';
    }
    std::cout << "total_elapsed_wall_time_seconds = "
              << total_elapsed_seconds << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_union_graph_single_step failed: "
              << error.what() << '\n';
    return 1;
  }
}
