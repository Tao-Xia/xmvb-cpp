#include <chrono>
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
#include "vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace {

constexpr double kContributionTolerance = 1.0e-15;

struct Options {
  std::string input_path;
  int subspace_size = 0;
  int repeat = 1;
  double tolerance = 1.0e-8;
};

void print_usage() {
  std::cerr
      << "usage: check_biorthogonal_opposite_spin_matrix_backward <input.xmi>"
         " [--subspace-size N]"
         " [--repeat N]"
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
    if (argument_name == "--repeat") {
      options.repeat = std::stoi(argument_value);
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
  if (options.repeat <= 0) {
    throw std::invalid_argument("--repeat must be positive");
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

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& started_at) {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
}

double safe_speedup(
    double baseline_seconds,
    double candidate_seconds) {
  if (candidate_seconds <= 0.0) {
    return 0.0;
  }
  return baseline_seconds / candidate_seconds;
}

struct ProjectionPayloadSummary {
  int first_order_materialized_pair_count = 0;
  int inverse_materialized_pair_count = 0;
};

void accumulate_projection_payload_summary(
    const std::vector<xmvb::vb::SpinDeterminantPairEvaluation>& pair_cache,
    ProjectionPayloadSummary* summary) {
  if (summary == nullptr) {
    throw std::invalid_argument("summary must not be null");
  }
  for (const auto& pair_evaluation : pair_cache) {
    if (!pair_evaluation
             .opposite_spin_pair_cache
             .first_order_cofactor_projection
             .projected_pair_values.empty()) {
      ++summary->first_order_materialized_pair_count;
    }
    if (!pair_evaluation
             .opposite_spin_pair_cache
             .inverse_overlap_projection
             .projected_pair_values.empty()) {
      ++summary->inverse_materialized_pair_count;
    }
  }
}

ProjectionPayloadSummary summarize_projection_payloads(
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache) {
  ProjectionPayloadSummary summary;
  accumulate_projection_payload_summary(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      &summary);
  if (!same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache()) {
    accumulate_projection_payload_summary(
        same_spin_pair_cache.beta_pair_cache_ref(),
        &summary);
  }
  return summary;
}

struct ContributionErrorSummary {
  double overlap_gradient_max_abs = 0.0;
  double two_electron_gradient_max_abs = 0.0;

  double worst_max_abs() const {
    return std::max(overlap_gradient_max_abs, two_electron_gradient_max_abs);
  }
};

ContributionErrorSummary compute_contribution_error_summary(
    const xmvb::vb::OppositeSpinMatrixBackwardContribution& left,
    const xmvb::vb::OppositeSpinMatrixBackwardContribution& right) {
  ContributionErrorSummary summary;
  summary.overlap_gradient_max_abs =
      max_abs_vector_difference(
          left.active_orbital_overlap_gradient,
          right.active_orbital_overlap_gradient);
  summary.two_electron_gradient_max_abs =
      max_abs_vector_difference(
          left.packed_active_two_electron_gradient,
          right.packed_active_two_electron_gradient);
  return summary;
}

void accumulate_opposite_spin_two_electron_gradient_contribution(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const Eigen::MatrixXd& alpha_cofactor_1st,
    const xmvb::vb::OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const Eigen::MatrixXd& beta_cofactor_1st,
    const xmvb::vb::OppositeSpinPairCache* beta_pair_cache,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  if (alpha_pair_cache != nullptr && beta_pair_cache != nullptr &&
      xmvb::vb::has_opposite_spin_first_order_projection(*alpha_pair_cache) &&
      xmvb::vb::has_opposite_spin_first_order_projection(*beta_pair_cache) &&
      alpha_pair_cache->n_packed_active_pairs ==
          beta_pair_cache->n_packed_active_pairs) {
    const auto& alpha_projection = alpha_pair_cache->first_order_cofactor_projection;
    const auto& beta_projection = beta_pair_cache->first_order_cofactor_projection;
    for (std::size_t alpha_entry = 0;
         alpha_entry < alpha_projection.packed_pair_indices.size();
         ++alpha_entry) {
      const int alpha_packed_pair_index =
          alpha_projection.packed_pair_indices[alpha_entry];
      const double weighted_alpha_value =
          weight * alpha_projection.packed_pair_values[alpha_entry];
      for (std::size_t beta_entry = 0;
           beta_entry < beta_projection.packed_pair_indices.size();
           ++beta_entry) {
        const int beta_packed_pair_index =
            beta_projection.packed_pair_indices[beta_entry];
        const int packed_pair_of_pairs_index =
            xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                beta_packed_pair_index,
                alpha_packed_pair_index);
        (*packed_active_two_electron_gradient)[xmvb::to_size(
            packed_pair_of_pairs_index)] +=
            weighted_alpha_value * beta_projection.packed_pair_values[beta_entry];
      }
    }
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left = alpha_occ_L[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right = alpha_occ_R[xmvb::to_size(alpha_right_row)];
      const double weighted_alpha_cofactor =
          weight * alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left = beta_occ_L[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right = beta_occ_R[xmvb::to_size(beta_right_row)];
          const int packed_pair_of_pairs_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_orbital_right,
                  beta_orbital_left,
                  alpha_orbital_right,
                  alpha_orbital_left);
          (*packed_active_two_electron_gradient)[xmvb::to_size(
              packed_pair_of_pairs_index)] +=
              weighted_alpha_cofactor *
              beta_cofactor_1st(beta_right_row, beta_left_column);
        }
      }
    }
  }
}

xmvb::vb::OppositeSpinMatrixBackwardContribution build_reference_opposite_spin_contribution(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const xmvb::vb::biorthogonal_vbscf::
        BiorthogonalDeterminantPairWeightTablesFromCoefficients& pair_weights,
    int n_active_orbitals) {
  xmvb::vb::OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient.assign(
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
      if (std::abs(hamiltonian_weight) <= kContributionTolerance) {
        continue;
      }

      const auto& alpha_occ_R =
          input.structure_data.alpha_det[xmvb::to_size(determinant_index_right)];
      const auto& beta_occ_R =
          input.structure_data.beta_det[xmvb::to_size(determinant_index_right)];
      if (alpha_occ_L.empty() || beta_occ_L.empty()) {
        continue;
      }

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
        continue;
      }

      Eigen::MatrixXd alpha_opposite_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(
              static_cast<int>(alpha_occ_L.size()),
              static_cast<int>(alpha_occ_L.size()));
      Eigen::MatrixXd beta_opposite_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(
              static_cast<int>(beta_occ_L.size()),
              static_cast<int>(beta_occ_L.size()));
      const double opposite_spin_phi =
          xmvb::vb::compute_opposite_spin_original_phi(
              alpha_occ_L,
              alpha_occ_R,
              alpha_pair.overlap_result,
              &alpha_pair.opposite_spin_pair_cache,
              beta_occ_L,
              beta_occ_R,
              beta_pair.overlap_result,
              &beta_pair.opposite_spin_pair_cache,
              active_space_two_electron_result,
              &alpha_opposite_spin_inverse_overlap_gradient,
              &beta_opposite_spin_inverse_overlap_gradient);
      if (std::abs(opposite_spin_phi) <= kContributionTolerance &&
          alpha_opposite_spin_inverse_overlap_gradient.cwiseAbs().maxCoeff() <=
              kContributionTolerance &&
          beta_opposite_spin_inverse_overlap_gradient.cwiseAbs().maxCoeff() <=
              kContributionTolerance) {
        continue;
      }

      const double alpha_determinant_overlap_weight =
          hamiltonian_weight *
          beta_pair.overlap_result.overlap_determinant *
          opposite_spin_phi;
      const double beta_determinant_overlap_weight =
          hamiltonian_weight *
          alpha_pair.overlap_result.overlap_determinant *
          opposite_spin_phi;
      xmvb::vb::accumulate_spin_overlap_gradient(
          alpha_occ_L,
          alpha_occ_R,
          alpha_pair.overlap_result,
          alpha_determinant_overlap_weight,
          hamiltonian_weight * beta_pair.overlap_result.overlap_determinant *
              alpha_opposite_spin_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);
      xmvb::vb::accumulate_spin_overlap_gradient(
          beta_occ_L,
          beta_occ_R,
          beta_pair.overlap_result,
          beta_determinant_overlap_weight,
          hamiltonian_weight * alpha_pair.overlap_result.overlap_determinant *
              beta_opposite_spin_inverse_overlap_gradient,
          n_active_orbitals,
          &result.active_orbital_overlap_gradient);

      const Eigen::MatrixXd alpha_cofactor_1st =
          xmvb::vb::calc_cofactor_1st(alpha_pair.overlap_result);
      const Eigen::MatrixXd beta_cofactor_1st =
          xmvb::vb::calc_cofactor_1st(beta_pair.overlap_result);
      accumulate_opposite_spin_two_electron_gradient_contribution(
          alpha_occ_L,
          alpha_occ_R,
          alpha_cofactor_1st,
          &alpha_pair.opposite_spin_pair_cache,
          beta_occ_L,
          beta_occ_R,
          beta_cofactor_1st,
          &beta_pair.opposite_spin_pair_cache,
          hamiltonian_weight,
          &result.packed_active_two_electron_gradient);
    }
  }

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
    const auto prepared_input =
        xmvb::vb::biorthogonal_vbscf::prepare_biorthogonal_input(load_result.input);
    const auto& prepared_active_space = prepared_input.prepared_active_space;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    const std::vector<int> selected_structure_indices =
        build_selected_structure_indices(
            load_result.input.structure_data.n_structures,
            options.subspace_size);
    const xmvb::vb::ActiveSpaceTwoElectronView active_space_two_electron_view =
        xmvb::vb::make_active_space_two_electron_view(
            prepared_active_space.active_space_two_electron_result);
    const xmvb::vb::FullDeterminantPairEvaluator pair_evaluator;

    const auto sparse_cache_started_at = std::chrono::steady_clock::now();
    const xmvb::vb::SameSpinPairCacheContext sparse_same_spin_pair_cache =
        xmvb::vb::build_same_spin_pair_cache_context(
            load_result.input.structure_data.alpha_det,
            load_result.input.structure_data.beta_det,
            pair_evaluator,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            prepared_active_space.active_space_two_electron_result,
            xmvb::vb::SameSpinPairCacheBuildOptions{false});
    const double sparse_same_spin_cache_build_seconds =
        elapsed_seconds(sparse_cache_started_at);

    const auto dense_cache_started_at = std::chrono::steady_clock::now();
    const xmvb::vb::SameSpinPairCacheContext dense_same_spin_pair_cache =
        xmvb::vb::build_same_spin_pair_cache_context(
            load_result.input.structure_data.alpha_det,
            load_result.input.structure_data.beta_det,
            pair_evaluator,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            prepared_active_space.active_space_one_electron_result.h1e_act,
            n_active_orbitals,
            prepared_active_space.active_space_two_electron_result,
            xmvb::vb::SameSpinPairCacheBuildOptions{true});
    const double dense_same_spin_cache_build_seconds =
        elapsed_seconds(dense_cache_started_at);

    const ProjectionPayloadSummary sparse_projection_summary =
        summarize_projection_payloads(sparse_same_spin_pair_cache);
    const ProjectionPayloadSummary dense_projection_summary =
        summarize_projection_payloads(dense_same_spin_pair_cache);

    double subspace_evaluation_seconds = 0.0;
    double selected_state_matrix_seconds = 0.0;
    double pair_weight_seconds = 0.0;
    double sparse_matrix_form_backward_seconds = 0.0;
    double dense_matrix_form_backward_seconds = 0.0;
    double reference_backward_seconds = 0.0;
    int n_determinants = 0;
    int n_unique_alpha = 0;
    int n_unique_beta = 0;
    ContributionErrorSummary sparse_reference_error_summary;
    ContributionErrorSummary dense_reference_error_summary;
    ContributionErrorSummary sparse_dense_error_summary;

    for (int repeat_index = 0; repeat_index < options.repeat; ++repeat_index) {
      const auto evaluation_started_at = std::chrono::steady_clock::now();
      const auto evaluation_result =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_exact_selected_structure_subspace(
              load_result.input,
              prepared_active_space,
              selected_structure_indices);
      subspace_evaluation_seconds += elapsed_seconds(evaluation_started_at);

      const auto selected_state_started_at = std::chrono::steady_clock::now();
      const auto selected_state_matrices =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_selected_state_matrices(
              evaluation_result,
              sparse_same_spin_pair_cache,
              {0},
              {1.0});
      selected_state_matrix_seconds += elapsed_seconds(selected_state_started_at);

      const auto pair_weight_started_at = std::chrono::steady_clock::now();
      const auto pair_weights =
          xmvb::vb::biorthogonal_vbscf::
              build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
                  selected_state_matrices);
      pair_weight_seconds += elapsed_seconds(pair_weight_started_at);

      const auto sparse_backward_started_at = std::chrono::steady_clock::now();
      const auto sparse_matrix_form_contribution =
          xmvb::vb::biorthogonal_vbscf::
              build_biorthogonal_opposite_spin_matrix_backward_contribution(
                  sparse_same_spin_pair_cache,
                  selected_state_matrices,
                  active_space_two_electron_view,
                  n_active_orbitals);
      sparse_matrix_form_backward_seconds += elapsed_seconds(
          sparse_backward_started_at);

      const auto dense_backward_started_at = std::chrono::steady_clock::now();
      const auto dense_matrix_form_contribution =
          xmvb::vb::biorthogonal_vbscf::
              build_biorthogonal_opposite_spin_matrix_backward_contribution(
                  dense_same_spin_pair_cache,
                  selected_state_matrices,
                  active_space_two_electron_view,
                  n_active_orbitals);
      dense_matrix_form_backward_seconds += elapsed_seconds(
          dense_backward_started_at);

      const auto reference_started_at = std::chrono::steady_clock::now();
      const auto reference_contribution =
          build_reference_opposite_spin_contribution(
              load_result.input,
              prepared_active_space.active_space_two_electron_result,
              sparse_same_spin_pair_cache,
              pair_weights,
              n_active_orbitals);
      reference_backward_seconds += elapsed_seconds(reference_started_at);

      const ContributionErrorSummary sparse_reference_error =
          compute_contribution_error_summary(
              sparse_matrix_form_contribution,
              reference_contribution);
      sparse_reference_error_summary.overlap_gradient_max_abs =
          std::max(
              sparse_reference_error_summary.overlap_gradient_max_abs,
              sparse_reference_error.overlap_gradient_max_abs);
      sparse_reference_error_summary.two_electron_gradient_max_abs =
          std::max(
              sparse_reference_error_summary.two_electron_gradient_max_abs,
              sparse_reference_error.two_electron_gradient_max_abs);

      const ContributionErrorSummary dense_reference_error =
          compute_contribution_error_summary(
              dense_matrix_form_contribution,
              reference_contribution);
      dense_reference_error_summary.overlap_gradient_max_abs =
          std::max(
              dense_reference_error_summary.overlap_gradient_max_abs,
              dense_reference_error.overlap_gradient_max_abs);
      dense_reference_error_summary.two_electron_gradient_max_abs =
          std::max(
              dense_reference_error_summary.two_electron_gradient_max_abs,
              dense_reference_error.two_electron_gradient_max_abs);

      const ContributionErrorSummary sparse_dense_error =
          compute_contribution_error_summary(
              sparse_matrix_form_contribution,
              dense_matrix_form_contribution);
      sparse_dense_error_summary.overlap_gradient_max_abs =
          std::max(
              sparse_dense_error_summary.overlap_gradient_max_abs,
              sparse_dense_error.overlap_gradient_max_abs);
      sparse_dense_error_summary.two_electron_gradient_max_abs =
          std::max(
              sparse_dense_error_summary.two_electron_gradient_max_abs,
              sparse_dense_error.two_electron_gradient_max_abs);

      n_determinants = evaluation_result.n_determinants;
      n_unique_alpha = selected_state_matrices.n_unique_alpha;
      n_unique_beta = selected_state_matrices.n_unique_beta;
    }

    const double average_subspace_evaluation_seconds =
        subspace_evaluation_seconds / options.repeat;
    const double average_selected_state_matrix_seconds =
        selected_state_matrix_seconds / options.repeat;
    const double average_pair_weight_seconds =
        pair_weight_seconds / options.repeat;
    const double average_sparse_matrix_form_backward_seconds =
        sparse_matrix_form_backward_seconds / options.repeat;
    const double average_dense_matrix_form_backward_seconds =
        dense_matrix_form_backward_seconds / options.repeat;
    const double average_reference_backward_seconds =
        reference_backward_seconds / options.repeat;
    const double sparse_reference_speedup =
        safe_speedup(
            average_reference_backward_seconds,
            average_sparse_matrix_form_backward_seconds);
    const double dense_reference_speedup =
        safe_speedup(
            average_reference_backward_seconds,
            average_dense_matrix_form_backward_seconds);
    const double sparse_cache_build_speedup_vs_dense =
        safe_speedup(
            dense_same_spin_cache_build_seconds,
            sparse_same_spin_cache_build_seconds);
    const double worst_max_abs = std::max(
        sparse_reference_error_summary.worst_max_abs(),
        dense_reference_error_summary.worst_max_abs());

    std::cout << "selected_structure_indices = "
              << format_indices(selected_structure_indices) << '\n';
    std::cout << "repeat = " << options.repeat << '\n';
    std::cout << "n_determinants = " << n_determinants << '\n';
    std::cout << "n_unique_alpha = " << n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << n_unique_beta << '\n';
    std::cout << "sparse_cache_build_seconds = "
              << sparse_same_spin_cache_build_seconds << '\n';
    std::cout << "dense_cache_build_seconds = "
              << dense_same_spin_cache_build_seconds << '\n';
    std::cout << "sparse_cache_build_speedup_vs_dense = "
              << sparse_cache_build_speedup_vs_dense << '\n';
    std::cout << "sparse_cache_materialized_first_order_pair_count = "
              << sparse_projection_summary.first_order_materialized_pair_count << '\n';
    std::cout << "dense_cache_materialized_first_order_pair_count = "
              << dense_projection_summary.first_order_materialized_pair_count << '\n';
    std::cout << "sparse_cache_materialized_inverse_pair_count = "
              << sparse_projection_summary.inverse_materialized_pair_count << '\n';
    std::cout << "dense_cache_materialized_inverse_pair_count = "
              << dense_projection_summary.inverse_materialized_pair_count << '\n';
    std::cout << "avg_subspace_evaluation_seconds = "
              << average_subspace_evaluation_seconds << '\n';
    std::cout << "avg_selected_state_matrix_seconds = "
              << average_selected_state_matrix_seconds << '\n';
    std::cout << "avg_pair_weight_seconds = "
              << average_pair_weight_seconds << '\n';
    std::cout << "avg_sparse_matrix_form_backward_seconds = "
              << average_sparse_matrix_form_backward_seconds << '\n';
    std::cout << "avg_dense_matrix_form_backward_seconds = "
              << average_dense_matrix_form_backward_seconds << '\n';
    std::cout << "avg_reference_backward_seconds = "
              << average_reference_backward_seconds << '\n';
    std::cout << "sparse_matrix_form_speedup_vs_reference = "
              << sparse_reference_speedup << '\n';
    std::cout << "dense_matrix_form_speedup_vs_reference = "
              << dense_reference_speedup << '\n';
    std::cout << "sparse_reference_overlap_gradient_max_abs = "
              << sparse_reference_error_summary.overlap_gradient_max_abs << '\n';
    std::cout << "sparse_reference_two_electron_gradient_max_abs = "
              << sparse_reference_error_summary.two_electron_gradient_max_abs << '\n';
    std::cout << "dense_reference_overlap_gradient_max_abs = "
              << dense_reference_error_summary.overlap_gradient_max_abs << '\n';
    std::cout << "dense_reference_two_electron_gradient_max_abs = "
              << dense_reference_error_summary.two_electron_gradient_max_abs << '\n';
    std::cout << "sparse_dense_overlap_gradient_max_abs = "
              << sparse_dense_error_summary.overlap_gradient_max_abs << '\n';
    std::cout << "sparse_dense_two_electron_gradient_max_abs = "
              << sparse_dense_error_summary.two_electron_gradient_max_abs << '\n';
    std::cout << "worst_max_abs = " << worst_max_abs << '\n';

    if (worst_max_abs > options.tolerance) {
      std::cerr << "biorthogonal opposite-spin matrix backward check failed: worst_max_abs = "
                << worst_max_abs
                << " > tolerance = " << options.tolerance << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "check_biorthogonal_opposite_spin_matrix_backward failed: "
              << exception.what() << '\n';
    return 1;
  }
}
