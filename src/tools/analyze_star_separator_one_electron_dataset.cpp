#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <limits>

#include <Eigen/LU>
#include <Eigen/QR>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/one_electron.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/union_graph_screening.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using Pair = std::pair<int, int>;
using OrbitalPair = xmvb::vb::OrbitalPair;
using CanonicalDeterminantKey = std::pair<std::vector<int>, std::vector<int>>;
using SpinMaskKey = std::pair<std::uint32_t, std::uint32_t>;
using CofactorKey = std::pair<int, int>;
using FullDeterminantPairKey =
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>;
using ZeroedBlockDeterminantKey = std::tuple<
    std::vector<int>,
    std::vector<int>,
    std::vector<int>,
    std::vector<int>>;
using OrientationTerm = xmvb::vb::exact_separator::OrientationTerm;
using ComponentData = xmvb::vb::exact_separator::ComponentData;
using CollapsedOneElectronStarPairStats =
    xmvb::vb::exact_separator::CollapsedOneElectronStarPairStats;

enum class PairOrder {
  Lexicographic,
  Random,
};

enum class ActiveOverlapSource {
  Input,
  OptimizedVbscf,
};

struct Options {
  std::string input_path;
  PairOrder pair_order = PairOrder::Lexicographic;
  ActiveOverlapSource active_overlap_source = ActiveOverlapSource::Input;
  int filter_left_structure = -1;
  int filter_right_structure = -1;
  int max_pairs = 0;
  int report_every = 0;
  int top_examples = 12;
  int benchmark_repeats = 0;
  int optimizer_max_iterations = 25;
  std::uint32_t seed = 0;
  double optimizer_gradient_tolerance = 2.0e-3;
  double optimizer_energy_tolerance = 1.0e-7;
  double singular_value_threshold = 1.0e-8;
  double edge_max_abs_threshold = 0.0;
  double tolerance = 1.0e-12;
  int dump_root_pair_target_matrices = 0;
  bool skip_explicit_one_leaf = false;
  bool benchmark_full_matrix_build = false;
};

struct OptimizedOverlapSummary {
  bool converged = false;
  std::string termination_reason;
  int accepted_iterations = 0;
  int objective_evaluations = 0;
  double initial_total_energy = 0.0;
  double final_total_energy = 0.0;
  double final_gradient_inf_norm = 0.0;
  double final_gradient_l2_norm = 0.0;
  double total_wall_time_seconds = 0.0;
};

struct ActiveOverlapSelectionResult {
  std::vector<double> active_overlap_matrix;
  std::optional<OptimizedOverlapSummary> optimized_overlap_summary;
};

struct PerStructureCache {
  std::vector<OrbitalPair> active_pairs;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> legacy_terms;
  std::map<CanonicalDeterminantKey, double> coefficient_lookup;
};

struct StructurePairExactValues {
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct SameSpinDeterminantOneElectronValues {
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct FullPairMatrixBenchmarkEntry {
  int left_structure = -1;
  int right_structure = -1;
  int support_size = 0;
  std::vector<double> support_overlap_storage;
  std::vector<double> support_one_electron_storage;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> left_terms;
  std::vector<xmvb::vb::LegacyStructureDeterminantTerm> right_terms;
  std::vector<ComponentData> ordered_components;
};

struct FullPairMatrixBenchmarkResult {
  int selected_pair_count = 0;
  int covered_pair_count = 0;
  int repeats = 0;
  double preparation_wall_time_seconds = 0.0;
  double total_exact_detpair_seconds = 0.0;
  double total_open_state_recurrence_seconds = 0.0;
};

struct LeafMessageEntry {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  double value = 0.0;
  std::vector<int> alpha_block_rows;
  std::vector<int> alpha_block_cols;
  std::vector<int> beta_block_rows;
  std::vector<int> beta_block_cols;
  std::vector<int> left_alpha_occ;
  std::vector<int> left_beta_occ;
  std::vector<int> right_alpha_occ;
  std::vector<int> right_beta_occ;
  double left_term_coefficient = 0.0;
  double right_term_coefficient = 0.0;
};

struct CollapsedLeafMessageEntry {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  double value = 0.0;
};

struct ExactSeparatorLeafStateKey {
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
};

bool operator<(
    const ExactSeparatorLeafStateKey& left,
    const ExactSeparatorLeafStateKey& right) {
  return std::tie(
             left.left_pairs,
             left.right_pairs,
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.left_pairs,
             right.right_pairs,
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

struct ExactSeparatorStateKey {
  std::vector<OrbitalPair> root_left_pairs;
  std::vector<OrbitalPair> root_right_pairs;
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  std::vector<ExactSeparatorLeafStateKey> leaf_states;
};

bool operator<(
    const ExactSeparatorStateKey& left,
    const ExactSeparatorStateKey& right) {
  return std::tie(
             left.root_left_pairs,
             left.root_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.leaf_states) <
      std::tie(
             right.root_left_pairs,
             right.root_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.leaf_states);
}

struct ExactSeparatorStateCollector {
  std::set<ExactSeparatorStateKey> unique_exact_separator_states;
  struct ExactLeafMessageStateKey {
    std::vector<OrbitalPair> leaf_left_pairs;
    std::vector<OrbitalPair> leaf_right_pairs;
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
    std::uint32_t alpha_row_mask = 0;
    std::uint32_t alpha_col_mask = 0;
    std::uint32_t beta_row_mask = 0;
    std::uint32_t beta_col_mask = 0;
  };

  struct ExactLeafMessageBundleKey {
    std::vector<OrbitalPair> leaf_left_pairs;
    std::vector<OrbitalPair> leaf_right_pairs;
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
  };

  struct ExactMergeStateKey {
    std::vector<int> left_root_alpha_occ;
    std::vector<int> left_root_beta_occ;
    std::vector<int> right_root_alpha_occ;
    std::vector<int> right_root_beta_occ;
    int leaf_index = 0;
    std::uint32_t used_alpha_row_mask = 0;
    std::uint32_t used_alpha_col_mask = 0;
    std::uint32_t used_beta_row_mask = 0;
    std::uint32_t used_beta_col_mask = 0;
  };

  std::set<ExactLeafMessageStateKey> unique_leaf_message_states;
  std::set<ExactLeafMessageBundleKey> unique_leaf_message_bundles;
  std::set<ExactMergeStateKey> unique_merge_states;
};

bool operator<(
    const ExactSeparatorStateCollector::ExactLeafMessageStateKey& left,
    const ExactSeparatorStateCollector::ExactLeafMessageStateKey& right) {
  return std::tie(
             left.leaf_left_pairs,
             left.leaf_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.leaf_left_pairs,
             right.leaf_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

bool operator<(
    const ExactSeparatorStateCollector::ExactMergeStateKey& left,
    const ExactSeparatorStateCollector::ExactMergeStateKey& right) {
  return std::tie(
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ,
             left.leaf_index,
             left.used_alpha_row_mask,
             left.used_alpha_col_mask,
             left.used_beta_row_mask,
             left.used_beta_col_mask) <
      std::tie(
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ,
             right.leaf_index,
             right.used_alpha_row_mask,
             right.used_alpha_col_mask,
             right.used_beta_row_mask,
             right.used_beta_col_mask);
}

bool operator<(
    const ExactSeparatorStateCollector::ExactLeafMessageBundleKey& left,
    const ExactSeparatorStateCollector::ExactLeafMessageBundleKey& right) {
  return std::tie(
             left.leaf_left_pairs,
             left.leaf_right_pairs,
             left.left_root_alpha_occ,
             left.left_root_beta_occ,
             left.right_root_alpha_occ,
             left.right_root_beta_occ) <
      std::tie(
             right.leaf_left_pairs,
             right.leaf_right_pairs,
             right.left_root_alpha_occ,
             right.left_root_beta_occ,
             right.right_root_alpha_occ,
             right.right_root_beta_occ);
}

struct SpinMaskDeterminantEntry {
  std::uint32_t row_mask = 0;
  std::uint32_t col_mask = 0;
  double value = 0.0;
};

struct SpinMaskOneElectronEntry {
  std::uint32_t row_mask = 0;
  std::uint32_t col_mask = 0;
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct SpinOverlapOneElectronValue {
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct SpinOneElectronBlockBreakdown {
  double overlap = 0.0;
  double local_one_electron = 0.0;
  double mixed_one_electron = 0.0;
};

struct SpinOneElectronThreeBlockBreakdown {
  double overlap = 0.0;
  std::array<std::array<double, 3>, 3> block_one_electron = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
};

struct SpinOneElectronLaplaceTransferBreakdown {
  double exact_total = 0.0;
  std::array<std::array<double, 3>, 3> block_one_electron = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<double, 4> transfer_in_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> side_case_total = {{0.0, 0.0, 0.0, 0.0}};
  // `transfer_one_exit_hist` resolves the transfer=1 sector by the message
  // column class that gets expelled into the root-side minor:
  //   [none, selected-root, leaf, multiple/other].
  std::array<double, 4> transfer_one_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  // `transfer_zero_exit_hist` uses the same bins for the transfer=0 sector.
  std::array<double, 4> transfer_zero_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
};

struct SpinCrossCofactorRankBreakdown {
  int message_to_root_rank = 0;
  int root_to_message_rank = 0;
};

struct SpinOneElectronOpenStatePrototypeBreakdown {
  double overlap = 0.0;
  double exact_total = 0.0;
  double zeroed_closed_total = 0.0;
  double exact_cross_total = 0.0;
  double residual_closed_total = 0.0;
  // `residual_leaf_leaf_cofactor` stores the raw residual closed cofactor on
  // the leaf-leaf block before contraction with the one-electron operator.
  // Dimensions: (leaf_size, leaf_size) in the common
  //   [selected root | leaf | root remainder]
  // ordering used throughout this diagnostic.
  Matrix residual_leaf_leaf_cofactor;
  std::array<std::array<double, 3>, 3> residual_closed_block_one_electron = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  int residual_message_rank = 0;
  int residual_leaf_leaf_rank = 0;
};

struct RootPairTargetMatrixDumpEntry {
  // These occupied-orbital lists identify the root completion that generated
  // one exact local residual block. They use the same support-orbital labels
  // as the surrounding explicit one-leaf diagnostic.
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  double exact_total = 0.0;
  double captured_total = 0.0;
  Matrix alpha_target_matrix;
  Matrix beta_target_matrix;
  int alpha_target_rank = 0;
  int beta_target_rank = 0;
  std::map<SpinMaskKey, double> alpha_features;
  std::map<SpinMaskKey, double> beta_features;
  std::map<SpinMaskKey, double> alpha_full_features;
  std::map<SpinMaskKey, double> beta_full_features;
};

struct SpinRootStateKey {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
};

bool operator<(const SpinRootStateKey& left, const SpinRootStateKey& right) {
  return std::tie(left.alpha_occ, left.beta_occ) <
      std::tie(right.alpha_occ, right.beta_occ);
}

struct RootStatePairKey {
  SpinRootStateKey left_state;
  SpinRootStateKey right_state;
};

bool operator<(const RootStatePairKey& left, const RootStatePairKey& right) {
  return std::tie(left.left_state, left.right_state) <
      std::tie(right.left_state, right.right_state);
}

struct SpinCoupledRootChannelLayout {
  // `left_states` / `right_states` enumerate the distinct root completions that
  // remain after the leaf has been removed.  In the covered one-leaf cases the
  // exact recurrence only needs a width-1 (1x2) or width-2 (2x2) binary table.
  std::vector<SpinRootStateKey> left_states;
  std::vector<SpinRootStateKey> right_states;
  std::map<SpinRootStateKey, int> left_state_index;
  std::map<SpinRootStateKey, int> right_state_index;
  int channel_count = 0;
  bool supported = false;
};

struct SpinCoupledRootChannelBasis {
  // Each channel matrix has the same dimensions as the exact residual leaf x
  // leaf cofactor block K_{rho,mu}^{(sigma)} for one spin sector.
  Matrix b00;
  Matrix b_left;
  Matrix b_right;
  Matrix b_left_right;
};

struct ExplicitOneLeafOneElectronBreakdown {
  double captured = 0.0;
  double mixed_bridge = 0.0;
  double classified_total = 0.0;
  double block_diagonal_laplace_total = 0.0;
  double zeroed_block_laplace_total = 0.0;
  double leaf_local_message_corrected_total = 0.0;
  double open_state_prototype_zeroed_closed_total = 0.0;
  double open_state_prototype_exact_cross_total = 0.0;
  double open_state_prototype_residual_closed_total = 0.0;
  double scalar_mask_closed_correction_fit_total = 0.0;
  double scalar_mask_closed_correction_fit_abs_error = 0.0;
  // These matrix-fit diagnostics are stricter than the scalar fit above.
  // They test whether the exact leaf-level residual closed cofactor family can
  // be represented by a mask-indexed payload basis before contraction with the
  // one-electron operator.
  double alpha_matrix_current_feature_fit_sum_frobenius_abs_error = 0.0;
  double alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double beta_matrix_current_feature_fit_sum_frobenius_abs_error = 0.0;
  double beta_matrix_current_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double alpha_matrix_full_feature_fit_sum_frobenius_abs_error = 0.0;
  double alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double beta_matrix_full_feature_fit_sum_frobenius_abs_error = 0.0;
  double beta_matrix_full_feature_fit_max_root_pair_frobenius_residual = 0.0;
  int matrix_mask_closed_correction_fit_equation_count = 0;
  int alpha_matrix_mask_closed_correction_fit_unknown_count = 0;
  int beta_matrix_mask_closed_correction_fit_unknown_count = 0;
  int alpha_matrix_target_span_rank = 0;
  int beta_matrix_target_span_rank = 0;
  int spin_coupled_root_channel_left_state_count = 0;
  int spin_coupled_root_channel_right_state_count = 0;
  int spin_coupled_root_channel_count = 0;
  double spin_coupled_root_channel_max_frobenius_residual = 0.0;
  double spin_coupled_root_channel_total_frobenius_abs_error = 0.0;
  bool spin_coupled_root_channel_supported = false;
  int direct_spin_coupled_root_channel_left_state_count = 0;
  int direct_spin_coupled_root_channel_right_state_count = 0;
  int direct_spin_coupled_root_channel_count = 0;
  double direct_spin_coupled_root_channel_max_frobenius_residual = 0.0;
  double direct_spin_coupled_root_channel_total_frobenius_abs_error = 0.0;
  double direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement = 0.0;
  bool direct_spin_coupled_root_channel_supported = false;
  double root_channel_swap_covariance_max_frobenius_residual = 0.0;
  double root_channel_swap_covariance_max_relative_residual = 0.0;
  int root_channel_swap_covariance_pair_count = 0;
  double alpha_width2_walsh_small_component_fraction = 0.0;
  double beta_width2_walsh_small_component_fraction = 0.0;
  bool width2_root_channel_grid_detected = false;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
      scalar_mask_closed_correction_fit_alpha_solution;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
      scalar_mask_closed_correction_fit_beta_solution;
  std::array<std::array<double, 3>, 3> exact_three_block = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<std::array<double, 3>, 3> zeroed_block_three_block = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<double, 4> exact_transfer_in_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> exact_transfer_one_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> exact_transfer_zero_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> exact_side_case_total = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> zeroed_block_transfer_in_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> zeroed_block_transfer_one_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> zeroed_block_transfer_zero_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> zeroed_block_side_case_total = {{0.0, 0.0, 0.0, 0.0}};
  std::array<std::array<double, 3>, 3> open_state_prototype_residual_closed_blocks = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  int max_full_cross_message_to_root_rank = 0;
  int max_full_cross_root_to_message_rank = 0;
  int max_open_state_prototype_residual_message_rank = 0;
  int max_open_state_prototype_residual_leaf_leaf_rank = 0;
  double scalar_mask_closed_correction_fit_max_root_pair_abs_residual = 0.0;
  int scalar_mask_closed_correction_fit_equation_count = 0;
  int scalar_mask_closed_correction_fit_unknown_count = 0;
  std::vector<RootPairTargetMatrixDumpEntry> root_pair_target_matrix_dumps;
};

struct StarPairStats {
  bool covered = false;
  double exact_overlap = 0.0;
  double star_overlap = 0.0;
  double absolute_error = 0.0;
  std::uint64_t reference_determinant_pair_count = 0;
  std::uint64_t local_term_pair_visits = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::uint64_t coefficient_factorization_checks = 0;
  std::uint64_t coefficient_factorization_mismatches = 0;
  double max_coefficient_factorization_abs_error = 0.0;
  int metric_width_upper_bound = 0;
  int node_count = 0;
  int root_node = -1;
};

struct CollapsedStarPairStats {
  double exact_overlap = 0.0;
  double collapsed_overlap = 0.0;
  double absolute_error = 0.0;
  std::uint64_t collapsed_leaf_state_count = 0;
  std::uint64_t hypercube_assignment_count = 0;
  std::uint64_t schur_fast_spin_assignment_count = 0;
  std::uint64_t rectangular_fallback_spin_assignment_count = 0;
  std::uint64_t singular_fallback_spin_assignment_count = 0;
  std::uint64_t schur_fast_subdeterminant_evaluation_count = 0;
  std::uint64_t rectangular_fallback_subdeterminant_evaluation_count = 0;
  std::uint64_t singular_fallback_subdeterminant_evaluation_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
};

struct PairExample {
  int left_structure = 0;
  int right_structure = 0;
  int node_count = 0;
  int root_node = -1;
  int metric_width_upper_bound = 0;
  std::uint64_t reference_determinant_pair_count = 0;
  std::uint64_t local_term_pair_visits = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::uint64_t one_electron_collapsed_leaf_state_count = 0;
  std::uint64_t one_electron_hypercube_assignment_count = 0;
  std::uint64_t one_electron_subdeterminant_evaluations = 0;
  std::uint64_t one_electron_dp_transition_count = 0;
  double exact_overlap = 0.0;
  double star_overlap = 0.0;
  double absolute_error = 0.0;
  double exact_one_electron = 0.0;
  double collapsed_one_electron = 0.0;
  double one_electron_absolute_error = 0.0;
  double constructed_one_electron = 0.0;
  double constructed_one_electron_absolute_error = 0.0;
  double benchmark_exact_detpair_seconds = 0.0;
  double benchmark_open_state_recurrence_seconds = 0.0;
  double benchmark_open_state_speedup_over_detpair = 0.0;
  double explicit_one_leaf_captured = 0.0;
  double explicit_one_leaf_mixed_bridge = 0.0;
  double explicit_one_leaf_classified_total = 0.0;
  double explicit_one_leaf_block_diagonal_laplace_total = 0.0;
  double explicit_one_leaf_zeroed_block_laplace_total = 0.0;
  double explicit_one_leaf_leaf_local_message_corrected_total = 0.0;
  double explicit_one_leaf_open_state_prototype_zeroed_closed_total = 0.0;
  double explicit_one_leaf_open_state_prototype_exact_cross_total = 0.0;
  double explicit_one_leaf_open_state_prototype_residual_closed_total = 0.0;
  double explicit_one_leaf_scalar_mask_closed_correction_fit_total = 0.0;
  double explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error = 0.0;
  double explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error = 0.0;
  double explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error = 0.0;
  double explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual = 0.0;
  double explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error = 0.0;
  double explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual = 0.0;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
      explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
      explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution;
  std::array<std::array<double, 3>, 3> explicit_one_leaf_exact_three_block = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<std::array<double, 3>, 3> explicit_one_leaf_zeroed_block_three_block = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<double, 4> explicit_one_leaf_exact_transfer_in_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_exact_transfer_one_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_exact_transfer_zero_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_exact_side_case_total = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_in_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_one_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_zero_exit_hist = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> explicit_one_leaf_zeroed_block_side_case_total = {{0.0, 0.0, 0.0, 0.0}};
  std::array<std::array<double, 3>, 3> explicit_one_leaf_open_state_prototype_residual_closed_blocks = {{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  int explicit_one_leaf_max_full_cross_message_to_root_rank = 0;
  int explicit_one_leaf_max_full_cross_root_to_message_rank = 0;
  int explicit_one_leaf_max_open_state_prototype_residual_message_rank = 0;
  int explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank = 0;
  double explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual = 0.0;
  int explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count = 0;
  int explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count = 0;
  int explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count = 0;
  int explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count = 0;
  int explicit_one_leaf_alpha_matrix_target_span_rank = 0;
  int explicit_one_leaf_beta_matrix_target_span_rank = 0;
  int explicit_one_leaf_spin_coupled_root_channel_left_state_count = 0;
  int explicit_one_leaf_spin_coupled_root_channel_right_state_count = 0;
  int explicit_one_leaf_spin_coupled_root_channel_count = 0;
  double explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual = 0.0;
  double explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error = 0.0;
  bool explicit_one_leaf_spin_coupled_root_channel_supported = false;
  int explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count = 0;
  int explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count = 0;
  int explicit_one_leaf_direct_spin_coupled_root_channel_count = 0;
  double explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual = 0.0;
  double explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error = 0.0;
  double explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement =
      0.0;
  bool explicit_one_leaf_direct_spin_coupled_root_channel_supported = false;
  double explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual = 0.0;
  double explicit_one_leaf_root_channel_swap_covariance_max_relative_residual = 0.0;
  int explicit_one_leaf_root_channel_swap_covariance_pair_count = 0;
  double explicit_one_leaf_alpha_width2_walsh_small_component_fraction = 0.0;
  double explicit_one_leaf_beta_width2_walsh_small_component_fraction = 0.0;
  bool explicit_one_leaf_width2_root_channel_grid_detected = false;
  double explicit_one_leaf_one_electron_absolute_error = 0.0;
  double explicit_one_leaf_bridge_recovered_absolute_error = 0.0;
  double explicit_one_leaf_classified_total_absolute_error = 0.0;
  double explicit_one_leaf_block_diagonal_laplace_absolute_error = 0.0;
  double explicit_one_leaf_zeroed_block_laplace_absolute_error = 0.0;
  double explicit_one_leaf_leaf_local_message_corrected_absolute_error = 0.0;
  double explicit_one_leaf_scalar_mask_closed_correction_fit_absolute_error = 0.0;
};

struct ExactWorkCollector {
  // `reference_full_determinant_pairs` stores canonical raw-VB determinant-pair
  // contributions exactly as the legacy overlap code would visit them:
  //   (left alpha occ, left beta occ, right alpha occ, right beta occ).
  // `reference_spin_determinants` stores the alpha/beta determinant subproblems
  // that appear in those raw determinant pairs.
  // `separator_root_spin_determinants` stores ordinary root-remainder minors
  // visited by the separator recurrence, and
  // `separator_zeroed_block_spin_determinants` stores leaf/root minors where
  // the selected root-root overlap block has been zeroed.
  std::set<FullDeterminantPairKey> reference_full_determinant_pairs;
  std::set<CanonicalDeterminantKey> reference_spin_determinants;
  std::set<CanonicalDeterminantKey> separator_root_spin_determinants;
  std::set<ZeroedBlockDeterminantKey> separator_zeroed_block_spin_determinants;
};

struct RootPairClosedCorrectionFitData {
  std::vector<int> left_root_alpha_occ;
  std::vector<int> left_root_beta_occ;
  std::vector<int> right_root_alpha_occ;
  std::vector<int> right_root_beta_occ;
  double exact_total = 0.0;
  double captured_total = 0.0;
  std::map<SpinMaskKey, double> alpha_features;
  std::map<SpinMaskKey, double> beta_features;
  std::map<SpinMaskKey, double> alpha_full_features;
  std::map<SpinMaskKey, double> beta_full_features;
  Matrix alpha_target_matrix;
  Matrix beta_target_matrix;
};

enum class MatrixMaskFeatureModel {
  FactorizedOppositeSpin,
  FullOppositeSpin,
};

struct SpinMaskMatrixFitMetrics {
  double total_frobenius_abs_error = 0.0;
  double max_root_pair_frobenius_residual = 0.0;
  int equation_count = 0;
  int unknown_count = 0;
  int target_span_rank = 0;
};

std::vector<double> flatten_column_major_matrix(const Matrix& matrix);

std::string format_scalar_mask_solution_entries(
    const std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>& entries) {
  std::ostringstream buffer;
  buffer << "[";
  for (std::size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
    const auto& [row_mask, col_mask, value] = entries[entry_index];
    if (entry_index > 0) {
      buffer << ",";
    }
    buffer << "(" << row_mask << "," << col_mask << "," << value << ")";
  }
  buffer << "]";
  return buffer.str();
}

std::string format_int_vector(const std::vector<int>& values) {
  std::ostringstream buffer;
  buffer << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      buffer << ",";
    }
    buffer << values[index];
  }
  buffer << "]";
  return buffer.str();
}

std::string format_spin_mask_feature_map(
    const std::map<SpinMaskKey, double>& entries) {
  std::ostringstream buffer;
  buffer << "[";
  bool first = true;
  for (const auto& [mask_key, value] : entries) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    if (!first) {
      buffer << ",";
    }
    first = false;
    buffer << "(" << mask_key.first << "," << mask_key.second << "," << value << ")";
  }
  buffer << "]";
  return buffer.str();
}

std::string format_matrix_compact(const Matrix& matrix) {
  std::ostringstream buffer;
  buffer << "[";
  for (int row = 0; row < matrix.rows(); ++row) {
    if (row > 0) {
      buffer << ",";
    }
    buffer << "[";
    for (int col = 0; col < matrix.cols(); ++col) {
      if (col > 0) {
        buffer << ",";
      }
      buffer << matrix(row, col);
    }
    buffer << "]";
  }
  buffer << "]";
  return buffer.str();
}

SpinMaskMatrixFitMetrics fit_spin_mask_matrix_payloads(
    const std::vector<RootPairClosedCorrectionFitData>& equations,
    bool alpha_channel,
    MatrixMaskFeatureModel feature_model) {
  // The scalar mask fit can be accidentally exact when the number of root-pair
  // equations matches the number of unknown mask coefficients. This stronger
  // diagnostic instead fits the full leaf-leaf residual cofactor payload,
  // coordinate by coordinate, against the same mask-indexed feature model. If
  // the fitted matrices still fail, the candidate reduced state is not really
  // available at this aggregation layer.
  SpinMaskMatrixFitMetrics metrics;
  std::set<SpinMaskKey> mask_keys;
  int target_rows = 0;
  int target_cols = 0;
  for (const auto& equation : equations) {
    const auto& feature_map = [&]() -> const std::map<SpinMaskKey, double>& {
      if (alpha_channel) {
        return (feature_model == MatrixMaskFeatureModel::FactorizedOppositeSpin)
            ? equation.alpha_features
            : equation.alpha_full_features;
      }
      return (feature_model == MatrixMaskFeatureModel::FactorizedOppositeSpin)
          ? equation.beta_features
          : equation.beta_full_features;
    }();
    const auto& target_matrix =
        alpha_channel ? equation.alpha_target_matrix : equation.beta_target_matrix;
    if (target_rows == 0 && target_matrix.rows() > 0 && target_matrix.cols() > 0) {
      target_rows = target_matrix.rows();
      target_cols = target_matrix.cols();
    }
    for (const auto& [mask_key, value] : feature_map) {
      if (std::abs(value) > 1.0e-15) {
        mask_keys.insert(mask_key);
      }
    }
  }

  metrics.equation_count = static_cast<int>(equations.size());
  metrics.unknown_count = static_cast<int>(mask_keys.size());
  if (equations.empty() || target_rows == 0 || target_cols == 0) {
    return metrics;
  }

  std::map<SpinMaskKey, int> mask_index;
  int next_index = 0;
  for (const auto& mask_key : mask_keys) {
    mask_index.emplace(mask_key, next_index++);
  }

  if (mask_index.empty()) {
    Matrix exact_total = Matrix::Zero(target_rows, target_cols);
    for (const auto& equation : equations) {
      const auto& target_matrix =
          alpha_channel ? equation.alpha_target_matrix : equation.beta_target_matrix;
      if (target_matrix.rows() == target_rows && target_matrix.cols() == target_cols) {
        exact_total += target_matrix;
        metrics.max_root_pair_frobenius_residual = std::max(
            metrics.max_root_pair_frobenius_residual,
            target_matrix.norm());
      }
    }
    metrics.total_frobenius_abs_error = exact_total.norm();
    return metrics;
  }

  Matrix design = Matrix::Zero(metrics.equation_count, metrics.unknown_count);
  Matrix target_coordinate_matrix =
      Matrix::Zero(metrics.equation_count, target_rows * target_cols);
  std::vector<Matrix> targets;
  targets.reserve(equations.size());
  for (int equation_index = 0; equation_index < metrics.equation_count; ++equation_index) {
    const auto& equation = equations[xmvb::to_size(equation_index)];
    const auto& feature_map = [&]() -> const std::map<SpinMaskKey, double>& {
      if (alpha_channel) {
        return (feature_model == MatrixMaskFeatureModel::FactorizedOppositeSpin)
            ? equation.alpha_features
            : equation.alpha_full_features;
      }
      return (feature_model == MatrixMaskFeatureModel::FactorizedOppositeSpin)
          ? equation.beta_features
          : equation.beta_full_features;
    }();
    for (const auto& [mask_key, value] : feature_map) {
      const auto iterator = mask_index.find(mask_key);
      if (iterator != mask_index.end()) {
        design(equation_index, iterator->second) = value;
      }
    }

    Matrix target = Matrix::Zero(target_rows, target_cols);
    const auto& source_target =
        alpha_channel ? equation.alpha_target_matrix : equation.beta_target_matrix;
    if (source_target.rows() == target_rows && source_target.cols() == target_cols) {
      target = source_target;
    }
    for (int row = 0; row < target_rows; ++row) {
      for (int col = 0; col < target_cols; ++col) {
        target_coordinate_matrix(
            equation_index,
            row + col * target_rows) = target(row, col);
      }
    }
    targets.push_back(std::move(target));
  }
  metrics.target_span_rank =
      Eigen::FullPivLU<Matrix>(target_coordinate_matrix).rank();

  std::vector<Matrix> predicted(
      xmvb::to_size(metrics.equation_count),
      Matrix::Zero(target_rows, target_cols));
  Matrix predicted_total = Matrix::Zero(target_rows, target_cols);
  Matrix exact_total = Matrix::Zero(target_rows, target_cols);
  for (int row = 0; row < target_rows; ++row) {
    for (int col = 0; col < target_cols; ++col) {
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(metrics.equation_count);
      for (int equation_index = 0; equation_index < metrics.equation_count; ++equation_index) {
        rhs(equation_index) = targets[xmvb::to_size(equation_index)](row, col);
      }
      const Eigen::VectorXd solution =
          design.colPivHouseholderQr().solve(rhs);
      const Eigen::VectorXd predicted_coordinate = design * solution;
      exact_total(row, col) = rhs.sum();
      predicted_total(row, col) = predicted_coordinate.sum();
      for (int equation_index = 0; equation_index < metrics.equation_count; ++equation_index) {
        predicted[xmvb::to_size(equation_index)](row, col) =
            predicted_coordinate(equation_index);
      }
    }
  }

  for (int equation_index = 0; equation_index < metrics.equation_count; ++equation_index) {
    metrics.max_root_pair_frobenius_residual = std::max(
        metrics.max_root_pair_frobenius_residual,
        (predicted[xmvb::to_size(equation_index)] -
         targets[xmvb::to_size(equation_index)])
            .norm());
  }
  metrics.total_frobenius_abs_error = (predicted_total - exact_total).norm();
  return metrics;
}

void insert_spin_root_state(
    const SpinRootStateKey& state,
    std::vector<SpinRootStateKey>* states,
    std::map<SpinRootStateKey, int>* state_index) {
  if (states == nullptr || state_index == nullptr) {
    throw std::invalid_argument("spin root state containers must not be null");
  }
  if (state_index->find(state) != state_index->end()) {
    return;
  }
  state_index->emplace(state, static_cast<int>(states->size()));
  states->push_back(state);
}

void finalize_spin_coupled_root_channel_layout(
    SpinCoupledRootChannelLayout* layout) {
  if (layout == nullptr) {
    throw std::invalid_argument("layout must not be null");
  }
  layout->supported =
      ((layout->left_states.size() == 1 && layout->right_states.size() == 2) ||
       (layout->left_states.size() == 2 && layout->right_states.size() == 2));
  if (!layout->supported) {
    layout->channel_count = 0;
    return;
  }
  layout->channel_count = (layout->left_states.size() == 1) ? 2 : 4;
}

SpinCoupledRootChannelLayout build_spin_coupled_root_channel_layout(
    const std::vector<RootPairTargetMatrixDumpEntry>& entries) {
  SpinCoupledRootChannelLayout layout;
  for (const auto& entry : entries) {
    insert_spin_root_state(
        SpinRootStateKey{entry.left_root_alpha_occ, entry.left_root_beta_occ},
        &layout.left_states,
        &layout.left_state_index);
    insert_spin_root_state(
        SpinRootStateKey{entry.right_root_alpha_occ, entry.right_root_beta_occ},
        &layout.right_states,
        &layout.right_state_index);
  }
  finalize_spin_coupled_root_channel_layout(&layout);
  return layout;
}

SpinCoupledRootChannelLayout build_spin_coupled_root_channel_layout(
    const ComponentData& root_component) {
  SpinCoupledRootChannelLayout layout;
  for (const auto& left_root_term : root_component.left_orientation_terms) {
    insert_spin_root_state(
        SpinRootStateKey{left_root_term.alpha_occ, left_root_term.beta_occ},
        &layout.left_states,
        &layout.left_state_index);
  }
  for (const auto& right_root_term : root_component.right_orientation_terms) {
    insert_spin_root_state(
        SpinRootStateKey{right_root_term.alpha_occ, right_root_term.beta_occ},
        &layout.right_states,
        &layout.right_state_index);
  }
  finalize_spin_coupled_root_channel_layout(&layout);
  return layout;
}

void accumulate_scaled_matrix(
    Matrix* target,
    const Matrix& contribution,
    double scale) {
  // `contribution` is one exact residual leaf-leaf cofactor block.  We keep
  // all channel matrices in the same basis and dimensions, so this helper only
  // needs to allocate the accumulator once and then add scaled updates.
  if (target == nullptr) {
    throw std::invalid_argument("target must not be null");
  }
  if (contribution.size() == 0 || std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (target->size() == 0) {
    *target = Matrix::Zero(contribution.rows(), contribution.cols());
  }
  target->noalias() += scale * contribution;
}

double matrix_component_disagreement(
    const Matrix& left,
    const Matrix& right) {
  if (left.size() == 0 && right.size() == 0) {
    return 0.0;
  }
  if (left.size() == 0) {
    return right.norm();
  }
  if (right.size() == 0) {
    return left.norm();
  }
  return (left - right).norm();
}

void accumulate_direct_spin_coupled_root_channel_contribution(
    const SpinCoupledRootChannelLayout& layout,
    int left_state_index,
    int right_state_index,
    const Matrix& contribution,
    double scale,
    bool alpha_channel,
    SpinCoupledRootChannelBasis* basis) {
  // This is the direct channel-generation step.  Each exact residual block
  // contribution is routed immediately into the width-1 / width-2 Walsh basis,
  // so we do not need to build the full root-pair matrix table first.
  if (basis == nullptr) {
    throw std::invalid_argument("basis must not be null");
  }
  if (!layout.supported || contribution.size() == 0 || std::abs(scale) <= 1.0e-15) {
    return;
  }

  const double s_l = (left_state_index == 0) ? 1.0 : -1.0;
  const double s_r = (right_state_index == 0) ? 1.0 : -1.0;
  const double odd_channel_sign = alpha_channel ? 1.0 : -1.0;
  if (layout.left_states.size() == 1) {
    accumulate_scaled_matrix(&basis->b00, contribution, 0.5 * scale);
    accumulate_scaled_matrix(
        &basis->b_right,
        contribution,
        0.5 * odd_channel_sign * s_r * scale);
    return;
  }

  accumulate_scaled_matrix(&basis->b00, contribution, 0.25 * scale);
  accumulate_scaled_matrix(
      &basis->b_left,
      contribution,
      0.25 * odd_channel_sign * s_l * scale);
  accumulate_scaled_matrix(
      &basis->b_right,
      contribution,
      0.25 * odd_channel_sign * s_r * scale);
  accumulate_scaled_matrix(
      &basis->b_left_right,
      contribution,
      0.25 * s_l * s_r * scale);
}

Matrix reconstruct_spin_coupled_root_target(
    const SpinCoupledRootChannelLayout& layout,
    int left_state_index,
    int right_state_index,
    const SpinCoupledRootChannelBasis& basis,
    bool alpha_channel) {
  if (!layout.supported || basis.b00.size() == 0) {
    return Matrix();
  }

  const double s_l = (left_state_index == 0) ? 1.0 : -1.0;
  const double s_r = (right_state_index == 0) ? 1.0 : -1.0;
  if (layout.left_states.size() == 1) {
    if (alpha_channel) {
      return (basis.b00 + s_r * basis.b_right).eval();
    }
    return (basis.b00 - s_r * basis.b_right).eval();
  }

  if (alpha_channel) {
    return (basis.b00 +
            s_l * basis.b_left +
            s_r * basis.b_right +
            (s_l * s_r) * basis.b_left_right)
        .eval();
  }
  return (basis.b00 -
          s_l * basis.b_left -
          s_r * basis.b_right +
          (s_l * s_r) * basis.b_left_right)
      .eval();
}

std::vector<RootPairTargetMatrixDumpEntry> aggregate_root_pair_target_matrix_dumps_by_state(
    const std::vector<RootPairTargetMatrixDumpEntry>& entries) {
  // The exact recurrence is indexed by unique left/right root spin states, not
  // by the raw determinant-level root-term pairs.  This aggregation provides a
  // canonical state table that can be compared against the direct channel
  // accumulators without assuming term-level uniqueness.
  std::map<RootStatePairKey, RootPairTargetMatrixDumpEntry> aggregated;
  for (const auto& entry : entries) {
    const RootStatePairKey key{
        SpinRootStateKey{entry.left_root_alpha_occ, entry.left_root_beta_occ},
        SpinRootStateKey{entry.right_root_alpha_occ, entry.right_root_beta_occ},
    };
    auto& combined = aggregated[key];
    if (combined.left_root_alpha_occ.empty() && combined.left_root_beta_occ.empty() &&
        combined.right_root_alpha_occ.empty() && combined.right_root_beta_occ.empty()) {
      combined.left_root_alpha_occ = entry.left_root_alpha_occ;
      combined.left_root_beta_occ = entry.left_root_beta_occ;
      combined.right_root_alpha_occ = entry.right_root_alpha_occ;
      combined.right_root_beta_occ = entry.right_root_beta_occ;
    }
    combined.exact_total += entry.exact_total;
    combined.captured_total += entry.captured_total;
    for (const auto& [mask_key, value] : entry.alpha_features) {
      combined.alpha_features[mask_key] += value;
    }
    for (const auto& [mask_key, value] : entry.beta_features) {
      combined.beta_features[mask_key] += value;
    }
    for (const auto& [mask_key, value] : entry.alpha_full_features) {
      combined.alpha_full_features[mask_key] += value;
    }
    for (const auto& [mask_key, value] : entry.beta_full_features) {
      combined.beta_full_features[mask_key] += value;
    }
    accumulate_scaled_matrix(&combined.alpha_target_matrix, entry.alpha_target_matrix, 1.0);
    accumulate_scaled_matrix(&combined.beta_target_matrix, entry.beta_target_matrix, 1.0);
  }

  std::vector<RootPairTargetMatrixDumpEntry> aggregated_entries;
  aggregated_entries.reserve(aggregated.size());
  for (auto& [key, entry] : aggregated) {
    entry.alpha_target_rank =
        (entry.alpha_target_matrix.size() == 0)
            ? 0
            : Eigen::FullPivLU<Matrix>(entry.alpha_target_matrix).rank();
    entry.beta_target_rank =
        (entry.beta_target_matrix.size() == 0)
            ? 0
            : Eigen::FullPivLU<Matrix>(entry.beta_target_matrix).rank();
    aggregated_entries.push_back(std::move(entry));
  }
  return aggregated_entries;
}

void accumulate_direct_spin_coupled_root_channel_metrics(
    const std::vector<RootPairTargetMatrixDumpEntry>& state_entries,
    const SpinCoupledRootChannelLayout& layout,
    const SpinCoupledRootChannelBasis& alpha_basis,
    const SpinCoupledRootChannelBasis& beta_basis,
    ExplicitOneLeafOneElectronBreakdown* breakdown) {
  // `state_entries` is the exact state-resolved target table extracted from the
  // explicit enumeration, while `alpha_basis` / `beta_basis` were accumulated
  // directly inside the recurrence loop.  Matching them to machine precision is
  // the validation that the covered case has moved beyond post-hoc table
  // factorization into true direct channel generation.
  if (breakdown == nullptr) {
    throw std::invalid_argument("breakdown must not be null");
  }

  breakdown->direct_spin_coupled_root_channel_left_state_count =
      static_cast<int>(layout.left_states.size());
  breakdown->direct_spin_coupled_root_channel_right_state_count =
      static_cast<int>(layout.right_states.size());
  if (!layout.supported) {
    return;
  }

  std::map<std::pair<int, int>, const RootPairTargetMatrixDumpEntry*> table;
  for (const auto& entry : state_entries) {
    const auto left_iterator = layout.left_state_index.find(
        SpinRootStateKey{entry.left_root_alpha_occ, entry.left_root_beta_occ});
    const auto right_iterator = layout.right_state_index.find(
        SpinRootStateKey{entry.right_root_alpha_occ, entry.right_root_beta_occ});
    if (left_iterator == layout.left_state_index.end() ||
        right_iterator == layout.right_state_index.end()) {
      return;
    }
    table[std::make_pair(left_iterator->second, right_iterator->second)] = &entry;
  }
  if (table.size() != state_entries.size() ||
      table.size() != layout.left_states.size() * layout.right_states.size()) {
    return;
  }
  if (alpha_basis.b00.size() == 0) {
    return;
  }

  breakdown->direct_spin_coupled_root_channel_supported = true;
  breakdown->direct_spin_coupled_root_channel_count = layout.channel_count;
  for (const auto& [index_pair, entry] : table) {
    const Matrix predicted_alpha = reconstruct_spin_coupled_root_target(
        layout,
        index_pair.first,
        index_pair.second,
        alpha_basis,
        true);
    const Matrix predicted_beta = reconstruct_spin_coupled_root_target(
        layout,
        index_pair.first,
        index_pair.second,
        alpha_basis,
        false);
    const Matrix alpha_residual = predicted_alpha - entry->alpha_target_matrix;
    const Matrix beta_residual = predicted_beta - entry->beta_target_matrix;
    breakdown->direct_spin_coupled_root_channel_max_frobenius_residual = std::max(
        breakdown->direct_spin_coupled_root_channel_max_frobenius_residual,
        std::max(alpha_residual.norm(), beta_residual.norm()));
    breakdown->direct_spin_coupled_root_channel_total_frobenius_abs_error +=
        alpha_residual.norm() + beta_residual.norm();
  }

  breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement =
      matrix_component_disagreement(alpha_basis.b00, beta_basis.b00);
  breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement = std::max(
      breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement,
      matrix_component_disagreement(alpha_basis.b_right, beta_basis.b_right));
  if (layout.left_states.size() == 2) {
    breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement = std::max(
        breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement,
        matrix_component_disagreement(alpha_basis.b_left, beta_basis.b_left));
    breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement = std::max(
        breakdown->direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement,
        matrix_component_disagreement(alpha_basis.b_left_right, beta_basis.b_left_right));
  }
}

void accumulate_width2_walsh_small_component_fraction(
    const std::vector<RootPairTargetMatrixDumpEntry>& entries,
    bool alpha_channel,
    ExplicitOneLeafOneElectronBreakdown* breakdown) {
  // When the root context exposes exactly two left completions and two right
  // completions, the residual payload forms a 2x2 root-channel table.  The
  // Walsh basis separates parity-like channels from left/right skew channels,
  // which lets us quantify how close the exact data are to a pure parity-only
  // organization without claiming exactness.
  if (breakdown == nullptr) {
    throw std::invalid_argument("breakdown must not be null");
  }

  std::vector<SpinRootStateKey> left_states;
  std::vector<SpinRootStateKey> right_states;
  std::map<SpinRootStateKey, int> left_state_index;
  std::map<SpinRootStateKey, int> right_state_index;
  for (const auto& entry : entries) {
    const SpinRootStateKey left_state{
        entry.left_root_alpha_occ,
        entry.left_root_beta_occ,
    };
    const SpinRootStateKey right_state{
        entry.right_root_alpha_occ,
        entry.right_root_beta_occ,
    };
    if (left_state_index.find(left_state) == left_state_index.end()) {
      left_state_index.emplace(left_state, static_cast<int>(left_states.size()));
      left_states.push_back(left_state);
    }
    if (right_state_index.find(right_state) == right_state_index.end()) {
      right_state_index.emplace(right_state, static_cast<int>(right_states.size()));
      right_states.push_back(right_state);
    }
  }
  if (left_states.size() != 2 || right_states.size() != 2) {
    return;
  }

  std::map<std::pair<int, int>, Matrix> table;
  for (const auto& entry : entries) {
    const SpinRootStateKey left_state{
        entry.left_root_alpha_occ,
        entry.left_root_beta_occ,
    };
    const SpinRootStateKey right_state{
        entry.right_root_alpha_occ,
        entry.right_root_beta_occ,
    };
    const int left_index = left_state_index.at(left_state);
    const int right_index = right_state_index.at(right_state);
    const Matrix& target =
        alpha_channel ? entry.alpha_target_matrix : entry.beta_target_matrix;
    if (target.size() == 0) {
      return;
    }
    table[std::make_pair(left_index, right_index)] = target;
  }
  if (table.size() != 4) {
    return;
  }

  breakdown->width2_root_channel_grid_detected = true;
  const Matrix& k00 = table.at(std::make_pair(0, 0));
  const Matrix& k01 = table.at(std::make_pair(0, 1));
  const Matrix& k10 = table.at(std::make_pair(1, 0));
  const Matrix& k11 = table.at(std::make_pair(1, 1));
  const Matrix b00 = 0.25 * (k00 + k01 + k10 + k11);
  const Matrix b_left = 0.25 * (k00 + k01 - k10 - k11);
  const Matrix b_right = 0.25 * (k00 - k01 + k10 - k11);
  const Matrix b_left_right = 0.25 * (k00 - k01 - k10 + k11);
  const double dominant_norm = b00.norm() + b_left_right.norm();
  const double skew_norm = b_left.norm() + b_right.norm();
  const double fraction =
      skew_norm / std::max(1.0e-15, dominant_norm + skew_norm);
  if (alpha_channel) {
    breakdown->alpha_width2_walsh_small_component_fraction = fraction;
  } else {
    breakdown->beta_width2_walsh_small_component_fraction = fraction;
  }
}

void accumulate_root_channel_covariance_metrics(
    const std::vector<RootPairTargetMatrixDumpEntry>& entries,
    ExplicitOneLeafOneElectronBreakdown* breakdown) {
  // The width-1/2 dump data suggest that the residual matrices transform
  // covariantly under a simultaneous alpha/beta swap of the left and right
  // root completions:
  //   K_alpha(L, R) ?= K_beta(swap(L), swap(R)).
  // This diagnostic measures the strongest violation of that hypothesis on the
  // exact root-pair targets themselves.
  if (breakdown == nullptr) {
    throw std::invalid_argument("breakdown must not be null");
  }

  struct PairKey {
    SpinRootStateKey left_state;
    SpinRootStateKey right_state;
  };
  auto pair_key_less = [](
                           const PairKey& left,
                           const PairKey& right) {
    return std::tie(left.left_state, left.right_state) <
        std::tie(right.left_state, right.right_state);
  };

  std::map<PairKey, const RootPairTargetMatrixDumpEntry*, decltype(pair_key_less)> lookup(
      pair_key_less);
  for (const auto& entry : entries) {
    lookup.emplace(
        PairKey{
            SpinRootStateKey{entry.left_root_alpha_occ, entry.left_root_beta_occ},
            SpinRootStateKey{entry.right_root_alpha_occ, entry.right_root_beta_occ},
        },
        &entry);
  }

  for (const auto& entry : entries) {
    const PairKey swapped_key{
        SpinRootStateKey{entry.left_root_beta_occ, entry.left_root_alpha_occ},
        SpinRootStateKey{entry.right_root_beta_occ, entry.right_root_alpha_occ},
    };
    const auto swapped_iterator = lookup.find(swapped_key);
    if (swapped_iterator == lookup.end()) {
      continue;
    }
    const auto* swapped_entry = swapped_iterator->second;
    if (entry.alpha_target_matrix.size() == 0 || entry.beta_target_matrix.size() == 0 ||
        swapped_entry->alpha_target_matrix.size() == 0 ||
        swapped_entry->beta_target_matrix.size() == 0) {
      continue;
    }

    const Matrix alpha_residual =
        entry.alpha_target_matrix - swapped_entry->beta_target_matrix;
    const Matrix beta_residual =
        entry.beta_target_matrix - swapped_entry->alpha_target_matrix;
    const double residual_norm =
        std::max(alpha_residual.norm(), beta_residual.norm());
    const double reference_norm = std::max(
        1.0e-15,
        std::max(
            std::max(entry.alpha_target_matrix.norm(), entry.beta_target_matrix.norm()),
            std::max(
                swapped_entry->alpha_target_matrix.norm(),
                swapped_entry->beta_target_matrix.norm())));
    breakdown->root_channel_swap_covariance_max_frobenius_residual = std::max(
        breakdown->root_channel_swap_covariance_max_frobenius_residual,
        residual_norm);
    breakdown->root_channel_swap_covariance_max_relative_residual = std::max(
        breakdown->root_channel_swap_covariance_max_relative_residual,
        residual_norm / reference_norm);
    ++breakdown->root_channel_swap_covariance_pair_count;
  }

  accumulate_width2_walsh_small_component_fraction(entries, true, breakdown);
  accumulate_width2_walsh_small_component_fraction(entries, false, breakdown);
}

void accumulate_spin_coupled_root_channel_metrics(
    const std::vector<RootPairTargetMatrixDumpEntry>& entries,
    ExplicitOneLeafOneElectronBreakdown* breakdown) {
  // This is the current candidate "true recurrence" for the covered width-1/2
  // one-leaf cases.  Instead of fitting mask-only scalar corrections, we build
  // the exact residual table in a spin-coupled root-channel basis:
  //
  // width 1:
  //   K_alpha(j) = B0 + s_R(j) BR
  //   K_beta (j) = B0 - s_R(j) BR
  //
  // width 2:
  //   K_alpha(i,j) = B00 + s_L(i) BL + s_R(j) BR + s_L(i)s_R(j) BLR
  //   K_beta (i,j) = B00 - s_L(i) BL - s_R(j) BR + s_L(i)s_R(j) BLR
  //
  // The resulting residual quantifies whether the observed root-channel
  // skeleton already closes into an exact recurrence on the currently covered
  // star-like cases.
  if (breakdown == nullptr) {
    throw std::invalid_argument("breakdown must not be null");
  }

  std::vector<SpinRootStateKey> left_states;
  std::vector<SpinRootStateKey> right_states;
  std::map<SpinRootStateKey, int> left_state_index;
  std::map<SpinRootStateKey, int> right_state_index;
  for (const auto& entry : entries) {
    const SpinRootStateKey left_state{
        entry.left_root_alpha_occ,
        entry.left_root_beta_occ,
    };
    const SpinRootStateKey right_state{
        entry.right_root_alpha_occ,
        entry.right_root_beta_occ,
    };
    if (left_state_index.find(left_state) == left_state_index.end()) {
      left_state_index.emplace(left_state, static_cast<int>(left_states.size()));
      left_states.push_back(left_state);
    }
    if (right_state_index.find(right_state) == right_state_index.end()) {
      right_state_index.emplace(right_state, static_cast<int>(right_states.size()));
      right_states.push_back(right_state);
    }
  }

  breakdown->spin_coupled_root_channel_left_state_count =
      static_cast<int>(left_states.size());
  breakdown->spin_coupled_root_channel_right_state_count =
      static_cast<int>(right_states.size());

  if (left_states.empty() || right_states.empty()) {
    return;
  }

  if (!((left_states.size() == 1 && right_states.size() == 2) ||
        (left_states.size() == 2 && right_states.size() == 2))) {
    return;
  }

  struct EntryRef {
    const RootPairTargetMatrixDumpEntry* entry = nullptr;
  };
  std::map<std::pair<int, int>, EntryRef> table;
  for (const auto& entry : entries) {
    const SpinRootStateKey left_state{
        entry.left_root_alpha_occ,
        entry.left_root_beta_occ,
    };
    const SpinRootStateKey right_state{
        entry.right_root_alpha_occ,
        entry.right_root_beta_occ,
    };
    table[std::make_pair(
        left_state_index.at(left_state),
        right_state_index.at(right_state))] = EntryRef{&entry};
  }
  if (table.size() != entries.size()) {
    return;
  }

  breakdown->spin_coupled_root_channel_supported = true;
  if (left_states.size() == 1) {
    breakdown->spin_coupled_root_channel_count = 2;
    const auto* entry0 = table.at(std::make_pair(0, 0)).entry;
    const auto* entry1 = table.at(std::make_pair(0, 1)).entry;
    const Matrix alpha_b0 =
        0.5 * (entry0->alpha_target_matrix + entry1->alpha_target_matrix);
    const Matrix alpha_br =
        0.5 * (entry0->alpha_target_matrix - entry1->alpha_target_matrix);

    for (const auto& [index_pair, ref] : table) {
      const int right_index = index_pair.second;
      const double s_r = (right_index == 0) ? 1.0 : -1.0;
      const Matrix predicted_alpha = alpha_b0 + s_r * alpha_br;
      const Matrix predicted_beta = alpha_b0 - s_r * alpha_br;
      const Matrix alpha_residual =
          predicted_alpha - ref.entry->alpha_target_matrix;
      const Matrix beta_residual =
          predicted_beta - ref.entry->beta_target_matrix;
      breakdown->spin_coupled_root_channel_max_frobenius_residual = std::max(
          breakdown->spin_coupled_root_channel_max_frobenius_residual,
          std::max(alpha_residual.norm(), beta_residual.norm()));
      breakdown->spin_coupled_root_channel_total_frobenius_abs_error +=
          alpha_residual.norm() + beta_residual.norm();
    }
    return;
  }

  breakdown->spin_coupled_root_channel_count = 4;
  const auto* entry00 = table.at(std::make_pair(0, 0)).entry;
  const auto* entry01 = table.at(std::make_pair(0, 1)).entry;
  const auto* entry10 = table.at(std::make_pair(1, 0)).entry;
  const auto* entry11 = table.at(std::make_pair(1, 1)).entry;
  const Matrix alpha_b00 =
      0.25 * (entry00->alpha_target_matrix +
              entry01->alpha_target_matrix +
              entry10->alpha_target_matrix +
              entry11->alpha_target_matrix);
  const Matrix alpha_bl =
      0.25 * (entry00->alpha_target_matrix +
              entry01->alpha_target_matrix -
              entry10->alpha_target_matrix -
              entry11->alpha_target_matrix);
  const Matrix alpha_br =
      0.25 * (entry00->alpha_target_matrix -
              entry01->alpha_target_matrix +
              entry10->alpha_target_matrix -
              entry11->alpha_target_matrix);
  const Matrix alpha_blr =
      0.25 * (entry00->alpha_target_matrix -
              entry01->alpha_target_matrix -
              entry10->alpha_target_matrix +
              entry11->alpha_target_matrix);

  for (const auto& [index_pair, ref] : table) {
    const int left_index = index_pair.first;
    const int right_index = index_pair.second;
    const double s_l = (left_index == 0) ? 1.0 : -1.0;
    const double s_r = (right_index == 0) ? 1.0 : -1.0;
    const Matrix predicted_alpha =
        alpha_b00 +
        s_l * alpha_bl +
        s_r * alpha_br +
        (s_l * s_r) * alpha_blr;
    const Matrix predicted_beta =
        alpha_b00 -
        s_l * alpha_bl -
        s_r * alpha_br +
        (s_l * s_r) * alpha_blr;
    const Matrix alpha_residual =
        predicted_alpha - ref.entry->alpha_target_matrix;
    const Matrix beta_residual =
        predicted_beta - ref.entry->beta_target_matrix;
    breakdown->spin_coupled_root_channel_max_frobenius_residual = std::max(
        breakdown->spin_coupled_root_channel_max_frobenius_residual,
        std::max(alpha_residual.norm(), beta_residual.norm()));
    breakdown->spin_coupled_root_channel_total_frobenius_abs_error +=
        alpha_residual.norm() + beta_residual.norm();
  }
}

void print_root_pair_target_matrix_dumps(
    const ExplicitOneLeafOneElectronBreakdown& breakdown,
    int max_entries) {
  // This printer is purely diagnostic. It exposes the exact root-pair-resolved
  // residual leaf-leaf cofactor blocks K_{rho,mu}^{(sigma)} so we can inspect
  // candidate exact basis matrices directly on the known nontrivial examples.
  if (max_entries <= 0) {
    return;
  }
  const int count = std::min(
      max_entries,
      static_cast<int>(breakdown.root_pair_target_matrix_dumps.size()));
  for (int entry_index = 0; entry_index < count; ++entry_index) {
    const auto& entry =
        breakdown.root_pair_target_matrix_dumps[xmvb::to_size(entry_index)];
    std::cout << "root_pair_target[" << entry_index << "]"
              << " left_root_alpha_occ=" << format_int_vector(entry.left_root_alpha_occ)
              << " left_root_beta_occ=" << format_int_vector(entry.left_root_beta_occ)
              << " right_root_alpha_occ=" << format_int_vector(entry.right_root_alpha_occ)
              << " right_root_beta_occ=" << format_int_vector(entry.right_root_beta_occ)
              << " exact_total=" << entry.exact_total
              << " captured_total=" << entry.captured_total
              << " alpha_target_rank=" << entry.alpha_target_rank
              << " beta_target_rank=" << entry.beta_target_rank
              << " alpha_target_matrix=" << format_matrix_compact(entry.alpha_target_matrix)
              << " beta_target_matrix=" << format_matrix_compact(entry.beta_target_matrix)
              << " alpha_features=" << format_spin_mask_feature_map(entry.alpha_features)
              << " beta_features=" << format_spin_mask_feature_map(entry.beta_features)
              << " alpha_full_features=" << format_spin_mask_feature_map(entry.alpha_full_features)
              << " beta_full_features=" << format_spin_mask_feature_map(entry.beta_full_features)
              << '\n';
  }
}

Matrix build_support_submatrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& active_matrix_storage,
    int n_active_orbitals);

StructurePairExactValues compute_exact_structure_pair_values(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    const std::vector<double>& active_overlap_matrix,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    ExactWorkCollector* work_collector);

SpinOverlapOneElectronValue evaluate_occ_list_spin_values(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOverlapOneElectronValue evaluate_zeroed_selected_root_block_spin_values(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOverlapOneElectronValue evaluate_block_diagonalized_zeroed_selected_root_spin_values(
    const std::vector<int>& left_selected_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& left_root_remainder_occ,
    const std::vector<int>& right_selected_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<int>& right_root_remainder_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOneElectronBlockBreakdown evaluate_spin_block_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int message_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOneElectronThreeBlockBreakdown evaluate_spin_three_block_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOneElectronLaplaceTransferBreakdown evaluate_spin_laplace_transfer_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOneElectronLaplaceTransferBreakdown evaluate_spin_zeroed_block_laplace_transfer_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinCrossCofactorRankBreakdown evaluate_spin_full_cross_cofactor_ranks(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

SpinOneElectronOpenStatePrototypeBreakdown evaluate_spin_open_state_prototype_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

double evaluate_dense_leaf_leaf_block_one_electron(
    const Matrix& overlap_block,
    const Matrix& one_electron_block,
    int selected_root_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

double determinant_of_dense_matrix(
    const Matrix& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

void record_reference_determinant_work(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    ExactWorkCollector* work_collector);

StructurePairExactValues compute_exact_structure_pair_values_one_electron_only(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver);

bool is_connected_star_graph(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    int* root_node);

std::vector<int> remap_occ_to_global_labels(
    const std::vector<int>& local_occ,
    const std::vector<int>& local_to_global_orbitals);

std::vector<OrbitalPair> remap_pairs_to_global_labels(
    const std::vector<OrbitalPair>& local_pairs,
    const std::vector<int>& local_to_global_orbitals);

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes);

std::vector<CollapsedLeafMessageEntry> build_collapsed_leaf_messages_component_ordered(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    CollapsedStarPairStats* stats,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count);

void print_usage() {
  std::cerr << "usage: analyze_star_separator_one_electron_dataset <input.xmi>"
               " [--pair-order lexicographic|random]"
               " [--active-overlap-source input]"
               " [--left-structure I --right-structure J]"
               " [--max-pairs N]"
               " [--seed S]"
               " [--report-every N]"
               " [--top-examples N]"
               " [--benchmark-repeats N]"
               " [--benchmark-full-matrix-build 0|1]"
               " [--dump-root-pair-target-matrices N]"
               " [--skip-explicit-one-leaf 0|1]"
               " [--optimizer-max-iterations N]"
               " [--optimizer-gradient-tolerance F]"
               " [--optimizer-energy-tolerance F]"
               " [--singular-value-threshold F]"
               " [--edge-max-abs-threshold F]"
               " [--tolerance F]\n";
}

PairOrder parse_pair_order(const std::string& value) {
  if (value == "lexicographic") {
    return PairOrder::Lexicographic;
  }
  if (value == "random") {
    return PairOrder::Random;
  }
  throw std::invalid_argument("unsupported --pair-order value: " + value);
}

const char* pair_order_name(PairOrder order) {
  switch (order) {
    case PairOrder::Lexicographic:
      return "lexicographic";
    case PairOrder::Random:
      return "random";
  }
  return "unknown";
}

ActiveOverlapSource parse_active_overlap_source(const std::string& value) {
  if (value == "input") {
    return ActiveOverlapSource::Input;
  }
  if (value == "optimized_vbscf") {
    return ActiveOverlapSource::OptimizedVbscf;
  }
  throw std::invalid_argument("unsupported --active-overlap-source value: " + value);
}

const char* active_overlap_source_name(ActiveOverlapSource source) {
  switch (source) {
    case ActiveOverlapSource::Input:
      return "input";
    case ActiveOverlapSource::OptimizedVbscf:
      return "optimized_vbscf";
  }
  return "unknown";
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
    if (argument_name == "--pair-order") {
      options.pair_order = parse_pair_order(argument_value);
      continue;
    }
    if (argument_name == "--active-overlap-source") {
      options.active_overlap_source = parse_active_overlap_source(argument_value);
      continue;
    }
    if (argument_name == "--left-structure") {
      options.filter_left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.filter_right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--max-pairs") {
      options.max_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(argument_value));
      continue;
    }
    if (argument_name == "--report-every") {
      options.report_every = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-examples") {
      options.top_examples = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--benchmark-repeats") {
      options.benchmark_repeats = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--benchmark-full-matrix-build") {
      options.benchmark_full_matrix_build = (std::stoi(argument_value) != 0);
      continue;
    }
    if (argument_name == "--dump-root-pair-target-matrices") {
      options.dump_root_pair_target_matrices = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--skip-explicit-one-leaf") {
      options.skip_explicit_one_leaf = (std::stoi(argument_value) != 0);
      continue;
    }
    if (argument_name == "--optimizer-max-iterations") {
      options.optimizer_max_iterations = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--optimizer-gradient-tolerance") {
      options.optimizer_gradient_tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--optimizer-energy-tolerance") {
      options.optimizer_energy_tolerance = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--singular-value-threshold") {
      options.singular_value_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--edge-max-abs-threshold") {
      options.edge_max_abs_threshold = std::stod(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if ((options.filter_left_structure < 0) != (options.filter_right_structure < 0)) {
    throw std::invalid_argument(
        "--left-structure and --right-structure must be provided together");
  }
  if (options.report_every < 0) {
    throw std::invalid_argument("--report-every must be >= 0");
  }
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.benchmark_repeats < 0) {
    throw std::invalid_argument("--benchmark-repeats must be >= 0");
  }
  if (options.dump_root_pair_target_matrices < 0) {
    throw std::invalid_argument("--dump-root-pair-target-matrices must be >= 0");
  }
  if (options.optimizer_max_iterations <= 0) {
    throw std::invalid_argument("--optimizer-max-iterations must be positive");
  }
  if (options.optimizer_gradient_tolerance <= 0.0) {
    throw std::invalid_argument("--optimizer-gradient-tolerance must be positive");
  }
  if (options.optimizer_energy_tolerance <= 0.0) {
    throw std::invalid_argument("--optimizer-energy-tolerance must be positive");
  }
  if (options.singular_value_threshold <= 0.0) {
    throw std::invalid_argument("--singular-value-threshold must be positive");
  }
  if (options.edge_max_abs_threshold < 0.0) {
    throw std::invalid_argument("--edge-max-abs-threshold must be non-negative");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<Pair> build_pair_list(
    int structure_count,
    PairOrder pair_order,
    std::uint32_t seed,
    int max_pairs,
    int filter_left_structure,
    int filter_right_structure) {
  if (filter_left_structure >= 0 && filter_right_structure >= 0) {
    if (filter_left_structure >= structure_count ||
        filter_right_structure >= structure_count ||
        filter_left_structure <= filter_right_structure ||
        filter_right_structure < 0) {
      throw std::invalid_argument("requested structure filter is out of range");
    }
    return {{filter_left_structure, filter_right_structure}};
  }
  if (structure_count < 2) {
    return {};
  }
  std::vector<Pair> pairs;
  pairs.reserve(xmvb::to_size(structure_count) *
                xmvb::to_size(structure_count - 1) / 2);
  for (int left_structure = 0; left_structure < structure_count; ++left_structure) {
    for (int right_structure = 0; right_structure < left_structure; ++right_structure) {
      pairs.emplace_back(left_structure, right_structure);
    }
  }
  if (pair_order == PairOrder::Random) {
    std::mt19937 rng(seed);
    std::shuffle(pairs.begin(), pairs.end(), rng);
  }
  if (max_pairs > 0 && static_cast<int>(pairs.size()) > max_pairs) {
    pairs.resize(xmvb::to_size(max_pairs));
  }
  return pairs;
}

template <typename Callback>
double benchmark_repeated_wall_time_seconds(
    int repeats,
    Callback&& callback) {
  if (repeats <= 0) {
    return 0.0;
  }
  volatile double benchmark_sink = 0.0;
  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    benchmark_sink += callback();
  }
  const auto finished_at = std::chrono::steady_clock::now();
  static_cast<void>(benchmark_sink);
  return std::chrono::duration<double>(finished_at - started_at).count();
}

ActiveOverlapSelectionResult select_active_overlap_matrix(
    const Options& options,
    const xmvb::vb::CppVbInputLoadResult& load_result) {
  ActiveOverlapSelectionResult result;
  if (options.active_overlap_source == ActiveOverlapSource::Input) {
    result.active_overlap_matrix =
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix;
    return result;
  }

  xmvb::vb::CppVbScfOptimizerOptions optimizer_options;
  optimizer_options.backend = xmvb::vb::CppVbScfOptimizerBackend::Lbfgspp;
  optimizer_options.max_iterations = options.optimizer_max_iterations;
  optimizer_options.gradient_tolerance = options.optimizer_gradient_tolerance;
  optimizer_options.energy_tolerance = options.optimizer_energy_tolerance;
  optimizer_options.verbose = false;
  optimizer_options.retain_accepted_iteration_trace = true;

  xmvb::vb::CppVbScfOptimizer optimizer(optimizer_options);
  const auto optimization_result = optimizer.optimize(
      load_result.input,
      load_result.nuclear_repulsion_energy);
  if (optimization_result.accepted_iteration_trace.empty()) {
    throw std::runtime_error(
        "optimized_vbscf overlap source requires at least one accepted optimizer snapshot");
  }

  const auto& final_snapshot = optimization_result.accepted_iteration_trace.back();
  result.active_overlap_matrix = final_snapshot.active_orbital_overlap_matrix;

  OptimizedOverlapSummary summary;
  summary.converged = optimization_result.converged;
  summary.termination_reason = optimization_result.termination_reason;
  summary.accepted_iterations = optimization_result.n_iterations;
  summary.objective_evaluations =
      static_cast<int>(optimization_result.total_energy_history.size());
  summary.initial_total_energy = optimization_result.initial_total_energy;
  summary.final_total_energy = optimization_result.final_total_energy;
  summary.final_gradient_inf_norm = optimization_result.final_gradient_inf_norm;
  summary.final_gradient_l2_norm = optimization_result.final_gradient_l2_norm;
  summary.total_wall_time_seconds = optimization_result.total_wall_time_seconds;
  result.optimized_overlap_summary = std::move(summary);
  return result;
}

ComponentData build_component_data(
    int graph_node,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    const std::vector<int>& support_orbitals) {
  ComponentData component;
  component.graph_node = graph_node;
  const auto& union_component = union_components[xmvb::to_size(graph_node)];
  for (const auto& pair : union_component.left_pairs) {
    component.left_pairs.emplace_back(
        support_orbitals[xmvb::to_size(pair.first)],
        support_orbitals[xmvb::to_size(pair.second)]);
  }
  for (const auto& pair : union_component.right_pairs) {
    component.right_pairs.emplace_back(
        support_orbitals[xmvb::to_size(pair.first)],
        support_orbitals[xmvb::to_size(pair.second)]);
  }
  return component;
}

ComponentData build_local_component_data(
    int graph_node,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components) {
  ComponentData component;
  component.graph_node = graph_node;
  const auto& union_component = union_components[xmvb::to_size(graph_node)];
  component.left_pairs = union_component.left_pairs;
  component.right_pairs = union_component.right_pairs;
  return component;
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& union_components,
    int root_node,
    const std::vector<int>& leaf_nodes) {
  // The collapsed prototype rewrites the support basis so that the root
  // component occupies the leading contiguous orbital block and the leaf
  // components follow in the DP merge order. Exact raw-VB overlap is invariant
  // under a consistent orbital relabeling, so this does not change the value
  // of the overlap. It does, however, remove inter-component orbital
  // interleaving, which is what makes the block-parity collapse below possible.
  std::vector<int> ordered_support_orbitals;
  ordered_support_orbitals.reserve(support_orbitals.size());

  const auto append_component = [&](int component_index) {
    std::vector<int> local_vertices =
        union_components[xmvb::to_size(component_index)].local_vertices;
    std::sort(local_vertices.begin(), local_vertices.end());
    for (const int local_vertex : local_vertices) {
      ordered_support_orbitals.push_back(
          support_orbitals[xmvb::to_size(local_vertex)]);
    }
  };

  append_component(root_node);
  for (const int leaf_node : leaf_nodes) {
    append_component(leaf_node);
  }
  return ordered_support_orbitals;
}

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  // The separator recurrence must use the same local determinant expansion as
  // the reference raw-VB overlap. Enumerating only pair-orientation flips is
  // not enough, because the exact determinant-term coefficients also include
  // the permutation sign required to canonicalize the alpha and beta occupied
  // lists. Reusing the legacy exact enumerator keeps the prototype on the same
  // determinant basis as `legacy_structure_overlap`.
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    OrientationTerm term;
    term.alpha_occ = legacy_term.alpha_occ;
    term.beta_occ = legacy_term.beta_occ;
    term.coefficient = legacy_term.coefficient;
    terms.push_back(std::move(term));
  }
  return terms;
}

std::optional<FullPairMatrixBenchmarkEntry> prepare_full_pair_matrix_benchmark_entry(
    int left_structure,
    int right_structure,
    const std::vector<PerStructureCache>& structure_cache,
    const std::vector<double>& active_overlap_storage,
    const std::vector<double>& active_one_electron_storage,
    int n_active_orbitals,
    double singular_value_threshold,
    double edge_max_abs_threshold) {
  // Prepare the minimal per-pair context required to rebuild one structure
  // matrix entry repeatedly. The returned object contains only the exact
  // determinant expansion on the component-ordered support plus the component
  // metadata needed by the separator recurrence.
  const auto& left_cache = structure_cache[xmvb::to_size(left_structure)];
  const auto& right_cache = structure_cache[xmvb::to_size(right_structure)];

  const auto support_orbitals =
      xmvb::vb::build_support_orbitals(left_cache.active_pairs, right_cache.active_pairs);
  const auto support_index =
      xmvb::vb::build_support_index(support_orbitals);
  const auto left_pairs_local =
      xmvb::vb::remap_pairs_to_support(left_cache.active_pairs, support_index);
  const auto right_pairs_local =
      xmvb::vb::remap_pairs_to_support(right_cache.active_pairs, support_index);
  const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
      support_orbitals,
      active_overlap_storage,
      n_active_orbitals);
  const auto union_components = xmvb::vb::build_union_graph_components(
      left_pairs_local,
      right_pairs_local,
      support_orbitals);
  const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
      support_overlap,
      union_components,
      singular_value_threshold);
  xmvb::vb::MetricAwareComponentGraphOptions graph_options;
  graph_options.edge_max_abs_threshold = edge_max_abs_threshold;
  const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
      union_components,
      cross_blocks,
      graph_options);
  const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
      metric_graph,
      union_components);
  if (metric_summary.connected_component_count != 1) {
    return std::nullopt;
  }

  int root_node = -1;
  if (!is_connected_star_graph(metric_graph, &root_node)) {
    return std::nullopt;
  }
  if (metric_graph.node_count == 2) {
    const int node0_size = static_cast<int>(union_components[0].active_orbitals.size());
    const int node1_size = static_cast<int>(union_components[1].active_orbitals.size());
    if (node1_size < node0_size) {
      root_node = 1;
    }
  }

  std::vector<ComponentData> components(
      xmvb::to_size(metric_graph.node_count));
  for (int graph_node = 0; graph_node < metric_graph.node_count; ++graph_node) {
    auto& component = components[xmvb::to_size(graph_node)];
    component = build_component_data(graph_node, union_components, support_orbitals);
    component.left_orientation_terms =
        enumerate_orientation_terms(component.left_pairs);
    component.right_orientation_terms =
        enumerate_orientation_terms(component.right_pairs);
  }
  if (metric_graph.node_count == 2) {
    const auto estimate_local_term_visits = [&](int candidate_root) {
      const int candidate_leaf = 1 - candidate_root;
      const auto& root_component = components[xmvb::to_size(candidate_root)];
      const auto& leaf_component = components[xmvb::to_size(candidate_leaf)];
      const std::uint64_t root_term_pairs =
          static_cast<std::uint64_t>(root_component.left_orientation_terms.size()) *
          static_cast<std::uint64_t>(root_component.right_orientation_terms.size());
      const std::uint64_t leaf_term_pairs =
          static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
          static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
      return root_term_pairs * (1ULL + leaf_term_pairs);
    };
    const std::uint64_t root0_visits = estimate_local_term_visits(0);
    const std::uint64_t root1_visits = estimate_local_term_visits(1);
    if (root1_visits < root0_visits) {
      root_node = 1;
    } else if (root0_visits < root1_visits) {
      root_node = 0;
    }
  }

  const auto component_ordered_support_orbitals =
      build_component_ordered_support_orbitals(
          support_orbitals,
          union_components,
          root_node,
          metric_graph.adjacency[xmvb::to_size(root_node)]);
  const auto component_ordered_support_index =
      xmvb::vb::build_support_index(component_ordered_support_orbitals);
  const auto component_ordered_left_pairs_local =
      xmvb::vb::remap_pairs_to_support(
          left_cache.active_pairs,
          component_ordered_support_index);
  const auto component_ordered_right_pairs_local =
      xmvb::vb::remap_pairs_to_support(
          right_cache.active_pairs,
          component_ordered_support_index);
  const auto component_ordered_support_overlap =
      xmvb::vb::build_support_overlap_matrix(
          component_ordered_support_orbitals,
          active_overlap_storage,
          n_active_orbitals);
  const auto component_ordered_support_one_electron =
      build_support_submatrix(
          component_ordered_support_orbitals,
          active_one_electron_storage,
          n_active_orbitals);
  const auto component_ordered_union_components =
      xmvb::vb::build_union_graph_components(
          component_ordered_left_pairs_local,
          component_ordered_right_pairs_local,
          component_ordered_support_orbitals);
  if (component_ordered_union_components.size() != union_components.size()) {
    throw std::runtime_error(
        "component-ordered support relabeling changed the union-component count");
  }

  FullPairMatrixBenchmarkEntry entry;
  entry.left_structure = left_structure;
  entry.right_structure = right_structure;
  entry.support_size = component_ordered_support_overlap.rows();
  entry.support_overlap_storage =
      flatten_column_major_matrix(component_ordered_support_overlap);
  entry.support_one_electron_storage =
      flatten_column_major_matrix(component_ordered_support_one_electron);
  entry.left_terms =
      xmvb::vb::enumerate_legacy_determinant_terms(
          component_ordered_left_pairs_local);
  entry.right_terms =
      xmvb::vb::enumerate_legacy_determinant_terms(
          component_ordered_right_pairs_local);
  entry.ordered_components.reserve(component_ordered_union_components.size());
  for (std::size_t component_index = 0;
       component_index < component_ordered_union_components.size();
       ++component_index) {
    auto component = build_local_component_data(
        static_cast<int>(component_index),
        component_ordered_union_components);
    component.left_orientation_terms =
        enumerate_orientation_terms(component.left_pairs);
    component.right_orientation_terms =
        enumerate_orientation_terms(component.right_pairs);
    entry.ordered_components.push_back(std::move(component));
  }
  return entry;
}

FullPairMatrixBenchmarkResult benchmark_full_pair_matrix_build(
    const std::vector<Pair>& pair_list,
    const std::vector<PerStructureCache>& structure_cache,
    const std::vector<double>& active_overlap_storage,
    const std::vector<double>& active_one_electron_storage,
    int n_active_orbitals,
    double singular_value_threshold,
    double edge_max_abs_threshold,
    int benchmark_repeats) {
  // Time the matrix build in the same way the real solver would use it:
  // first prepare every covered structure pair once, then repeatedly rebuild
  // all covered entries with either the exact determinant expansion or the
  // exact-separator recurrence.
  FullPairMatrixBenchmarkResult result;
  result.selected_pair_count = static_cast<int>(pair_list.size());
  result.repeats = benchmark_repeats;

  std::vector<FullPairMatrixBenchmarkEntry> benchmark_entries;
  benchmark_entries.reserve(pair_list.size());
  const auto prepare_started_at = std::chrono::steady_clock::now();
  for (const auto& [left_structure, right_structure] : pair_list) {
    auto entry = prepare_full_pair_matrix_benchmark_entry(
        left_structure,
        right_structure,
        structure_cache,
        active_overlap_storage,
        active_one_electron_storage,
        n_active_orbitals,
        singular_value_threshold,
        edge_max_abs_threshold);
    if (entry.has_value()) {
      benchmark_entries.push_back(std::move(*entry));
    }
  }
  const auto prepare_finished_at = std::chrono::steady_clock::now();
  result.covered_pair_count = static_cast<int>(benchmark_entries.size());
  result.preparation_wall_time_seconds =
      std::chrono::duration<double>(prepare_finished_at - prepare_started_at).count();
  if (benchmark_entries.empty() || benchmark_repeats <= 0) {
    return result;
  }

  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  result.total_exact_detpair_seconds = benchmark_repeated_wall_time_seconds(
      benchmark_repeats,
      [&]() {
        double accumulator = 0.0;
        for (const auto& entry : benchmark_entries) {
          const auto exact_values = compute_exact_structure_pair_values_one_electron_only(
              entry.left_terms,
              entry.right_terms,
              entry.support_overlap_storage,
              entry.support_one_electron_storage,
              entry.support_size,
              overlap_resolver);
          accumulator += exact_values.overlap + exact_values.one_electron;
        }
        return accumulator;
      });
  result.total_open_state_recurrence_seconds = benchmark_repeated_wall_time_seconds(
      benchmark_repeats,
      [&]() {
        double accumulator = 0.0;
        for (const auto& entry : benchmark_entries) {
          const auto recurrence_values =
              evaluate_component_ordered_open_state_star_pair_one_electron(
                  std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::quiet_NaN(),
                  entry.support_overlap_storage,
                  entry.support_one_electron_storage,
                  entry.support_size,
                  entry.ordered_components,
                  overlap_resolver);
          accumulator +=
              recurrence_values.collapsed_overlap +
              recurrence_values.collapsed_one_electron;
        }
        return accumulator;
      });
  return result;
}

std::vector<int> remap_occ_to_global_labels(
    const std::vector<int>& local_occ,
    const std::vector<int>& local_to_global_orbitals) {
  std::vector<int> global_occ;
  global_occ.reserve(local_occ.size());
  for (const int local_orbital : local_occ) {
    if (local_orbital < 0 ||
        local_orbital >= static_cast<int>(local_to_global_orbitals.size())) {
      throw std::out_of_range("local occupied orbital is out of range");
    }
    global_occ.push_back(local_to_global_orbitals[xmvb::to_size(local_orbital)]);
  }
  return global_occ;
}

std::vector<OrbitalPair> remap_pairs_to_global_labels(
    const std::vector<OrbitalPair>& local_pairs,
    const std::vector<int>& local_to_global_orbitals) {
  std::vector<OrbitalPair> global_pairs;
  global_pairs.reserve(local_pairs.size());
  for (const auto& [left_orbital, right_orbital] : local_pairs) {
    if (left_orbital < 0 ||
        left_orbital >= static_cast<int>(local_to_global_orbitals.size()) ||
        right_orbital < 0 ||
        right_orbital >= static_cast<int>(local_to_global_orbitals.size())) {
      throw std::out_of_range("local pair orbital is out of range");
    }
    global_pairs.emplace_back(
        local_to_global_orbitals[xmvb::to_size(left_orbital)],
        local_to_global_orbitals[xmvb::to_size(right_orbital)]);
  }
  return global_pairs;
}

std::map<CanonicalDeterminantKey, double> build_coefficient_lookup(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& terms) {
  std::map<CanonicalDeterminantKey, double> lookup;
  for (const auto& term : terms) {
    lookup[{term.alpha_occ, term.beta_occ}] = term.coefficient;
  }
  return lookup;
}

double determinant_for_occ_lists(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactWorkCollector* work_collector = nullptr) {
  if (left_occ.size() != right_occ.size()) {
    return 0.0;
  }
  if (work_collector != nullptr) {
    work_collector->separator_root_spin_determinants.insert({left_occ, right_occ});
  }
  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      active_overlap_storage,
      n_active_orbitals);
  return overlap_resolver
      .resolve(overlap_submatrix, static_cast<int>(left_occ.size()))
      .overlap_determinant;
}

double determinant_with_zeroed_selected_root_block(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::vector<int>* ordered_left_occ,
    std::vector<int>* ordered_right_occ,
    ExactWorkCollector* work_collector = nullptr) {
  if (ordered_left_occ == nullptr || ordered_right_occ == nullptr) {
    throw std::invalid_argument("ordered occupied-orbital outputs must not be null");
  }

  std::vector<std::pair<int, bool>> left_occ_with_root_flag;
  left_occ_with_root_flag.reserve(left_root_occ.size() + left_leaf_occ.size());
  for (const int orbital : left_root_occ) {
    left_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : left_leaf_occ) {
    left_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(left_occ_with_root_flag.begin(), left_occ_with_root_flag.end());

  std::vector<std::pair<int, bool>> right_occ_with_root_flag;
  right_occ_with_root_flag.reserve(right_root_occ.size() + right_leaf_occ.size());
  for (const int orbital : right_root_occ) {
    right_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : right_leaf_occ) {
    right_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(right_occ_with_root_flag.begin(), right_occ_with_root_flag.end());

  ordered_left_occ->clear();
  ordered_right_occ->clear();
  ordered_left_occ->reserve(left_occ_with_root_flag.size());
  ordered_right_occ->reserve(right_occ_with_root_flag.size());
  for (const auto& [orbital, is_root] : left_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_left_occ->push_back(orbital);
  }
  for (const auto& [orbital, is_root] : right_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_right_occ->push_back(orbital);
  }
  if (ordered_left_occ->size() != ordered_right_occ->size()) {
    return 0.0;
  }
  if (work_collector != nullptr) {
    std::vector<int> left_root_flags;
    std::vector<int> right_root_flags;
    left_root_flags.reserve(left_occ_with_root_flag.size());
    right_root_flags.reserve(right_occ_with_root_flag.size());
    for (const auto& [orbital, is_root] : left_occ_with_root_flag) {
      static_cast<void>(orbital);
      left_root_flags.push_back(is_root ? 1 : 0);
    }
    for (const auto& [orbital, is_root] : right_occ_with_root_flag) {
      static_cast<void>(orbital);
      right_root_flags.push_back(is_root ? 1 : 0);
    }
    work_collector->separator_zeroed_block_spin_determinants.insert(
        {*ordered_left_occ, *ordered_right_occ, left_root_flags, right_root_flags});
  }

  auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      *ordered_left_occ,
      *ordered_right_occ,
      active_overlap_storage,
      n_active_orbitals);
  const int dimension = static_cast<int>(ordered_left_occ->size());
  for (int column = 0; column < dimension; ++column) {
    if (!left_occ_with_root_flag[xmvb::to_size(column)].second) {
      continue;
    }
    for (int row = 0; row < dimension; ++row) {
      if (!right_occ_with_root_flag[xmvb::to_size(row)].second) {
        continue;
      }
      overlap_submatrix[xmvb::to_size(column) * dimension + row] = 0.0;
    }
  }
  return overlap_resolver
      .resolve(overlap_submatrix, dimension)
      .overlap_determinant;
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occ,
    std::uint32_t mask) {
  std::vector<int> selected;
  selected.reserve(occ.size());
  for (int index = 0; index < static_cast<int>(occ.size()); ++index) {
    if ((mask & (static_cast<std::uint32_t>(1) << index)) != 0U) {
      selected.push_back(occ[xmvb::to_size(index)]);
    }
  }
  return selected;
}

double lookup_global_coefficient(
    const std::vector<int>& alpha_occ,
    const std::vector<int>& beta_occ,
    const std::map<CanonicalDeterminantKey, double>& coefficient_lookup) {
  std::vector<int> alpha_key = alpha_occ;
  std::vector<int> beta_key = beta_occ;
  std::sort(alpha_key.begin(), alpha_key.end());
  std::sort(beta_key.begin(), beta_key.end());
  const auto iterator = coefficient_lookup.find({alpha_key, beta_key});
  if (iterator == coefficient_lookup.end()) {
    return 0.0;
  }
  return iterator->second;
}

int popcount(std::uint32_t mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
  // The leaf/root minors are evaluated in block order. The exact raw-VB
  // reference, however, builds determinant overlaps in canonical occupied
  // order. This parity counts the sign of sorting the block-ordered occupied
  // list into the canonical ascending order.
  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals.size();
         ++right_index) {
      if (occupied_orbitals[left_index] > occupied_orbitals[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

struct CoefficientFactorPart {
  double coefficient = 0.0;
  const std::vector<int>* alpha_occ = nullptr;
  const std::vector<int>* beta_occ = nullptr;
};

std::vector<int> concatenate_occ_lists(const std::vector<std::vector<int>>& occ_lists) {
  std::size_t total_size = 0;
  for (const auto& occ : occ_lists) {
    total_size += occ.size();
  }
  std::vector<int> concatenated;
  concatenated.reserve(total_size);
  for (const auto& occ : occ_lists) {
    concatenated.insert(concatenated.end(), occ.begin(), occ.end());
  }
  return concatenated;
}

double factorized_global_coefficient(
    const std::vector<CoefficientFactorPart>& component_parts) {
  double coefficient = 1.0;
  std::vector<std::vector<int>> alpha_occ_lists;
  std::vector<std::vector<int>> beta_occ_lists;
  alpha_occ_lists.reserve(component_parts.size());
  beta_occ_lists.reserve(component_parts.size());
  for (const auto& part : component_parts) {
    if (part.alpha_occ == nullptr || part.beta_occ == nullptr) {
      throw std::invalid_argument("component occupied-orbital lists must not be null");
    }
    coefficient *= part.coefficient;
    alpha_occ_lists.push_back(*part.alpha_occ);
    beta_occ_lists.push_back(*part.beta_occ);
  }
  const auto alpha_concat = concatenate_occ_lists(alpha_occ_lists);
  const auto beta_concat = concatenate_occ_lists(beta_occ_lists);
  const int inter_component_parity =
      canonicalization_parity(alpha_concat) ^ canonicalization_parity(beta_concat);
  return coefficient * parity_sign(inter_component_parity);
}

void record_coefficient_factorization_check(
    double reference_coefficient,
    double factorized_coefficient,
    StarPairStats* stats) {
  if (stats == nullptr) {
    throw std::invalid_argument("stats must not be null");
  }
  ++stats->coefficient_factorization_checks;
  const double absolute_error =
      std::abs(reference_coefficient - factorized_coefficient);
  stats->max_coefficient_factorization_abs_error = std::max(
      stats->max_coefficient_factorization_abs_error,
      absolute_error);
  if (absolute_error > 1.0e-12) {
    ++stats->coefficient_factorization_mismatches;
  }
}

std::vector<LeafMessageEntry> build_leaf_message_entries(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    ExactWorkCollector* work_collector) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
  const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
  const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
  const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

  std::vector<LeafMessageEntry> entries;
  for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
    for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
      for (std::uint32_t alpha_row_mask = 0;
           alpha_row_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_rows);
           ++alpha_row_mask) {
        const int alpha_row_count = popcount(alpha_row_mask);
        for (std::uint32_t alpha_col_mask = 0;
             alpha_col_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_cols);
             ++alpha_col_mask) {
          const int alpha_col_count = popcount(alpha_col_mask);
          if (alpha_row_count + static_cast<int>(right_leaf_term.alpha_occ.size()) !=
              alpha_col_count + static_cast<int>(left_leaf_term.alpha_occ.size())) {
            continue;
          }

          const auto alpha_rows_root =
              select_occ_by_mask(right_root_term.alpha_occ, alpha_row_mask);
          const auto alpha_cols_root =
              select_occ_by_mask(left_root_term.alpha_occ, alpha_col_mask);
          std::vector<int> ordered_alpha_cols;
          std::vector<int> ordered_alpha_rows;
          const double alpha_determinant = determinant_with_zeroed_selected_root_block(
              alpha_cols_root,
              left_leaf_term.alpha_occ,
              alpha_rows_root,
              right_leaf_term.alpha_occ,
              active_overlap_storage,
              n_active_orbitals,
              overlap_resolver,
              &ordered_alpha_cols,
              &ordered_alpha_rows,
              work_collector);
          ++(*subdeterminant_evaluations);
          if (std::abs(alpha_determinant) <= 1.0e-15) {
            continue;
          }

          for (std::uint32_t beta_row_mask = 0;
               beta_row_mask < (static_cast<std::uint32_t>(1) << n_beta_root_rows);
               ++beta_row_mask) {
            const int beta_row_count = popcount(beta_row_mask);
            for (std::uint32_t beta_col_mask = 0;
                 beta_col_mask < (static_cast<std::uint32_t>(1) << n_beta_root_cols);
                 ++beta_col_mask) {
              const int beta_col_count = popcount(beta_col_mask);
              if (beta_row_count + static_cast<int>(right_leaf_term.beta_occ.size()) !=
                  beta_col_count + static_cast<int>(left_leaf_term.beta_occ.size())) {
                continue;
              }

              const auto beta_rows_root =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_mask);
              const auto beta_cols_root =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_mask);
              std::vector<int> ordered_beta_cols;
              std::vector<int> ordered_beta_rows;
              const double beta_determinant = determinant_with_zeroed_selected_root_block(
                  beta_cols_root,
                  left_leaf_term.beta_occ,
                  beta_rows_root,
                  right_leaf_term.beta_occ,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  &ordered_beta_cols,
                  &ordered_beta_rows,
                  work_collector);
              ++(*subdeterminant_evaluations);
              if (std::abs(beta_determinant) <= 1.0e-15) {
                continue;
              }

              LeafMessageEntry entry;
              entry.alpha_row_mask = alpha_row_mask;
              entry.alpha_col_mask = alpha_col_mask;
              entry.beta_row_mask = beta_row_mask;
              entry.beta_col_mask = beta_col_mask;
              entry.value = alpha_determinant * beta_determinant;
              entry.alpha_block_rows = ordered_alpha_rows;
              entry.alpha_block_cols = ordered_alpha_cols;
              entry.beta_block_rows = std::move(ordered_beta_rows);
              entry.beta_block_cols = std::move(ordered_beta_cols);
              entry.left_alpha_occ = left_leaf_term.alpha_occ;
              entry.left_beta_occ = left_leaf_term.beta_occ;
              entry.right_alpha_occ = right_leaf_term.alpha_occ;
              entry.right_beta_occ = right_leaf_term.beta_occ;
              entry.left_term_coefficient = left_leaf_term.coefficient;
              entry.right_term_coefficient = right_leaf_term.coefficient;
              entries.push_back(std::move(entry));
            }
          }
        }
      }
    }
  }
  return entries;
}

bool is_connected_star_graph(
    const xmvb::vb::MetricAwareComponentGraph& graph,
    int* root_node) {
  if (root_node == nullptr) {
    throw std::invalid_argument("root_node must not be null");
  }
  *root_node = -1;
  if (graph.node_count <= 0) {
    return false;
  }
  if (graph.node_count == 1) {
    *root_node = 0;
    return true;
  }

  for (int candidate_root = 0; candidate_root < graph.node_count; ++candidate_root) {
    bool is_star = true;
    for (int node = 0; node < graph.node_count; ++node) {
      const int degree = static_cast<int>(graph.adjacency[xmvb::to_size(node)].size());
      if (node == candidate_root) {
        if (degree != graph.node_count - 1) {
          is_star = false;
          break;
        }
      } else if (degree != 1) {
        is_star = false;
        break;
      }
    }
    if (is_star) {
      *root_node = candidate_root;
      return true;
    }
  }

  if (graph.node_count == 2) {
    *root_node = 0;
    return static_cast<int>(graph.adjacency[0].size()) == 1 &&
        static_cast<int>(graph.adjacency[1].size()) == 1;
  }
  return false;
}

StarPairStats evaluate_star_pair(
    double exact_overlap,
    std::uint64_t reference_determinant_pair_count,
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals,
    const xmvb::vb::MetricAwareGraphSummary& metric_summary,
    const xmvb::vb::MetricAwareComponentGraph& metric_graph,
    const std::vector<ComponentData>& components,
    int root_node,
    const std::map<CanonicalDeterminantKey, double>& left_coefficient_lookup,
    const std::map<CanonicalDeterminantKey, double>& right_coefficient_lookup,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactWorkCollector* work_collector) {
  StarPairStats stats;
  stats.covered = true;
  stats.exact_overlap = exact_overlap;
  stats.metric_width_upper_bound = metric_summary.weighted_min_degree_width_upper_bound;
  stats.node_count = metric_graph.node_count;
  stats.root_node = root_node;
  stats.reference_determinant_pair_count = reference_determinant_pair_count;

  const auto& root_component = components[xmvb::to_size(root_node)];
  std::vector<int> leaf_nodes = metric_graph.adjacency[xmvb::to_size(root_node)];
  std::sort(leaf_nodes.begin(), leaf_nodes.end());
  int total_alpha_electrons = static_cast<int>(root_component.right_orientation_terms.empty()
                                                   ? 0
                                                   : root_component.right_orientation_terms.front()
                                                         .alpha_occ.size());
  int total_beta_electrons = static_cast<int>(root_component.right_orientation_terms.empty()
                                                  ? 0
                                                  : root_component.right_orientation_terms.front()
                                                        .beta_occ.size());
  for (const int leaf_node : leaf_nodes) {
    const auto& leaf_component = components[xmvb::to_size(leaf_node)];
    total_alpha_electrons +=
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().alpha_occ.size());
    total_beta_electrons +=
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().beta_occ.size());
  }

  for (const auto& left_root_term : root_component.left_orientation_terms) {
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      ++stats.local_term_pair_visits;

      const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
      const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
      const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
      const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

      const std::uint32_t alpha_row_full_mask =
          (n_alpha_root_rows == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_rows) - 1U);
      const std::uint32_t alpha_col_full_mask =
          (n_alpha_root_cols == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_cols) - 1U);
      const std::uint32_t beta_row_full_mask =
          (n_beta_root_rows == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_rows) - 1U);
      const std::uint32_t beta_col_full_mask =
          (n_beta_root_cols == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_cols) - 1U);

      if (leaf_nodes.size() == 1) {
        const auto& leaf_component = components[xmvb::to_size(leaf_nodes[0])];
        stats.local_term_pair_visits +=
            static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
            static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
        for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
          for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
            std::vector<int> left_alpha_occ = left_root_term.alpha_occ;
            std::vector<int> left_beta_occ = left_root_term.beta_occ;
            std::vector<int> right_alpha_occ = right_root_term.alpha_occ;
            std::vector<int> right_beta_occ = right_root_term.beta_occ;
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                left_leaf_term.alpha_occ.begin(),
                left_leaf_term.alpha_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                left_leaf_term.beta_occ.begin(),
                left_leaf_term.beta_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                right_leaf_term.alpha_occ.begin(),
                right_leaf_term.alpha_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                right_leaf_term.beta_occ.begin(),
                right_leaf_term.beta_occ.end());

            const double left_coefficient = lookup_global_coefficient(
                left_alpha_occ,
                left_beta_occ,
                left_coefficient_lookup);
            const double right_coefficient = lookup_global_coefficient(
                right_alpha_occ,
                right_beta_occ,
                right_coefficient_lookup);
            const double left_factorized_coefficient = factorized_global_coefficient(
                {{left_root_term.coefficient, &left_root_term.alpha_occ, &left_root_term.beta_occ},
                 {left_leaf_term.coefficient, &left_leaf_term.alpha_occ, &left_leaf_term.beta_occ}});
            const double right_factorized_coefficient = factorized_global_coefficient(
                {{right_root_term.coefficient,
                  &right_root_term.alpha_occ,
                  &right_root_term.beta_occ},
                 {right_leaf_term.coefficient,
                  &right_leaf_term.alpha_occ,
                  &right_leaf_term.beta_occ}});
            record_coefficient_factorization_check(
                left_coefficient,
                left_factorized_coefficient,
                &stats);
            record_coefficient_factorization_check(
                right_coefficient,
                right_factorized_coefficient,
                &stats);
            if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                std::abs(right_factorized_coefficient) <= 1.0e-15) {
              continue;
            }

            for (std::uint32_t alpha_row_mask = 0;
                 alpha_row_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_rows);
                 ++alpha_row_mask) {
              const int alpha_row_count = popcount(alpha_row_mask);
              for (std::uint32_t alpha_col_mask = 0;
                   alpha_col_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_cols);
                   ++alpha_col_mask) {
                const int alpha_col_count = popcount(alpha_col_mask);
                if (alpha_row_count + static_cast<int>(right_leaf_term.alpha_occ.size()) !=
                    alpha_col_count + static_cast<int>(left_leaf_term.alpha_occ.size())) {
                  continue;
                }

                const auto alpha_rows_root =
                    select_occ_by_mask(right_root_term.alpha_occ, alpha_row_mask);
                const auto alpha_cols_root =
                    select_occ_by_mask(left_root_term.alpha_occ, alpha_col_mask);
                std::vector<int> ordered_alpha_cols;
                std::vector<int> ordered_alpha_rows;
                const double alpha_leaf_determinant =
                    determinant_with_zeroed_selected_root_block(
                        alpha_cols_root,
                        left_leaf_term.alpha_occ,
                        alpha_rows_root,
                        right_leaf_term.alpha_occ,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        &ordered_alpha_cols,
                        &ordered_alpha_rows,
                        work_collector);
                ++stats.subdeterminant_evaluations;
                if (std::abs(alpha_leaf_determinant) <= 1.0e-15) {
                  continue;
                }

                for (std::uint32_t beta_row_mask = 0;
                     beta_row_mask < (static_cast<std::uint32_t>(1) << n_beta_root_rows);
                     ++beta_row_mask) {
                  const int beta_row_count = popcount(beta_row_mask);
                  for (std::uint32_t beta_col_mask = 0;
                       beta_col_mask < (static_cast<std::uint32_t>(1) << n_beta_root_cols);
                       ++beta_col_mask) {
                    const int beta_col_count = popcount(beta_col_mask);
                    if (beta_row_count + static_cast<int>(right_leaf_term.beta_occ.size()) !=
                        beta_col_count + static_cast<int>(left_leaf_term.beta_occ.size())) {
                      continue;
                    }

                    const auto beta_rows_root =
                        select_occ_by_mask(right_root_term.beta_occ, beta_row_mask);
                    const auto beta_cols_root =
                        select_occ_by_mask(left_root_term.beta_occ, beta_col_mask);
                    std::vector<int> ordered_beta_cols;
                    std::vector<int> ordered_beta_rows;
                    const double beta_leaf_determinant =
                        determinant_with_zeroed_selected_root_block(
                            beta_cols_root,
                            left_leaf_term.beta_occ,
                            beta_rows_root,
                            right_leaf_term.beta_occ,
                            active_overlap_storage,
                            n_active_orbitals,
                            overlap_resolver,
                            &ordered_beta_cols,
                            &ordered_beta_rows,
                            work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(beta_leaf_determinant) <= 1.0e-15) {
                      continue;
                    }

                    const std::uint32_t alpha_row_remainder =
                        alpha_row_full_mask ^ alpha_row_mask;
                    const std::uint32_t alpha_col_remainder =
                        alpha_col_full_mask ^ alpha_col_mask;
                    const std::uint32_t beta_row_remainder =
                        beta_row_full_mask ^ beta_row_mask;
                    const std::uint32_t beta_col_remainder =
                        beta_col_full_mask ^ beta_col_mask;
                    const auto alpha_root_rows_remainder =
                        select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
                    const auto alpha_root_cols_remainder =
                        select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
                    const auto beta_root_rows_remainder =
                        select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
                    const auto beta_root_cols_remainder =
                        select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

                    const double alpha_root_determinant = determinant_for_occ_lists(
                        alpha_root_cols_remainder,
                        alpha_root_rows_remainder,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(alpha_root_determinant) <= 1.0e-15) {
                      continue;
                    }

                    const double beta_root_determinant = determinant_for_occ_lists(
                        beta_root_cols_remainder,
                        beta_root_rows_remainder,
                        active_overlap_storage,
                        n_active_orbitals,
                        overlap_resolver,
                        work_collector);
                    ++stats.subdeterminant_evaluations;
                    if (std::abs(beta_root_determinant) <= 1.0e-15) {
                      continue;
                    }

                    std::vector<int> alpha_rows = ordered_alpha_rows;
                    std::vector<int> alpha_cols = ordered_alpha_cols;
                    std::vector<int> beta_rows = ordered_beta_rows;
                    std::vector<int> beta_cols = ordered_beta_cols;
                    alpha_rows.insert(
                        alpha_rows.end(),
                        alpha_root_rows_remainder.begin(),
                        alpha_root_rows_remainder.end());
                    alpha_cols.insert(
                        alpha_cols.end(),
                        alpha_root_cols_remainder.begin(),
                        alpha_root_cols_remainder.end());
                    beta_rows.insert(
                        beta_rows.end(),
                        beta_root_rows_remainder.begin(),
                        beta_root_rows_remainder.end());
                    beta_cols.insert(
                        beta_cols.end(),
                        beta_root_cols_remainder.begin(),
                        beta_root_cols_remainder.end());

                    int parity = 0;
                    parity ^= canonicalization_parity(alpha_rows);
                    parity ^= canonicalization_parity(alpha_cols);
                    parity ^= canonicalization_parity(beta_rows);
                    parity ^= canonicalization_parity(beta_cols);

                    ++stats.dp_transition_count;
                    stats.star_overlap +=
                        left_factorized_coefficient * right_factorized_coefficient *
                        parity_sign(parity) *
                        alpha_leaf_determinant * beta_leaf_determinant *
                        alpha_root_determinant * beta_root_determinant;
                  }
                }
              }
            }
          }
        }
        continue;
      }

      if (leaf_nodes.size() == 2) {
        std::array<std::vector<LeafMessageEntry>, 2> leaf_messages;
        for (int leaf_slot = 0; leaf_slot < 2; ++leaf_slot) {
          const auto& leaf_component =
              components[xmvb::to_size(leaf_nodes[xmvb::to_size(leaf_slot)])];
          stats.local_term_pair_visits +=
              static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
          leaf_messages[leaf_slot] = build_leaf_message_entries(
              left_root_term,
              right_root_term,
              leaf_component,
              active_overlap_storage,
              n_active_orbitals,
              overlap_resolver,
              &stats.subdeterminant_evaluations,
              work_collector);
        }

        for (const auto& first_entry : leaf_messages[0]) {
          for (const auto& second_entry : leaf_messages[1]) {
            if ((first_entry.alpha_row_mask & second_entry.alpha_row_mask) != 0U ||
                (first_entry.alpha_col_mask & second_entry.alpha_col_mask) != 0U ||
                (first_entry.beta_row_mask & second_entry.beta_row_mask) != 0U ||
                (first_entry.beta_col_mask & second_entry.beta_col_mask) != 0U) {
              continue;
            }

            const std::uint32_t used_alpha_row_mask =
                first_entry.alpha_row_mask | second_entry.alpha_row_mask;
            const std::uint32_t used_alpha_col_mask =
                first_entry.alpha_col_mask | second_entry.alpha_col_mask;
            const std::uint32_t used_beta_row_mask =
                first_entry.beta_row_mask | second_entry.beta_row_mask;
            const std::uint32_t used_beta_col_mask =
                first_entry.beta_col_mask | second_entry.beta_col_mask;

            const std::uint32_t alpha_row_remainder =
                alpha_row_full_mask ^ used_alpha_row_mask;
            const std::uint32_t alpha_col_remainder =
                alpha_col_full_mask ^ used_alpha_col_mask;
            const std::uint32_t beta_row_remainder =
                beta_row_full_mask ^ used_beta_row_mask;
            const std::uint32_t beta_col_remainder =
                beta_col_full_mask ^ used_beta_col_mask;

            const auto alpha_root_rows =
                select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
            const auto alpha_root_cols =
                select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
            const auto beta_root_rows =
                select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
            const auto beta_root_cols =
                select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

            const double alpha_root_determinant = determinant_for_occ_lists(
                alpha_root_cols,
                alpha_root_rows,
                active_overlap_storage,
                n_active_orbitals,
                overlap_resolver,
                work_collector);
            ++stats.subdeterminant_evaluations;
            if (std::abs(alpha_root_determinant) <= 1.0e-15) {
              continue;
            }

            const double beta_root_determinant = determinant_for_occ_lists(
                beta_root_cols,
                beta_root_rows,
                active_overlap_storage,
                n_active_orbitals,
                overlap_resolver,
                work_collector);
            ++stats.subdeterminant_evaluations;
            if (std::abs(beta_root_determinant) <= 1.0e-15) {
              continue;
            }

            std::vector<int> alpha_rows = first_entry.alpha_block_rows;
            std::vector<int> alpha_cols = first_entry.alpha_block_cols;
            std::vector<int> beta_rows = first_entry.beta_block_rows;
            std::vector<int> beta_cols = first_entry.beta_block_cols;
            alpha_rows.insert(
                alpha_rows.end(),
                second_entry.alpha_block_rows.begin(),
                second_entry.alpha_block_rows.end());
            alpha_cols.insert(
                alpha_cols.end(),
                second_entry.alpha_block_cols.begin(),
                second_entry.alpha_block_cols.end());
            beta_rows.insert(
                beta_rows.end(),
                second_entry.beta_block_rows.begin(),
                second_entry.beta_block_rows.end());
            beta_cols.insert(
                beta_cols.end(),
                second_entry.beta_block_cols.begin(),
                second_entry.beta_block_cols.end());
            alpha_rows.insert(alpha_rows.end(), alpha_root_rows.begin(), alpha_root_rows.end());
            alpha_cols.insert(alpha_cols.end(), alpha_root_cols.begin(), alpha_root_cols.end());
            beta_rows.insert(beta_rows.end(), beta_root_rows.begin(), beta_root_rows.end());
            beta_cols.insert(beta_cols.end(), beta_root_cols.begin(), beta_root_cols.end());

            int parity = 0;
            parity ^= canonicalization_parity(alpha_rows);
            parity ^= canonicalization_parity(alpha_cols);
            parity ^= canonicalization_parity(beta_rows);
            parity ^= canonicalization_parity(beta_cols);

            std::vector<int> left_alpha_occ = left_root_term.alpha_occ;
            std::vector<int> left_beta_occ = left_root_term.beta_occ;
            std::vector<int> right_alpha_occ = right_root_term.alpha_occ;
            std::vector<int> right_beta_occ = right_root_term.beta_occ;
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                first_entry.left_alpha_occ.begin(),
                first_entry.left_alpha_occ.end());
            left_alpha_occ.insert(
                left_alpha_occ.end(),
                second_entry.left_alpha_occ.begin(),
                second_entry.left_alpha_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                first_entry.left_beta_occ.begin(),
                first_entry.left_beta_occ.end());
            left_beta_occ.insert(
                left_beta_occ.end(),
                second_entry.left_beta_occ.begin(),
                second_entry.left_beta_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                first_entry.right_alpha_occ.begin(),
                first_entry.right_alpha_occ.end());
            right_alpha_occ.insert(
                right_alpha_occ.end(),
                second_entry.right_alpha_occ.begin(),
                second_entry.right_alpha_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                first_entry.right_beta_occ.begin(),
                first_entry.right_beta_occ.end());
            right_beta_occ.insert(
                right_beta_occ.end(),
                second_entry.right_beta_occ.begin(),
                second_entry.right_beta_occ.end());

            const double left_coefficient = lookup_global_coefficient(
                left_alpha_occ,
                left_beta_occ,
                left_coefficient_lookup);
            const double right_coefficient = lookup_global_coefficient(
                right_alpha_occ,
                right_beta_occ,
                right_coefficient_lookup);
            const double left_factorized_coefficient = factorized_global_coefficient(
                {{left_root_term.coefficient,
                  &left_root_term.alpha_occ,
                  &left_root_term.beta_occ},
                 {first_entry.left_term_coefficient,
                  &first_entry.left_alpha_occ,
                  &first_entry.left_beta_occ},
                 {second_entry.left_term_coefficient,
                  &second_entry.left_alpha_occ,
                  &second_entry.left_beta_occ}});
            const double right_factorized_coefficient = factorized_global_coefficient(
                {{right_root_term.coefficient,
                  &right_root_term.alpha_occ,
                  &right_root_term.beta_occ},
                 {first_entry.right_term_coefficient,
                  &first_entry.right_alpha_occ,
                  &first_entry.right_beta_occ},
                 {second_entry.right_term_coefficient,
                  &second_entry.right_alpha_occ,
                  &second_entry.right_beta_occ}});
            record_coefficient_factorization_check(
                left_coefficient,
                left_factorized_coefficient,
                &stats);
            record_coefficient_factorization_check(
                right_coefficient,
                right_factorized_coefficient,
                &stats);
            if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                std::abs(right_factorized_coefficient) <= 1.0e-15) {
              continue;
            }

            ++stats.dp_transition_count;
            stats.star_overlap +=
                left_factorized_coefficient * right_factorized_coefficient *
                parity_sign(parity) *
                first_entry.value * second_entry.value *
                alpha_root_determinant * beta_root_determinant;
          }
        }
        continue;
      }

      std::vector<std::vector<LeafMessageEntry>> leaf_messages(leaf_nodes.size());
      for (std::size_t leaf_order = 0; leaf_order < leaf_nodes.size(); ++leaf_order) {
        const auto& leaf_component = components[xmvb::to_size(leaf_nodes[leaf_order])];
        stats.local_term_pair_visits +=
            static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
            static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
        leaf_messages[leaf_order] = build_leaf_message_entries(
            left_root_term,
            right_root_term,
            leaf_component,
            active_overlap_storage,
            n_active_orbitals,
            overlap_resolver,
            &stats.subdeterminant_evaluations,
            work_collector);
      }

      std::vector<int> alpha_block_rows;
      std::vector<int> alpha_block_cols;
      std::vector<int> beta_block_rows;
      std::vector<int> beta_block_cols;
      std::vector<int> left_alpha_occ;
      std::vector<int> left_beta_occ;
      std::vector<int> right_alpha_occ;
      std::vector<int> right_beta_occ;
      std::vector<CoefficientFactorPart> left_coefficient_parts;
      std::vector<CoefficientFactorPart> right_coefficient_parts;
      alpha_block_rows.reserve(xmvb::to_size(total_alpha_electrons));
      alpha_block_cols.reserve(xmvb::to_size(total_alpha_electrons));
      beta_block_rows.reserve(xmvb::to_size(total_beta_electrons));
      beta_block_cols.reserve(xmvb::to_size(total_beta_electrons));
      left_alpha_occ.reserve(xmvb::to_size(total_alpha_electrons));
      left_beta_occ.reserve(xmvb::to_size(total_beta_electrons));
      right_alpha_occ.reserve(xmvb::to_size(total_alpha_electrons));
      right_beta_occ.reserve(xmvb::to_size(total_beta_electrons));
      left_coefficient_parts.reserve(1 + leaf_messages.size());
      right_coefficient_parts.reserve(1 + leaf_messages.size());
      left_alpha_occ.insert(
          left_alpha_occ.end(),
          left_root_term.alpha_occ.begin(),
          left_root_term.alpha_occ.end());
      left_beta_occ.insert(
          left_beta_occ.end(),
          left_root_term.beta_occ.begin(),
          left_root_term.beta_occ.end());
      right_alpha_occ.insert(
          right_alpha_occ.end(),
          right_root_term.alpha_occ.begin(),
          right_root_term.alpha_occ.end());
      right_beta_occ.insert(
          right_beta_occ.end(),
          right_root_term.beta_occ.begin(),
          right_root_term.beta_occ.end());
      left_coefficient_parts.push_back(
          {left_root_term.coefficient, &left_root_term.alpha_occ, &left_root_term.beta_occ});
      right_coefficient_parts.push_back(
          {right_root_term.coefficient,
           &right_root_term.alpha_occ,
           &right_root_term.beta_occ});

      const auto accumulate_leaf_assignments =
          [&](const auto& self,
              std::size_t leaf_order,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double state_value) -> void {
            if (std::abs(state_value) <= 1.0e-15) {
              return;
            }

            if (leaf_order == leaf_messages.size()) {
              const std::uint32_t alpha_row_remainder =
                  alpha_row_full_mask ^ used_alpha_row_mask;
              const std::uint32_t alpha_col_remainder =
                  alpha_col_full_mask ^ used_alpha_col_mask;
              const std::uint32_t beta_row_remainder =
                  beta_row_full_mask ^ used_beta_row_mask;
              const std::uint32_t beta_col_remainder =
                  beta_col_full_mask ^ used_beta_col_mask;

              const auto alpha_root_rows =
                  select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
              const auto alpha_root_cols =
                  select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
              const auto beta_root_rows =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
              const auto beta_root_cols =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

              const double alpha_root_determinant = determinant_for_occ_lists(
                  alpha_root_cols,
                  alpha_root_rows,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  work_collector);
              ++stats.subdeterminant_evaluations;
              if (std::abs(alpha_root_determinant) <= 1.0e-15) {
                return;
              }

              const double beta_root_determinant = determinant_for_occ_lists(
                  beta_root_cols,
                  beta_root_rows,
                  active_overlap_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  work_collector);
              ++stats.subdeterminant_evaluations;
              if (std::abs(beta_root_determinant) <= 1.0e-15) {
                return;
              }

              std::vector<int> alpha_rows = alpha_block_rows;
              std::vector<int> alpha_cols = alpha_block_cols;
              std::vector<int> beta_rows = beta_block_rows;
              std::vector<int> beta_cols = beta_block_cols;
              alpha_rows.insert(alpha_rows.end(), alpha_root_rows.begin(), alpha_root_rows.end());
              alpha_cols.insert(alpha_cols.end(), alpha_root_cols.begin(), alpha_root_cols.end());
              beta_rows.insert(beta_rows.end(), beta_root_rows.begin(), beta_root_rows.end());
              beta_cols.insert(beta_cols.end(), beta_root_cols.begin(), beta_root_cols.end());

              int parity = 0;
              parity ^= canonicalization_parity(alpha_rows);
              parity ^= canonicalization_parity(alpha_cols);
              parity ^= canonicalization_parity(beta_rows);
              parity ^= canonicalization_parity(beta_cols);

              const double left_coefficient = lookup_global_coefficient(
                  left_alpha_occ,
                  left_beta_occ,
                  left_coefficient_lookup);
              const double right_coefficient = lookup_global_coefficient(
                  right_alpha_occ,
                  right_beta_occ,
                  right_coefficient_lookup);
              const double left_factorized_coefficient =
                  factorized_global_coefficient(left_coefficient_parts);
              const double right_factorized_coefficient =
                  factorized_global_coefficient(right_coefficient_parts);
              record_coefficient_factorization_check(
                  left_coefficient,
                  left_factorized_coefficient,
                  &stats);
              record_coefficient_factorization_check(
                  right_coefficient,
                  right_factorized_coefficient,
                  &stats);
              if (std::abs(left_factorized_coefficient) <= 1.0e-15 ||
                  std::abs(right_factorized_coefficient) <= 1.0e-15) {
                return;
              }

              stats.star_overlap +=
                  left_factorized_coefficient * right_factorized_coefficient *
                  parity_sign(parity) * state_value *
                  alpha_root_determinant * beta_root_determinant;
              return;
            }

            for (const auto& entry : leaf_messages[leaf_order]) {
              if ((used_alpha_row_mask & entry.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & entry.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & entry.beta_row_mask) != 0U ||
                  (used_beta_col_mask & entry.beta_col_mask) != 0U) {
                continue;
              }

              const std::size_t alpha_row_size = alpha_block_rows.size();
              const std::size_t alpha_col_size = alpha_block_cols.size();
              const std::size_t beta_row_size = beta_block_rows.size();
              const std::size_t beta_col_size = beta_block_cols.size();
              const std::size_t left_alpha_size = left_alpha_occ.size();
              const std::size_t left_beta_size = left_beta_occ.size();
              const std::size_t right_alpha_size = right_alpha_occ.size();
              const std::size_t right_beta_size = right_beta_occ.size();
              const std::size_t left_coefficient_size = left_coefficient_parts.size();
              const std::size_t right_coefficient_size = right_coefficient_parts.size();
              alpha_block_rows.insert(
                  alpha_block_rows.end(),
                  entry.alpha_block_rows.begin(),
                  entry.alpha_block_rows.end());
              alpha_block_cols.insert(
                  alpha_block_cols.end(),
                  entry.alpha_block_cols.begin(),
                  entry.alpha_block_cols.end());
              beta_block_rows.insert(
                  beta_block_rows.end(),
                  entry.beta_block_rows.begin(),
                  entry.beta_block_rows.end());
              beta_block_cols.insert(
                  beta_block_cols.end(),
                  entry.beta_block_cols.begin(),
                  entry.beta_block_cols.end());
              left_alpha_occ.insert(
                  left_alpha_occ.end(),
                  entry.left_alpha_occ.begin(),
                  entry.left_alpha_occ.end());
              left_beta_occ.insert(
                  left_beta_occ.end(),
                  entry.left_beta_occ.begin(),
                  entry.left_beta_occ.end());
              right_alpha_occ.insert(
                  right_alpha_occ.end(),
                  entry.right_alpha_occ.begin(),
                  entry.right_alpha_occ.end());
              right_beta_occ.insert(
                  right_beta_occ.end(),
                  entry.right_beta_occ.begin(),
                  entry.right_beta_occ.end());
              left_coefficient_parts.push_back(
                  {entry.left_term_coefficient, &entry.left_alpha_occ, &entry.left_beta_occ});
              right_coefficient_parts.push_back(
                  {entry.right_term_coefficient,
                   &entry.right_alpha_occ,
                   &entry.right_beta_occ});

              ++stats.dp_transition_count;
              self(
                  self,
                  leaf_order + 1,
                  used_alpha_row_mask | entry.alpha_row_mask,
                  used_alpha_col_mask | entry.alpha_col_mask,
                  used_beta_row_mask | entry.beta_row_mask,
                  used_beta_col_mask | entry.beta_col_mask,
                  state_value * entry.value);

              alpha_block_rows.resize(alpha_row_size);
              alpha_block_cols.resize(alpha_col_size);
              beta_block_rows.resize(beta_row_size);
              beta_block_cols.resize(beta_col_size);
              left_alpha_occ.resize(left_alpha_size);
              left_beta_occ.resize(left_beta_size);
              right_alpha_occ.resize(right_alpha_size);
              right_beta_occ.resize(right_beta_size);
              left_coefficient_parts.resize(left_coefficient_size);
              right_coefficient_parts.resize(right_coefficient_size);
            }
          };
      accumulate_leaf_assignments(
          accumulate_leaf_assignments,
          0,
          0U,
          0U,
          0U,
          0U,
          1.0);
    }
  }
  stats.absolute_error = std::abs(stats.exact_overlap - stats.star_overlap);
  return stats;
}

ExplicitOneLeafOneElectronBreakdown evaluate_star_pair_one_electron_explicit_one_leaf(
    const std::vector<double>& active_overlap_storage,
    const std::vector<double>& active_one_electron_storage,
    int n_active_orbitals,
    const ComponentData& root_component,
    const ComponentData& leaf_component,
    const std::map<CanonicalDeterminantKey, double>& left_coefficient_lookup,
    const std::map<CanonicalDeterminantKey, double>& right_coefficient_lookup,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count) {
  // This routine keeps the exact star recurrence at the explicit leaf-term
  // level for the single-leaf case. It answers a sharper question than the
  // collapsed mask test: if one-electron already fails here, then the missing
  // information is not caused by mask aggregation, but by missing bridge/open
  // states in the recurrence itself.
  if (subdeterminant_evaluations == nullptr || dp_transition_count == nullptr) {
    throw std::invalid_argument("explicit one-electron counters must not be null");
  }

  ExplicitOneLeafOneElectronBreakdown breakdown;
  std::vector<RootPairClosedCorrectionFitData> closed_correction_fit_equations;
  const SpinCoupledRootChannelLayout direct_root_channel_layout =
      build_spin_coupled_root_channel_layout(root_component);
  SpinCoupledRootChannelBasis direct_alpha_root_channel_basis;
  SpinCoupledRootChannelBasis direct_beta_root_channel_basis;
  for (const auto& left_root_term : root_component.left_orientation_terms) {
    const SpinRootStateKey left_root_state{
        left_root_term.alpha_occ,
        left_root_term.beta_occ,
    };
    const int left_root_state_index =
        direct_root_channel_layout.supported
            ? direct_root_channel_layout.left_state_index.at(left_root_state)
            : 0;
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      const SpinRootStateKey right_root_state{
          right_root_term.alpha_occ,
          right_root_term.beta_occ,
      };
      const int right_root_state_index =
          direct_root_channel_layout.supported
              ? direct_root_channel_layout.right_state_index.at(right_root_state)
              : 0;
      RootPairClosedCorrectionFitData root_pair_fit_data;
      root_pair_fit_data.left_root_alpha_occ = left_root_term.alpha_occ;
      root_pair_fit_data.left_root_beta_occ = left_root_term.beta_occ;
      root_pair_fit_data.right_root_alpha_occ = right_root_term.alpha_occ;
      root_pair_fit_data.right_root_beta_occ = right_root_term.beta_occ;
      const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
      const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
      const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
      const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

      const std::uint32_t alpha_row_full_mask =
          (n_alpha_root_rows == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_rows) - 1U);
      const std::uint32_t alpha_col_full_mask =
          (n_alpha_root_cols == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_cols) - 1U);
      const std::uint32_t beta_row_full_mask =
          (n_beta_root_rows == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_rows) - 1U);
      const std::uint32_t beta_col_full_mask =
          (n_beta_root_cols == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_cols) - 1U);

      for (const auto& left_leaf_term : leaf_component.left_orientation_terms) {
        for (const auto& right_leaf_term : leaf_component.right_orientation_terms) {
          std::vector<int> left_alpha_occ = left_root_term.alpha_occ;
          std::vector<int> left_beta_occ = left_root_term.beta_occ;
          std::vector<int> right_alpha_occ = right_root_term.alpha_occ;
          std::vector<int> right_beta_occ = right_root_term.beta_occ;
          left_alpha_occ.insert(
              left_alpha_occ.end(),
              left_leaf_term.alpha_occ.begin(),
              left_leaf_term.alpha_occ.end());
          left_beta_occ.insert(
              left_beta_occ.end(),
              left_leaf_term.beta_occ.begin(),
              left_leaf_term.beta_occ.end());
          right_alpha_occ.insert(
              right_alpha_occ.end(),
              right_leaf_term.alpha_occ.begin(),
              right_leaf_term.alpha_occ.end());
          right_beta_occ.insert(
              right_beta_occ.end(),
              right_leaf_term.beta_occ.begin(),
              right_leaf_term.beta_occ.end());

          const double left_coefficient = lookup_global_coefficient(
              left_alpha_occ,
              left_beta_occ,
              left_coefficient_lookup);
          const double right_coefficient = lookup_global_coefficient(
              right_alpha_occ,
              right_beta_occ,
              right_coefficient_lookup);
          const double left_factorized_coefficient = factorized_global_coefficient(
              {{left_root_term.coefficient, &left_root_term.alpha_occ, &left_root_term.beta_occ},
               {left_leaf_term.coefficient, &left_leaf_term.alpha_occ, &left_leaf_term.beta_occ}});
          const double right_factorized_coefficient = factorized_global_coefficient(
              {{right_root_term.coefficient,
                &right_root_term.alpha_occ,
                &right_root_term.beta_occ},
               {right_leaf_term.coefficient,
                &right_leaf_term.alpha_occ,
                &right_leaf_term.beta_occ}});
          const double coefficient =
              left_factorized_coefficient * right_factorized_coefficient;
          if (std::abs(left_coefficient - left_factorized_coefficient) > 1.0e-12 ||
              std::abs(right_coefficient - right_factorized_coefficient) > 1.0e-12 ||
              std::abs(coefficient) <= 1.0e-15) {
            continue;
          }

          {
            const auto alpha_exact = evaluate_occ_list_spin_values(
                left_alpha_occ,
                right_alpha_occ,
                active_overlap_storage,
                active_one_electron_storage,
                n_active_orbitals,
                overlap_resolver,
                subdeterminant_evaluations);
            const auto beta_exact = evaluate_occ_list_spin_values(
                left_beta_occ,
                right_beta_occ,
                active_overlap_storage,
                active_one_electron_storage,
                n_active_orbitals,
                overlap_resolver,
                subdeterminant_evaluations);
            root_pair_fit_data.exact_total +=
                coefficient *
                (alpha_exact.one_electron * beta_exact.overlap +
                 beta_exact.one_electron * alpha_exact.overlap);
          }

          for (std::uint32_t alpha_row_mask = 0;
               alpha_row_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_rows);
               ++alpha_row_mask) {
            const int alpha_row_count = popcount(alpha_row_mask);
            for (std::uint32_t alpha_col_mask = 0;
                 alpha_col_mask < (static_cast<std::uint32_t>(1) << n_alpha_root_cols);
                 ++alpha_col_mask) {
              const int alpha_col_count = popcount(alpha_col_mask);
              if (alpha_row_count + static_cast<int>(right_leaf_term.alpha_occ.size()) !=
                  alpha_col_count + static_cast<int>(left_leaf_term.alpha_occ.size())) {
                continue;
              }

              const auto alpha_rows_root =
                  select_occ_by_mask(right_root_term.alpha_occ, alpha_row_mask);
              const auto alpha_cols_root =
                  select_occ_by_mask(left_root_term.alpha_occ, alpha_col_mask);
              const auto alpha_leaf = evaluate_zeroed_selected_root_block_spin_values(
                  alpha_cols_root,
                  left_leaf_term.alpha_occ,
                  alpha_rows_root,
                  right_leaf_term.alpha_occ,
                  active_overlap_storage,
                  active_one_electron_storage,
                  n_active_orbitals,
                  overlap_resolver,
                  subdeterminant_evaluations);
              if (std::abs(alpha_leaf.overlap) <= 1.0e-15 &&
                  std::abs(alpha_leaf.one_electron) <= 1.0e-15) {
                continue;
              }

              for (std::uint32_t beta_row_mask = 0;
                   beta_row_mask < (static_cast<std::uint32_t>(1) << n_beta_root_rows);
                   ++beta_row_mask) {
                const int beta_row_count = popcount(beta_row_mask);
                for (std::uint32_t beta_col_mask = 0;
                     beta_col_mask < (static_cast<std::uint32_t>(1) << n_beta_root_cols);
                     ++beta_col_mask) {
                  const int beta_col_count = popcount(beta_col_mask);
                  if (beta_row_count + static_cast<int>(right_leaf_term.beta_occ.size()) !=
                      beta_col_count + static_cast<int>(left_leaf_term.beta_occ.size())) {
                    continue;
                  }

                  const auto beta_rows_root =
                      select_occ_by_mask(right_root_term.beta_occ, beta_row_mask);
                  const auto beta_cols_root =
                      select_occ_by_mask(left_root_term.beta_occ, beta_col_mask);
                  const auto beta_leaf = evaluate_zeroed_selected_root_block_spin_values(
                      beta_cols_root,
                      left_leaf_term.beta_occ,
                      beta_rows_root,
                      right_leaf_term.beta_occ,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  if (std::abs(beta_leaf.overlap) <= 1.0e-15 &&
                      std::abs(beta_leaf.one_electron) <= 1.0e-15) {
                    continue;
                  }

                  const std::uint32_t alpha_row_remainder =
                      alpha_row_full_mask ^ alpha_row_mask;
                  const std::uint32_t alpha_col_remainder =
                      alpha_col_full_mask ^ alpha_col_mask;
                  const std::uint32_t beta_row_remainder =
                      beta_row_full_mask ^ beta_row_mask;
                  const std::uint32_t beta_col_remainder =
                      beta_col_full_mask ^ beta_col_mask;

                  const auto alpha_root_rows =
                      select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
                  const auto alpha_root_cols =
                      select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
                  const auto beta_root_rows =
                      select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
                  const auto beta_root_cols =
                      select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);

                  const auto alpha_root = evaluate_occ_list_spin_values(
                      alpha_root_cols,
                      alpha_root_rows,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto beta_root = evaluate_occ_list_spin_values(
                      beta_root_cols,
                      beta_root_rows,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);

                  if ((std::abs(alpha_root.overlap) <= 1.0e-15 &&
                       std::abs(alpha_root.one_electron) <= 1.0e-15) ||
                      (std::abs(beta_root.overlap) <= 1.0e-15 &&
                       std::abs(beta_root.one_electron) <= 1.0e-15)) {
                    continue;
                  }

                  std::vector<int> alpha_rows = alpha_rows_root;
                  std::vector<int> alpha_cols = alpha_cols_root;
                  std::vector<int> beta_rows = beta_rows_root;
                  std::vector<int> beta_cols = beta_cols_root;
                  alpha_rows.insert(
                      alpha_rows.end(),
                      right_leaf_term.alpha_occ.begin(),
                      right_leaf_term.alpha_occ.end());
                  alpha_cols.insert(
                      alpha_cols.end(),
                      left_leaf_term.alpha_occ.begin(),
                      left_leaf_term.alpha_occ.end());
                  beta_rows.insert(
                      beta_rows.end(),
                      right_leaf_term.beta_occ.begin(),
                      right_leaf_term.beta_occ.end());
                  beta_cols.insert(
                      beta_cols.end(),
                      left_leaf_term.beta_occ.begin(),
                      left_leaf_term.beta_occ.end());
                  alpha_rows.insert(alpha_rows.end(), alpha_root_rows.begin(), alpha_root_rows.end());
                  alpha_cols.insert(alpha_cols.end(), alpha_root_cols.begin(), alpha_root_cols.end());
                  beta_rows.insert(beta_rows.end(), beta_root_rows.begin(), beta_root_rows.end());
                  beta_cols.insert(beta_cols.end(), beta_root_cols.begin(), beta_root_cols.end());

                  const int alpha_message_size =
                      static_cast<int>(alpha_rows_root.size() + right_leaf_term.alpha_occ.size());
                  const int beta_message_size =
                      static_cast<int>(beta_rows_root.size() + right_leaf_term.beta_occ.size());
                  const auto alpha_full_breakdown = evaluate_spin_block_breakdown(
                      alpha_cols,
                      alpha_rows,
                      alpha_message_size,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto alpha_three_block = evaluate_spin_three_block_breakdown(
                      alpha_cols,
                      alpha_rows,
                      static_cast<int>(alpha_rows_root.size()),
                      static_cast<int>(right_leaf_term.alpha_occ.size()),
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto alpha_block_diagonal = evaluate_block_diagonalized_zeroed_selected_root_spin_values(
                      alpha_cols_root,
                      left_leaf_term.alpha_occ,
                      alpha_root_cols,
                      alpha_rows_root,
                      right_leaf_term.alpha_occ,
                      alpha_root_rows,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto alpha_transfer_breakdown = evaluate_spin_laplace_transfer_breakdown(
                      alpha_cols,
                      alpha_rows,
                      static_cast<int>(alpha_rows_root.size()),
                      static_cast<int>(right_leaf_term.alpha_occ.size()),
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto alpha_zeroed_transfer_breakdown =
                      evaluate_spin_zeroed_block_laplace_transfer_breakdown(
                          alpha_cols,
                          alpha_rows,
                          static_cast<int>(alpha_rows_root.size()),
                          static_cast<int>(right_leaf_term.alpha_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto alpha_cross_rank_breakdown =
                      evaluate_spin_full_cross_cofactor_ranks(
                          alpha_cols,
                          alpha_rows,
                          static_cast<int>(alpha_rows_root.size()),
                          static_cast<int>(right_leaf_term.alpha_occ.size()),
                          active_overlap_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto alpha_open_state_prototype =
                      evaluate_spin_open_state_prototype_breakdown(
                          alpha_cols,
                          alpha_rows,
                          static_cast<int>(alpha_rows_root.size()),
                          static_cast<int>(right_leaf_term.alpha_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  std::vector<int> alpha_message_rows = alpha_rows_root;
                  std::vector<int> alpha_message_cols = alpha_cols_root;
                  alpha_message_rows.insert(
                      alpha_message_rows.end(),
                      right_leaf_term.alpha_occ.begin(),
                      right_leaf_term.alpha_occ.end());
                  alpha_message_cols.insert(
                      alpha_message_cols.end(),
                      left_leaf_term.alpha_occ.begin(),
                      left_leaf_term.alpha_occ.end());
                  const auto alpha_message_full_breakdown =
                      evaluate_spin_laplace_transfer_breakdown(
                          alpha_message_cols,
                          alpha_message_rows,
                          static_cast<int>(alpha_rows_root.size()),
                          static_cast<int>(right_leaf_term.alpha_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto alpha_message_zeroed_breakdown =
                      evaluate_spin_zeroed_block_laplace_transfer_breakdown(
                          alpha_message_cols,
                          alpha_message_rows,
                          static_cast<int>(alpha_rows_root.size()),
                          static_cast<int>(right_leaf_term.alpha_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto beta_full_breakdown = evaluate_spin_block_breakdown(
                      beta_cols,
                      beta_rows,
                      beta_message_size,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto beta_transfer_breakdown = evaluate_spin_laplace_transfer_breakdown(
                      beta_cols,
                      beta_rows,
                      static_cast<int>(beta_rows_root.size()),
                      static_cast<int>(right_leaf_term.beta_occ.size()),
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto beta_zeroed_transfer_breakdown =
                      evaluate_spin_zeroed_block_laplace_transfer_breakdown(
                          beta_cols,
                          beta_rows,
                          static_cast<int>(beta_rows_root.size()),
                          static_cast<int>(right_leaf_term.beta_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto beta_cross_rank_breakdown =
                      evaluate_spin_full_cross_cofactor_ranks(
                          beta_cols,
                          beta_rows,
                          static_cast<int>(beta_rows_root.size()),
                          static_cast<int>(right_leaf_term.beta_occ.size()),
                          active_overlap_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto beta_open_state_prototype =
                      evaluate_spin_open_state_prototype_breakdown(
                          beta_cols,
                          beta_rows,
                          static_cast<int>(beta_rows_root.size()),
                          static_cast<int>(right_leaf_term.beta_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  std::vector<int> beta_message_rows = beta_rows_root;
                  std::vector<int> beta_message_cols = beta_cols_root;
                  beta_message_rows.insert(
                      beta_message_rows.end(),
                      right_leaf_term.beta_occ.begin(),
                      right_leaf_term.beta_occ.end());
                  beta_message_cols.insert(
                      beta_message_cols.end(),
                      left_leaf_term.beta_occ.begin(),
                      left_leaf_term.beta_occ.end());
                  const auto beta_message_full_breakdown =
                      evaluate_spin_laplace_transfer_breakdown(
                          beta_message_cols,
                          beta_message_rows,
                          static_cast<int>(beta_rows_root.size()),
                          static_cast<int>(right_leaf_term.beta_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto beta_message_zeroed_breakdown =
                      evaluate_spin_zeroed_block_laplace_transfer_breakdown(
                          beta_message_cols,
                          beta_message_rows,
                          static_cast<int>(beta_rows_root.size()),
                          static_cast<int>(right_leaf_term.beta_occ.size()),
                          active_overlap_storage,
                          active_one_electron_storage,
                          n_active_orbitals,
                          overlap_resolver,
                          subdeterminant_evaluations);
                  const auto beta_block_diagonal = evaluate_block_diagonalized_zeroed_selected_root_spin_values(
                      beta_cols_root,
                      left_leaf_term.beta_occ,
                      beta_root_cols,
                      beta_rows_root,
                      right_leaf_term.beta_occ,
                      beta_root_rows,
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  const auto beta_three_block = evaluate_spin_three_block_breakdown(
                      beta_cols,
                      beta_rows,
                      static_cast<int>(beta_rows_root.size()),
                      static_cast<int>(right_leaf_term.beta_occ.size()),
                      active_overlap_storage,
                      active_one_electron_storage,
                      n_active_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations);

                  int parity = 0;
                  parity ^= canonicalization_parity(alpha_rows);
                  parity ^= canonicalization_parity(alpha_cols);
                  parity ^= canonicalization_parity(beta_rows);
                  parity ^= canonicalization_parity(beta_cols);

                  ++(*dp_transition_count);
                  const double signed_coefficient =
                      coefficient * parity_sign(parity);
                  root_pair_fit_data.captured_total +=
                      signed_coefficient *
                      ((alpha_leaf.one_electron * beta_leaf.overlap +
                        beta_leaf.one_electron * alpha_leaf.overlap) *
                           alpha_root.overlap * beta_root.overlap +
                       alpha_leaf.overlap * beta_leaf.overlap *
                           (alpha_root.one_electron * beta_root.overlap +
                            beta_root.one_electron * alpha_root.overlap));
                  root_pair_fit_data.alpha_features[SpinMaskKey{alpha_row_mask, alpha_col_mask}] +=
                      signed_coefficient *
                      alpha_root.overlap * beta_root.overlap * beta_leaf.overlap;
                  root_pair_fit_data.beta_features[SpinMaskKey{beta_row_mask, beta_col_mask}] +=
                      signed_coefficient *
                      alpha_root.overlap * beta_root.overlap * alpha_leaf.overlap;
                  root_pair_fit_data.alpha_full_features[SpinMaskKey{alpha_row_mask, alpha_col_mask}] +=
                      signed_coefficient *
                      alpha_root.overlap * beta_full_breakdown.overlap;
                  root_pair_fit_data.beta_full_features[SpinMaskKey{beta_row_mask, beta_col_mask}] +=
                      signed_coefficient *
                      beta_root.overlap * alpha_full_breakdown.overlap;
                  if (alpha_open_state_prototype.residual_leaf_leaf_cofactor.size() > 0) {
                    const double alpha_target_scale =
                        signed_coefficient * beta_full_breakdown.overlap;
                    if (root_pair_fit_data.alpha_target_matrix.size() == 0) {
                      root_pair_fit_data.alpha_target_matrix = Matrix::Zero(
                          alpha_open_state_prototype.residual_leaf_leaf_cofactor.rows(),
                          alpha_open_state_prototype.residual_leaf_leaf_cofactor.cols());
                    }
                    root_pair_fit_data.alpha_target_matrix.noalias() +=
                        alpha_target_scale *
                        alpha_open_state_prototype.residual_leaf_leaf_cofactor;
                    accumulate_direct_spin_coupled_root_channel_contribution(
                        direct_root_channel_layout,
                        left_root_state_index,
                        right_root_state_index,
                        alpha_open_state_prototype.residual_leaf_leaf_cofactor,
                        alpha_target_scale,
                        true,
                        &direct_alpha_root_channel_basis);
                  }
                  if (beta_open_state_prototype.residual_leaf_leaf_cofactor.size() > 0) {
                    const double beta_target_scale =
                        signed_coefficient * alpha_full_breakdown.overlap;
                    if (root_pair_fit_data.beta_target_matrix.size() == 0) {
                      root_pair_fit_data.beta_target_matrix = Matrix::Zero(
                          beta_open_state_prototype.residual_leaf_leaf_cofactor.rows(),
                          beta_open_state_prototype.residual_leaf_leaf_cofactor.cols());
                    }
                    root_pair_fit_data.beta_target_matrix.noalias() +=
                        beta_target_scale *
                        beta_open_state_prototype.residual_leaf_leaf_cofactor;
                    accumulate_direct_spin_coupled_root_channel_contribution(
                        direct_root_channel_layout,
                        left_root_state_index,
                        right_root_state_index,
                        beta_open_state_prototype.residual_leaf_leaf_cofactor,
                        beta_target_scale,
                        false,
                        &direct_beta_root_channel_basis);
                  }
                  breakdown.captured +=
                      signed_coefficient *
                      ((alpha_leaf.one_electron * beta_leaf.overlap +
                        beta_leaf.one_electron * alpha_leaf.overlap) *
                           alpha_root.overlap * beta_root.overlap +
                       alpha_leaf.overlap * beta_leaf.overlap *
                           (alpha_root.one_electron * beta_root.overlap +
                            beta_root.one_electron * alpha_root.overlap));
                  breakdown.mixed_bridge +=
                      signed_coefficient *
                      (alpha_full_breakdown.mixed_one_electron *
                           beta_full_breakdown.overlap +
                       beta_full_breakdown.mixed_one_electron *
                           alpha_full_breakdown.overlap);
                  breakdown.classified_total +=
                      signed_coefficient *
                      ((alpha_full_breakdown.local_one_electron +
                        alpha_full_breakdown.mixed_one_electron) *
                           beta_full_breakdown.overlap +
                       (beta_full_breakdown.local_one_electron +
                        beta_full_breakdown.mixed_one_electron) *
                           alpha_full_breakdown.overlap);
                  breakdown.block_diagonal_laplace_total +=
                      signed_coefficient *
                      (alpha_block_diagonal.one_electron * beta_block_diagonal.overlap +
                       beta_block_diagonal.one_electron * alpha_block_diagonal.overlap);
                  breakdown.zeroed_block_laplace_total +=
                      signed_coefficient *
                      (alpha_zeroed_transfer_breakdown.exact_total * beta_full_breakdown.overlap +
                       beta_zeroed_transfer_breakdown.exact_total * alpha_full_breakdown.overlap);
                  const double alpha_leaf_local_message_correction =
                      (alpha_message_full_breakdown.block_one_electron[1][1] -
                       alpha_message_zeroed_breakdown.block_one_electron[1][1]) *
                      alpha_root.overlap;
                  const double beta_leaf_local_message_correction =
                      (beta_message_full_breakdown.block_one_electron[1][1] -
                       beta_message_zeroed_breakdown.block_one_electron[1][1]) *
                      beta_root.overlap;
                  breakdown.leaf_local_message_corrected_total +=
                      signed_coefficient *
                      ((alpha_zeroed_transfer_breakdown.exact_total +
                        alpha_leaf_local_message_correction) *
                           beta_full_breakdown.overlap +
                       (beta_zeroed_transfer_breakdown.exact_total +
                        beta_leaf_local_message_correction) *
                           alpha_full_breakdown.overlap);
                  breakdown.open_state_prototype_zeroed_closed_total +=
                      signed_coefficient *
                      (alpha_open_state_prototype.zeroed_closed_total *
                           beta_full_breakdown.overlap +
                       beta_open_state_prototype.zeroed_closed_total *
                           alpha_full_breakdown.overlap);
                  breakdown.open_state_prototype_exact_cross_total +=
                      signed_coefficient *
                      (alpha_open_state_prototype.exact_cross_total *
                           beta_full_breakdown.overlap +
                       beta_open_state_prototype.exact_cross_total *
                           alpha_full_breakdown.overlap);
                  breakdown.open_state_prototype_residual_closed_total +=
                      signed_coefficient *
                      (alpha_open_state_prototype.residual_closed_total *
                           beta_full_breakdown.overlap +
                       beta_open_state_prototype.residual_closed_total *
                           alpha_full_breakdown.overlap);
                  breakdown.max_full_cross_message_to_root_rank = std::max(
                      breakdown.max_full_cross_message_to_root_rank,
                      std::max(
                          alpha_cross_rank_breakdown.message_to_root_rank,
                          beta_cross_rank_breakdown.message_to_root_rank));
                  breakdown.max_full_cross_root_to_message_rank = std::max(
                      breakdown.max_full_cross_root_to_message_rank,
                      std::max(
                          alpha_cross_rank_breakdown.root_to_message_rank,
                          beta_cross_rank_breakdown.root_to_message_rank));
                  breakdown.max_open_state_prototype_residual_message_rank = std::max(
                      breakdown.max_open_state_prototype_residual_message_rank,
                      std::max(
                          alpha_open_state_prototype.residual_message_rank,
                          beta_open_state_prototype.residual_message_rank));
                  breakdown.max_open_state_prototype_residual_leaf_leaf_rank = std::max(
                      breakdown.max_open_state_prototype_residual_leaf_leaf_rank,
                      std::max(
                          alpha_open_state_prototype.residual_leaf_leaf_rank,
                          beta_open_state_prototype.residual_leaf_leaf_rank));
                  for (int row_block = 0; row_block < 3; ++row_block) {
                    for (int col_block = 0; col_block < 3; ++col_block) {
                      breakdown.exact_three_block[xmvb::to_size(row_block)]
                                                 [xmvb::to_size(col_block)] +=
                          signed_coefficient *
                          (alpha_three_block.block_one_electron[xmvb::to_size(row_block)]
                                                               [xmvb::to_size(col_block)] *
                               beta_three_block.overlap +
                               beta_three_block.block_one_electron[xmvb::to_size(row_block)]
                                                              [xmvb::to_size(col_block)] *
                               alpha_three_block.overlap);
                      breakdown.zeroed_block_three_block[xmvb::to_size(row_block)]
                                                        [xmvb::to_size(col_block)] +=
                          signed_coefficient *
                          (alpha_zeroed_transfer_breakdown.block_one_electron[xmvb::to_size(row_block)]
                                                                             [xmvb::to_size(col_block)] *
                               beta_full_breakdown.overlap +
                           beta_zeroed_transfer_breakdown.block_one_electron[xmvb::to_size(row_block)]
                                                                            [xmvb::to_size(col_block)] *
                               alpha_full_breakdown.overlap);
                      breakdown.open_state_prototype_residual_closed_blocks
                               [xmvb::to_size(row_block)]
                               [xmvb::to_size(col_block)] +=
                          signed_coefficient *
                          (alpha_open_state_prototype.residual_closed_block_one_electron
                                   [xmvb::to_size(row_block)]
                                   [xmvb::to_size(col_block)] *
                               beta_full_breakdown.overlap +
                           beta_open_state_prototype.residual_closed_block_one_electron
                                   [xmvb::to_size(row_block)]
                                   [xmvb::to_size(col_block)] *
                               alpha_full_breakdown.overlap);
                    }
                  }
                  for (int transfer_bin = 0; transfer_bin < 4; ++transfer_bin) {
                    breakdown.exact_transfer_in_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_transfer_breakdown.transfer_in_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_transfer_breakdown.transfer_in_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                    breakdown.exact_transfer_one_exit_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_transfer_breakdown.transfer_one_exit_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_transfer_breakdown.transfer_one_exit_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                    breakdown.exact_transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_transfer_breakdown.transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_transfer_breakdown.transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                  }
                  for (int side_case = 0; side_case < 4; ++side_case) {
                    breakdown.exact_side_case_total[xmvb::to_size(side_case)] +=
                        signed_coefficient *
                        (alpha_transfer_breakdown.side_case_total[xmvb::to_size(side_case)] *
                             beta_full_breakdown.overlap +
                         beta_transfer_breakdown.side_case_total[xmvb::to_size(side_case)] *
                             alpha_full_breakdown.overlap);
                  }
                  for (int transfer_bin = 0; transfer_bin < 4; ++transfer_bin) {
                    breakdown.zeroed_block_transfer_in_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_zeroed_transfer_breakdown.transfer_in_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_zeroed_transfer_breakdown.transfer_in_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                    breakdown.zeroed_block_transfer_one_exit_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_zeroed_transfer_breakdown.transfer_one_exit_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_zeroed_transfer_breakdown.transfer_one_exit_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                    breakdown.zeroed_block_transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] +=
                        signed_coefficient *
                        (alpha_zeroed_transfer_breakdown.transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] *
                             beta_full_breakdown.overlap +
                         beta_zeroed_transfer_breakdown.transfer_zero_exit_hist[xmvb::to_size(transfer_bin)] *
                             alpha_full_breakdown.overlap);
                  }
                  for (int side_case = 0; side_case < 4; ++side_case) {
                    breakdown.zeroed_block_side_case_total[xmvb::to_size(side_case)] +=
                        signed_coefficient *
                        (alpha_zeroed_transfer_breakdown.side_case_total[xmvb::to_size(side_case)] *
                             beta_full_breakdown.overlap +
                         beta_zeroed_transfer_breakdown.side_case_total[xmvb::to_size(side_case)] *
                             alpha_full_breakdown.overlap);
                  }
                }
              }
            }
          }
        }
      }
      closed_correction_fit_equations.push_back(std::move(root_pair_fit_data));
    }
  }

  {
    std::set<SpinMaskKey> alpha_mask_keys;
    std::set<SpinMaskKey> beta_mask_keys;
    for (const auto& equation : closed_correction_fit_equations) {
      for (const auto& [mask_key, value] : equation.alpha_features) {
        if (std::abs(value) > 1.0e-15) {
          alpha_mask_keys.insert(mask_key);
        }
      }
      for (const auto& [mask_key, value] : equation.beta_features) {
        if (std::abs(value) > 1.0e-15) {
          beta_mask_keys.insert(mask_key);
        }
      }
    }

    std::map<SpinMaskKey, int> alpha_mask_index;
    std::map<SpinMaskKey, int> beta_mask_index;
    int next_index = 0;
    for (const auto& mask_key : alpha_mask_keys) {
      alpha_mask_index.emplace(mask_key, next_index++);
    }
    for (const auto& mask_key : beta_mask_keys) {
      beta_mask_index.emplace(mask_key, next_index++);
    }

    const int equation_count = static_cast<int>(closed_correction_fit_equations.size());
    const int unknown_count = next_index;
    breakdown.scalar_mask_closed_correction_fit_equation_count = equation_count;
    breakdown.scalar_mask_closed_correction_fit_unknown_count = unknown_count;
    double exact_total_sum = 0.0;
    for (const auto& equation : closed_correction_fit_equations) {
      exact_total_sum += equation.exact_total;
    }
    if (equation_count > 0 && unknown_count > 0) {
      Matrix design = Matrix::Zero(equation_count, unknown_count);
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(equation_count);
      for (int equation_index = 0; equation_index < equation_count; ++equation_index) {
        const auto& equation = closed_correction_fit_equations[xmvb::to_size(equation_index)];
        rhs(equation_index) = equation.exact_total - equation.captured_total;
        for (const auto& [mask_key, value] : equation.alpha_features) {
          const auto iterator = alpha_mask_index.find(mask_key);
          if (iterator != alpha_mask_index.end()) {
            design(equation_index, iterator->second) = value;
          }
        }
        for (const auto& [mask_key, value] : equation.beta_features) {
          const auto iterator = beta_mask_index.find(mask_key);
          if (iterator != beta_mask_index.end()) {
            design(equation_index, iterator->second) = value;
          }
        }
      }

      const Eigen::VectorXd solution =
          design.colPivHouseholderQr().solve(rhs);
      const Eigen::VectorXd predicted_delta = design * solution;
      const Eigen::VectorXd residual = predicted_delta - rhs;
      breakdown.scalar_mask_closed_correction_fit_alpha_solution.clear();
      breakdown.scalar_mask_closed_correction_fit_beta_solution.clear();
      for (const auto& [mask_key, solution_index] : alpha_mask_index) {
        breakdown.scalar_mask_closed_correction_fit_alpha_solution.emplace_back(
            mask_key.first,
            mask_key.second,
            solution(solution_index));
      }
      for (const auto& [mask_key, solution_index] : beta_mask_index) {
        breakdown.scalar_mask_closed_correction_fit_beta_solution.emplace_back(
            mask_key.first,
            mask_key.second,
            solution(solution_index));
      }
      for (int equation_index = 0; equation_index < equation_count; ++equation_index) {
        breakdown.scalar_mask_closed_correction_fit_max_root_pair_abs_residual = std::max(
            breakdown.scalar_mask_closed_correction_fit_max_root_pair_abs_residual,
            std::abs(residual(equation_index)));
      }
      breakdown.scalar_mask_closed_correction_fit_total =
          breakdown.captured + predicted_delta.sum();
      breakdown.scalar_mask_closed_correction_fit_abs_error =
          std::abs(breakdown.scalar_mask_closed_correction_fit_total - exact_total_sum);
    } else {
      breakdown.scalar_mask_closed_correction_fit_total = breakdown.captured;
      breakdown.scalar_mask_closed_correction_fit_abs_error =
          std::abs(breakdown.scalar_mask_closed_correction_fit_total - exact_total_sum);
    }
  }

  {
    const auto alpha_current_metrics = fit_spin_mask_matrix_payloads(
        closed_correction_fit_equations,
        true,
        MatrixMaskFeatureModel::FactorizedOppositeSpin);
    const auto beta_current_metrics = fit_spin_mask_matrix_payloads(
        closed_correction_fit_equations,
        false,
        MatrixMaskFeatureModel::FactorizedOppositeSpin);
    const auto alpha_full_metrics = fit_spin_mask_matrix_payloads(
        closed_correction_fit_equations,
        true,
        MatrixMaskFeatureModel::FullOppositeSpin);
    const auto beta_full_metrics = fit_spin_mask_matrix_payloads(
        closed_correction_fit_equations,
        false,
        MatrixMaskFeatureModel::FullOppositeSpin);
    breakdown.alpha_matrix_current_feature_fit_sum_frobenius_abs_error =
        alpha_current_metrics.total_frobenius_abs_error;
    breakdown.alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual =
        alpha_current_metrics.max_root_pair_frobenius_residual;
    breakdown.beta_matrix_current_feature_fit_sum_frobenius_abs_error =
        beta_current_metrics.total_frobenius_abs_error;
    breakdown.beta_matrix_current_feature_fit_max_root_pair_frobenius_residual =
        beta_current_metrics.max_root_pair_frobenius_residual;
    breakdown.alpha_matrix_full_feature_fit_sum_frobenius_abs_error =
        alpha_full_metrics.total_frobenius_abs_error;
    breakdown.alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual =
        alpha_full_metrics.max_root_pair_frobenius_residual;
    breakdown.beta_matrix_full_feature_fit_sum_frobenius_abs_error =
        beta_full_metrics.total_frobenius_abs_error;
    breakdown.beta_matrix_full_feature_fit_max_root_pair_frobenius_residual =
        beta_full_metrics.max_root_pair_frobenius_residual;
    breakdown.matrix_mask_closed_correction_fit_equation_count =
        alpha_current_metrics.equation_count;
    breakdown.alpha_matrix_mask_closed_correction_fit_unknown_count =
        alpha_current_metrics.unknown_count;
    breakdown.beta_matrix_mask_closed_correction_fit_unknown_count =
        beta_current_metrics.unknown_count;
    breakdown.alpha_matrix_target_span_rank =
        alpha_current_metrics.target_span_rank;
    breakdown.beta_matrix_target_span_rank =
        beta_current_metrics.target_span_rank;
  }

  breakdown.root_pair_target_matrix_dumps.clear();
  breakdown.root_pair_target_matrix_dumps.reserve(closed_correction_fit_equations.size());
  for (const auto& equation : closed_correction_fit_equations) {
    RootPairTargetMatrixDumpEntry entry;
    entry.left_root_alpha_occ = equation.left_root_alpha_occ;
    entry.left_root_beta_occ = equation.left_root_beta_occ;
    entry.right_root_alpha_occ = equation.right_root_alpha_occ;
    entry.right_root_beta_occ = equation.right_root_beta_occ;
    entry.exact_total = equation.exact_total;
    entry.captured_total = equation.captured_total;
    entry.alpha_target_matrix = equation.alpha_target_matrix;
    entry.beta_target_matrix = equation.beta_target_matrix;
    entry.alpha_target_rank =
        (entry.alpha_target_matrix.size() == 0)
            ? 0
            : Eigen::FullPivLU<Matrix>(entry.alpha_target_matrix).rank();
    entry.beta_target_rank =
        (entry.beta_target_matrix.size() == 0)
            ? 0
            : Eigen::FullPivLU<Matrix>(entry.beta_target_matrix).rank();
    entry.alpha_features = equation.alpha_features;
    entry.beta_features = equation.beta_features;
    entry.alpha_full_features = equation.alpha_full_features;
    entry.beta_full_features = equation.beta_full_features;
    breakdown.root_pair_target_matrix_dumps.push_back(std::move(entry));
  }

  accumulate_root_channel_covariance_metrics(
      breakdown.root_pair_target_matrix_dumps,
      &breakdown);
  accumulate_spin_coupled_root_channel_metrics(
      breakdown.root_pair_target_matrix_dumps,
      &breakdown);
  const auto aggregated_root_state_target_matrix_dumps =
      aggregate_root_pair_target_matrix_dumps_by_state(
          breakdown.root_pair_target_matrix_dumps);
  accumulate_direct_spin_coupled_root_channel_metrics(
      aggregated_root_state_target_matrix_dumps,
      direct_root_channel_layout,
      direct_alpha_root_channel_basis,
      direct_beta_root_channel_basis,
      &breakdown);

  return breakdown;
}

CollapsedStarPairStats evaluate_component_ordered_collapsed_star_pair(
    double exact_overlap,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const std::vector<int>& ordered_support_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    ExactSeparatorStateCollector* state_collector) {
  // This routine evaluates the same exact star-separated overlap as the
  // explicit local-term recurrence above, but in the component-ordered support
  // basis and with leaf messages already aggregated onto separator masks. The
  // remaining root merge therefore depends only on mask compatibility,
  // root-remainder minors, and a block-level canonicalization parity.
  CollapsedStarPairStats stats;
  stats.exact_overlap = exact_overlap;
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (ordered_support_orbitals.size() != xmvb::to_size(support_size)) {
    throw std::invalid_argument("ordered_support_orbitals size must match support_size");
  }

  const auto& root_component = ordered_components.front();
  const auto root_left_pairs_global =
      remap_pairs_to_global_labels(root_component.left_pairs, ordered_support_orbitals);
  const auto root_right_pairs_global =
      remap_pairs_to_global_labels(root_component.right_pairs, ordered_support_orbitals);
  const int n_leaves = static_cast<int>(ordered_components.size()) - 1;
  if (n_leaves == 0) {
    stats.collapsed_overlap = exact_overlap;
    if (state_collector != nullptr) {
      for (const auto& left_root_term : root_component.left_orientation_terms) {
        for (const auto& right_root_term : root_component.right_orientation_terms) {
          ExactSeparatorStateKey state_key;
          state_key.root_left_pairs = root_left_pairs_global;
          state_key.root_right_pairs = root_right_pairs_global;
          state_key.left_root_alpha_occ =
              remap_occ_to_global_labels(
                  left_root_term.alpha_occ,
                  ordered_support_orbitals);
          state_key.left_root_beta_occ =
              remap_occ_to_global_labels(
                  left_root_term.beta_occ,
                  ordered_support_orbitals);
          state_key.right_root_alpha_occ =
              remap_occ_to_global_labels(
                  right_root_term.alpha_occ,
                  ordered_support_orbitals);
          state_key.right_root_beta_occ =
              remap_occ_to_global_labels(
                  right_root_term.beta_occ,
                  ordered_support_orbitals);
          state_collector->unique_exact_separator_states.insert(std::move(state_key));
        }
      }
    }
    return stats;
  }

  std::vector<int> left_leaf_alpha_sizes;
  std::vector<int> left_leaf_beta_sizes;
  std::vector<int> right_leaf_alpha_sizes;
  std::vector<int> right_leaf_beta_sizes;
  left_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  left_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_alpha_sizes.reserve(xmvb::to_size(n_leaves));
  right_leaf_beta_sizes.reserve(xmvb::to_size(n_leaves));
  for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
    const auto& leaf_component = ordered_components[xmvb::to_size(leaf_index + 1)];
    left_leaf_alpha_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().alpha_occ.size()));
    left_leaf_beta_sizes.push_back(
        leaf_component.left_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.left_orientation_terms.front().beta_occ.size()));
    right_leaf_alpha_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().alpha_occ.size()));
    right_leaf_beta_sizes.push_back(
        leaf_component.right_orientation_terms.empty()
            ? 0
            : static_cast<int>(leaf_component.right_orientation_terms.front().beta_occ.size()));
  }

  for (const auto& left_root_term : root_component.left_orientation_terms) {
    for (const auto& right_root_term : root_component.right_orientation_terms) {
      const auto left_root_alpha_occ_global =
          remap_occ_to_global_labels(
              left_root_term.alpha_occ,
              ordered_support_orbitals);
      const auto left_root_beta_occ_global =
          remap_occ_to_global_labels(
              left_root_term.beta_occ,
              ordered_support_orbitals);
      const auto right_root_alpha_occ_global =
          remap_occ_to_global_labels(
              right_root_term.alpha_occ,
              ordered_support_orbitals);
      const auto right_root_beta_occ_global =
          remap_occ_to_global_labels(
              right_root_term.beta_occ,
              ordered_support_orbitals);
      std::map<
          std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
          std::pair<double, double>>
          root_remainder_cache;
      std::vector<std::vector<CollapsedLeafMessageEntry>> collapsed_leaf_messages(
          xmvb::to_size(n_leaves));
      for (int leaf_index = 0; leaf_index < n_leaves; ++leaf_index) {
        const auto& leaf_component = ordered_components[xmvb::to_size(leaf_index + 1)];
        collapsed_leaf_messages[xmvb::to_size(leaf_index)] =
            build_collapsed_leaf_messages_component_ordered(
                left_root_term,
                right_root_term,
                leaf_component,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &stats,
                &stats.subdeterminant_evaluations,
                &stats.hypercube_assignment_count);
        stats.collapsed_leaf_state_count +=
            static_cast<std::uint64_t>(
                collapsed_leaf_messages[xmvb::to_size(leaf_index)].size());
        if (state_collector != nullptr) {
          const auto leaf_left_pairs_global =
              remap_pairs_to_global_labels(
                  leaf_component.left_pairs,
                  ordered_support_orbitals);
          const auto leaf_right_pairs_global =
              remap_pairs_to_global_labels(
                  leaf_component.right_pairs,
                  ordered_support_orbitals);
          if (!collapsed_leaf_messages[xmvb::to_size(leaf_index)].empty()) {
            ExactSeparatorStateCollector::ExactLeafMessageBundleKey bundle_key;
            bundle_key.leaf_left_pairs = leaf_left_pairs_global;
            bundle_key.leaf_right_pairs = leaf_right_pairs_global;
            bundle_key.left_root_alpha_occ = left_root_alpha_occ_global;
            bundle_key.left_root_beta_occ = left_root_beta_occ_global;
            bundle_key.right_root_alpha_occ = right_root_alpha_occ_global;
            bundle_key.right_root_beta_occ = right_root_beta_occ_global;
            state_collector->unique_leaf_message_bundles.insert(std::move(bundle_key));
          }
          for (const auto& entry :
               collapsed_leaf_messages[xmvb::to_size(leaf_index)]) {
            ExactSeparatorStateCollector::ExactLeafMessageStateKey leaf_state_key;
            leaf_state_key.leaf_left_pairs = leaf_left_pairs_global;
            leaf_state_key.leaf_right_pairs = leaf_right_pairs_global;
            leaf_state_key.left_root_alpha_occ = left_root_alpha_occ_global;
            leaf_state_key.left_root_beta_occ = left_root_beta_occ_global;
            leaf_state_key.right_root_alpha_occ = right_root_alpha_occ_global;
            leaf_state_key.right_root_beta_occ = right_root_beta_occ_global;
            leaf_state_key.alpha_row_mask = entry.alpha_row_mask;
            leaf_state_key.alpha_col_mask = entry.alpha_col_mask;
            leaf_state_key.beta_row_mask = entry.beta_row_mask;
            leaf_state_key.beta_col_mask = entry.beta_col_mask;
            state_collector->unique_leaf_message_states.insert(std::move(leaf_state_key));
          }
        }
      }

      const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
      const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
      const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
      const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());

      const std::uint32_t alpha_row_full_mask =
          (n_alpha_root_rows == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_rows) - 1U);
      const std::uint32_t alpha_col_full_mask =
          (n_alpha_root_cols == 0) ? 0U
                                   : ((static_cast<std::uint32_t>(1) << n_alpha_root_cols) - 1U);
      const std::uint32_t beta_row_full_mask =
          (n_beta_root_rows == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_rows) - 1U);
      const std::uint32_t beta_col_full_mask =
          (n_beta_root_cols == 0) ? 0U
                                  : ((static_cast<std::uint32_t>(1) << n_beta_root_cols) - 1U);

      std::vector<std::uint32_t> selected_alpha_row_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_beta_row_masks(xmvb::to_size(n_leaves), 0U);
      std::vector<std::uint32_t> selected_beta_col_masks(xmvb::to_size(n_leaves), 0U);

      const auto accumulate_collapsed_messages =
          [&](const auto& self,
              int leaf_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double state_value) -> void {
            if (std::abs(state_value) <= 1.0e-15) {
              return;
            }
            if (state_collector != nullptr) {
              ExactSeparatorStateCollector::ExactMergeStateKey merge_state_key;
              merge_state_key.left_root_alpha_occ = left_root_alpha_occ_global;
              merge_state_key.left_root_beta_occ = left_root_beta_occ_global;
              merge_state_key.right_root_alpha_occ = right_root_alpha_occ_global;
              merge_state_key.right_root_beta_occ = right_root_beta_occ_global;
              merge_state_key.leaf_index = leaf_index;
              merge_state_key.used_alpha_row_mask = used_alpha_row_mask;
              merge_state_key.used_alpha_col_mask = used_alpha_col_mask;
              merge_state_key.used_beta_row_mask = used_beta_row_mask;
              merge_state_key.used_beta_col_mask = used_beta_col_mask;
              state_collector->unique_merge_states.insert(std::move(merge_state_key));
            }
            if (leaf_index == n_leaves) {
              const std::uint32_t alpha_row_remainder =
                  alpha_row_full_mask ^ used_alpha_row_mask;
              const std::uint32_t alpha_col_remainder =
                  alpha_col_full_mask ^ used_alpha_col_mask;
              const std::uint32_t beta_row_remainder =
                  beta_row_full_mask ^ used_beta_row_mask;
              const std::uint32_t beta_col_remainder =
                  beta_col_full_mask ^ used_beta_col_mask;

              const auto alpha_root_rows =
                  select_occ_by_mask(right_root_term.alpha_occ, alpha_row_remainder);
              const auto alpha_root_cols =
                  select_occ_by_mask(left_root_term.alpha_occ, alpha_col_remainder);
              const auto beta_root_rows =
                  select_occ_by_mask(right_root_term.beta_occ, beta_row_remainder);
              const auto beta_root_cols =
                  select_occ_by_mask(left_root_term.beta_occ, beta_col_remainder);
              const auto cache_key = std::make_tuple(
                  used_alpha_row_mask,
                  used_alpha_col_mask,
                  used_beta_row_mask,
                  used_beta_col_mask);
              auto cache_iterator = root_remainder_cache.find(cache_key);
              double alpha_root_determinant = 0.0;
              double beta_root_determinant = 0.0;
              if (cache_iterator == root_remainder_cache.end()) {
                alpha_root_determinant = determinant_for_occ_lists(
                    alpha_root_cols,
                    alpha_root_rows,
                    support_overlap_storage,
                    support_size,
                    overlap_resolver,
                    nullptr);
                ++stats.subdeterminant_evaluations;
                if (std::abs(alpha_root_determinant) > 1.0e-15) {
                  beta_root_determinant = determinant_for_occ_lists(
                      beta_root_cols,
                      beta_root_rows,
                      support_overlap_storage,
                      support_size,
                      overlap_resolver,
                      nullptr);
                  ++stats.subdeterminant_evaluations;
                }
                cache_iterator = root_remainder_cache.emplace(
                    cache_key,
                    std::make_pair(alpha_root_determinant, beta_root_determinant)).first;
              }
              alpha_root_determinant = cache_iterator->second.first;
              beta_root_determinant = cache_iterator->second.second;
              if (std::abs(alpha_root_determinant) <= 1.0e-15 ||
                  std::abs(beta_root_determinant) <= 1.0e-15) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  n_alpha_root_rows,
                  selected_alpha_row_masks,
                  right_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  n_alpha_root_cols,
                  selected_alpha_col_masks,
                  left_leaf_alpha_sizes);
              parity ^= component_ordered_block_parity(
                  n_beta_root_rows,
                  selected_beta_row_masks,
                  right_leaf_beta_sizes);
              parity ^= component_ordered_block_parity(
                  n_beta_root_cols,
                  selected_beta_col_masks,
                  left_leaf_beta_sizes);

              stats.collapsed_overlap +=
                  left_root_term.coefficient * right_root_term.coefficient *
                  parity_sign(parity) * state_value *
                  alpha_root_determinant * beta_root_determinant;
              if (state_collector != nullptr) {
                // This key is the strongest exact terminal state that the
                // current collapsed star recurrence reaches without falling
                // back to individual determinant-pair identities: root
                // component identity, root determinant terms, and one
                // aggregated separator mask tuple per leaf component.
                ExactSeparatorStateKey state_key;
                state_key.root_left_pairs = root_left_pairs_global;
                state_key.root_right_pairs = root_right_pairs_global;
                state_key.left_root_alpha_occ = left_root_alpha_occ_global;
                state_key.left_root_beta_occ = left_root_beta_occ_global;
                state_key.right_root_alpha_occ = right_root_alpha_occ_global;
                state_key.right_root_beta_occ = right_root_beta_occ_global;
                state_key.leaf_states.reserve(xmvb::to_size(n_leaves));
                for (int selected_leaf_index = 0;
                     selected_leaf_index < n_leaves;
                     ++selected_leaf_index) {
                  const auto& leaf_component =
                      ordered_components[xmvb::to_size(selected_leaf_index + 1)];
                  ExactSeparatorLeafStateKey leaf_key;
                  leaf_key.left_pairs =
                      remap_pairs_to_global_labels(
                          leaf_component.left_pairs,
                          ordered_support_orbitals);
                  leaf_key.right_pairs =
                      remap_pairs_to_global_labels(
                          leaf_component.right_pairs,
                          ordered_support_orbitals);
                  leaf_key.alpha_row_mask =
                      selected_alpha_row_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.alpha_col_mask =
                      selected_alpha_col_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.beta_row_mask =
                      selected_beta_row_masks[xmvb::to_size(selected_leaf_index)];
                  leaf_key.beta_col_mask =
                      selected_beta_col_masks[xmvb::to_size(selected_leaf_index)];
                  state_key.leaf_states.push_back(std::move(leaf_key));
                }
                state_collector->unique_exact_separator_states.insert(std::move(state_key));
              }
              return;
            }

            for (const auto& entry :
                 collapsed_leaf_messages[xmvb::to_size(leaf_index)]) {
              if ((used_alpha_row_mask & entry.alpha_row_mask) != 0U ||
                  (used_alpha_col_mask & entry.alpha_col_mask) != 0U ||
                  (used_beta_row_mask & entry.beta_row_mask) != 0U ||
                  (used_beta_col_mask & entry.beta_col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] =
                  entry.alpha_row_mask;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] =
                  entry.alpha_col_mask;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] =
                  entry.beta_row_mask;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] =
                  entry.beta_col_mask;

              ++stats.dp_transition_count;
              self(
                  self,
                  leaf_index + 1,
                  used_alpha_row_mask | entry.alpha_row_mask,
                  used_alpha_col_mask | entry.alpha_col_mask,
                  used_beta_row_mask | entry.beta_row_mask,
                  used_beta_col_mask | entry.beta_col_mask,
                  state_value * entry.value);

              selected_alpha_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_alpha_col_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_row_masks[xmvb::to_size(leaf_index)] = 0U;
              selected_beta_col_masks[xmvb::to_size(leaf_index)] = 0U;
            }
          };
      accumulate_collapsed_messages(
          accumulate_collapsed_messages,
          0,
          0U,
          0U,
          0U,
          0U,
          1.0);
    }
  }

  stats.absolute_error = std::abs(stats.exact_overlap - stats.collapsed_overlap);
  return stats;
}

Matrix build_full_active_overlap_matrix(
    const std::vector<double>& active_overlap_storage,
    int n_active_orbitals) {
  Matrix overlap(n_active_orbitals, n_active_orbitals);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      overlap(row, column) =
          active_overlap_storage[xmvb::to_size(column) *
                                     xmvb::to_size(n_active_orbitals) +
                                 row];
    }
  }
  return overlap;
}

std::string format_ratio(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0) {
    return "0";
  }
  return std::to_string(numerator) + "/" + std::to_string(denominator);
}

std::uint64_t integer_power(std::uint64_t base, int exponent) {
  if (exponent < 0) {
    throw std::invalid_argument("exponent must be non-negative");
  }
  std::uint64_t result = 1;
  for (int power = 0; power < exponent; ++power) {
    result *= base;
  }
  return result;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

Matrix build_support_submatrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& active_matrix_storage,
    int n_active_orbitals) {
  // `active_matrix_storage` uses the repository-wide column-major convention
  // `storage[column * n_active_orbitals + row]`. This helper extracts the
  // square support-local block for either the overlap metric or the active
  // one-electron Hamiltonian without changing that convention.
  const int support_size = static_cast<int>(support_orbitals.size());
  Matrix support_matrix(support_size, support_size);
  for (int row = 0; row < support_size; ++row) {
    for (int column = 0; column < support_size; ++column) {
      support_matrix(row, column) =
          active_matrix_storage[xmvb::to_size(
                                    support_orbitals[xmvb::to_size(column)]) *
                                    xmvb::to_size(n_active_orbitals) +
                                support_orbitals[xmvb::to_size(row)]];
    }
  }
  return support_matrix;
}

StructurePairExactValues compute_exact_structure_pair_values(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    const std::vector<double>& active_overlap_matrix,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    ExactWorkCollector* work_collector) {
  // This is the determinant-pair reference used to validate the separator
  // recurrence. Each raw-VB structure pair is expanded on the legacy
  // determinant basis, and every determinant pair is evaluated by the same
  // exact C++ kernel used elsewhere in the codebase.
  xmvb::vb::FullDeterminantPairEvaluator pair_evaluator;
  record_reference_determinant_work(left_terms, right_terms, work_collector);

  StructurePairExactValues values;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const auto pair_value = pair_evaluator.evaluate(
          left_term.alpha_occ,
          right_term.alpha_occ,
          left_term.beta_occ,
          right_term.beta_occ,
          active_overlap_matrix,
          active_one_electron_matrix,
          n_active_orbitals,
          packed_active_two_electron_integrals);
      const double coefficient =
          left_term.coefficient * right_term.coefficient;
      values.overlap += coefficient * pair_value.overlap_determinant;
      values.one_electron += coefficient * pair_value.one_electron_hamiltonian;
    }
  }
  return values;
}

SameSpinDeterminantOneElectronValues
evaluate_same_spin_determinant_overlap_and_one_electron(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  // This helper is the fair exact reference for the benchmarked quantity in
  // this tool: determinant overlap plus the same-spin one-electron matrix
  // element. It intentionally skips the two-electron Hamiltonian work that the
  // full determinant evaluator performs, so the benchmark matches the actual
  // recurrence target instead of timing unrelated ERI contractions.
  if (left_occ.size() != right_occ.size()) {
    throw std::invalid_argument("left and right occupied lists must have the same size");
  }

  SameSpinDeterminantOneElectronValues values;
  if (left_occ.empty()) {
    values.overlap = 1.0;
    return values;
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      support_overlap_storage,
      support_size);
  const auto overlap_result = overlap_resolver.resolve(
      overlap_submatrix,
      static_cast<int>(left_occ.size()));
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);
  const xmvb::vb::ConstMatrixMap support_one_electron_matrix(
      support_one_electron_storage.data(),
      support_size,
      support_size);

  values.overlap = overlap_result.overlap_determinant;
  for (int left_column = 0; left_column < static_cast<int>(left_occ.size()); ++left_column) {
    const int left_orbital = left_occ[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(right_occ.size()); ++right_row) {
      const int right_orbital = right_occ[xmvb::to_size(right_row)];
      values.one_electron +=
          support_one_electron_matrix(right_orbital, left_orbital) *
          cofactor_1st(right_row, left_column);
    }
  }
  return values;
}

StructurePairExactValues compute_exact_structure_pair_values_one_electron_only(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  // Build one raw-structure matrix entry from the legacy determinant expansion
  // without touching the two-electron Hamiltonian. This is the exact baseline
  // used by the full matrix-build benchmark.
  StructurePairExactValues values;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const auto alpha_values = evaluate_same_spin_determinant_overlap_and_one_electron(
          left_term.alpha_occ,
          right_term.alpha_occ,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          overlap_resolver);
      const auto beta_values = evaluate_same_spin_determinant_overlap_and_one_electron(
          left_term.beta_occ,
          right_term.beta_occ,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          overlap_resolver);
      const double coefficient =
          left_term.coefficient * right_term.coefficient;
      values.overlap +=
          coefficient * alpha_values.overlap * beta_values.overlap;
      values.one_electron +=
          coefficient *
          (alpha_values.one_electron * beta_values.overlap +
           beta_values.one_electron * alpha_values.overlap);
    }
  }
  return values;
}

int count_covalent_pairs(const std::vector<OrbitalPair>& pairs) {
  int count = 0;
  for (const auto& [left_orbital, right_orbital] : pairs) {
    if (left_orbital != right_orbital) {
      ++count;
    }
  }
  return count;
}

Matrix build_signed_selector_matrix(
    const std::vector<OrbitalPair>& pairs,
    std::uint64_t sign_mask,
    bool is_alpha,
    int support_size) {
  Matrix selector = Matrix::Zero(static_cast<int>(pairs.size()), support_size);
  int covalent_index = 0;
  for (int pair_index = 0; pair_index < static_cast<int>(pairs.size()); ++pair_index) {
    const auto& [left_orbital, right_orbital] = pairs[xmvb::to_size(pair_index)];
    if (left_orbital == right_orbital) {
      selector(pair_index, left_orbital) = 1.0;
      continue;
    }
    const double sign =
        ((sign_mask & (static_cast<std::uint64_t>(1) << covalent_index)) == 0U) ? -1.0 : 1.0;
    if (is_alpha) {
      selector(pair_index, left_orbital) = 1.0;
      selector(pair_index, right_orbital) = sign;
    } else {
      selector(pair_index, right_orbital) = 1.0;
      selector(pair_index, left_orbital) = sign;
    }
    ++covalent_index;
  }
  return selector;
}

Matrix build_basis_selector_matrix(
    const std::vector<int>& occupied_orbitals,
    int support_size) {
  Matrix selector = Matrix::Zero(static_cast<int>(occupied_orbitals.size()), support_size);
  for (int selector_row = 0;
       selector_row < static_cast<int>(occupied_orbitals.size());
       ++selector_row) {
    selector(selector_row, occupied_orbitals[xmvb::to_size(selector_row)]) = 1.0;
  }
  return selector;
}

Matrix select_selector_rows_by_mask(
    const Matrix& full_selector,
    std::uint32_t mask) {
  Matrix selected_selector(popcount(mask), full_selector.cols());
  int selected_row = 0;
  for (int row_index = 0; row_index < full_selector.rows(); ++row_index) {
    if ((mask & (static_cast<std::uint32_t>(1) << row_index)) == 0U) {
      continue;
    }
    selected_selector.row(selected_row) = full_selector.row(row_index);
    ++selected_row;
  }
  return selected_selector;
}

Matrix build_transformed_block_matrix(
    const Matrix& left_selector,
    const Matrix& right_selector,
    const Matrix& support_matrix) {
  // `left_selector` and `right_selector` are row-selector matrices whose rows
  // span the left and right occupied subspaces for one spin channel. The
  // transformed block therefore has the determinant-kernel convention
  // `rows = right occupied rows`, `columns = left occupied columns`.
  return right_selector * support_matrix * left_selector.transpose();
}

SpinOverlapOneElectronValue evaluate_dense_spin_block(
    const Matrix& overlap_block,
    const Matrix& one_electron_block,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This routine is the exact spin-determinant analogue of the determinant
  // pair one-electron kernel. `overlap_block` is the dense overlap matrix for
  // the current spin subproblem after any separator-imposed zeroing. The
  // matching `one_electron_block` uses the same row/column ordering but is not
  // zeroed, because the separator modifies only the overlap minors.
  if (overlap_block.rows() != overlap_block.cols() ||
      one_electron_block.rows() != overlap_block.rows() ||
      one_electron_block.cols() != overlap_block.cols()) {
    throw std::invalid_argument(
        "evaluate_dense_spin_block requires square, dimension-matched matrices");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  ++(*subdeterminant_evaluations);
  SpinOverlapOneElectronValue result;
  const auto overlap_result = overlap_resolver.resolve(
      flatten_column_major_matrix(overlap_block),
      overlap_block.rows());
  result.overlap = overlap_result.overlap_determinant;
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);
  result.one_electron =
      (one_electron_block.cwiseProduct(cofactor_1st)).sum();
  return result;
}

SpinOverlapOneElectronValue evaluate_occ_list_spin_values(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (left_occ.size() != right_occ.size()) {
    return {};
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      left_occ,
      right_occ,
      one_electron_storage,
      n_orbitals);

  const Eigen::Map<const Matrix> overlap_block(
      overlap_submatrix.data(),
      static_cast<int>(left_occ.size()),
      static_cast<int>(left_occ.size()));
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      static_cast<int>(left_occ.size()),
      static_cast<int>(left_occ.size()));
  return evaluate_dense_spin_block(
      overlap_block,
      one_electron_block,
      overlap_resolver,
      subdeterminant_evaluations);
}

SpinOverlapOneElectronValue evaluate_block_diagonalized_zeroed_selected_root_spin_values(
    const std::vector<int>& left_selected_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& left_root_remainder_occ,
    const std::vector<int>& right_selected_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<int>& right_root_remainder_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This helper evaluates the exact one-electron kernel on the Laplace
  // factorized overlap object
  //
  //   diag(B_zeroed(selected-root, leaf), D_root-remainder),
  //
  // while keeping the full one-electron matrix in the common occupied-orbital
  // order [selected root | leaf | root remainder]. If this matches the exact
  // determinant-pair one-electron value after mask summation, then the correct
  // separator recurrence should be built from zeroed-block interface states,
  // not from the cofactor matrix of the fully reassembled overlap block.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const std::size_t total_left_size =
      left_selected_root_occ.size() + left_leaf_occ.size() + left_root_remainder_occ.size();
  const std::size_t total_right_size =
      right_selected_root_occ.size() + right_leaf_occ.size() + right_root_remainder_occ.size();
  if (total_left_size != total_right_size) {
    return {};
  }

  std::vector<int> ordered_left_occ;
  std::vector<int> ordered_right_occ;
  ordered_left_occ.reserve(total_left_size);
  ordered_right_occ.reserve(total_right_size);
  ordered_left_occ.insert(
      ordered_left_occ.end(),
      left_selected_root_occ.begin(),
      left_selected_root_occ.end());
  ordered_left_occ.insert(
      ordered_left_occ.end(),
      left_leaf_occ.begin(),
      left_leaf_occ.end());
  ordered_left_occ.insert(
      ordered_left_occ.end(),
      left_root_remainder_occ.begin(),
      left_root_remainder_occ.end());
  ordered_right_occ.insert(
      ordered_right_occ.end(),
      right_selected_root_occ.begin(),
      right_selected_root_occ.end());
  ordered_right_occ.insert(
      ordered_right_occ.end(),
      right_leaf_occ.begin(),
      right_leaf_occ.end());
  ordered_right_occ.insert(
      ordered_right_occ.end(),
      right_root_remainder_occ.begin(),
      right_root_remainder_occ.end());

  const int message_size =
      static_cast<int>(right_selected_root_occ.size() + right_leaf_occ.size());
  const int root_remainder_size = static_cast<int>(right_root_remainder_occ.size());
  const int dimension = message_size + root_remainder_size;
  if (dimension == 0) {
    SpinOverlapOneElectronValue result;
    result.overlap = 1.0;
    return result;
  }

  Matrix block_diagonal_overlap = Matrix::Zero(dimension, dimension);
  if (message_size > 0) {
    auto message_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
        std::vector<int>(ordered_left_occ.begin(), ordered_left_occ.begin() + message_size),
        std::vector<int>(ordered_right_occ.begin(), ordered_right_occ.begin() + message_size),
        overlap_storage,
        n_orbitals);
    const int selected_root_size = static_cast<int>(right_selected_root_occ.size());
    for (int column = 0; column < selected_root_size; ++column) {
      for (int row = 0; row < selected_root_size; ++row) {
        message_overlap_submatrix[xmvb::to_size(column) * message_size + row] = 0.0;
      }
    }
    const Eigen::Map<const Matrix> message_overlap_block(
        message_overlap_submatrix.data(),
        message_size,
        message_size);
    block_diagonal_overlap.topLeftCorner(message_size, message_size) = message_overlap_block;
  }
  if (root_remainder_size > 0) {
    const auto root_overlap_submatrix = xmvb::vb::build_overlap_submatrix(
        left_root_remainder_occ,
        right_root_remainder_occ,
        overlap_storage,
        n_orbitals);
    const Eigen::Map<const Matrix> root_overlap_block(
        root_overlap_submatrix.data(),
        root_remainder_size,
        root_remainder_size);
    block_diagonal_overlap.bottomRightCorner(root_remainder_size, root_remainder_size) =
        root_overlap_block;
  }

  const auto full_one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);
  const Eigen::Map<const Matrix> full_one_electron_block(
      full_one_electron_submatrix.data(),
      dimension,
      dimension);
  return evaluate_dense_spin_block(
      block_diagonal_overlap,
      full_one_electron_block,
      overlap_resolver,
      subdeterminant_evaluations);
}

SpinOneElectronBlockBreakdown evaluate_spin_block_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int message_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // `message_size` splits the block-ordered occupied lists into
  //   [selected-root + leaf | root-remainder].
  // The one-electron contribution can then be partitioned into
  //   local  : row and column both on the same side of the split,
  //   mixed  : row and column on opposite sides.
  // The mixed part is the exact bridge contribution that the current
  // overlap-style recurrence does not represent.
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  if (message_size < 0 ||
      message_size > static_cast<int>(ordered_left_occ.size())) {
    throw std::invalid_argument("message_size is out of range");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);

  ++(*subdeterminant_evaluations);
  const int dimension = static_cast<int>(ordered_left_occ.size());
  const auto overlap_result = overlap_resolver.resolve(overlap_submatrix, dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);

  SpinOneElectronBlockBreakdown result;
  result.overlap = overlap_result.overlap_determinant;
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col < dimension; ++col) {
      const double contribution =
          one_electron_block(row, col) * cofactor_1st(row, col);
      const bool row_in_message = row < message_size;
      const bool col_in_message = col < message_size;
      if (row_in_message == col_in_message) {
        result.local_one_electron += contribution;
      } else {
        result.mixed_one_electron += contribution;
      }
    }
  }
  return result;
}

SpinOneElectronThreeBlockBreakdown evaluate_spin_three_block_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // The block order is fixed by the one-leaf explicit separator construction:
  //   [selected root | leaf | root remainder].
  // Grouping the exact cofactor contraction by these three blocks reveals
  // which subblocks of the full spin matrix are missing from the current
  // scalar leaf/root one-electron recurrence.
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  if (selected_root_size < 0 || leaf_size < 0) {
    throw std::invalid_argument("block sizes must be non-negative");
  }
  const int dimension = static_cast<int>(ordered_left_occ.size());
  if (selected_root_size + leaf_size > dimension) {
    throw std::invalid_argument("selected-root and leaf sizes exceed matrix dimension");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);

  ++(*subdeterminant_evaluations);
  const auto overlap_result = overlap_resolver.resolve(overlap_submatrix, dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);

  const auto block_index = [selected_root_size, leaf_size](const int index) -> int {
    if (index < selected_root_size) {
      return 0;
    }
    if (index < selected_root_size + leaf_size) {
      return 1;
    }
    return 2;
  };

  SpinOneElectronThreeBlockBreakdown result;
  result.overlap = overlap_result.overlap_determinant;
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col < dimension; ++col) {
      const double contribution =
          one_electron_block(row, col) * cofactor_1st(row, col);
      result.block_one_electron[xmvb::to_size(block_index(row))]
                               [xmvb::to_size(block_index(col))] += contribution;
    }
  }
  return result;
}

SpinOneElectronLaplaceTransferBreakdown evaluate_spin_laplace_transfer_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This diagnostic expands each exact cofactor minor by Laplace expansion over
  // the message rows that remain after deleting `(row, col)`. The chosen
  // message columns can then be classified by how many remainder columns are
  // transferred into the message-side subdeterminant. If only the `0/1`
  // transfer sectors survive, the next exact separator recurrence should be
  // built from off-by-one transfer states rather than higher-rank states.
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  const int message_size = selected_root_size + leaf_size;
  if (message_size < 0 || message_size > static_cast<int>(ordered_left_occ.size())) {
    throw std::invalid_argument("message_size is out of range");
  }
  if (selected_root_size < 0 || leaf_size < 0 ||
      selected_root_size + leaf_size > static_cast<int>(ordered_left_occ.size())) {
    throw std::invalid_argument("selected-root or leaf size is out of range");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);
  const int dimension = static_cast<int>(ordered_left_occ.size());
  const Eigen::Map<const Matrix> overlap_block(
      overlap_submatrix.data(),
      dimension,
      dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);

  auto build_submatrix = [](const Matrix& source,
                            const std::vector<int>& rows,
                            const std::vector<int>& cols) {
    Matrix submatrix(rows.size(), cols.size());
    for (int col_index = 0; col_index < static_cast<int>(cols.size()); ++col_index) {
      for (int row_index = 0; row_index < static_cast<int>(rows.size()); ++row_index) {
        submatrix(row_index, col_index) =
            source(rows[xmvb::to_size(row_index)],
                   cols[xmvb::to_size(col_index)]);
      }
    }
    return submatrix;
  };

  auto enumerate_column_subsets = [](const int n_columns,
                                     const int n_selected) {
    std::vector<std::vector<int>> subsets;
    std::vector<int> current;
    const auto recurse =
        [&](const auto& self, int next_column) -> void {
          if (static_cast<int>(current.size()) == n_selected) {
            subsets.push_back(current);
            return;
          }
          if (next_column >= n_columns) {
            return;
          }
          const int remaining_slots = n_selected - static_cast<int>(current.size());
          const int remaining_columns = n_columns - next_column;
          if (remaining_columns < remaining_slots) {
            return;
          }
          current.push_back(next_column);
          self(self, next_column + 1);
          current.pop_back();
          self(self, next_column + 1);
        };
    recurse(recurse, 0);
    return subsets;
  };
  const auto block_index = [selected_root_size, leaf_size](const int index) -> int {
    if (index < selected_root_size) {
      return 0;
    }
    if (index < selected_root_size + leaf_size) {
      return 1;
    }
    return 2;
  };
  SpinOneElectronLaplaceTransferBreakdown result;
  for (int row = 0; row < dimension; ++row) {
    const bool row_in_message = row < message_size;
    const int row_block = block_index(row);
    for (int col = 0; col < dimension; ++col) {
      const bool col_in_message = col < message_size;
      const int col_block = block_index(col);
      const double h_value = one_electron_block(row, col);
      if (std::abs(h_value) <= 1.0e-15) {
        continue;
      }

      std::vector<int> remaining_rows;
      std::vector<int> remaining_cols;
      std::vector<bool> remaining_col_in_message;
      remaining_rows.reserve(dimension - 1);
      remaining_cols.reserve(dimension - 1);
      remaining_col_in_message.reserve(dimension - 1);
      for (int row_index = 0; row_index < dimension; ++row_index) {
        if (row_index != row) {
          remaining_rows.push_back(row_index);
        }
      }
      for (int col_index = 0; col_index < dimension; ++col_index) {
        if (col_index != col) {
          remaining_cols.push_back(col_index);
          remaining_col_in_message.push_back(col_index < message_size);
        }
      }

      const int remaining_message_row_count = message_size - (row_in_message ? 1 : 0);
      const auto subsets = enumerate_column_subsets(
          static_cast<int>(remaining_cols.size()),
          remaining_message_row_count);
      const double cofactor_sign = ((row + col) & 1) ? -1.0 : 1.0;
      const int side_case_index =
          (row_in_message ? 0 : 2) + (col_in_message ? 0 : 1);

      for (const auto& selected_column_positions : subsets) {
        int transfer_in = 0;
        int sum_selected_column_positions = 0;
        std::vector<int> message_minor_columns;
        std::vector<int> root_minor_columns;
        message_minor_columns.reserve(selected_column_positions.size());
        root_minor_columns.reserve(remaining_cols.size() - selected_column_positions.size());

        std::vector<bool> is_selected_column(remaining_cols.size(), false);
        for (const int selected_position : selected_column_positions) {
          is_selected_column[xmvb::to_size(selected_position)] = true;
          sum_selected_column_positions += selected_position;
          message_minor_columns.push_back(
              remaining_cols[xmvb::to_size(selected_position)]);
          if (!remaining_col_in_message[xmvb::to_size(selected_position)]) {
            ++transfer_in;
          }
        }
        for (int position = 0; position < static_cast<int>(remaining_cols.size()); ++position) {
          if (!is_selected_column[xmvb::to_size(position)]) {
            root_minor_columns.push_back(
                remaining_cols[xmvb::to_size(position)]);
          }
        }

        std::vector<int> message_minor_rows;
        std::vector<int> root_minor_rows;
        message_minor_rows.reserve(remaining_message_row_count);
        root_minor_rows.reserve(remaining_rows.size() - remaining_message_row_count);
        for (int position = 0; position < static_cast<int>(remaining_rows.size()); ++position) {
          if (position < remaining_message_row_count) {
            message_minor_rows.push_back(remaining_rows[xmvb::to_size(position)]);
          } else {
            root_minor_rows.push_back(remaining_rows[xmvb::to_size(position)]);
          }
        }

        const Matrix message_minor =
            build_submatrix(overlap_block, message_minor_rows, message_minor_columns);
        const Matrix root_minor =
            build_submatrix(overlap_block, root_minor_rows, root_minor_columns);
        const double message_det =
            determinant_of_dense_matrix(message_minor, overlap_resolver, subdeterminant_evaluations);
        if (std::abs(message_det) <= 1.0e-15) {
          continue;
        }
        const double root_det =
            determinant_of_dense_matrix(root_minor, overlap_resolver, subdeterminant_evaluations);
        if (std::abs(root_det) <= 1.0e-15) {
          continue;
        }

        const int sum_selected_row_positions =
            remaining_message_row_count * (remaining_message_row_count - 1) / 2;
        const double laplace_sign =
            ((sum_selected_row_positions + sum_selected_column_positions) & 1) ? -1.0 : 1.0;
        const double term =
            h_value * cofactor_sign * laplace_sign * message_det * root_det;
        result.exact_total += term;
        result.block_one_electron[xmvb::to_size(row_block)]
                                 [xmvb::to_size(col_block)] += term;
        result.side_case_total[xmvb::to_size(side_case_index)] += term;
        const int transfer_bin = std::min(transfer_in, 3);
        result.transfer_in_hist[xmvb::to_size(transfer_bin)] += term;
        int expelled_message_column_count = 0;
        int exit_bin = 0;
        for (int position = 0; position < static_cast<int>(remaining_cols.size()); ++position) {
          if (is_selected_column[xmvb::to_size(position)]) {
            continue;
          }
          const int original_col = remaining_cols[xmvb::to_size(position)];
          if (original_col >= message_size) {
            continue;
          }
          ++expelled_message_column_count;
          if (expelled_message_column_count == 1) {
            exit_bin = (original_col < selected_root_size) ? 1 : 2;
          } else {
            exit_bin = 3;
          }
        }
        if (expelled_message_column_count > 1) {
          exit_bin = 3;
        }
        if (transfer_in == 0) {
          result.transfer_zero_exit_hist[xmvb::to_size(exit_bin)] += term;
        } else if (transfer_in == 1) {
          result.transfer_one_exit_hist[xmvb::to_size(exit_bin)] += term;
        }
      }
    }
  }
  return result;
}

SpinOneElectronLaplaceTransferBreakdown evaluate_spin_zeroed_block_laplace_transfer_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This diagnostic repeats the transfer-count Laplace expansion, but the
  // message-side minor uses the zeroed selected-root block that underlies the
  // exact overlap recurrence.
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  const int message_size = selected_root_size + leaf_size;
  if (selected_root_size < 0 || leaf_size < 0 ||
      message_size > static_cast<int>(ordered_left_occ.size())) {
    throw std::invalid_argument("selected-root or leaf size is out of range");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);
  const int dimension = static_cast<int>(ordered_left_occ.size());
  const Eigen::Map<const Matrix> overlap_block(
      overlap_submatrix.data(),
      dimension,
      dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);

  auto build_submatrix = [](const Matrix& source,
                            const std::vector<int>& rows,
                            const std::vector<int>& cols) {
    Matrix submatrix(rows.size(), cols.size());
    for (int col_index = 0; col_index < static_cast<int>(cols.size()); ++col_index) {
      for (int row_index = 0; row_index < static_cast<int>(rows.size()); ++row_index) {
        submatrix(row_index, col_index) =
            source(rows[xmvb::to_size(row_index)],
                   cols[xmvb::to_size(col_index)]);
      }
    }
    return submatrix;
  };

  auto enumerate_column_subsets = [](const int n_columns,
                                     const int n_selected) {
    std::vector<std::vector<int>> subsets;
    std::vector<int> current;
    const auto recurse =
        [&](const auto& self, int next_column) -> void {
          if (static_cast<int>(current.size()) == n_selected) {
            subsets.push_back(current);
            return;
          }
          if (next_column >= n_columns) {
            return;
          }
          const int remaining_slots = n_selected - static_cast<int>(current.size());
          const int remaining_columns = n_columns - next_column;
          if (remaining_columns < remaining_slots) {
            return;
          }
          current.push_back(next_column);
          self(self, next_column + 1);
          current.pop_back();
          self(self, next_column + 1);
        };
    recurse(recurse, 0);
    return subsets;
  };
  const auto block_index = [selected_root_size, leaf_size](const int index) -> int {
    if (index < selected_root_size) {
      return 0;
    }
    if (index < selected_root_size + leaf_size) {
      return 1;
    }
    return 2;
  };

  SpinOneElectronLaplaceTransferBreakdown result;
  for (int row = 0; row < dimension; ++row) {
    const bool row_in_message = row < message_size;
    const int row_block = block_index(row);
    for (int col = 0; col < dimension; ++col) {
      const bool col_in_message = col < message_size;
      const int col_block = block_index(col);
      const double h_value = one_electron_block(row, col);
      if (std::abs(h_value) <= 1.0e-15) {
        continue;
      }

      std::vector<int> remaining_rows;
      std::vector<int> remaining_cols;
      std::vector<bool> remaining_col_in_message;
      remaining_rows.reserve(dimension - 1);
      remaining_cols.reserve(dimension - 1);
      remaining_col_in_message.reserve(dimension - 1);
      for (int row_index = 0; row_index < dimension; ++row_index) {
        if (row_index != row) {
          remaining_rows.push_back(row_index);
        }
      }
      for (int col_index = 0; col_index < dimension; ++col_index) {
        if (col_index != col) {
          remaining_cols.push_back(col_index);
          remaining_col_in_message.push_back(col_index < message_size);
        }
      }

      const int remaining_message_row_count = message_size - (row_in_message ? 1 : 0);
      const auto subsets = enumerate_column_subsets(
          static_cast<int>(remaining_cols.size()),
          remaining_message_row_count);
      const double cofactor_sign = ((row + col) & 1) ? -1.0 : 1.0;
      const int side_case_index =
          (row_in_message ? 0 : 2) + (col_in_message ? 0 : 1);

      for (const auto& selected_column_positions : subsets) {
        int transfer_in = 0;
        int sum_selected_column_positions = 0;
        std::vector<int> message_minor_columns;
        std::vector<int> root_minor_columns;
        std::vector<bool> message_minor_column_is_selected_root;
        message_minor_columns.reserve(selected_column_positions.size());
        root_minor_columns.reserve(remaining_cols.size() - selected_column_positions.size());
        message_minor_column_is_selected_root.reserve(selected_column_positions.size());

        std::vector<bool> is_selected_column(remaining_cols.size(), false);
        for (const int selected_position : selected_column_positions) {
          is_selected_column[xmvb::to_size(selected_position)] = true;
          sum_selected_column_positions += selected_position;
          const int original_col = remaining_cols[xmvb::to_size(selected_position)];
          message_minor_columns.push_back(original_col);
          message_minor_column_is_selected_root.push_back(original_col < selected_root_size);
          if (!remaining_col_in_message[xmvb::to_size(selected_position)]) {
            ++transfer_in;
          }
        }
        for (int position = 0; position < static_cast<int>(remaining_cols.size()); ++position) {
          if (!is_selected_column[xmvb::to_size(position)]) {
            root_minor_columns.push_back(
                remaining_cols[xmvb::to_size(position)]);
          }
        }

        std::vector<int> message_minor_rows;
        std::vector<int> root_minor_rows;
        std::vector<bool> message_minor_row_is_selected_root;
        message_minor_rows.reserve(remaining_message_row_count);
        root_minor_rows.reserve(remaining_rows.size() - remaining_message_row_count);
        message_minor_row_is_selected_root.reserve(remaining_message_row_count);
        for (int position = 0; position < static_cast<int>(remaining_rows.size()); ++position) {
          const int original_row = remaining_rows[xmvb::to_size(position)];
          if (position < remaining_message_row_count) {
            message_minor_rows.push_back(original_row);
            message_minor_row_is_selected_root.push_back(original_row < selected_root_size);
          } else {
            root_minor_rows.push_back(original_row);
          }
        }

        Matrix message_minor =
            build_submatrix(overlap_block, message_minor_rows, message_minor_columns);
        for (int minor_col = 0; minor_col < static_cast<int>(message_minor_columns.size()); ++minor_col) {
          if (!message_minor_column_is_selected_root[xmvb::to_size(minor_col)]) {
            continue;
          }
          for (int minor_row = 0; minor_row < static_cast<int>(message_minor_rows.size()); ++minor_row) {
            if (message_minor_row_is_selected_root[xmvb::to_size(minor_row)]) {
              message_minor(minor_row, minor_col) = 0.0;
            }
          }
        }
        const Matrix root_minor =
            build_submatrix(overlap_block, root_minor_rows, root_minor_columns);
        const double message_det =
            determinant_of_dense_matrix(message_minor, overlap_resolver, subdeterminant_evaluations);
        if (std::abs(message_det) <= 1.0e-15) {
          continue;
        }
        const double root_det =
            determinant_of_dense_matrix(root_minor, overlap_resolver, subdeterminant_evaluations);
        if (std::abs(root_det) <= 1.0e-15) {
          continue;
        }

        const int sum_selected_row_positions =
            remaining_message_row_count * (remaining_message_row_count - 1) / 2;
        const double laplace_sign =
            ((sum_selected_row_positions + sum_selected_column_positions) & 1) ? -1.0 : 1.0;
        const double term =
            h_value * cofactor_sign * laplace_sign * message_det * root_det;
        result.exact_total += term;
        result.block_one_electron[xmvb::to_size(row_block)]
                                 [xmvb::to_size(col_block)] += term;
        result.side_case_total[xmvb::to_size(side_case_index)] += term;
        result.transfer_in_hist[xmvb::to_size(std::min(transfer_in, 3))] += term;
        int expelled_message_column_count = 0;
        int exit_bin = 0;
        for (int position = 0; position < static_cast<int>(remaining_cols.size()); ++position) {
          if (is_selected_column[xmvb::to_size(position)]) {
            continue;
          }
          const int original_col = remaining_cols[xmvb::to_size(position)];
          if (original_col >= message_size) {
            continue;
          }
          ++expelled_message_column_count;
          if (expelled_message_column_count == 1) {
            exit_bin = (original_col < selected_root_size) ? 1 : 2;
          } else {
            exit_bin = 3;
          }
        }
        if (expelled_message_column_count > 1) {
          exit_bin = 3;
        }
        if (transfer_in == 0) {
          result.transfer_zero_exit_hist[xmvb::to_size(exit_bin)] += term;
        } else if (transfer_in == 1) {
          result.transfer_one_exit_hist[xmvb::to_size(exit_bin)] += term;
        }
      }
    }
  }
  return result;
}

SpinCrossCofactorRankBreakdown evaluate_spin_full_cross_cofactor_ranks(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This diagnostic asks whether the exact full-block cross cofactor kernel is
  // rank-1 across the `[message] x [root remainder]` split. If the rank is
  // larger than 1, then scalar row-open / column-open payloads are
  // insufficient, and the exact recurrence must carry an extra interface
  // index.
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  const int message_size = selected_root_size + leaf_size;
  const int dimension = static_cast<int>(ordered_left_occ.size());
  if (selected_root_size < 0 || leaf_size < 0 || message_size > dimension) {
    throw std::invalid_argument("selected-root or leaf size is out of range");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  ++(*subdeterminant_evaluations);
  const auto overlap_result = overlap_resolver.resolve(overlap_submatrix, dimension);
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);

  SpinCrossCofactorRankBreakdown result;
  const int root_size = dimension - message_size;
  if (message_size == 0 || root_size == 0) {
    return result;
  }
  const Matrix message_to_root =
      cofactor_1st.topRightCorner(message_size, root_size);
  const Matrix root_to_message =
      cofactor_1st.bottomLeftCorner(root_size, message_size);
  result.message_to_root_rank =
      Eigen::FullPivLU<Matrix>(message_to_root).rank();
  result.root_to_message_rank =
      Eigen::FullPivLU<Matrix>(root_to_message).rank();
  return result;
}

SpinOneElectronOpenStatePrototypeBreakdown evaluate_spin_open_state_prototype_breakdown(
    const std::vector<int>& ordered_left_occ,
    const std::vector<int>& ordered_right_occ,
    int selected_root_size,
    int leaf_size,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This prototype does not yet claim to be a reusable separator recurrence.
  // It provides an exact cofactor decomposition for one spin block:
  //
  //   full cofactor
  // = zeroed closed part
  // + exact message<->root cross blocks
  // + residual closed correction.
  //
  // The purpose is to isolate which Hamiltonian information is still missing
  // after the current exact-overlap-inspired zeroed-block construction. The
  // three block classes follow the common occupied-orbital order
  //   [selected root | leaf | root remainder].
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    throw std::invalid_argument("ordered occupied lists must have the same size");
  }
  const int dimension = static_cast<int>(ordered_left_occ.size());
  const int message_size = selected_root_size + leaf_size;
  const int root_size = dimension - message_size;
  if (selected_root_size < 0 || leaf_size < 0 || message_size > dimension) {
    throw std::invalid_argument("selected-root or leaf size is out of range");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);
  const Eigen::Map<const Matrix> overlap_block(
      overlap_submatrix.data(),
      dimension,
      dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);

  ++(*subdeterminant_evaluations);
  const auto full_overlap_result = overlap_resolver.resolve(overlap_submatrix, dimension);
  const Matrix full_cofactor_1st = xmvb::vb::calc_cofactor_1st(full_overlap_result);

  Matrix zeroed_closed_cofactor = Matrix::Zero(dimension, dimension);
  if (message_size > 0) {
    Matrix zeroed_message_overlap = overlap_block.topLeftCorner(message_size, message_size);
    if (selected_root_size > 0) {
      zeroed_message_overlap.topLeftCorner(selected_root_size, selected_root_size).setZero();
    }
    ++(*subdeterminant_evaluations);
    const auto zeroed_message_result = overlap_resolver.resolve(
        flatten_column_major_matrix(zeroed_message_overlap),
        message_size);
    const Matrix zeroed_message_cofactor =
        xmvb::vb::calc_cofactor_1st(zeroed_message_result);
    std::optional<xmvb::vb::DeterminantOverlapResult> root_result;
    double root_overlap = 1.0;
    if (root_size > 0) {
      ++(*subdeterminant_evaluations);
      root_result = overlap_resolver.resolve(
          flatten_column_major_matrix(overlap_block.bottomRightCorner(root_size, root_size)),
          root_size);
      root_overlap = root_result->overlap_determinant;
    }
    zeroed_closed_cofactor.topLeftCorner(message_size, message_size) =
        zeroed_message_cofactor * root_overlap;
    if (root_size > 0) {
      const Matrix root_cofactor = xmvb::vb::calc_cofactor_1st(*root_result);
      zeroed_closed_cofactor.bottomRightCorner(root_size, root_size) =
          root_cofactor * zeroed_message_result.overlap_determinant;
    }
  } else if (root_size > 0) {
    ++(*subdeterminant_evaluations);
    const auto root_result = overlap_resolver.resolve(
        flatten_column_major_matrix(overlap_block.bottomRightCorner(root_size, root_size)),
        root_size);
    zeroed_closed_cofactor.bottomRightCorner(root_size, root_size) =
        xmvb::vb::calc_cofactor_1st(root_result);
  }

  Matrix exact_cross_cofactor = Matrix::Zero(dimension, dimension);
  if (message_size > 0 && root_size > 0) {
    exact_cross_cofactor.topRightCorner(message_size, root_size) =
        full_cofactor_1st.topRightCorner(message_size, root_size);
    exact_cross_cofactor.bottomLeftCorner(root_size, message_size) =
        full_cofactor_1st.bottomLeftCorner(root_size, message_size);
  }

  const Matrix residual_closed_cofactor =
      full_cofactor_1st - zeroed_closed_cofactor - exact_cross_cofactor;

  const auto block_index = [selected_root_size, leaf_size](const int index) -> int {
    if (index < selected_root_size) {
      return 0;
    }
    if (index < selected_root_size + leaf_size) {
      return 1;
    }
    return 2;
  };

  SpinOneElectronOpenStatePrototypeBreakdown result;
  result.overlap = full_overlap_result.overlap_determinant;
  if (leaf_size > 0) {
    result.residual_leaf_leaf_cofactor =
        residual_closed_cofactor.block(
            selected_root_size,
            selected_root_size,
            leaf_size,
            leaf_size);
  }
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col < dimension; ++col) {
      const double h_value = one_electron_block(row, col);
      const double full_term = h_value * full_cofactor_1st(row, col);
      const double zeroed_closed_term = h_value * zeroed_closed_cofactor(row, col);
      const double exact_cross_term = h_value * exact_cross_cofactor(row, col);
      const double residual_closed_term = h_value * residual_closed_cofactor(row, col);
      result.exact_total += full_term;
      result.zeroed_closed_total += zeroed_closed_term;
      result.exact_cross_total += exact_cross_term;
      result.residual_closed_total += residual_closed_term;
      result.residual_closed_block_one_electron[xmvb::to_size(block_index(row))]
                                               [xmvb::to_size(block_index(col))] +=
          residual_closed_term;
    }
  }

  if (message_size > 0) {
    result.residual_message_rank =
        Eigen::FullPivLU<Matrix>(
            residual_closed_cofactor.topLeftCorner(message_size, message_size))
            .rank();
  }
  if (leaf_size > 0) {
    result.residual_leaf_leaf_rank =
        Eigen::FullPivLU<Matrix>(
            residual_closed_cofactor.block(
                selected_root_size,
                selected_root_size,
                leaf_size,
                leaf_size))
            .rank();
  }
  return result;
}

SpinOverlapOneElectronValue evaluate_zeroed_selected_root_block_spin_values(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // The separator leaf block is assembled in the same block order used by the
  // overlap recurrence: selected root rows/cols first, then the leaf rows/cols.
  // Only the selected root-root overlap subblock is zeroed. The one-electron
  // block keeps its full values because the Hamiltonian insertion is evaluated
  // on the corresponding cofactor matrix of the zeroed overlap block.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  std::vector<std::pair<int, bool>> left_occ_with_root_flag;
  left_occ_with_root_flag.reserve(left_root_occ.size() + left_leaf_occ.size());
  for (const int orbital : left_root_occ) {
    left_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : left_leaf_occ) {
    left_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(left_occ_with_root_flag.begin(), left_occ_with_root_flag.end());

  std::vector<std::pair<int, bool>> right_occ_with_root_flag;
  right_occ_with_root_flag.reserve(right_root_occ.size() + right_leaf_occ.size());
  for (const int orbital : right_root_occ) {
    right_occ_with_root_flag.emplace_back(orbital, true);
  }
  for (const int orbital : right_leaf_occ) {
    right_occ_with_root_flag.emplace_back(orbital, false);
  }
  std::sort(right_occ_with_root_flag.begin(), right_occ_with_root_flag.end());

  std::vector<int> ordered_left_occ;
  std::vector<int> ordered_right_occ;
  ordered_left_occ.reserve(left_occ_with_root_flag.size());
  ordered_right_occ.reserve(right_occ_with_root_flag.size());
  for (const auto& [orbital, is_root] : left_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_left_occ.push_back(orbital);
  }
  for (const auto& [orbital, is_root] : right_occ_with_root_flag) {
    static_cast<void>(is_root);
    ordered_right_occ.push_back(orbital);
  }
  if (ordered_left_occ.size() != ordered_right_occ.size()) {
    return {};
  }

  auto overlap_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      overlap_storage,
      n_orbitals);
  const auto one_electron_submatrix = xmvb::vb::build_overlap_submatrix(
      ordered_left_occ,
      ordered_right_occ,
      one_electron_storage,
      n_orbitals);
  const int dimension = static_cast<int>(ordered_left_occ.size());
  for (int column = 0; column < dimension; ++column) {
    if (!left_occ_with_root_flag[xmvb::to_size(column)].second) {
      continue;
    }
    for (int row = 0; row < dimension; ++row) {
      if (!right_occ_with_root_flag[xmvb::to_size(row)].second) {
        continue;
      }
      overlap_submatrix[xmvb::to_size(column) * dimension + row] = 0.0;
    }
  }

  const Eigen::Map<const Matrix> overlap_block(
      overlap_submatrix.data(),
      dimension,
      dimension);
  const Eigen::Map<const Matrix> one_electron_block(
      one_electron_submatrix.data(),
      dimension,
      dimension);
  return evaluate_dense_spin_block(
      overlap_block,
      one_electron_block,
      overlap_resolver,
      subdeterminant_evaluations);
}

double evaluate_dense_leaf_leaf_block_one_electron(
    const Matrix& overlap_block,
    const Matrix& one_electron_block,
    int selected_root_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // This helper extracts only the leaf-leaf contribution to the exact
  // first-cofactor contraction on a dense message block ordered as
  //   [selected root | leaf].
  // The returned scalar therefore isolates the closed leaf sector that
  // survives after removing message<->root cross terms.
  if (overlap_block.rows() != overlap_block.cols() ||
      one_electron_block.rows() != overlap_block.rows() ||
      one_electron_block.cols() != overlap_block.cols()) {
    throw std::invalid_argument(
        "evaluate_dense_leaf_leaf_block_one_electron requires square, dimension-matched matrices");
  }
  if (selected_root_size < 0 || selected_root_size > overlap_block.rows()) {
    throw std::invalid_argument("selected_root_size is out of range");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  ++(*subdeterminant_evaluations);
  const auto overlap_result = overlap_resolver.resolve(
      flatten_column_major_matrix(overlap_block),
      overlap_block.rows());
  const Matrix cofactor_1st = xmvb::vb::calc_cofactor_1st(overlap_result);
  double leaf_leaf_total = 0.0;
  for (int row = selected_root_size; row < overlap_block.rows(); ++row) {
    for (int col = selected_root_size; col < overlap_block.cols(); ++col) {
      leaf_leaf_total += one_electron_block(row, col) * cofactor_1st(row, col);
    }
  }
  return leaf_leaf_total;
}

std::vector<SpinMaskOneElectronEntry> build_spin_mask_one_electron_entries_from_selectors(
    const Matrix& left_root_full_selector,
    const Matrix& left_leaf_selector,
    const Matrix& right_root_full_selector,
    const Matrix& right_leaf_selector,
    const Matrix& support_overlap,
    const Matrix& support_one_electron,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // One-electron validation deliberately uses the direct exact path for every
  // compatible mask pair. This keeps the logic transparent: if the recurrence
  // is correct, it must match the determinant-pair reference before any
  // Schur-complement optimization is introduced.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  const int n_root_rows = right_root_full_selector.rows();
  const int n_root_cols = left_root_full_selector.rows();
  const int n_leaf_rows = right_leaf_selector.rows();
  const int n_leaf_cols = left_leaf_selector.rows();
  std::vector<SpinMaskOneElectronEntry> entries;

  for (std::uint32_t row_mask = 0;
       row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
       ++row_mask) {
    const int row_count = popcount(row_mask);
    for (std::uint32_t col_mask = 0;
         col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
         ++col_mask) {
      const int col_count = popcount(col_mask);
      if (row_count + n_leaf_rows != col_count + n_leaf_cols) {
        continue;
      }

      const Matrix selected_left_root =
          select_selector_rows_by_mask(left_root_full_selector, col_mask);
      const Matrix selected_right_root =
          select_selector_rows_by_mask(right_root_full_selector, row_mask);
      const int dimension = row_count + n_leaf_rows;
      Matrix right_selector(dimension, support_overlap.rows());
      Matrix left_selector(dimension, support_overlap.cols());
      right_selector.setZero();
      left_selector.setZero();
      if (row_count > 0) {
        right_selector.topRows(row_count) = selected_right_root;
      }
      if (n_leaf_rows > 0) {
        right_selector.bottomRows(n_leaf_rows) = right_leaf_selector;
      }
      if (col_count > 0) {
        left_selector.topRows(col_count) = selected_left_root;
      }
      if (n_leaf_cols > 0) {
        left_selector.bottomRows(n_leaf_cols) = left_leaf_selector;
      }

      Matrix overlap_block =
          build_transformed_block_matrix(left_selector, right_selector, support_overlap);
      const Matrix one_electron_block =
          build_transformed_block_matrix(left_selector, right_selector, support_one_electron);
      if (row_count > 0 && col_count > 0) {
        overlap_block.topLeftCorner(row_count, col_count).setZero();
      }

      const auto value = evaluate_dense_spin_block(
          overlap_block,
          one_electron_block,
          overlap_resolver,
          subdeterminant_evaluations);
      if (std::abs(value.overlap) <= 1.0e-15 &&
          std::abs(value.one_electron) <= 1.0e-15) {
        continue;
      }

      entries.push_back({
          .row_mask = row_mask,
          .col_mask = col_mask,
          .overlap = value.overlap,
          .one_electron = value.one_electron,
      });
    }
  }
  return entries;
}

double determinant_with_zeroed_selected_root_block_from_selectors(
    const Matrix& left_root_selector,
    const Matrix& left_leaf_selector,
    const Matrix& right_root_selector,
    const Matrix& right_leaf_selector,
    const Matrix& support_overlap,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver) {
  const int n_root_rows = right_root_selector.rows();
  const int n_root_cols = left_root_selector.rows();
  const int n_leaf_rows = right_leaf_selector.rows();
  const int n_leaf_cols = left_leaf_selector.rows();
  const int dimension = n_root_rows + n_leaf_rows;
  if (dimension != n_root_cols + n_leaf_cols) {
    return 0.0;
  }

  Matrix right_selector(dimension, support_overlap.rows());
  Matrix left_selector(dimension, support_overlap.cols());
  right_selector.setZero();
  left_selector.setZero();
  if (n_root_rows > 0) {
    right_selector.topRows(n_root_rows) = right_root_selector;
  }
  if (n_leaf_rows > 0) {
    right_selector.bottomRows(n_leaf_rows) = right_leaf_selector;
  }
  if (n_root_cols > 0) {
    left_selector.topRows(n_root_cols) = left_root_selector;
  }
  if (n_leaf_cols > 0) {
    left_selector.bottomRows(n_leaf_cols) = left_leaf_selector;
  }

  Matrix overlap_submatrix = right_selector * support_overlap * left_selector.transpose();
  if (n_root_rows > 0 && n_root_cols > 0) {
    overlap_submatrix.topLeftCorner(n_root_rows, n_root_cols).setZero();
  }
  return overlap_resolver
      .resolve(flatten_column_major_matrix(overlap_submatrix), dimension)
      .overlap_determinant;
}

double determinant_of_dense_matrix(
    const Matrix& matrix,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // `matrix` is the explicit dense overlap/minor matrix whose determinant is
  // needed by the separator recurrence. In the collapsed Schur path these
  // minors are tiny root-mask blocks, so evaluating them by direct closed
  // forms avoids the overhead of dispatching through the generic determinant
  // resolver for 1x1, 2x2, and 3x3 cases.
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  ++(*subdeterminant_evaluations);
  switch (matrix.rows()) {
    case 0:
      return 1.0;
    case 1:
      return matrix(0, 0);
    case 2:
      return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    case 3:
      return matrix(0, 0) *
                 (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
             matrix(0, 1) *
                 (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
             matrix(0, 2) *
                 (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    default:
      break;
  }
  return overlap_resolver
      .resolve(flatten_column_major_matrix(matrix), matrix.rows())
      .overlap_determinant;
}

std::vector<SpinMaskDeterminantEntry> build_spin_mask_determinants_from_selectors(
    const Matrix& left_root_full_selector,
    const Matrix& left_leaf_selector,
    const Matrix& right_root_full_selector,
    const Matrix& right_leaf_selector,
    const Matrix& support_overlap,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* schur_fast_spin_assignment_count,
    std::uint64_t* rectangular_fallback_spin_assignment_count,
    std::uint64_t* singular_fallback_spin_assignment_count,
    std::uint64_t* schur_fast_subdeterminant_evaluation_count,
    std::uint64_t* rectangular_fallback_subdeterminant_evaluation_count,
    std::uint64_t* singular_fallback_subdeterminant_evaluation_count) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (schur_fast_spin_assignment_count == nullptr ||
      rectangular_fallback_spin_assignment_count == nullptr ||
      singular_fallback_spin_assignment_count == nullptr ||
      schur_fast_subdeterminant_evaluation_count == nullptr ||
      rectangular_fallback_subdeterminant_evaluation_count == nullptr ||
      singular_fallback_subdeterminant_evaluation_count == nullptr) {
    throw std::invalid_argument("collapsed spin-assignment counters must not be null");
  }

  const int n_root_rows = right_root_full_selector.rows();
  const int n_root_cols = left_root_full_selector.rows();
  const int n_leaf_rows = right_leaf_selector.rows();
  const int n_leaf_cols = left_leaf_selector.rows();

  if (n_leaf_rows == 0 && n_leaf_cols == 0) {
    std::vector<SpinMaskDeterminantEntry> entries;
    entries.push_back({0U, 0U, 1.0});
    return entries;
  }

  if (n_leaf_rows != n_leaf_cols) {
    // Rectangular leaf blocks cannot use the square leaf-leaf Schur collapse.
    // We therefore fall back to the exact zeroed-root-block determinant for
    // every compatible root mask pair.
    ++(*rectangular_fallback_spin_assignment_count);
    const std::uint64_t branch_start = *subdeterminant_evaluations;
    std::vector<SpinMaskDeterminantEntry> entries;
    for (std::uint32_t row_mask = 0;
         row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
         ++row_mask) {
      const int row_count = popcount(row_mask);
      for (std::uint32_t col_mask = 0;
           col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
           ++col_mask) {
        const int col_count = popcount(col_mask);
        if (row_count + n_leaf_rows != col_count + n_leaf_cols) {
          continue;
        }
        const double determinant =
            determinant_with_zeroed_selected_root_block_from_selectors(
                select_selector_rows_by_mask(left_root_full_selector, col_mask),
                left_leaf_selector,
                select_selector_rows_by_mask(right_root_full_selector, row_mask),
                right_leaf_selector,
                support_overlap,
                overlap_resolver);
        ++(*subdeterminant_evaluations);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        entries.push_back({row_mask, col_mask, determinant});
      }
    }
    *rectangular_fallback_subdeterminant_evaluation_count +=
        (*subdeterminant_evaluations - branch_start);
    return entries;
  }

  const int n_leaf = n_leaf_cols;
  const Matrix leaf_leaf_block =
      right_leaf_selector * support_overlap * left_leaf_selector.transpose();
  const auto leaf_leaf_result = overlap_resolver.resolve(
      flatten_column_major_matrix(leaf_leaf_block),
      n_leaf);
  ++(*subdeterminant_evaluations);

  if (std::abs(leaf_leaf_result.overlap_determinant) <= 1.0e-15 ||
      leaf_leaf_result.inverse_overlap_submatrix.rows() != n_leaf ||
      leaf_leaf_result.inverse_overlap_submatrix.cols() != n_leaf) {
    // Singular square leaf blocks also lose the Schur path. This branch keeps
    // the recurrence exact, but it pays the full masked zeroed-block
    // determinant cost for the current spin assignment.
    ++(*singular_fallback_spin_assignment_count);
    const std::uint64_t branch_start = *subdeterminant_evaluations - 1U;
    std::vector<SpinMaskDeterminantEntry> entries;
    for (std::uint32_t row_mask = 0;
         row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
         ++row_mask) {
      for (std::uint32_t col_mask = 0;
           col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
           ++col_mask) {
        if (popcount(row_mask) != popcount(col_mask)) {
          continue;
        }
        const double determinant =
            determinant_with_zeroed_selected_root_block_from_selectors(
                select_selector_rows_by_mask(left_root_full_selector, col_mask),
                left_leaf_selector,
                select_selector_rows_by_mask(right_root_full_selector, row_mask),
                right_leaf_selector,
                support_overlap,
                overlap_resolver);
        ++(*subdeterminant_evaluations);
        if (std::abs(determinant) <= 1.0e-15) {
          continue;
        }
        entries.push_back({row_mask, col_mask, determinant});
      }
    }
    *singular_fallback_subdeterminant_evaluation_count +=
        (*subdeterminant_evaluations - branch_start);
    return entries;
  }

  const Matrix root_to_leaf_block =
      right_root_full_selector * support_overlap * left_leaf_selector.transpose();
  const Matrix leaf_to_root_block =
      right_leaf_selector * support_overlap * left_root_full_selector.transpose();
  Matrix schur_mask_matrix =
      -(root_to_leaf_block * leaf_leaf_result.inverse_overlap_submatrix * leaf_to_root_block);
  ++(*schur_fast_spin_assignment_count);
  const std::uint64_t branch_start = *subdeterminant_evaluations - 1U;

  std::vector<SpinMaskDeterminantEntry> entries;
  for (std::uint32_t row_mask = 0;
       row_mask < (static_cast<std::uint32_t>(1) << n_root_rows);
       ++row_mask) {
    const int row_count = popcount(row_mask);
    for (std::uint32_t col_mask = 0;
         col_mask < (static_cast<std::uint32_t>(1) << n_root_cols);
         ++col_mask) {
      const int col_count = popcount(col_mask);
      if (row_count != col_count) {
        continue;
      }
      if (row_count == 0) {
        entries.push_back({row_mask, col_mask, leaf_leaf_result.overlap_determinant});
        continue;
      }

      Matrix minor(row_count, col_count);
      int minor_row = 0;
      for (int row_index = 0; row_index < n_root_rows; ++row_index) {
        if ((row_mask & (static_cast<std::uint32_t>(1) << row_index)) == 0U) {
          continue;
        }
        int minor_col = 0;
        for (int col_index = 0; col_index < n_root_cols; ++col_index) {
          if ((col_mask & (static_cast<std::uint32_t>(1) << col_index)) == 0U) {
            continue;
          }
          minor(minor_row, minor_col) = schur_mask_matrix(row_index, col_index);
          ++minor_col;
        }
        ++minor_row;
      }

      const double determinant =
          leaf_leaf_result.overlap_determinant *
          determinant_of_dense_matrix(minor, overlap_resolver, subdeterminant_evaluations);
      if (std::abs(determinant) <= 1.0e-15) {
        continue;
      }
      entries.push_back({row_mask, col_mask, determinant});
    }
  }
  *schur_fast_subdeterminant_evaluation_count +=
      (*subdeterminant_evaluations - branch_start);
  return entries;
}

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  // After component ordering, the canonicalization sign for one spin channel
  // depends only on how many selected root orbitals are inserted ahead of each
  // leaf block and how many root orbitals remain for the final remainder
  // block. The detailed leaf occupied-orbital identities no longer enter.
  if (selected_masks.size() != leaf_occ_sizes.size()) {
    throw std::invalid_argument("selected_masks and leaf_occ_sizes must have the same length");
  }
  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ) +
      std::accumulate(leaf_occ_sizes.begin(), leaf_occ_sizes.end(), 0));

  std::uint32_t used_mask = 0U;
  int next_leaf_label = n_root_occ;
  for (std::size_t leaf_index = 0; leaf_index < selected_masks.size(); ++leaf_index) {
    const std::uint32_t mask = selected_masks[leaf_index];
    used_mask |= mask;
    for (int root_position = 0; root_position < n_root_occ; ++root_position) {
      if ((mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0;
         leaf_position < leaf_occ_sizes[leaf_index];
         ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint32_t full_mask =
      (n_root_occ == 0) ? 0U
                        : ((static_cast<std::uint32_t>(1) << n_root_occ) - 1U);
  const std::uint32_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (static_cast<std::uint32_t>(1) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

std::vector<CollapsedLeafMessageEntry> build_collapsed_leaf_messages_component_ordered(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    const ComponentData& leaf_component,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    CollapsedStarPairStats* stats,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* hypercube_assignment_count) {
  // First-stage leaf-message collapse: for one fixed root determinant-term
  // pair, sum every exact local leaf contribution that lands on the same
  // separator mask tuple
  //   (alpha_row_mask, alpha_col_mask, beta_row_mask, beta_col_mask).
  // The returned message list therefore carries only boundary-state data and a
  // scalar value, not explicit leaf occupied-orbital lists.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (hypercube_assignment_count == nullptr) {
    throw std::invalid_argument("hypercube_assignment_count must not be null");
  }
  if (stats == nullptr) {
    throw std::invalid_argument("collapsed stats must not be null");
  }

  const int n_alpha_root_rows = static_cast<int>(right_root_term.alpha_occ.size());
  const int n_alpha_root_cols = static_cast<int>(left_root_term.alpha_occ.size());
  const int n_beta_root_rows = static_cast<int>(right_root_term.beta_occ.size());
  const int n_beta_root_cols = static_cast<int>(left_root_term.beta_occ.size());
  const int n_left_covariant_labels = count_covalent_pairs(leaf_component.left_pairs);
  const int n_right_covariant_labels = count_covalent_pairs(leaf_component.right_pairs);
  const std::uint64_t n_left_sign_assignments =
      static_cast<std::uint64_t>(1) << n_left_covariant_labels;
  const std::uint64_t n_right_sign_assignments =
      static_cast<std::uint64_t>(1) << n_right_covariant_labels;
  const Eigen::Map<const Matrix> support_overlap(
      support_overlap_storage.data(),
      support_size,
      support_size);
  const Matrix left_alpha_root_full_selector =
      build_basis_selector_matrix(left_root_term.alpha_occ, support_size);
  const Matrix right_alpha_root_full_selector =
      build_basis_selector_matrix(right_root_term.alpha_occ, support_size);
  const Matrix left_beta_root_full_selector =
      build_basis_selector_matrix(left_root_term.beta_occ, support_size);
  const Matrix right_beta_root_full_selector =
      build_basis_selector_matrix(right_root_term.beta_occ, support_size);

  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>, double>
      aggregated_messages;
  for (std::uint64_t left_sign_mask = 0;
       left_sign_mask < n_left_sign_assignments;
       ++left_sign_mask) {
    const Matrix left_alpha_selector =
        build_signed_selector_matrix(leaf_component.left_pairs, left_sign_mask, true, support_size);
    const Matrix left_beta_selector =
        build_signed_selector_matrix(leaf_component.left_pairs, left_sign_mask, false, support_size);
    for (std::uint64_t right_sign_mask = 0;
         right_sign_mask < n_right_sign_assignments;
         ++right_sign_mask) {
      ++(*hypercube_assignment_count);
      const Matrix right_alpha_selector =
          build_signed_selector_matrix(
              leaf_component.right_pairs,
              right_sign_mask,
              true,
              support_size);
      const Matrix right_beta_selector =
          build_signed_selector_matrix(
              leaf_component.right_pairs,
              right_sign_mask,
              false,
              support_size);
      const auto alpha_entries = build_spin_mask_determinants_from_selectors(
          left_alpha_root_full_selector,
          left_alpha_selector,
          right_alpha_root_full_selector,
          right_alpha_selector,
          support_overlap,
          overlap_resolver,
          subdeterminant_evaluations,
          &stats->schur_fast_spin_assignment_count,
          &stats->rectangular_fallback_spin_assignment_count,
          &stats->singular_fallback_spin_assignment_count,
          &stats->schur_fast_subdeterminant_evaluation_count,
          &stats->rectangular_fallback_subdeterminant_evaluation_count,
          &stats->singular_fallback_subdeterminant_evaluation_count);
      const auto beta_entries = build_spin_mask_determinants_from_selectors(
          left_beta_root_full_selector,
          left_beta_selector,
          right_beta_root_full_selector,
          right_beta_selector,
          support_overlap,
          overlap_resolver,
          subdeterminant_evaluations,
          &stats->schur_fast_spin_assignment_count,
          &stats->rectangular_fallback_spin_assignment_count,
          &stats->singular_fallback_spin_assignment_count,
          &stats->schur_fast_subdeterminant_evaluation_count,
          &stats->rectangular_fallback_subdeterminant_evaluation_count,
          &stats->singular_fallback_subdeterminant_evaluation_count);

      for (const auto& alpha_entry : alpha_entries) {
        for (const auto& beta_entry : beta_entries) {
          aggregated_messages[{
              alpha_entry.row_mask,
              alpha_entry.col_mask,
              beta_entry.row_mask,
              beta_entry.col_mask}] +=
              alpha_entry.value * beta_entry.value;
        }
      }
    }
  }

  const double average_factor =
      1.0 / static_cast<double>(
                static_cast<std::uint64_t>(1)
                << (n_left_covariant_labels + n_right_covariant_labels));
  for (auto& [key, value] : aggregated_messages) {
    value *= average_factor;
  }

  std::vector<CollapsedLeafMessageEntry> entries;
  entries.reserve(aggregated_messages.size());
  for (const auto& [key, value] : aggregated_messages) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    CollapsedLeafMessageEntry entry;
    entry.alpha_row_mask = std::get<0>(key);
    entry.alpha_col_mask = std::get<1>(key);
    entry.beta_row_mask = std::get<2>(key);
    entry.beta_col_mask = std::get<3>(key);
    entry.value = value;
    entries.push_back(std::move(entry));
  }
  return entries;
}

void record_reference_determinant_work(
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<xmvb::vb::LegacyStructureDeterminantTerm>& right_terms,
    ExactWorkCollector* work_collector) {
  if (work_collector == nullptr) {
    return;
  }
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      work_collector->reference_full_determinant_pairs.insert(
          {left_term.alpha_occ,
           left_term.beta_occ,
           right_term.alpha_occ,
           right_term.beta_occ});
      work_collector->reference_spin_determinants.insert(
          {left_term.alpha_occ, right_term.alpha_occ});
      work_collector->reference_spin_determinants.insert(
          {left_term.beta_occ, right_term.beta_occ});
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto started_at = std::chrono::steady_clock::now();
    xmvb::vb::CppVbInputLoadOptions load_options;
    // This analyzer does not require the legacy HF/VB guess runtime. Request
    // the pure C++ orbital guess path so the tool remains runnable in lighter
    // builds that omit the legacy HF setup.
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto& raw_structure_data = load_result.raw_structure_data;
    if (options.active_overlap_source != ActiveOverlapSource::Input) {
      throw std::runtime_error(
          "analyze_star_separator_one_electron_dataset currently supports only "
          "--active-overlap-source input because the one-electron integrals must "
          "match the overlap metric");
    }
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "analyze_star_separator_one_electron_dataset currently supports only singlet closed-shell structures");
    }

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto timed_active_space = xmvb::vb::prepare_timed_active_space_context(
        load_result.input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);
    const auto& prepared_active_space = timed_active_space.prepared_active_space;
    const auto& active_overlap_storage =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const auto& active_one_electron_storage =
        prepared_active_space.active_space_one_electron_result.h1e_act;
    const auto& packed_active_two_electron_integrals =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;

    std::vector<PerStructureCache> structure_cache(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      auto& cache = structure_cache[xmvb::to_size(structure_index)];
      cache.active_pairs =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
      cache.legacy_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(cache.active_pairs);
      cache.coefficient_lookup = build_coefficient_lookup(cache.legacy_terms);
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.pair_order,
        options.seed,
        options.max_pairs,
        options.filter_left_structure,
        options.filter_right_structure);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    const Matrix full_active_overlap =
        build_full_active_overlap_matrix(active_overlap_storage, n_active_orbitals);
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    xmvb::vb::MetricAwareComponentGraphOptions graph_options;
    graph_options.edge_max_abs_threshold = options.edge_max_abs_threshold;

    int covered_pair_count = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    double max_abs_error = 0.0;
    int collapsed_exact_match_count = 0;
    int collapsed_mismatch_count = 0;
    double collapsed_max_abs_error = 0.0;
    int one_electron_exact_match_count = 0;
    int one_electron_mismatch_count = 0;
    double one_electron_max_abs_error = 0.0;
    int explicit_one_leaf_one_electron_exact_match_count = 0;
    int explicit_one_leaf_one_electron_mismatch_count = 0;
    double explicit_one_leaf_one_electron_max_abs_error = 0.0;
    int explicit_one_leaf_block_diagonal_laplace_exact_match_count = 0;
    int explicit_one_leaf_block_diagonal_laplace_mismatch_count = 0;
    double explicit_one_leaf_block_diagonal_laplace_max_abs_error = 0.0;
    int explicit_one_leaf_zeroed_block_laplace_exact_match_count = 0;
    int explicit_one_leaf_zeroed_block_laplace_mismatch_count = 0;
    double explicit_one_leaf_zeroed_block_laplace_max_abs_error = 0.0;
    int explicit_one_leaf_leaf_local_message_corrected_exact_match_count = 0;
    int explicit_one_leaf_leaf_local_message_corrected_mismatch_count = 0;
    double explicit_one_leaf_leaf_local_message_corrected_max_abs_error = 0.0;
    int explicit_one_leaf_bridge_recovered_exact_match_count = 0;
    int explicit_one_leaf_bridge_recovered_mismatch_count = 0;
    double explicit_one_leaf_bridge_recovered_max_abs_error = 0.0;
    int explicit_one_leaf_classified_total_exact_match_count = 0;
    int explicit_one_leaf_classified_total_mismatch_count = 0;
    double explicit_one_leaf_classified_total_max_abs_error = 0.0;
    double component_ordered_exact_overlap_max_abs_error = 0.0;
    double component_ordered_exact_one_electron_max_abs_error = 0.0;
    std::uint64_t total_reference_determinant_pair_count = 0;
    std::uint64_t total_local_term_pair_visits = 0;
    std::uint64_t total_subdeterminant_evaluations = 0;
    std::uint64_t total_dp_transition_count = 0;
    std::uint64_t total_binary_separator_state_count = 0;
    std::uint64_t total_ternary_separator_state_count = 0;
    std::uint64_t total_collapsed_leaf_state_count = 0;
    std::uint64_t total_collapsed_hypercube_assignment_count = 0;
    std::uint64_t total_collapsed_schur_fast_spin_assignment_count = 0;
    std::uint64_t total_collapsed_rectangular_fallback_spin_assignment_count = 0;
    std::uint64_t total_collapsed_singular_fallback_spin_assignment_count = 0;
    std::uint64_t total_collapsed_schur_fast_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_rectangular_fallback_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_singular_fallback_subdeterminant_evaluation_count = 0;
    std::uint64_t total_collapsed_subdeterminant_evaluations = 0;
    std::uint64_t total_collapsed_dp_transition_count = 0;
    std::uint64_t total_coefficient_factorization_checks = 0;
    std::uint64_t total_coefficient_factorization_mismatches = 0;
    double max_coefficient_factorization_abs_error = 0.0;
    std::uint64_t total_one_electron_collapsed_leaf_state_count = 0;
    std::uint64_t total_one_electron_hypercube_assignment_count = 0;
    std::uint64_t total_one_electron_subdeterminant_evaluations = 0;
    std::uint64_t total_one_electron_dp_transition_count = 0;
    std::uint64_t total_explicit_one_leaf_one_electron_subdeterminant_evaluations = 0;
    std::uint64_t total_explicit_one_leaf_one_electron_dp_transition_count = 0;
    double total_benchmark_exact_detpair_seconds = 0.0;
    double total_benchmark_open_state_recurrence_seconds = 0.0;
    int benchmarked_pair_count = 0;
    std::optional<FullPairMatrixBenchmarkResult> full_pair_matrix_benchmark_result;
    ExactWorkCollector work_collector;
    ExactSeparatorStateCollector exact_separator_state_collector;
    std::map<int, int> covered_width_histogram;
    std::map<int, int> covered_node_count_histogram;
    std::vector<PairExample> examples;

    for (std::size_t pair_index = 0; pair_index < pair_list.size(); ++pair_index) {
      const auto [left_structure, right_structure] = pair_list[pair_index];
      const auto& left_cache = structure_cache[xmvb::to_size(left_structure)];
      const auto& right_cache = structure_cache[xmvb::to_size(right_structure)];

      const auto support_orbitals =
          xmvb::vb::build_support_orbitals(left_cache.active_pairs, right_cache.active_pairs);
      const auto support_index =
          xmvb::vb::build_support_index(support_orbitals);
      const auto left_pairs_local =
          xmvb::vb::remap_pairs_to_support(left_cache.active_pairs, support_index);
      const auto right_pairs_local =
          xmvb::vb::remap_pairs_to_support(right_cache.active_pairs, support_index);
      const auto support_overlap = xmvb::vb::build_support_overlap_matrix(
          support_orbitals,
          active_overlap_storage,
          n_active_orbitals);
      const auto union_components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      const auto cross_blocks = xmvb::vb::summarize_union_graph_cross_blocks(
          support_overlap,
          union_components,
          options.singular_value_threshold);
      const auto metric_graph = xmvb::vb::build_metric_aware_component_graph(
          union_components,
          cross_blocks,
          graph_options);
      const auto metric_summary = xmvb::vb::summarize_metric_aware_component_graph(
          metric_graph,
          union_components);
      if (metric_summary.connected_component_count != 1) {
        continue;
      }

      int root_node = -1;
      if (!is_connected_star_graph(metric_graph, &root_node)) {
        continue;
      }
      if (metric_graph.node_count == 2) {
        const int node0_size = static_cast<int>(union_components[0].active_orbitals.size());
        const int node1_size = static_cast<int>(union_components[1].active_orbitals.size());
        if (node1_size < node0_size) {
          root_node = 1;
        }
      }

      std::vector<ComponentData> components(
          xmvb::to_size(metric_graph.node_count));
      for (int graph_node = 0; graph_node < metric_graph.node_count; ++graph_node) {
        auto& component = components[xmvb::to_size(graph_node)];
        component = build_component_data(graph_node, union_components, support_orbitals);
        component.left_orientation_terms =
            enumerate_orientation_terms(component.left_pairs);
        component.right_orientation_terms =
            enumerate_orientation_terms(component.right_pairs);
      }
      if (metric_graph.node_count == 2) {
        const auto estimate_local_term_visits = [&](int candidate_root) {
          const int candidate_leaf = 1 - candidate_root;
          const auto& root_component = components[xmvb::to_size(candidate_root)];
          const auto& leaf_component = components[xmvb::to_size(candidate_leaf)];
          const std::uint64_t root_term_pairs =
              static_cast<std::uint64_t>(root_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(root_component.right_orientation_terms.size());
          const std::uint64_t leaf_term_pairs =
              static_cast<std::uint64_t>(leaf_component.left_orientation_terms.size()) *
              static_cast<std::uint64_t>(leaf_component.right_orientation_terms.size());
          return root_term_pairs * (1ULL + leaf_term_pairs);
        };
        const std::uint64_t root0_visits = estimate_local_term_visits(0);
        const std::uint64_t root1_visits = estimate_local_term_visits(1);
        if (root1_visits < root0_visits) {
          root_node = 1;
        } else if (root0_visits < root1_visits) {
          root_node = 0;
        }
      }

      const auto exact_structure_values = compute_exact_structure_pair_values(
          left_cache.legacy_terms,
          right_cache.legacy_terms,
          active_overlap_storage,
          active_one_electron_storage,
          n_active_orbitals,
          packed_active_two_electron_integrals,
          &work_collector);
      const double exact_overlap = exact_structure_values.overlap;
      const std::uint64_t reference_count =
          static_cast<std::uint64_t>(left_cache.legacy_terms.size()) *
          static_cast<std::uint64_t>(right_cache.legacy_terms.size());
      const StarPairStats stats = evaluate_star_pair(
          exact_overlap,
          reference_count,
          active_overlap_storage,
          n_active_orbitals,
          metric_summary,
          metric_graph,
          components,
          root_node,
          left_cache.coefficient_lookup,
          right_cache.coefficient_lookup,
          overlap_resolver,
          &work_collector);
      double explicit_one_leaf_error_for_example = 0.0;
      double explicit_one_leaf_captured_for_example = 0.0;
      double explicit_one_leaf_mixed_bridge_for_example = 0.0;
      double explicit_one_leaf_classified_total_for_example = 0.0;
      double explicit_one_leaf_block_diagonal_laplace_for_example = 0.0;
      double explicit_one_leaf_zeroed_block_laplace_for_example = 0.0;
      double explicit_one_leaf_leaf_local_message_corrected_for_example = 0.0;
      double explicit_one_leaf_open_state_prototype_zeroed_closed_for_example = 0.0;
      double explicit_one_leaf_open_state_prototype_exact_cross_for_example = 0.0;
      double explicit_one_leaf_open_state_prototype_residual_closed_for_example = 0.0;
      double explicit_one_leaf_scalar_mask_closed_correction_fit_for_example = 0.0;
      double explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error_for_example =
          0.0;
      double explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error_for_example =
          0.0;
      double explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error_for_example =
          0.0;
      double explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error_for_example =
          0.0;
      double explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example =
          0.0;
      std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
          explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution_for_example;
      std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>
          explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution_for_example;
      int explicit_one_leaf_max_full_cross_message_to_root_rank_for_example = 0;
      int explicit_one_leaf_max_full_cross_root_to_message_rank_for_example = 0;
      int explicit_one_leaf_max_open_state_prototype_residual_message_rank_for_example = 0;
      int explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank_for_example = 0;
      double explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual_for_example =
          0.0;
      int explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count_for_example = 0;
      int explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count_for_example = 0;
      int explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count_for_example = 0;
      int explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count_for_example = 0;
      int explicit_one_leaf_alpha_matrix_target_span_rank_for_example = 0;
      int explicit_one_leaf_beta_matrix_target_span_rank_for_example = 0;
      int explicit_one_leaf_spin_coupled_root_channel_left_state_count_for_example = 0;
      int explicit_one_leaf_spin_coupled_root_channel_right_state_count_for_example = 0;
      int explicit_one_leaf_spin_coupled_root_channel_count_for_example = 0;
      double explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error_for_example =
          0.0;
      bool explicit_one_leaf_spin_coupled_root_channel_supported_for_example = false;
      int explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count_for_example = 0;
      int explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count_for_example = 0;
      int explicit_one_leaf_direct_spin_coupled_root_channel_count_for_example = 0;
      double explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error_for_example =
          0.0;
      double explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement_for_example =
          0.0;
      bool explicit_one_leaf_direct_spin_coupled_root_channel_supported_for_example = false;
      double explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual_for_example =
          0.0;
      double explicit_one_leaf_root_channel_swap_covariance_max_relative_residual_for_example =
          0.0;
      int explicit_one_leaf_root_channel_swap_covariance_pair_count_for_example = 0;
      double explicit_one_leaf_alpha_width2_walsh_small_component_fraction_for_example = 0.0;
      double explicit_one_leaf_beta_width2_walsh_small_component_fraction_for_example = 0.0;
      bool explicit_one_leaf_width2_root_channel_grid_detected_for_example = false;
      std::array<std::array<double, 3>, 3> explicit_one_leaf_exact_three_block_for_example = {{
          {{0.0, 0.0, 0.0}},
          {{0.0, 0.0, 0.0}},
          {{0.0, 0.0, 0.0}},
      }};
      std::array<std::array<double, 3>, 3> explicit_one_leaf_zeroed_block_three_block_for_example = {{
          {{0.0, 0.0, 0.0}},
          {{0.0, 0.0, 0.0}},
          {{0.0, 0.0, 0.0}},
      }};
      std::array<double, 4> explicit_one_leaf_exact_transfer_in_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_exact_transfer_one_exit_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_exact_transfer_zero_exit_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_exact_side_case_total_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_in_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_one_exit_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_zeroed_block_transfer_zero_exit_hist_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<double, 4> explicit_one_leaf_zeroed_block_side_case_total_for_example = {{
          0.0, 0.0, 0.0, 0.0}};
      std::array<std::array<double, 3>, 3>
          explicit_one_leaf_open_state_prototype_residual_closed_blocks_for_example = {{
              {{0.0, 0.0, 0.0}},
              {{0.0, 0.0, 0.0}},
              {{0.0, 0.0, 0.0}},
          }};
      double explicit_one_leaf_bridge_recovered_error_for_example = 0.0;
      double explicit_one_leaf_classified_total_error_for_example = 0.0;
      double explicit_one_leaf_block_diagonal_laplace_error_for_example = 0.0;
      double explicit_one_leaf_zeroed_block_laplace_error_for_example = 0.0;
      double explicit_one_leaf_leaf_local_message_corrected_error_for_example = 0.0;
      double explicit_one_leaf_scalar_mask_closed_correction_fit_error_for_example = 0.0;
      if (!options.skip_explicit_one_leaf &&
          metric_graph.adjacency[xmvb::to_size(root_node)].size() == 1) {
        const auto& root_component = components[xmvb::to_size(root_node)];
        const int leaf_node = metric_graph.adjacency[xmvb::to_size(root_node)].front();
        const auto& leaf_component = components[xmvb::to_size(leaf_node)];
        std::uint64_t explicit_one_leaf_subdeterminant_evaluations = 0;
        std::uint64_t explicit_one_leaf_dp_transition_count = 0;
        const auto explicit_one_leaf_breakdown =
            evaluate_star_pair_one_electron_explicit_one_leaf(
                active_overlap_storage,
                active_one_electron_storage,
                n_active_orbitals,
                root_component,
                leaf_component,
                left_cache.coefficient_lookup,
                right_cache.coefficient_lookup,
                overlap_resolver,
                &explicit_one_leaf_subdeterminant_evaluations,
                &explicit_one_leaf_dp_transition_count);
        const double explicit_one_leaf_error = std::abs(
            explicit_one_leaf_breakdown.captured - exact_structure_values.one_electron);
        const double explicit_one_leaf_bridge_recovered_error = std::abs(
            explicit_one_leaf_breakdown.captured +
                explicit_one_leaf_breakdown.mixed_bridge -
            exact_structure_values.one_electron);
        const double explicit_one_leaf_classified_total_error = std::abs(
            explicit_one_leaf_breakdown.classified_total -
            exact_structure_values.one_electron);
        const double explicit_one_leaf_block_diagonal_laplace_error = std::abs(
            explicit_one_leaf_breakdown.block_diagonal_laplace_total -
            exact_structure_values.one_electron);
        const double explicit_one_leaf_zeroed_block_laplace_error = std::abs(
            explicit_one_leaf_breakdown.zeroed_block_laplace_total -
            exact_structure_values.one_electron);
        const double explicit_one_leaf_leaf_local_message_corrected_error = std::abs(
            explicit_one_leaf_breakdown.leaf_local_message_corrected_total -
            exact_structure_values.one_electron);
        explicit_one_leaf_error_for_example = explicit_one_leaf_error;
        explicit_one_leaf_captured_for_example = explicit_one_leaf_breakdown.captured;
        explicit_one_leaf_mixed_bridge_for_example = explicit_one_leaf_breakdown.mixed_bridge;
        explicit_one_leaf_classified_total_for_example =
            explicit_one_leaf_breakdown.classified_total;
        explicit_one_leaf_block_diagonal_laplace_for_example =
            explicit_one_leaf_breakdown.block_diagonal_laplace_total;
        explicit_one_leaf_zeroed_block_laplace_for_example =
            explicit_one_leaf_breakdown.zeroed_block_laplace_total;
        explicit_one_leaf_leaf_local_message_corrected_for_example =
            explicit_one_leaf_breakdown.leaf_local_message_corrected_total;
        explicit_one_leaf_open_state_prototype_zeroed_closed_for_example =
            explicit_one_leaf_breakdown.open_state_prototype_zeroed_closed_total;
        explicit_one_leaf_open_state_prototype_exact_cross_for_example =
            explicit_one_leaf_breakdown.open_state_prototype_exact_cross_total;
        explicit_one_leaf_open_state_prototype_residual_closed_for_example =
            explicit_one_leaf_breakdown.open_state_prototype_residual_closed_total;
        explicit_one_leaf_scalar_mask_closed_correction_fit_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_total;
        explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.alpha_matrix_current_feature_fit_sum_frobenius_abs_error;
        explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example =
            explicit_one_leaf_breakdown
                .alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual;
        explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.beta_matrix_current_feature_fit_sum_frobenius_abs_error;
        explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example =
            explicit_one_leaf_breakdown
                .beta_matrix_current_feature_fit_max_root_pair_frobenius_residual;
        explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.alpha_matrix_full_feature_fit_sum_frobenius_abs_error;
        explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example =
            explicit_one_leaf_breakdown
                .alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual;
        explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.beta_matrix_full_feature_fit_sum_frobenius_abs_error;
        explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example =
            explicit_one_leaf_breakdown
                .beta_matrix_full_feature_fit_max_root_pair_frobenius_residual;
        explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_alpha_solution;
        explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_beta_solution;
        explicit_one_leaf_max_full_cross_message_to_root_rank_for_example =
            explicit_one_leaf_breakdown.max_full_cross_message_to_root_rank;
        explicit_one_leaf_max_full_cross_root_to_message_rank_for_example =
            explicit_one_leaf_breakdown.max_full_cross_root_to_message_rank;
        explicit_one_leaf_max_open_state_prototype_residual_message_rank_for_example =
            explicit_one_leaf_breakdown.max_open_state_prototype_residual_message_rank;
        explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank_for_example =
            explicit_one_leaf_breakdown.max_open_state_prototype_residual_leaf_leaf_rank;
        explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_max_root_pair_abs_residual;
        explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_equation_count;
        explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_unknown_count;
        explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count_for_example =
            explicit_one_leaf_breakdown.alpha_matrix_mask_closed_correction_fit_unknown_count;
        explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count_for_example =
            explicit_one_leaf_breakdown.beta_matrix_mask_closed_correction_fit_unknown_count;
        explicit_one_leaf_alpha_matrix_target_span_rank_for_example =
            explicit_one_leaf_breakdown.alpha_matrix_target_span_rank;
        explicit_one_leaf_beta_matrix_target_span_rank_for_example =
            explicit_one_leaf_breakdown.beta_matrix_target_span_rank;
        explicit_one_leaf_spin_coupled_root_channel_left_state_count_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_left_state_count;
        explicit_one_leaf_spin_coupled_root_channel_right_state_count_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_right_state_count;
        explicit_one_leaf_spin_coupled_root_channel_count_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_count;
        explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_max_frobenius_residual;
        explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_total_frobenius_abs_error;
        explicit_one_leaf_spin_coupled_root_channel_supported_for_example =
            explicit_one_leaf_breakdown.spin_coupled_root_channel_supported;
        explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_left_state_count;
        explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_right_state_count;
        explicit_one_leaf_direct_spin_coupled_root_channel_count_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_count;
        explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_max_frobenius_residual;
        explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_total_frobenius_abs_error;
        explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement_for_example =
            explicit_one_leaf_breakdown
                .direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement;
        explicit_one_leaf_direct_spin_coupled_root_channel_supported_for_example =
            explicit_one_leaf_breakdown.direct_spin_coupled_root_channel_supported;
        explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual_for_example =
            explicit_one_leaf_breakdown.root_channel_swap_covariance_max_frobenius_residual;
        explicit_one_leaf_root_channel_swap_covariance_max_relative_residual_for_example =
            explicit_one_leaf_breakdown.root_channel_swap_covariance_max_relative_residual;
        explicit_one_leaf_root_channel_swap_covariance_pair_count_for_example =
            explicit_one_leaf_breakdown.root_channel_swap_covariance_pair_count;
        explicit_one_leaf_alpha_width2_walsh_small_component_fraction_for_example =
            explicit_one_leaf_breakdown.alpha_width2_walsh_small_component_fraction;
        explicit_one_leaf_beta_width2_walsh_small_component_fraction_for_example =
            explicit_one_leaf_breakdown.beta_width2_walsh_small_component_fraction;
        explicit_one_leaf_width2_root_channel_grid_detected_for_example =
            explicit_one_leaf_breakdown.width2_root_channel_grid_detected;
        explicit_one_leaf_exact_three_block_for_example =
            explicit_one_leaf_breakdown.exact_three_block;
        explicit_one_leaf_zeroed_block_three_block_for_example =
            explicit_one_leaf_breakdown.zeroed_block_three_block;
        explicit_one_leaf_exact_transfer_in_hist_for_example =
            explicit_one_leaf_breakdown.exact_transfer_in_hist;
        explicit_one_leaf_exact_transfer_one_exit_hist_for_example =
            explicit_one_leaf_breakdown.exact_transfer_one_exit_hist;
        explicit_one_leaf_exact_transfer_zero_exit_hist_for_example =
            explicit_one_leaf_breakdown.exact_transfer_zero_exit_hist;
        explicit_one_leaf_exact_side_case_total_for_example =
            explicit_one_leaf_breakdown.exact_side_case_total;
        explicit_one_leaf_zeroed_block_transfer_in_hist_for_example =
            explicit_one_leaf_breakdown.zeroed_block_transfer_in_hist;
        explicit_one_leaf_zeroed_block_transfer_one_exit_hist_for_example =
            explicit_one_leaf_breakdown.zeroed_block_transfer_one_exit_hist;
        explicit_one_leaf_zeroed_block_transfer_zero_exit_hist_for_example =
            explicit_one_leaf_breakdown.zeroed_block_transfer_zero_exit_hist;
        explicit_one_leaf_zeroed_block_side_case_total_for_example =
            explicit_one_leaf_breakdown.zeroed_block_side_case_total;
        explicit_one_leaf_open_state_prototype_residual_closed_blocks_for_example =
            explicit_one_leaf_breakdown.open_state_prototype_residual_closed_blocks;
        explicit_one_leaf_bridge_recovered_error_for_example =
            explicit_one_leaf_bridge_recovered_error;
        explicit_one_leaf_classified_total_error_for_example =
            explicit_one_leaf_classified_total_error;
        explicit_one_leaf_block_diagonal_laplace_error_for_example =
            explicit_one_leaf_block_diagonal_laplace_error;
        explicit_one_leaf_zeroed_block_laplace_error_for_example =
            explicit_one_leaf_zeroed_block_laplace_error;
        explicit_one_leaf_leaf_local_message_corrected_error_for_example =
            explicit_one_leaf_leaf_local_message_corrected_error;
        explicit_one_leaf_scalar_mask_closed_correction_fit_error_for_example =
            explicit_one_leaf_breakdown.scalar_mask_closed_correction_fit_abs_error;
        if (options.dump_root_pair_target_matrices > 0) {
          std::cout << "root_pair_target_dump"
                    << " input_file=" << options.input_path
                    << " left_structure=" << left_structure
                    << " right_structure=" << right_structure
                    << " root_node=" << root_node
                    << " width_upper_bound=" << stats.metric_width_upper_bound
                    << " dump_count="
                    << std::min(
                           options.dump_root_pair_target_matrices,
                           static_cast<int>(
                               explicit_one_leaf_breakdown.root_pair_target_matrix_dumps.size()))
                    << '\n';
          print_root_pair_target_matrix_dumps(
              explicit_one_leaf_breakdown,
              options.dump_root_pair_target_matrices);
        }
        explicit_one_leaf_one_electron_max_abs_error = std::max(
            explicit_one_leaf_one_electron_max_abs_error,
            explicit_one_leaf_error);
        explicit_one_leaf_block_diagonal_laplace_max_abs_error = std::max(
            explicit_one_leaf_block_diagonal_laplace_max_abs_error,
            explicit_one_leaf_block_diagonal_laplace_error);
        explicit_one_leaf_zeroed_block_laplace_max_abs_error = std::max(
            explicit_one_leaf_zeroed_block_laplace_max_abs_error,
            explicit_one_leaf_zeroed_block_laplace_error);
        explicit_one_leaf_leaf_local_message_corrected_max_abs_error = std::max(
            explicit_one_leaf_leaf_local_message_corrected_max_abs_error,
            explicit_one_leaf_leaf_local_message_corrected_error);
        explicit_one_leaf_bridge_recovered_max_abs_error = std::max(
            explicit_one_leaf_bridge_recovered_max_abs_error,
            explicit_one_leaf_bridge_recovered_error);
        explicit_one_leaf_classified_total_max_abs_error = std::max(
            explicit_one_leaf_classified_total_max_abs_error,
            explicit_one_leaf_classified_total_error);
        total_explicit_one_leaf_one_electron_subdeterminant_evaluations +=
            explicit_one_leaf_subdeterminant_evaluations;
        total_explicit_one_leaf_one_electron_dp_transition_count +=
            explicit_one_leaf_dp_transition_count;
        if (explicit_one_leaf_error <= options.tolerance) {
          ++explicit_one_leaf_one_electron_exact_match_count;
        } else {
          ++explicit_one_leaf_one_electron_mismatch_count;
        }
        if (explicit_one_leaf_block_diagonal_laplace_error <= options.tolerance) {
          ++explicit_one_leaf_block_diagonal_laplace_exact_match_count;
        } else {
          ++explicit_one_leaf_block_diagonal_laplace_mismatch_count;
        }
        if (explicit_one_leaf_zeroed_block_laplace_error <= options.tolerance) {
          ++explicit_one_leaf_zeroed_block_laplace_exact_match_count;
        } else {
          ++explicit_one_leaf_zeroed_block_laplace_mismatch_count;
        }
        if (explicit_one_leaf_leaf_local_message_corrected_error <= options.tolerance) {
          ++explicit_one_leaf_leaf_local_message_corrected_exact_match_count;
        } else {
          ++explicit_one_leaf_leaf_local_message_corrected_mismatch_count;
        }
        if (explicit_one_leaf_bridge_recovered_error <= options.tolerance) {
          ++explicit_one_leaf_bridge_recovered_exact_match_count;
        } else {
          ++explicit_one_leaf_bridge_recovered_mismatch_count;
        }
        if (explicit_one_leaf_classified_total_error <= options.tolerance) {
          ++explicit_one_leaf_classified_total_exact_match_count;
        } else {
          ++explicit_one_leaf_classified_total_mismatch_count;
        }
      }

      const auto component_ordered_support_orbitals =
          build_component_ordered_support_orbitals(
              support_orbitals,
              union_components,
              root_node,
              metric_graph.adjacency[xmvb::to_size(root_node)]);
      const auto component_ordered_support_index =
          xmvb::vb::build_support_index(component_ordered_support_orbitals);
      const auto component_ordered_left_pairs_local =
          xmvb::vb::remap_pairs_to_support(
              left_cache.active_pairs,
              component_ordered_support_index);
      const auto component_ordered_right_pairs_local =
          xmvb::vb::remap_pairs_to_support(
              right_cache.active_pairs,
              component_ordered_support_index);
      const auto component_ordered_support_overlap =
          xmvb::vb::build_support_overlap_matrix(
              component_ordered_support_orbitals,
              active_overlap_storage,
              n_active_orbitals);
      const auto component_ordered_support_one_electron =
          build_support_submatrix(
              component_ordered_support_orbitals,
              active_one_electron_storage,
              n_active_orbitals);
      const auto component_ordered_support_overlap_storage =
          flatten_column_major_matrix(component_ordered_support_overlap);
      const auto component_ordered_support_one_electron_storage =
          flatten_column_major_matrix(component_ordered_support_one_electron);
      const auto component_ordered_union_components =
          xmvb::vb::build_union_graph_components(
              component_ordered_left_pairs_local,
              component_ordered_right_pairs_local,
              component_ordered_support_orbitals);
      if (component_ordered_union_components.size() != union_components.size()) {
        throw std::runtime_error(
            "component-ordered support relabeling changed the union-component count");
      }

      std::vector<ComponentData> component_ordered_components;
      component_ordered_components.reserve(component_ordered_union_components.size());
      for (std::size_t component_index = 0;
           component_index < component_ordered_union_components.size();
           ++component_index) {
        auto component = build_local_component_data(
            static_cast<int>(component_index),
            component_ordered_union_components);
        component.left_orientation_terms =
            enumerate_orientation_terms(component.left_pairs);
        component.right_orientation_terms =
            enumerate_orientation_terms(component.right_pairs);
        component_ordered_components.push_back(std::move(component));
      }

      const auto component_ordered_left_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(
              component_ordered_left_pairs_local);
      const auto component_ordered_right_terms =
          xmvb::vb::enumerate_legacy_determinant_terms(
              component_ordered_right_pairs_local);
      const auto component_ordered_exact_structure_values =
          compute_exact_structure_pair_values(
              component_ordered_left_terms,
              component_ordered_right_terms,
              component_ordered_support_overlap_storage,
              component_ordered_support_one_electron_storage,
              component_ordered_support_overlap.rows(),
              packed_active_two_electron_integrals,
              nullptr);
      const double component_ordered_exact_overlap = xmvb::vb::legacy_structure_overlap(
          component_ordered_left_terms,
          component_ordered_right_terms,
          component_ordered_support_overlap,
          overlap_resolver);
      component_ordered_exact_overlap_max_abs_error = std::max(
          component_ordered_exact_overlap_max_abs_error,
          std::abs(component_ordered_exact_overlap - exact_structure_values.overlap));
      component_ordered_exact_one_electron_max_abs_error = std::max(
          component_ordered_exact_one_electron_max_abs_error,
          std::abs(component_ordered_exact_structure_values.one_electron -
                   exact_structure_values.one_electron));
      const auto collapsed_stats = evaluate_component_ordered_collapsed_star_pair(
          component_ordered_exact_overlap,
          component_ordered_support_overlap_storage,
          component_ordered_support_overlap.rows(),
          component_ordered_components,
          component_ordered_support_orbitals,
          overlap_resolver,
          &exact_separator_state_collector);
      const auto collapsed_one_electron_stats =
          evaluate_component_ordered_open_state_star_pair_one_electron(
              component_ordered_exact_structure_values.overlap,
              component_ordered_exact_structure_values.one_electron,
              component_ordered_support_overlap_storage,
              component_ordered_support_one_electron_storage,
              component_ordered_support_overlap.rows(),
              component_ordered_components,
              overlap_resolver);
      double benchmark_exact_detpair_seconds_for_example = 0.0;
      double benchmark_open_state_recurrence_seconds_for_example = 0.0;
      double benchmark_open_state_speedup_over_detpair_for_example = 0.0;
      if (options.benchmark_repeats > 0 &&
          !options.benchmark_full_matrix_build) {
        benchmark_exact_detpair_seconds_for_example = benchmark_repeated_wall_time_seconds(
            options.benchmark_repeats,
            [&]() {
              const auto values = compute_exact_structure_pair_values_one_electron_only(
                  component_ordered_left_terms,
                  component_ordered_right_terms,
                  component_ordered_support_overlap_storage,
                  component_ordered_support_one_electron_storage,
                  component_ordered_support_overlap.rows(),
                  overlap_resolver);
              return values.overlap + values.one_electron;
            });
        benchmark_open_state_recurrence_seconds_for_example =
            benchmark_repeated_wall_time_seconds(
                options.benchmark_repeats,
                [&]() {
                  const auto values =
                      evaluate_component_ordered_open_state_star_pair_one_electron(
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          component_ordered_support_overlap_storage,
                          component_ordered_support_one_electron_storage,
                          component_ordered_support_overlap.rows(),
                          component_ordered_components,
                          overlap_resolver);
                  return values.collapsed_overlap + values.collapsed_one_electron;
                });
        if (benchmark_open_state_recurrence_seconds_for_example > 0.0) {
          benchmark_open_state_speedup_over_detpair_for_example =
              benchmark_exact_detpair_seconds_for_example /
              benchmark_open_state_recurrence_seconds_for_example;
        }
        total_benchmark_exact_detpair_seconds +=
            benchmark_exact_detpair_seconds_for_example;
        total_benchmark_open_state_recurrence_seconds +=
            benchmark_open_state_recurrence_seconds_for_example;
        ++benchmarked_pair_count;
      }

      ++covered_pair_count;
      ++covered_width_histogram[stats.metric_width_upper_bound];
      ++covered_node_count_histogram[stats.node_count];
      total_reference_determinant_pair_count += stats.reference_determinant_pair_count;
      total_local_term_pair_visits += stats.local_term_pair_visits;
      total_subdeterminant_evaluations += stats.subdeterminant_evaluations;
      total_dp_transition_count += stats.dp_transition_count;
      total_binary_separator_state_count +=
          integer_power(2, stats.metric_width_upper_bound);
      total_ternary_separator_state_count +=
          integer_power(3, stats.metric_width_upper_bound);
      total_collapsed_leaf_state_count +=
          collapsed_stats.collapsed_leaf_state_count;
      total_collapsed_hypercube_assignment_count +=
          collapsed_stats.hypercube_assignment_count;
      total_collapsed_schur_fast_spin_assignment_count +=
          collapsed_stats.schur_fast_spin_assignment_count;
      total_collapsed_rectangular_fallback_spin_assignment_count +=
          collapsed_stats.rectangular_fallback_spin_assignment_count;
      total_collapsed_singular_fallback_spin_assignment_count +=
          collapsed_stats.singular_fallback_spin_assignment_count;
      total_collapsed_schur_fast_subdeterminant_evaluation_count +=
          collapsed_stats.schur_fast_subdeterminant_evaluation_count;
      total_collapsed_rectangular_fallback_subdeterminant_evaluation_count +=
          collapsed_stats.rectangular_fallback_subdeterminant_evaluation_count;
      total_collapsed_singular_fallback_subdeterminant_evaluation_count +=
          collapsed_stats.singular_fallback_subdeterminant_evaluation_count;
      total_collapsed_subdeterminant_evaluations +=
          collapsed_stats.subdeterminant_evaluations;
      total_collapsed_dp_transition_count +=
          collapsed_stats.dp_transition_count;
      total_one_electron_collapsed_leaf_state_count +=
          collapsed_one_electron_stats.collapsed_leaf_state_count;
      total_one_electron_hypercube_assignment_count +=
          collapsed_one_electron_stats.hypercube_assignment_count;
      total_one_electron_subdeterminant_evaluations +=
          collapsed_one_electron_stats.subdeterminant_evaluations;
      total_one_electron_dp_transition_count +=
          collapsed_one_electron_stats.dp_transition_count;
      total_coefficient_factorization_checks +=
          stats.coefficient_factorization_checks;
      total_coefficient_factorization_mismatches +=
          stats.coefficient_factorization_mismatches;
      max_coefficient_factorization_abs_error = std::max(
          max_coefficient_factorization_abs_error,
          stats.max_coefficient_factorization_abs_error);
      max_abs_error = std::max(max_abs_error, stats.absolute_error);
      collapsed_max_abs_error = std::max(
          collapsed_max_abs_error,
          collapsed_stats.absolute_error);
      one_electron_max_abs_error = std::max(
          one_electron_max_abs_error,
          collapsed_one_electron_stats.one_electron_absolute_error);
      if (stats.absolute_error <= options.tolerance) {
        ++exact_match_count;
      } else {
        ++mismatch_count;
      }
      if (collapsed_stats.absolute_error <= options.tolerance) {
        ++collapsed_exact_match_count;
      } else {
        ++collapsed_mismatch_count;
      }
      if (collapsed_one_electron_stats.one_electron_absolute_error <= options.tolerance) {
        ++one_electron_exact_match_count;
      } else {
        ++one_electron_mismatch_count;
      }

      PairExample example;
      example.left_structure = left_structure;
      example.right_structure = right_structure;
      example.node_count = stats.node_count;
      example.root_node = stats.root_node;
      example.metric_width_upper_bound = stats.metric_width_upper_bound;
      example.reference_determinant_pair_count = stats.reference_determinant_pair_count;
      example.local_term_pair_visits = stats.local_term_pair_visits;
      example.subdeterminant_evaluations = stats.subdeterminant_evaluations;
      example.dp_transition_count = stats.dp_transition_count;
      example.one_electron_collapsed_leaf_state_count =
          collapsed_one_electron_stats.collapsed_leaf_state_count;
      example.one_electron_hypercube_assignment_count =
          collapsed_one_electron_stats.hypercube_assignment_count;
      example.one_electron_subdeterminant_evaluations =
          collapsed_one_electron_stats.subdeterminant_evaluations;
      example.one_electron_dp_transition_count =
          collapsed_one_electron_stats.dp_transition_count;
      example.exact_overlap = stats.exact_overlap;
      example.star_overlap = stats.star_overlap;
      example.absolute_error = stats.absolute_error;
      example.exact_one_electron = exact_structure_values.one_electron;
      example.collapsed_one_electron =
          collapsed_one_electron_stats.collapsed_one_electron;
      example.one_electron_absolute_error =
          collapsed_one_electron_stats.one_electron_absolute_error;
      example.constructed_one_electron =
          collapsed_one_electron_stats.constructed_one_electron;
      example.constructed_one_electron_absolute_error =
          collapsed_one_electron_stats.constructed_one_electron_absolute_error;
      example.benchmark_exact_detpair_seconds =
          benchmark_exact_detpair_seconds_for_example;
      example.benchmark_open_state_recurrence_seconds =
          benchmark_open_state_recurrence_seconds_for_example;
      example.benchmark_open_state_speedup_over_detpair =
          benchmark_open_state_speedup_over_detpair_for_example;
      example.explicit_one_leaf_captured = explicit_one_leaf_captured_for_example;
      example.explicit_one_leaf_mixed_bridge = explicit_one_leaf_mixed_bridge_for_example;
      example.explicit_one_leaf_classified_total =
          explicit_one_leaf_classified_total_for_example;
      example.explicit_one_leaf_block_diagonal_laplace_total =
          explicit_one_leaf_block_diagonal_laplace_for_example;
      example.explicit_one_leaf_zeroed_block_laplace_total =
          explicit_one_leaf_zeroed_block_laplace_for_example;
      example.explicit_one_leaf_leaf_local_message_corrected_total =
          explicit_one_leaf_leaf_local_message_corrected_for_example;
      example.explicit_one_leaf_open_state_prototype_zeroed_closed_total =
          explicit_one_leaf_open_state_prototype_zeroed_closed_for_example;
      example.explicit_one_leaf_open_state_prototype_exact_cross_total =
          explicit_one_leaf_open_state_prototype_exact_cross_for_example;
      example.explicit_one_leaf_open_state_prototype_residual_closed_total =
          explicit_one_leaf_open_state_prototype_residual_closed_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_total =
          explicit_one_leaf_scalar_mask_closed_correction_fit_for_example;
      example.explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error =
          explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error_for_example;
      example.explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual =
          explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example;
      example.explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error =
          explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error_for_example;
      example.explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual =
          explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual_for_example;
      example.explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error =
          explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error_for_example;
      example.explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual =
          explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example;
      example.explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error =
          explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error_for_example;
      example.explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual =
          explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution =
          explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution =
          explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution_for_example;
      example.explicit_one_leaf_max_full_cross_message_to_root_rank =
          explicit_one_leaf_max_full_cross_message_to_root_rank_for_example;
      example.explicit_one_leaf_max_full_cross_root_to_message_rank =
          explicit_one_leaf_max_full_cross_root_to_message_rank_for_example;
      example.explicit_one_leaf_max_open_state_prototype_residual_message_rank =
          explicit_one_leaf_max_open_state_prototype_residual_message_rank_for_example;
      example.explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank =
          explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual =
          explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count =
          explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count =
          explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count_for_example;
      example.explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count =
          explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count_for_example;
      example.explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count =
          explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count_for_example;
      example.explicit_one_leaf_alpha_matrix_target_span_rank =
          explicit_one_leaf_alpha_matrix_target_span_rank_for_example;
      example.explicit_one_leaf_beta_matrix_target_span_rank =
          explicit_one_leaf_beta_matrix_target_span_rank_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_left_state_count =
          explicit_one_leaf_spin_coupled_root_channel_left_state_count_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_right_state_count =
          explicit_one_leaf_spin_coupled_root_channel_right_state_count_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_count =
          explicit_one_leaf_spin_coupled_root_channel_count_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual =
          explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error =
          explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error_for_example;
      example.explicit_one_leaf_spin_coupled_root_channel_supported =
          explicit_one_leaf_spin_coupled_root_channel_supported_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count =
          explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count =
          explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_count =
          explicit_one_leaf_direct_spin_coupled_root_channel_count_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual =
          explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error =
          explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement =
          explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement_for_example;
      example.explicit_one_leaf_direct_spin_coupled_root_channel_supported =
          explicit_one_leaf_direct_spin_coupled_root_channel_supported_for_example;
      example.explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual =
          explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual_for_example;
      example.explicit_one_leaf_root_channel_swap_covariance_max_relative_residual =
          explicit_one_leaf_root_channel_swap_covariance_max_relative_residual_for_example;
      example.explicit_one_leaf_root_channel_swap_covariance_pair_count =
          explicit_one_leaf_root_channel_swap_covariance_pair_count_for_example;
      example.explicit_one_leaf_alpha_width2_walsh_small_component_fraction =
          explicit_one_leaf_alpha_width2_walsh_small_component_fraction_for_example;
      example.explicit_one_leaf_beta_width2_walsh_small_component_fraction =
          explicit_one_leaf_beta_width2_walsh_small_component_fraction_for_example;
      example.explicit_one_leaf_width2_root_channel_grid_detected =
          explicit_one_leaf_width2_root_channel_grid_detected_for_example;
      example.explicit_one_leaf_exact_three_block =
          explicit_one_leaf_exact_three_block_for_example;
      example.explicit_one_leaf_zeroed_block_three_block =
          explicit_one_leaf_zeroed_block_three_block_for_example;
      example.explicit_one_leaf_exact_transfer_in_hist =
          explicit_one_leaf_exact_transfer_in_hist_for_example;
      example.explicit_one_leaf_exact_transfer_one_exit_hist =
          explicit_one_leaf_exact_transfer_one_exit_hist_for_example;
      example.explicit_one_leaf_exact_transfer_zero_exit_hist =
          explicit_one_leaf_exact_transfer_zero_exit_hist_for_example;
      example.explicit_one_leaf_exact_side_case_total =
          explicit_one_leaf_exact_side_case_total_for_example;
      example.explicit_one_leaf_zeroed_block_transfer_in_hist =
          explicit_one_leaf_zeroed_block_transfer_in_hist_for_example;
      example.explicit_one_leaf_zeroed_block_transfer_one_exit_hist =
          explicit_one_leaf_zeroed_block_transfer_one_exit_hist_for_example;
      example.explicit_one_leaf_zeroed_block_transfer_zero_exit_hist =
          explicit_one_leaf_zeroed_block_transfer_zero_exit_hist_for_example;
      example.explicit_one_leaf_zeroed_block_side_case_total =
          explicit_one_leaf_zeroed_block_side_case_total_for_example;
      example.explicit_one_leaf_open_state_prototype_residual_closed_blocks =
          explicit_one_leaf_open_state_prototype_residual_closed_blocks_for_example;
      example.explicit_one_leaf_one_electron_absolute_error =
          explicit_one_leaf_error_for_example;
      example.explicit_one_leaf_bridge_recovered_absolute_error =
          explicit_one_leaf_bridge_recovered_error_for_example;
      example.explicit_one_leaf_classified_total_absolute_error =
          explicit_one_leaf_classified_total_error_for_example;
      example.explicit_one_leaf_block_diagonal_laplace_absolute_error =
          explicit_one_leaf_block_diagonal_laplace_error_for_example;
      example.explicit_one_leaf_zeroed_block_laplace_absolute_error =
          explicit_one_leaf_zeroed_block_laplace_error_for_example;
      example.explicit_one_leaf_leaf_local_message_corrected_absolute_error =
          explicit_one_leaf_leaf_local_message_corrected_error_for_example;
      example.explicit_one_leaf_scalar_mask_closed_correction_fit_absolute_error =
          explicit_one_leaf_scalar_mask_closed_correction_fit_error_for_example;
      examples.push_back(std::move(example));

      if (options.report_every > 0 &&
          (pair_index + 1) % xmvb::to_size(options.report_every) == 0) {
        const auto elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at).count();
        std::cerr << "progress = " << (pair_index + 1) << "/" << pair_list.size()
                  << " covered_pairs = " << covered_pair_count
                  << " elapsed_s = " << std::fixed << std::setprecision(2)
                  << elapsed_seconds << '\n';
      }
    }

    std::sort(
        examples.begin(),
        examples.end(),
        [](const PairExample& left, const PairExample& right) {
          if (left.one_electron_absolute_error != right.one_electron_absolute_error) {
            return left.one_electron_absolute_error > right.one_electron_absolute_error;
          }
          if (left.absolute_error != right.absolute_error) {
            return left.absolute_error > right.absolute_error;
          }
          const auto left_saved =
              static_cast<std::int64_t>(left.reference_determinant_pair_count) -
              static_cast<std::int64_t>(left.local_term_pair_visits);
          const auto right_saved =
              static_cast<std::int64_t>(right.reference_determinant_pair_count) -
              static_cast<std::int64_t>(right.local_term_pair_visits);
          if (left_saved != right_saved) {
            return left_saved > right_saved;
          }
          if (left.left_structure != right.left_structure) {
            return left.left_structure < right.left_structure;
          }
          return left.right_structure < right.right_structure;
        });

    if (options.benchmark_full_matrix_build) {
      full_pair_matrix_benchmark_result = benchmark_full_pair_matrix_build(
          pair_list,
          structure_cache,
          active_overlap_storage,
          active_one_electron_storage,
          n_active_orbitals,
          options.singular_value_threshold,
          options.edge_max_abs_threshold,
          options.benchmark_repeats);
    }

    const auto elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_at).count();

    std::cout << std::setprecision(16);
    std::cout << "input_file = " << options.input_path << '\n';
    std::cout << "selected_pairs = " << pair_list.size() << '\n';
    std::cout << "pair_order = " << pair_order_name(options.pair_order) << '\n';
    std::cout << "active_overlap_source = "
              << active_overlap_source_name(options.active_overlap_source) << '\n';
    std::cout << "covered_star_pair_count = " << covered_pair_count << '\n';
    std::cout << "explicit_overlap_exact_match_count = " << exact_match_count << '\n';
    std::cout << "explicit_overlap_mismatch_count = " << mismatch_count << '\n';
    std::cout << "explicit_overlap_max_abs_error = " << max_abs_error << '\n';
    std::cout << "collapsed_overlap_exact_match_count = "
              << collapsed_exact_match_count << '\n';
    std::cout << "collapsed_overlap_mismatch_count = "
              << collapsed_mismatch_count << '\n';
    std::cout << "collapsed_overlap_max_abs_error = "
              << collapsed_max_abs_error << '\n';
    std::cout << "one_electron_exact_match_count = "
              << one_electron_exact_match_count << '\n';
    std::cout << "one_electron_mismatch_count = "
              << one_electron_mismatch_count << '\n';
    std::cout << "one_electron_max_abs_error = "
              << one_electron_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_one_electron_exact_match_count = "
              << explicit_one_leaf_one_electron_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_one_electron_mismatch_count = "
              << explicit_one_leaf_one_electron_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_one_electron_max_abs_error = "
              << explicit_one_leaf_one_electron_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_block_diagonal_laplace_exact_match_count = "
              << explicit_one_leaf_block_diagonal_laplace_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_block_diagonal_laplace_mismatch_count = "
              << explicit_one_leaf_block_diagonal_laplace_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_block_diagonal_laplace_max_abs_error = "
              << explicit_one_leaf_block_diagonal_laplace_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_zeroed_block_laplace_exact_match_count = "
              << explicit_one_leaf_zeroed_block_laplace_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_zeroed_block_laplace_mismatch_count = "
              << explicit_one_leaf_zeroed_block_laplace_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_zeroed_block_laplace_max_abs_error = "
              << explicit_one_leaf_zeroed_block_laplace_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_leaf_local_message_corrected_exact_match_count = "
              << explicit_one_leaf_leaf_local_message_corrected_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_leaf_local_message_corrected_mismatch_count = "
              << explicit_one_leaf_leaf_local_message_corrected_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_leaf_local_message_corrected_max_abs_error = "
              << explicit_one_leaf_leaf_local_message_corrected_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_bridge_recovered_exact_match_count = "
              << explicit_one_leaf_bridge_recovered_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_bridge_recovered_mismatch_count = "
              << explicit_one_leaf_bridge_recovered_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_bridge_recovered_max_abs_error = "
              << explicit_one_leaf_bridge_recovered_max_abs_error << '\n';
    std::cout << "explicit_one_leaf_classified_total_exact_match_count = "
              << explicit_one_leaf_classified_total_exact_match_count << '\n';
    std::cout << "explicit_one_leaf_classified_total_mismatch_count = "
              << explicit_one_leaf_classified_total_mismatch_count << '\n';
    std::cout << "explicit_one_leaf_classified_total_max_abs_error = "
              << explicit_one_leaf_classified_total_max_abs_error << '\n';
    std::cout << "component_ordered_exact_overlap_max_abs_error = "
              << component_ordered_exact_overlap_max_abs_error << '\n';
    std::cout << "component_ordered_exact_one_electron_max_abs_error = "
              << component_ordered_exact_one_electron_max_abs_error << '\n';
    std::cout << "skip_explicit_one_leaf = "
              << (options.skip_explicit_one_leaf ? "true" : "false") << '\n';
    std::cout << "benchmark_repeats = " << options.benchmark_repeats << '\n';
    std::cout << "benchmark_full_matrix_build = "
              << (options.benchmark_full_matrix_build ? "true" : "false") << '\n';
    std::cout << "benchmarked_pair_count = " << benchmarked_pair_count << '\n';
    std::cout << "benchmark_total_exact_detpair_seconds = "
              << total_benchmark_exact_detpair_seconds << '\n';
    std::cout << "benchmark_total_open_state_recurrence_seconds = "
              << total_benchmark_open_state_recurrence_seconds << '\n';
    std::cout << "benchmark_avg_exact_detpair_seconds = "
              << ((benchmarked_pair_count == 0 || options.benchmark_repeats <= 0)
                      ? 0.0
                      : total_benchmark_exact_detpair_seconds /
                            static_cast<double>(benchmarked_pair_count) /
                            static_cast<double>(options.benchmark_repeats))
              << '\n';
    std::cout << "benchmark_avg_open_state_recurrence_seconds = "
              << ((benchmarked_pair_count == 0 || options.benchmark_repeats <= 0)
                      ? 0.0
                      : total_benchmark_open_state_recurrence_seconds /
                            static_cast<double>(benchmarked_pair_count) /
                            static_cast<double>(options.benchmark_repeats))
              << '\n';
    std::cout << "benchmark_open_state_speedup_over_detpair = "
              << ((total_benchmark_open_state_recurrence_seconds <= 0.0)
                      ? 0.0
                      : total_benchmark_exact_detpair_seconds /
                            total_benchmark_open_state_recurrence_seconds)
              << '\n';
    if (full_pair_matrix_benchmark_result.has_value()) {
      const auto& result = *full_pair_matrix_benchmark_result;
      std::cout << "full_matrix_benchmark_selected_pair_count = "
                << result.selected_pair_count << '\n';
      std::cout << "full_matrix_benchmark_covered_pair_count = "
                << result.covered_pair_count << '\n';
      std::cout << "full_matrix_benchmark_prepare_seconds = "
                << result.preparation_wall_time_seconds << '\n';
      std::cout << "full_matrix_benchmark_total_exact_detpair_seconds = "
                << result.total_exact_detpair_seconds << '\n';
      std::cout << "full_matrix_benchmark_total_open_state_recurrence_seconds = "
                << result.total_open_state_recurrence_seconds << '\n';
      std::cout << "full_matrix_benchmark_avg_exact_detpair_seconds = "
                << ((result.covered_pair_count == 0 || result.repeats <= 0)
                        ? 0.0
                        : result.total_exact_detpair_seconds /
                              static_cast<double>(result.covered_pair_count) /
                              static_cast<double>(result.repeats))
                << '\n';
      std::cout << "full_matrix_benchmark_avg_open_state_recurrence_seconds = "
                << ((result.covered_pair_count == 0 || result.repeats <= 0)
                        ? 0.0
                        : result.total_open_state_recurrence_seconds /
                              static_cast<double>(result.covered_pair_count) /
                              static_cast<double>(result.repeats))
                << '\n';
      std::cout << "full_matrix_benchmark_open_state_speedup_over_detpair = "
                << ((result.total_open_state_recurrence_seconds <= 0.0)
                        ? 0.0
                        : result.total_exact_detpair_seconds /
                              result.total_open_state_recurrence_seconds)
                << '\n';
    }
    std::cout << "prepare_active_space_wall_time_seconds = "
              << (timed_active_space.timings.orbital_preparation_wall_time_seconds +
                  timed_active_space.timings.ao_effective_one_electron_wall_time_seconds +
                  timed_active_space.timings.active_one_electron_wall_time_seconds +
                  timed_active_space.timings.active_two_electron_wall_time_seconds)
              << '\n';
    std::cout << "covered_width_histogram = {";
    bool first = true;
    for (const auto& [width, count] : covered_width_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << width << ": " << count;
    }
    std::cout << "}\n";
    std::cout << "covered_node_count_histogram = {";
    first = true;
    for (const auto& [node_count, count] : covered_node_count_histogram) {
      if (!first) {
        std::cout << ", ";
      }
      first = false;
      std::cout << node_count << ": " << count;
    }
    std::cout << "}\n";
    std::cout << "total_reference_determinant_pair_count = "
              << total_reference_determinant_pair_count << '\n';
    std::cout << "unique_reference_determinant_pair_count = "
              << work_collector.reference_full_determinant_pairs.size() << '\n';
    std::cout << "unique_exact_separator_state_count = "
              << exact_separator_state_collector.unique_exact_separator_states.size() << '\n';
    std::cout << "separator_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_exact_separator_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_leaf_message_state_count = "
              << exact_separator_state_collector.unique_leaf_message_states.size() << '\n';
    std::cout << "leaf_message_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_leaf_message_bundle_count = "
              << exact_separator_state_collector.unique_leaf_message_bundles.size() << '\n';
    std::cout << "leaf_message_bundle_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_bundles.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_exact_merge_state_count = "
              << exact_separator_state_collector.unique_merge_states.size() << '\n';
    std::cout << "merge_state_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_merge_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "layered_state_total_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_states.size() +
                            exact_separator_state_collector.unique_merge_states.size()) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    const std::size_t unique_separator_spin_kernel_count =
        work_collector.separator_root_spin_determinants.size() +
        work_collector.separator_zeroed_block_spin_determinants.size();
    std::cout << "bundle_merge_spin_kernel_total_over_unique_reference_pair_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(
                            exact_separator_state_collector.unique_leaf_message_bundles.size() +
                            exact_separator_state_collector.unique_merge_states.size() +
                            unique_separator_spin_kernel_count) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "unique_reference_spin_determinant_count = "
              << work_collector.reference_spin_determinants.size() << '\n';
    std::cout << "total_binary_separator_state_count = "
              << total_binary_separator_state_count << '\n';
    std::cout << "total_ternary_separator_state_count = "
              << total_ternary_separator_state_count << '\n';
    std::cout << "total_local_term_pair_visits = "
              << total_local_term_pair_visits << '\n';
    std::cout << "total_collapsed_leaf_state_count = "
              << total_collapsed_leaf_state_count << '\n';
    std::cout << "total_collapsed_hypercube_assignment_count = "
              << total_collapsed_hypercube_assignment_count << '\n';
    std::cout << "total_one_electron_collapsed_leaf_state_count = "
              << total_one_electron_collapsed_leaf_state_count << '\n';
    std::cout << "total_one_electron_hypercube_assignment_count = "
              << total_one_electron_hypercube_assignment_count << '\n';
    std::cout << "total_collapsed_schur_fast_spin_assignment_count = "
              << total_collapsed_schur_fast_spin_assignment_count << '\n';
    std::cout << "total_collapsed_rectangular_fallback_spin_assignment_count = "
              << total_collapsed_rectangular_fallback_spin_assignment_count << '\n';
    std::cout << "total_collapsed_singular_fallback_spin_assignment_count = "
              << total_collapsed_singular_fallback_spin_assignment_count << '\n';
    std::cout << "total_collapsed_schur_fast_subdeterminant_evaluation_count = "
              << total_collapsed_schur_fast_subdeterminant_evaluation_count << '\n';
    std::cout << "total_collapsed_rectangular_fallback_subdeterminant_evaluation_count = "
              << total_collapsed_rectangular_fallback_subdeterminant_evaluation_count << '\n';
    std::cout << "total_collapsed_singular_fallback_subdeterminant_evaluation_count = "
              << total_collapsed_singular_fallback_subdeterminant_evaluation_count << '\n';
    std::cout << "total_subdeterminant_evaluations = "
              << total_subdeterminant_evaluations << '\n';
    std::cout << "total_collapsed_subdeterminant_evaluations = "
              << total_collapsed_subdeterminant_evaluations << '\n';
    std::cout << "total_one_electron_subdeterminant_evaluations = "
              << total_one_electron_subdeterminant_evaluations << '\n';
    std::cout << "total_explicit_one_leaf_one_electron_subdeterminant_evaluations = "
              << total_explicit_one_leaf_one_electron_subdeterminant_evaluations << '\n';
    std::cout << "unique_separator_root_spin_determinant_count = "
              << work_collector.separator_root_spin_determinants.size() << '\n';
    std::cout << "unique_separator_zeroed_block_spin_determinant_count = "
              << work_collector.separator_zeroed_block_spin_determinants.size() << '\n';
    std::cout << "unique_separator_total_spin_determinant_count = "
              << (work_collector.separator_root_spin_determinants.size() +
                  work_collector.separator_zeroed_block_spin_determinants.size())
              << '\n';
    std::cout << "total_dp_transition_count = "
              << total_dp_transition_count << '\n';
    std::cout << "total_collapsed_dp_transition_count = "
              << total_collapsed_dp_transition_count << '\n';
    std::cout << "total_one_electron_dp_transition_count = "
              << total_one_electron_dp_transition_count << '\n';
    std::cout << "total_explicit_one_leaf_one_electron_dp_transition_count = "
              << total_explicit_one_leaf_one_electron_dp_transition_count << '\n';
    std::cout << "total_coefficient_factorization_checks = "
              << total_coefficient_factorization_checks << '\n';
    std::cout << "total_coefficient_factorization_mismatches = "
              << total_coefficient_factorization_mismatches << '\n';
    std::cout << "max_coefficient_factorization_abs_error = "
              << max_coefficient_factorization_abs_error << '\n';
    std::cout << "reference_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "reference_over_ternary_state_ratio = "
              << (total_ternary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_ternary_separator_state_count))
              << '\n';
    std::cout << "reference_over_local_term_pair_ratio = "
              << (total_local_term_pair_visits == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_local_term_pair_visits))
              << '\n';
    std::cout << "reference_pairs_over_unique_reference_pairs_ratio = "
              << (work_collector.reference_full_determinant_pairs.empty()
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(
                                work_collector.reference_full_determinant_pairs.size()))
              << '\n';
    std::cout << "reference_spin_over_unique_reference_spin_ratio = "
              << (work_collector.reference_spin_determinants.empty()
                      ? 0.0
                      : (2.0 * static_cast<double>(total_reference_determinant_pair_count)) /
                            static_cast<double>(
                                work_collector.reference_spin_determinants.size()))
              << '\n';
    const std::size_t unique_separator_total_spin_determinant_count =
        work_collector.separator_root_spin_determinants.size() +
        work_collector.separator_zeroed_block_spin_determinants.size();
    std::cout << "separator_spin_over_unique_separator_spin_ratio = "
              << (unique_separator_total_spin_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(total_subdeterminant_evaluations) /
                            static_cast<double>(unique_separator_total_spin_determinant_count))
              << '\n';
    std::cout << "reference_unique_spin_over_separator_unique_spin_ratio = "
              << (unique_separator_total_spin_determinant_count == 0
                      ? 0.0
                      : static_cast<double>(
                            work_collector.reference_spin_determinants.size()) /
                            static_cast<double>(unique_separator_total_spin_determinant_count))
              << '\n';
    std::cout << "local_term_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_local_term_pair_visits) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "collapsed_leaf_state_over_binary_state_ratio = "
              << (total_binary_separator_state_count == 0
                      ? 0.0
                      : static_cast<double>(total_collapsed_leaf_state_count) /
                            static_cast<double>(total_binary_separator_state_count))
              << '\n';
    std::cout << "reference_over_subdeterminant_ratio = "
              << (total_subdeterminant_evaluations == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(total_subdeterminant_evaluations))
              << '\n';
    std::cout << "reference_over_collapsed_subdeterminant_ratio = "
              << (total_collapsed_subdeterminant_evaluations == 0
                      ? 0.0
                      : static_cast<double>(total_reference_determinant_pair_count) /
                            static_cast<double>(
                                total_collapsed_subdeterminant_evaluations))
              << '\n';
    const std::uint64_t total_collapsed_classified_spin_assignment_count =
        total_collapsed_schur_fast_spin_assignment_count +
        total_collapsed_rectangular_fallback_spin_assignment_count +
        total_collapsed_singular_fallback_spin_assignment_count;
    std::cout << "collapsed_schur_fast_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_schur_fast_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    std::cout << "collapsed_rectangular_fallback_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_rectangular_fallback_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    std::cout << "collapsed_singular_fallback_spin_assignment_fraction = "
              << (total_collapsed_classified_spin_assignment_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_singular_fallback_spin_assignment_count) /
                            static_cast<double>(
                                total_collapsed_classified_spin_assignment_count))
              << '\n';
    const std::uint64_t total_collapsed_classified_subdeterminant_evaluation_count =
        total_collapsed_schur_fast_subdeterminant_evaluation_count +
        total_collapsed_rectangular_fallback_subdeterminant_evaluation_count +
        total_collapsed_singular_fallback_subdeterminant_evaluation_count;
    std::cout << "collapsed_schur_fast_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_schur_fast_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "collapsed_rectangular_fallback_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_rectangular_fallback_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "collapsed_singular_fallback_subdeterminant_fraction = "
              << (total_collapsed_classified_subdeterminant_evaluation_count == 0
                      ? 0.0
                      : static_cast<double>(
                            total_collapsed_singular_fallback_subdeterminant_evaluation_count) /
                            static_cast<double>(
                                total_collapsed_classified_subdeterminant_evaluation_count))
              << '\n';
    std::cout << "elapsed_wall_time_seconds = " << elapsed_seconds << '\n';

    const int n_examples =
        std::min(options.top_examples, static_cast<int>(examples.size()));
    std::cout << "top_examples\n";
    for (int example_index = 0; example_index < n_examples; ++example_index) {
      const auto& example = examples[xmvb::to_size(example_index)];
      const auto saved =
          static_cast<std::int64_t>(example.reference_determinant_pair_count) -
          static_cast<std::int64_t>(example.local_term_pair_visits);
      std::cout << "example[" << example_index << "]"
                << " left_structure=" << example.left_structure
                << " right_structure=" << example.right_structure
                << " node_count=" << example.node_count
                << " root_node=" << example.root_node
                << " width_upper_bound=" << example.metric_width_upper_bound
                << " reference_pair_count=" << example.reference_determinant_pair_count
                << " local_term_pair_visits=" << example.local_term_pair_visits
                << " saved=" << saved
                << " subdeterminant_evaluations=" << example.subdeterminant_evaluations
                << " dp_transitions=" << example.dp_transition_count
                << " one_electron_collapsed_leaf_state_count="
                << example.one_electron_collapsed_leaf_state_count
                << " one_electron_hypercube_assignment_count="
                << example.one_electron_hypercube_assignment_count
                << " one_electron_subdeterminant_evaluations="
                << example.one_electron_subdeterminant_evaluations
                << " one_electron_dp_transitions="
                << example.one_electron_dp_transition_count
                << " exact_overlap=" << example.exact_overlap
                << " star_overlap=" << example.star_overlap
                << " abs_error=" << example.absolute_error
                << " exact_one_electron=" << example.exact_one_electron
                << " collapsed_one_electron=" << example.collapsed_one_electron
                << " one_electron_abs_error=" << example.one_electron_absolute_error
                << " constructed_one_electron=" << example.constructed_one_electron
                << " constructed_one_electron_abs_error="
                << example.constructed_one_electron_absolute_error
                << " benchmark_exact_detpair_seconds="
                << example.benchmark_exact_detpair_seconds
                << " benchmark_open_state_recurrence_seconds="
                << example.benchmark_open_state_recurrence_seconds
                << " benchmark_open_state_speedup_over_detpair="
                << example.benchmark_open_state_speedup_over_detpair
                << " explicit_one_leaf_captured="
                << example.explicit_one_leaf_captured
                << " explicit_one_leaf_mixed_bridge="
                << example.explicit_one_leaf_mixed_bridge
                << " explicit_one_leaf_classified_total="
                << example.explicit_one_leaf_classified_total
                << " explicit_one_leaf_block_diagonal_laplace_total="
                << example.explicit_one_leaf_block_diagonal_laplace_total
                << " explicit_one_leaf_zeroed_block_laplace_total="
                << example.explicit_one_leaf_zeroed_block_laplace_total
                << " explicit_one_leaf_leaf_local_message_corrected_total="
                << example.explicit_one_leaf_leaf_local_message_corrected_total
                << " explicit_one_leaf_open_state_prototype_zeroed_closed_total="
                << example.explicit_one_leaf_open_state_prototype_zeroed_closed_total
                << " explicit_one_leaf_open_state_prototype_exact_cross_total="
                << example.explicit_one_leaf_open_state_prototype_exact_cross_total
                << " explicit_one_leaf_open_state_prototype_residual_closed_total="
                << example.explicit_one_leaf_open_state_prototype_residual_closed_total
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_total="
                << example.explicit_one_leaf_scalar_mask_closed_correction_fit_total
                << " explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error="
                << example.explicit_one_leaf_alpha_matrix_current_feature_fit_sum_frobenius_abs_error
                << " explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual="
                << example.explicit_one_leaf_alpha_matrix_current_feature_fit_max_root_pair_frobenius_residual
                << " explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error="
                << example.explicit_one_leaf_beta_matrix_current_feature_fit_sum_frobenius_abs_error
                << " explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual="
                << example.explicit_one_leaf_beta_matrix_current_feature_fit_max_root_pair_frobenius_residual
                << " explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error="
                << example.explicit_one_leaf_alpha_matrix_full_feature_fit_sum_frobenius_abs_error
                << " explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual="
                << example.explicit_one_leaf_alpha_matrix_full_feature_fit_max_root_pair_frobenius_residual
                << " explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error="
                << example.explicit_one_leaf_beta_matrix_full_feature_fit_sum_frobenius_abs_error
                << " explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual="
                << example.explicit_one_leaf_beta_matrix_full_feature_fit_max_root_pair_frobenius_residual
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution="
                << format_scalar_mask_solution_entries(
                       example.explicit_one_leaf_scalar_mask_closed_correction_fit_alpha_solution)
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution="
                << format_scalar_mask_solution_entries(
                       example.explicit_one_leaf_scalar_mask_closed_correction_fit_beta_solution)
                << " explicit_one_leaf_exact_blocks="
                << "[["
                << example.explicit_one_leaf_exact_three_block[0][0] << ","
                << example.explicit_one_leaf_exact_three_block[0][1] << ","
                << example.explicit_one_leaf_exact_three_block[0][2] << "],["
                << example.explicit_one_leaf_exact_three_block[1][0] << ","
                << example.explicit_one_leaf_exact_three_block[1][1] << ","
                << example.explicit_one_leaf_exact_three_block[1][2] << "],["
                << example.explicit_one_leaf_exact_three_block[2][0] << ","
                << example.explicit_one_leaf_exact_three_block[2][1] << ","
                << example.explicit_one_leaf_exact_three_block[2][2] << "]]"
                << " explicit_one_leaf_zeroed_block_blocks="
                << "[["
                << example.explicit_one_leaf_zeroed_block_three_block[0][0] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[0][1] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[0][2] << "],["
                << example.explicit_one_leaf_zeroed_block_three_block[1][0] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[1][1] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[1][2] << "],["
                << example.explicit_one_leaf_zeroed_block_three_block[2][0] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[2][1] << ","
                << example.explicit_one_leaf_zeroed_block_three_block[2][2] << "]]"
                << " explicit_one_leaf_transfer_hist=["
                << example.explicit_one_leaf_exact_transfer_in_hist[0] << ","
                << example.explicit_one_leaf_exact_transfer_in_hist[1] << ","
                << example.explicit_one_leaf_exact_transfer_in_hist[2] << ","
                << example.explicit_one_leaf_exact_transfer_in_hist[3] << "]"
                << " explicit_one_leaf_transfer_one_exit_hist=["
                << example.explicit_one_leaf_exact_transfer_one_exit_hist[0] << ","
                << example.explicit_one_leaf_exact_transfer_one_exit_hist[1] << ","
                << example.explicit_one_leaf_exact_transfer_one_exit_hist[2] << ","
                << example.explicit_one_leaf_exact_transfer_one_exit_hist[3] << "]"
                << " explicit_one_leaf_transfer_zero_exit_hist=["
                << example.explicit_one_leaf_exact_transfer_zero_exit_hist[0] << ","
                << example.explicit_one_leaf_exact_transfer_zero_exit_hist[1] << ","
                << example.explicit_one_leaf_exact_transfer_zero_exit_hist[2] << ","
                << example.explicit_one_leaf_exact_transfer_zero_exit_hist[3] << "]"
                << " explicit_one_leaf_side_cases=["
                << example.explicit_one_leaf_exact_side_case_total[0] << ","
                << example.explicit_one_leaf_exact_side_case_total[1] << ","
                << example.explicit_one_leaf_exact_side_case_total[2] << ","
                << example.explicit_one_leaf_exact_side_case_total[3] << "]"
                << " explicit_one_leaf_zeroed_block_transfer_hist=["
                << example.explicit_one_leaf_zeroed_block_transfer_in_hist[0] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_in_hist[1] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_in_hist[2] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_in_hist[3] << "]"
                << " explicit_one_leaf_zeroed_block_transfer_one_exit_hist=["
                << example.explicit_one_leaf_zeroed_block_transfer_one_exit_hist[0] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_one_exit_hist[1] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_one_exit_hist[2] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_one_exit_hist[3] << "]"
                << " explicit_one_leaf_zeroed_block_transfer_zero_exit_hist=["
                << example.explicit_one_leaf_zeroed_block_transfer_zero_exit_hist[0] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_zero_exit_hist[1] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_zero_exit_hist[2] << ","
                << example.explicit_one_leaf_zeroed_block_transfer_zero_exit_hist[3] << "]"
                << " explicit_one_leaf_zeroed_block_side_cases=["
                << example.explicit_one_leaf_zeroed_block_side_case_total[0] << ","
                << example.explicit_one_leaf_zeroed_block_side_case_total[1] << ","
                << example.explicit_one_leaf_zeroed_block_side_case_total[2] << ","
                << example.explicit_one_leaf_zeroed_block_side_case_total[3] << "]"
                << " explicit_one_leaf_open_state_prototype_residual_closed_blocks="
                << "[["
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[0][0]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[0][1]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[0][2]
                << "],["
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[1][0]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[1][1]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[1][2]
                << "],["
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[2][0]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[2][1]
                << ","
                << example.explicit_one_leaf_open_state_prototype_residual_closed_blocks[2][2]
                << "]]"
                << " explicit_one_leaf_max_full_cross_message_to_root_rank="
                << example.explicit_one_leaf_max_full_cross_message_to_root_rank
                << " explicit_one_leaf_max_full_cross_root_to_message_rank="
                << example.explicit_one_leaf_max_full_cross_root_to_message_rank
                << " explicit_one_leaf_max_open_state_prototype_residual_message_rank="
                << example.explicit_one_leaf_max_open_state_prototype_residual_message_rank
                << " explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank="
                << example.explicit_one_leaf_max_open_state_prototype_residual_leaf_leaf_rank
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual="
                << example.explicit_one_leaf_scalar_mask_closed_correction_fit_max_root_pair_abs_residual
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count="
                << example.explicit_one_leaf_scalar_mask_closed_correction_fit_equation_count
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count="
                << example.explicit_one_leaf_scalar_mask_closed_correction_fit_unknown_count
                << " explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count="
                << example.explicit_one_leaf_alpha_matrix_mask_closed_correction_fit_unknown_count
                << " explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count="
                << example.explicit_one_leaf_beta_matrix_mask_closed_correction_fit_unknown_count
                << " explicit_one_leaf_alpha_matrix_target_span_rank="
                << example.explicit_one_leaf_alpha_matrix_target_span_rank
                << " explicit_one_leaf_beta_matrix_target_span_rank="
                << example.explicit_one_leaf_beta_matrix_target_span_rank
                << " explicit_one_leaf_spin_coupled_root_channel_left_state_count="
                << example.explicit_one_leaf_spin_coupled_root_channel_left_state_count
                << " explicit_one_leaf_spin_coupled_root_channel_right_state_count="
                << example.explicit_one_leaf_spin_coupled_root_channel_right_state_count
                << " explicit_one_leaf_spin_coupled_root_channel_count="
                << example.explicit_one_leaf_spin_coupled_root_channel_count
                << " explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual="
                << example.explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual
                << " explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error="
                << example.explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error
                << " explicit_one_leaf_spin_coupled_root_channel_supported="
                << (example.explicit_one_leaf_spin_coupled_root_channel_supported ? "true"
                                                                                 : "false")
                << " explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count
                << " explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count
                << " explicit_one_leaf_direct_spin_coupled_root_channel_count="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_count
                << " explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual
                << " explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error
                << " explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement="
                << example.explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement
                << " explicit_one_leaf_direct_spin_coupled_root_channel_supported="
                << (example.explicit_one_leaf_direct_spin_coupled_root_channel_supported ? "true"
                                                                                        : "false")
                << " explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual="
                << example.explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual
                << " explicit_one_leaf_root_channel_swap_covariance_max_relative_residual="
                << example.explicit_one_leaf_root_channel_swap_covariance_max_relative_residual
                << " explicit_one_leaf_root_channel_swap_covariance_pair_count="
                << example.explicit_one_leaf_root_channel_swap_covariance_pair_count
                << " explicit_one_leaf_alpha_width2_walsh_small_component_fraction="
                << example.explicit_one_leaf_alpha_width2_walsh_small_component_fraction
                << " explicit_one_leaf_beta_width2_walsh_small_component_fraction="
                << example.explicit_one_leaf_beta_width2_walsh_small_component_fraction
                << " explicit_one_leaf_width2_root_channel_grid_detected="
                << (example.explicit_one_leaf_width2_root_channel_grid_detected ? "true" : "false")
                << " explicit_one_leaf_one_electron_abs_error="
                << example.explicit_one_leaf_one_electron_absolute_error
                << " explicit_one_leaf_bridge_recovered_abs_error="
                << example.explicit_one_leaf_bridge_recovered_absolute_error
                << " explicit_one_leaf_classified_total_abs_error="
                << example.explicit_one_leaf_classified_total_absolute_error
                << " explicit_one_leaf_block_diagonal_laplace_abs_error="
                << example.explicit_one_leaf_block_diagonal_laplace_absolute_error
                << " explicit_one_leaf_zeroed_block_laplace_abs_error="
                << example.explicit_one_leaf_zeroed_block_laplace_absolute_error
                << " explicit_one_leaf_leaf_local_message_corrected_abs_error="
                << example.explicit_one_leaf_leaf_local_message_corrected_absolute_error
                << " explicit_one_leaf_scalar_mask_closed_correction_fit_abs_error="
                << example.explicit_one_leaf_scalar_mask_closed_correction_fit_absolute_error
                << '\n';
    }
    return (mismatch_count == 0 &&
            collapsed_mismatch_count == 0 &&
            one_electron_mismatch_count == 0)
        ? 0
        : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
