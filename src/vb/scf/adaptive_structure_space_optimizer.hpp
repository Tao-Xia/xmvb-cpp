#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace xmvb::vb {

enum class AdaptiveDeterminantScoreMode {
  ProposalAll,
  OutsideOnly,
};

inline const char* adaptive_determinant_score_mode_name(
    AdaptiveDeterminantScoreMode mode) {
  switch (mode) {
    case AdaptiveDeterminantScoreMode::ProposalAll:
      return "proposal_all";
    case AdaptiveDeterminantScoreMode::OutsideOnly:
      return "outside_only";
  }
  return "unknown";
}

struct AdaptiveStructureSpaceIterationSummary {
  int outer_iteration_index = 0;
  int selected_raw_structure_count = 0;
  int expanded_determinant_count = 0;
  int proposal_count = 0;
  int unique_proposal_determinant_count = 0;
  int unique_outside_determinant_count = 0;
  int unique_shared_determinant_count = 0;
  int added_structure_count = 0;
  int inner_iterations = 0;
  double best_candidate_score = 0.0;
  double total_energy = 0.0;
  std::uint64_t boundary_pair_evaluation_count = 0;
  double scoring_wall_time_seconds = 0.0;
  double outer_iteration_wall_time_seconds = 0.0;
};

struct AdaptiveStructureSpaceOptimizerOptions {
  RawStructureSelectionMode seed_selection =
      RawStructureSelectionMode::Covalent;
  AdaptiveDeterminantScoreMode determinant_score_mode =
      AdaptiveDeterminantScoreMode::ProposalAll;
  int max_outer_iterations = 8;
  int max_topology_distance = 1;
  int max_neighbors_per_structure = 8;
  int max_candidate_pool_size = 64;
  int batch_size = 4;
  int max_total_structures = 256;
  double minimum_candidate_score = 1.0e-6;
  bool verbose = true;
};

struct AdaptiveStructureSpaceOptimizerResult {
  bool converged = false;
  std::string termination_reason;
  std::vector<int> selected_raw_structure_indices;
  std::vector<AdaptiveStructureSpaceIterationSummary> iteration_summaries;
  double total_wall_time_seconds = 0.0;
  CppVbScfOptimizerResult inner_result;
};

inline const char* adaptive_structure_space_seed_selection_name(
    RawStructureSelectionMode mode) {
  return raw_structure_selection_mode_name(mode);
}

class AdaptiveStructureSpaceOptimizer {
public:
  AdaptiveStructureSpaceOptimizer(
      CppVbScfOptimizerOptions optimizer_options,
      AdaptiveStructureSpaceOptimizerOptions options = {});

  AdaptiveStructureSpaceOptimizerResult optimize(
      const CppVbInputLoadResult& load_result) const;

private:
  CppVbScfOptimizerOptions optimizer_options_;
  AdaptiveStructureSpaceOptimizerOptions options_;
};

}  // namespace xmvb::vb
