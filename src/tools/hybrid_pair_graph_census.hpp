#pragma once

#include <vector>

namespace xmvb::tools {

/**
 * @brief Numerical census for a Woodbury base with an inverse-free small core.
 *
 * The diagnostic constructs a certified stable completion of every occupied
 * overlap block.  Singular or ill-conditioned directions remain in a small
 * inverse-free core, while consecutive stable completions are connected by a
 * block-Woodbury update.  No electronic integral is evaluated.
 */
struct HybridPairGraphCensus {
  long long visited_pairs = 0;
  long long certified_regular_pairs = 0;
  long long uncertified_stable_bases = 0;
  long long low_rank_base_updates = 0;
  long long full_rank_base_updates = 0;
  long long failed_base_updates = 0;
  long long topology_breaks = 0;
  long long numerical_reanchors = 0;
  long long interpolation_node_equivalents = 0;
  std::vector<long long> core_rank_counts;
  std::vector<long long> base_update_rank_counts;
  double mean_core_rank = 0.0;
  double mean_base_update_rank = 0.0;
  double max_reconstruction_relative_error = 0.0;
  double max_base_inverse_backward_error = 0.0;
  double max_propagated_inverse_relative_error = 0.0;
};

HybridPairGraphCensus run_hybrid_pair_graph_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<int>& traversal_order,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    int max_pairs);

void print_hybrid_pair_graph_census(
    const char* label,
    const HybridPairGraphCensus& census);

}  // namespace xmvb::tools
