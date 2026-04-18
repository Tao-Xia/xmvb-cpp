#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb {

using OrbitalPair = std::pair<int, int>;

struct UnionGraphComponent {
  int index = 0;
  std::vector<int> local_vertices;
  std::vector<int> active_orbitals;
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::string type;
};

struct UnionGraphCrossBlockSummary {
  int left_component = 0;
  int right_component = 0;
  double frobenius_norm = 0.0;
  double spectral_norm = 0.0;
  double max_abs = 0.0;
  int numerical_rank = 0;
  int n_rows = 0;
  int n_columns = 0;
  std::vector<double> singular_values;
};

struct UnionGraphScreeningSummary {
  int support_size = 0;
  int component_count = 0;
  int cross_block_count = 0;
  int n_single_vertex_components = 0;
  int n_doubled_edge_components = 0;
  int n_alternating_cycle_components = 0;
  int n_general_components = 0;
  int max_component_size = 0;
  int max_numerical_rank = 0;
  double support_overlap_frobenius = 0.0;
  double offblock_overlap_frobenius = 0.0;
  double offblock_overlap_fraction = 0.0;
  double max_cross_block_frobenius = 0.0;
  double max_cross_block_spectral = 0.0;
  double max_cross_block_max_abs = 0.0;
  double max_cross_block_second_singular = 0.0;
  double max_cross_block_third_singular = 0.0;
  double sum_cross_block_frobenius = 0.0;
  std::string component_signature;
};

struct MetricAwareComponentGraphOptions {
  // Cross-block entries with max absolute value at or below this threshold are
  // dropped from the graph. Use 0.0 for the exact nonzero-pattern graph.
  double edge_max_abs_threshold = 0.0;
};

struct MetricAwareComponentGraphEdge {
  int left_component = 0;
  int right_component = 0;
  double frobenius_norm = 0.0;
  double spectral_norm = 0.0;
  double max_abs = 0.0;
  int numerical_rank = 0;
};

struct MetricAwareConnectedComponent {
  int index = 0;
  std::vector<int> graph_nodes;
  int total_active_orbitals = 0;
  int total_left_pairs = 0;
  int total_right_pairs = 0;
  int total_covalent_labels = 0;
};

struct MetricAwareSeparatorCandidate {
  int articulation_component = -1;
  int resulting_component_count = 0;
  int largest_piece_node_count = 0;
  int largest_piece_covalent_labels = 0;
};

struct MetricAwareComponentGraph {
  int node_count = 0;
  // Weight of one graph node in "exact bond labels", defined as the number of
  // covalent left pairs plus covalent right pairs carried by the union
  // component.
  std::vector<int> component_covalent_label_counts;
  std::vector<MetricAwareComponentGraphEdge> edges;
  std::vector<std::vector<int>> adjacency;
};

struct MetricAwareGraphSummary {
  int node_count = 0;
  int edge_count = 0;
  int connected_component_count = 0;
  int articulation_count = 0;
  int max_degree = 0;
  int max_component_node_count = 0;
  int max_component_covalent_labels = 0;
  // Min-degree elimination on the weighted component graph. This is only an
  // upper-bound surrogate for the exact separator width, not a proof of
  // optimal treewidth.
  int weighted_min_degree_width_upper_bound = 0;
  std::vector<int> weighted_min_degree_order;
  std::vector<MetricAwareConnectedComponent> connected_components;
  std::vector<MetricAwareSeparatorCandidate> articulation_candidates;
};

std::vector<OrbitalPair> extract_active_pairs(
    const RawStructureData& raw_structure_data,
    int structure_index);

std::vector<int> build_support_orbitals(
    const std::vector<OrbitalPair>& left_pairs,
    const std::vector<OrbitalPair>& right_pairs);

std::map<int, int> build_support_index(
    const std::vector<int>& support_orbitals);

std::vector<OrbitalPair> remap_pairs_to_support(
    const std::vector<OrbitalPair>& pairs,
    const std::map<int, int>& support_index);

Eigen::MatrixXd build_support_overlap_matrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& active_overlap_matrix,
    int n_active_orbitals);

std::vector<UnionGraphComponent> build_union_graph_components(
    const std::vector<OrbitalPair>& left_pairs,
    const std::vector<OrbitalPair>& right_pairs,
    const std::vector<int>& support_orbitals);

std::vector<UnionGraphCrossBlockSummary> summarize_union_graph_cross_blocks(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components,
    double singular_value_threshold);

UnionGraphScreeningSummary summarize_union_graph_screening(
    const Eigen::MatrixXd& support_overlap,
    const Eigen::MatrixXd& offblock_overlap,
    const std::vector<UnionGraphComponent>& components,
    const std::vector<UnionGraphCrossBlockSummary>& cross_blocks);

Eigen::MatrixXd build_block_diagonalized_support_overlap(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components);

Eigen::MatrixXd build_offblock_support_overlap(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components);

Eigen::MatrixXd build_blockwise_truncated_offblock(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components,
    int rank_cap);

MetricAwareComponentGraph build_metric_aware_component_graph(
    const std::vector<UnionGraphComponent>& components,
    const std::vector<UnionGraphCrossBlockSummary>& cross_blocks,
    MetricAwareComponentGraphOptions options = {});

MetricAwareGraphSummary summarize_metric_aware_component_graph(
    const MetricAwareComponentGraph& graph,
    const std::vector<UnionGraphComponent>& components);

}  // namespace xmvb::vb
