#include "vb/matrices/union_graph_screening.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/SVD>

namespace xmvb::vb {

namespace {

Eigen::MatrixXd extract_cross_block(
    const Eigen::MatrixXd& matrix,
    const UnionGraphComponent& left_component,
    const UnionGraphComponent& right_component) {
  Eigen::MatrixXd block(
      static_cast<int>(left_component.local_vertices.size()),
      static_cast<int>(right_component.local_vertices.size()));
  for (int row = 0; row < block.rows(); ++row) {
    for (int column = 0; column < block.cols(); ++column) {
      block(row, column) = matrix(
          left_component.local_vertices[row],
          right_component.local_vertices[column]);
    }
  }
  return block;
}

void assign_directed_cross_block(
    const Eigen::MatrixXd& block,
    const UnionGraphComponent& left_component,
    const UnionGraphComponent& right_component,
    Eigen::MatrixXd* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  for (int row = 0; row < block.rows(); ++row) {
    for (int column = 0; column < block.cols(); ++column) {
      (*destination)(
          left_component.local_vertices[row],
          right_component.local_vertices[column]) =
          block(row, column);
    }
  }
}

Eigen::MatrixXd truncate_block_rank(
    const Eigen::MatrixXd& block,
    int rank_cap) {
  if (rank_cap <= 0 || block.rows() == 0 || block.cols() == 0) {
    return Eigen::MatrixXd::Zero(block.rows(), block.cols());
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(block, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const int retained_rank = std::min(rank_cap, static_cast<int>(svd.singularValues().size()));
  if (retained_rank <= 0) {
    return Eigen::MatrixXd::Zero(block.rows(), block.cols());
  }

  return svd.matrixU().leftCols(retained_rank) *
      svd.singularValues().head(retained_rank).asDiagonal() *
      svd.matrixV().leftCols(retained_rank).transpose();
}

std::string build_component_signature(
    const std::vector<UnionGraphComponent>& components) {
  std::vector<std::string> labels;
  labels.reserve(components.size());
  for (const auto& component : components) {
    std::ostringstream label;
    label << component.type << ":" << component.active_orbitals.size();
    labels.push_back(label.str());
  }
  std::sort(labels.begin(), labels.end());

  std::ostringstream signature;
  for (std::size_t label_index = 0; label_index < labels.size(); ++label_index) {
    if (label_index > 0) {
      signature << "+";
    }
    signature << labels[label_index];
  }
  return signature.str();
}

UnionGraphCrossBlockSummary summarize_cross_block(
    const Eigen::MatrixXd& support_overlap,
    const UnionGraphComponent& left_component,
    const UnionGraphComponent& right_component,
    double singular_value_threshold) {
  UnionGraphCrossBlockSummary summary;
  summary.left_component = left_component.index;
  summary.right_component = right_component.index;
  summary.n_rows = static_cast<int>(left_component.local_vertices.size());
  summary.n_columns = static_cast<int>(right_component.local_vertices.size());

  const Eigen::MatrixXd forward_block = extract_cross_block(
      support_overlap,
      left_component,
      right_component);
  const Eigen::MatrixXd reverse_block = extract_cross_block(
      support_overlap,
      right_component,
      left_component);

  // The support-overlap matrix used by the determinant kernels is not
  // guaranteed to be symmetric in storage. For graph construction we therefore
  // need an undirected coupling summary: if either directed block is nonzero,
  // the two union components interact and must stay connected in the
  // metric-aware graph.
  summary.frobenius_norm = std::sqrt(
      forward_block.squaredNorm() + reverse_block.squaredNorm());
  const double forward_max_abs =
      summary.n_rows == 0 || summary.n_columns == 0
          ? 0.0
          : forward_block.cwiseAbs().maxCoeff();
  const double reverse_max_abs =
      summary.n_rows == 0 || summary.n_columns == 0
          ? 0.0
          : reverse_block.cwiseAbs().maxCoeff();
  summary.max_abs = std::max(forward_max_abs, reverse_max_abs);

  auto accumulate_singular_values =
      [&](const Eigen::MatrixXd& block,
          double* spectral_norm,
          int* numerical_rank,
          std::vector<double>* singular_values) {
        if (block.rows() == 0 || block.cols() == 0) {
          return;
        }
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(block, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto values = svd.singularValues();
        if (values.size() == 0) {
          return;
        }
        *spectral_norm = std::max(*spectral_norm, values(0));
        int directed_rank = 0;
        for (int singular_index = 0; singular_index < values.size(); ++singular_index) {
          singular_values->push_back(values(singular_index));
          if (values(singular_index) > singular_value_threshold) {
            ++directed_rank;
          }
        }
        *numerical_rank = std::max(*numerical_rank, directed_rank);
      };

  accumulate_singular_values(
      forward_block,
      &summary.spectral_norm,
      &summary.numerical_rank,
      &summary.singular_values);
  accumulate_singular_values(
      reverse_block,
      &summary.spectral_norm,
      &summary.numerical_rank,
      &summary.singular_values);
  std::sort(summary.singular_values.begin(), summary.singular_values.end(), std::greater<double>());
  return summary;
}

int count_component_covalent_labels(const UnionGraphComponent& component) {
  int covalent_labels = 0;
  for (const auto& pair : component.left_pairs) {
    if (pair.first != pair.second) {
      ++covalent_labels;
    }
  }
  for (const auto& pair : component.right_pairs) {
    if (pair.first != pair.second) {
      ++covalent_labels;
    }
  }
  return covalent_labels;
}

std::vector<std::vector<int>> build_metric_aware_adjacency(
    int node_count,
    const std::vector<MetricAwareComponentGraphEdge>& edges) {
  std::vector<std::vector<int>> adjacency(node_count);
  for (const auto& edge : edges) {
    adjacency[edge.left_component].push_back(edge.right_component);
    adjacency[edge.right_component].push_back(edge.left_component);
  }
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  return adjacency;
}

MetricAwareConnectedComponent summarize_metric_connected_component(
    int component_index,
    const std::vector<int>& graph_nodes,
    const std::vector<UnionGraphComponent>& components,
    const MetricAwareComponentGraph& graph) {
  MetricAwareConnectedComponent summary;
  summary.index = component_index;
  summary.graph_nodes = graph_nodes;
  for (const int graph_node : graph_nodes) {
    const auto& union_component = components[graph_node];
    summary.total_active_orbitals +=
        static_cast<int>(union_component.active_orbitals.size());
    summary.total_left_pairs +=
        static_cast<int>(union_component.left_pairs.size());
    summary.total_right_pairs +=
        static_cast<int>(union_component.right_pairs.size());
    summary.total_covalent_labels +=
        graph.component_covalent_label_counts[graph_node];
  }
  return summary;
}

std::vector<MetricAwareConnectedComponent> build_metric_connected_components(
    const MetricAwareComponentGraph& graph,
    const std::vector<UnionGraphComponent>& components,
    int removed_node) {
  std::vector<MetricAwareConnectedComponent> connected_components;
  std::vector<int> component_id(graph.node_count, -1);
  for (int start_node = 0; start_node < graph.node_count; ++start_node) {
    if (start_node == removed_node ||
        component_id[start_node] >= 0) {
      continue;
    }

    std::vector<int> stack{start_node};
    std::vector<int> graph_nodes;
    component_id[start_node] =
        static_cast<int>(connected_components.size());
    while (!stack.empty()) {
      const int node = stack.back();
      stack.pop_back();
      graph_nodes.push_back(node);
      for (const int neighbor : graph.adjacency[node]) {
        if (neighbor == removed_node ||
            component_id[neighbor] >= 0) {
          continue;
        }
        component_id[neighbor] =
            static_cast<int>(connected_components.size());
        stack.push_back(neighbor);
      }
    }
    std::sort(graph_nodes.begin(), graph_nodes.end());
    connected_components.push_back(summarize_metric_connected_component(
        static_cast<int>(connected_components.size()),
        graph_nodes,
        components,
        graph));
  }

  std::sort(
      connected_components.begin(),
      connected_components.end(),
      [](const MetricAwareConnectedComponent& left,
         const MetricAwareConnectedComponent& right) {
        if (left.graph_nodes.size() != right.graph_nodes.size()) {
          return left.graph_nodes.size() > right.graph_nodes.size();
        }
        if (left.total_covalent_labels != right.total_covalent_labels) {
          return left.total_covalent_labels > right.total_covalent_labels;
        }
        return left.graph_nodes < right.graph_nodes;
      });
  for (int component_index = 0;
       component_index < static_cast<int>(connected_components.size());
       ++component_index) {
    connected_components[component_index].index = component_index;
  }
  return connected_components;
}

MetricAwareSeparatorCandidate summarize_articulation_candidate(
    int removed_node,
    const std::vector<MetricAwareConnectedComponent>& connected_components) {
  MetricAwareSeparatorCandidate candidate;
  candidate.articulation_component = removed_node;
  candidate.resulting_component_count = static_cast<int>(connected_components.size());
  for (const auto& connected_component : connected_components) {
    candidate.largest_piece_node_count = std::max(
        candidate.largest_piece_node_count,
        static_cast<int>(connected_component.graph_nodes.size()));
    candidate.largest_piece_covalent_labels = std::max(
        candidate.largest_piece_covalent_labels,
        connected_component.total_covalent_labels);
  }
  return candidate;
}

std::pair<int, std::vector<int>> build_weighted_min_degree_order(
    const MetricAwareComponentGraph& graph) {
  const int node_count = graph.node_count;
  std::vector<std::vector<char>> live_adjacency(
      node_count,
      std::vector<char>(node_count, 0));
  for (const auto& edge : graph.edges) {
    live_adjacency[edge.left_component]
                  [edge.right_component] = 1;
    live_adjacency[edge.right_component]
                  [edge.left_component] = 1;
  }

  std::vector<char> eliminated(node_count, 0);
  std::vector<int> elimination_order;
  elimination_order.reserve(node_count);
  int width_upper_bound = 0;
  for (int step = 0; step < node_count; ++step) {
    int best_node = -1;
    int best_weighted_degree = std::numeric_limits<int>::max();
    int best_degree = std::numeric_limits<int>::max();
    for (int node = 0; node < node_count; ++node) {
      if (eliminated[node] != 0) {
        continue;
      }
      int weighted_degree = 0;
      int degree = 0;
      for (int neighbor = 0; neighbor < node_count; ++neighbor) {
        if (eliminated[neighbor] != 0 ||
            live_adjacency[node]
                          [neighbor] == 0) {
          continue;
        }
        weighted_degree +=
            graph.component_covalent_label_counts[neighbor];
        ++degree;
      }
      if (weighted_degree < best_weighted_degree ||
          (weighted_degree == best_weighted_degree && degree < best_degree) ||
          (weighted_degree == best_weighted_degree && degree == best_degree &&
           (best_node < 0 || node < best_node))) {
        best_node = node;
        best_weighted_degree = weighted_degree;
        best_degree = degree;
      }
    }

    if (best_node < 0) {
      throw std::runtime_error("failed to select a node for weighted min-degree elimination");
    }

    width_upper_bound = std::max(width_upper_bound, best_weighted_degree);
    elimination_order.push_back(best_node);

    std::vector<int> live_neighbors;
    for (int neighbor = 0; neighbor < node_count; ++neighbor) {
      if (eliminated[neighbor] != 0 ||
          live_adjacency[best_node]
                        [neighbor] == 0) {
        continue;
      }
      live_neighbors.push_back(neighbor);
    }

    // Fill in a clique on the live neighbors before eliminating the node.
    for (std::size_t left_index = 0; left_index < live_neighbors.size(); ++left_index) {
      for (std::size_t right_index = left_index + 1;
           right_index < live_neighbors.size();
           ++right_index) {
        const int left_neighbor = live_neighbors[left_index];
        const int right_neighbor = live_neighbors[right_index];
        live_adjacency[left_neighbor]
                      [right_neighbor] = 1;
        live_adjacency[right_neighbor]
                      [left_neighbor] = 1;
      }
    }

    for (int neighbor = 0; neighbor < node_count; ++neighbor) {
      live_adjacency[best_node]
                    [neighbor] = 0;
      live_adjacency[neighbor]
                    [best_node] = 0;
    }
    eliminated[best_node] = 1;
  }
  return {width_upper_bound, elimination_order};
}

}  // namespace

std::vector<OrbitalPair> extract_active_pairs(
    const RawStructureData& raw_structure_data,
    int structure_index) {
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  const int active_stop = active_start + raw_structure_data.n_active_electrons;
  if (structure_index < 0 || structure_index >= raw_structure_data.n_structures) {
    throw std::out_of_range("structure index out of range");
  }
  if (active_start < 0 || active_stop > raw_structure_data.n_total_electrons) {
    throw std::runtime_error("active-electron window is out of range for raw structures");
  }

  const int* structure_orbitals = raw_structure_data.structure_orbitals_data(structure_index);
  std::vector<OrbitalPair> pairs;
  pairs.reserve(n_active_beta_electrons);
  for (int pair_index = 0; pair_index < n_active_beta_electrons; ++pair_index) {
    const int left_orbital =
        structure_orbitals[active_start + 2 * pair_index] -
        n_inactive_doubly_occupied_orbitals - 1;
    const int right_orbital =
        structure_orbitals[active_start + 2 * pair_index + 1] -
        n_inactive_doubly_occupied_orbitals - 1;
    pairs.emplace_back(left_orbital, right_orbital);
  }
  return pairs;
}

std::vector<int> build_support_orbitals(
    const std::vector<OrbitalPair>& left_pairs,
    const std::vector<OrbitalPair>& right_pairs) {
  std::vector<int> support_orbitals;
  for (const auto& pair : left_pairs) {
    support_orbitals.push_back(pair.first);
    support_orbitals.push_back(pair.second);
  }
  for (const auto& pair : right_pairs) {
    support_orbitals.push_back(pair.first);
    support_orbitals.push_back(pair.second);
  }
  std::sort(support_orbitals.begin(), support_orbitals.end());
  support_orbitals.erase(
      std::unique(support_orbitals.begin(), support_orbitals.end()),
      support_orbitals.end());
  return support_orbitals;
}

std::map<int, int> build_support_index(
    const std::vector<int>& support_orbitals) {
  std::map<int, int> support_index;
  for (int support_position = 0; support_position < static_cast<int>(support_orbitals.size());
       ++support_position) {
    support_index.emplace(
        support_orbitals[support_position],
        support_position);
  }
  return support_index;
}

std::vector<OrbitalPair> remap_pairs_to_support(
    const std::vector<OrbitalPair>& pairs,
    const std::map<int, int>& support_index) {
  std::vector<OrbitalPair> remapped_pairs;
  remapped_pairs.reserve(pairs.size());
  for (const auto& pair : pairs) {
    const auto left_iterator = support_index.find(pair.first);
    const auto right_iterator = support_index.find(pair.second);
    if (left_iterator == support_index.end() || right_iterator == support_index.end()) {
      throw std::runtime_error("support orbital missing during pair remap");
    }
    remapped_pairs.emplace_back(left_iterator->second, right_iterator->second);
  }
  return remapped_pairs;
}

Eigen::MatrixXd build_support_overlap_matrix(
    const std::vector<int>& support_orbitals,
    const std::vector<double>& active_overlap_matrix,
    int n_active_orbitals) {
  const int support_size = static_cast<int>(support_orbitals.size());
  Eigen::MatrixXd support_overlap(support_size, support_size);
  for (int row = 0; row < support_size; ++row) {
    for (int column = 0; column < support_size; ++column) {
      support_overlap(row, column) =
          active_overlap_matrix[support_orbitals[column] *
                                    n_active_orbitals +
                                support_orbitals[row]];
    }
  }
  return support_overlap;
}

std::vector<UnionGraphComponent> build_union_graph_components(
    const std::vector<OrbitalPair>& left_pairs,
    const std::vector<OrbitalPair>& right_pairs,
    const std::vector<int>& support_orbitals) {
  const int support_size = static_cast<int>(support_orbitals.size());
  std::vector<std::vector<int>> adjacency(support_size);
  auto add_edge = [&](const OrbitalPair& pair) {
    if (pair.first == pair.second) {
      return;
    }
    adjacency[pair.first].push_back(pair.second);
    adjacency[pair.second].push_back(pair.first);
  };
  for (const auto& pair : left_pairs) {
    add_edge(pair);
  }
  for (const auto& pair : right_pairs) {
    add_edge(pair);
  }

  std::vector<int> component_id(support_size, -1);
  std::vector<UnionGraphComponent> components;
  for (int start_vertex = 0; start_vertex < support_size; ++start_vertex) {
    if (component_id[start_vertex] >= 0) {
      continue;
    }

    UnionGraphComponent component;
    component.index = static_cast<int>(components.size());
    std::vector<int> stack{start_vertex};
    component_id[start_vertex] = component.index;
    while (!stack.empty()) {
      const int vertex = stack.back();
      stack.pop_back();
      component.local_vertices.push_back(vertex);
      component.active_orbitals.push_back(support_orbitals[vertex]);
      for (const int neighbor : adjacency[vertex]) {
        if (component_id[neighbor] >= 0) {
          continue;
        }
        component_id[neighbor] = component.index;
        stack.push_back(neighbor);
      }
    }
    std::sort(component.local_vertices.begin(), component.local_vertices.end());
    std::sort(component.active_orbitals.begin(), component.active_orbitals.end());
    components.push_back(std::move(component));
  }

  for (const auto& pair : left_pairs) {
    components[component_id[pair.first]]
        .left_pairs.push_back(pair);
  }
  for (const auto& pair : right_pairs) {
    components[component_id[pair.first]]
        .right_pairs.push_back(pair);
  }

  for (auto& component : components) {
    int n_left_nondiagonal = 0;
    int n_right_nondiagonal = 0;
    bool all_covalent = true;
    for (const auto& pair : component.left_pairs) {
      if (pair.first == pair.second) {
        all_covalent = false;
      } else {
        ++n_left_nondiagonal;
      }
    }
    for (const auto& pair : component.right_pairs) {
      if (pair.first == pair.second) {
        all_covalent = false;
      } else {
        ++n_right_nondiagonal;
      }
    }

    if (component.local_vertices.size() == 1) {
      component.type = "single_vertex";
      continue;
    }

    if (all_covalent &&
        n_left_nondiagonal == 1 &&
        n_right_nondiagonal == 1 &&
        component.left_pairs.size() == 1 &&
        component.right_pairs.size() == 1) {
      const auto left_edge = std::minmax(
          component.left_pairs.front().first,
          component.left_pairs.front().second);
      const auto right_edge = std::minmax(
          component.right_pairs.front().first,
          component.right_pairs.front().second);
      if (left_edge == right_edge) {
        component.type = "doubled_edge";
        continue;
      }
    }

    if (all_covalent &&
        static_cast<int>(component.local_vertices.size()) == 2 * n_left_nondiagonal &&
        n_left_nondiagonal == n_right_nondiagonal) {
      component.type = "alternating_cycle";
      continue;
    }

    component.type = "general";
  }

  return components;
}

std::vector<UnionGraphCrossBlockSummary> summarize_union_graph_cross_blocks(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components,
    double singular_value_threshold) {
  std::vector<UnionGraphCrossBlockSummary> cross_blocks;
  for (std::size_t left_component = 0; left_component < components.size(); ++left_component) {
    for (std::size_t right_component = left_component + 1;
         right_component < components.size();
         ++right_component) {
      cross_blocks.push_back(summarize_cross_block(
          support_overlap,
          components[left_component],
          components[right_component],
          singular_value_threshold));
    }
  }
  std::sort(
      cross_blocks.begin(),
      cross_blocks.end(),
      [](const UnionGraphCrossBlockSummary& left, const UnionGraphCrossBlockSummary& right) {
        if (left.spectral_norm != right.spectral_norm) {
          return left.spectral_norm > right.spectral_norm;
        }
        if (left.frobenius_norm != right.frobenius_norm) {
          return left.frobenius_norm > right.frobenius_norm;
        }
        if (left.left_component != right.left_component) {
          return left.left_component < right.left_component;
        }
        return left.right_component < right.right_component;
      });
  return cross_blocks;
}

UnionGraphScreeningSummary summarize_union_graph_screening(
    const Eigen::MatrixXd& support_overlap,
    const Eigen::MatrixXd& offblock_overlap,
    const std::vector<UnionGraphComponent>& components,
    const std::vector<UnionGraphCrossBlockSummary>& cross_blocks) {
  UnionGraphScreeningSummary summary;
  summary.support_size = support_overlap.rows();
  summary.component_count = static_cast<int>(components.size());
  summary.cross_block_count = static_cast<int>(cross_blocks.size());
  summary.support_overlap_frobenius = support_overlap.norm();
  summary.offblock_overlap_frobenius = offblock_overlap.norm();
  summary.offblock_overlap_fraction =
      summary.offblock_overlap_frobenius /
      std::max(1.0, summary.support_overlap_frobenius);
  summary.component_signature = build_component_signature(components);

  for (const auto& component : components) {
    summary.max_component_size = std::max(
        summary.max_component_size,
        static_cast<int>(component.active_orbitals.size()));
    if (component.type == "single_vertex") {
      ++summary.n_single_vertex_components;
    } else if (component.type == "doubled_edge") {
      ++summary.n_doubled_edge_components;
    } else if (component.type == "alternating_cycle") {
      ++summary.n_alternating_cycle_components;
    } else if (component.type == "general") {
      ++summary.n_general_components;
    }
  }

  for (const auto& cross_block : cross_blocks) {
    summary.max_numerical_rank = std::max(summary.max_numerical_rank, cross_block.numerical_rank);
    summary.max_cross_block_frobenius = std::max(
        summary.max_cross_block_frobenius,
        cross_block.frobenius_norm);
    summary.max_cross_block_spectral = std::max(
        summary.max_cross_block_spectral,
        cross_block.spectral_norm);
    summary.max_cross_block_max_abs = std::max(
        summary.max_cross_block_max_abs,
        cross_block.max_abs);
    summary.sum_cross_block_frobenius += cross_block.frobenius_norm;
    if (cross_block.singular_values.size() >= 2) {
      summary.max_cross_block_second_singular = std::max(
          summary.max_cross_block_second_singular,
          cross_block.singular_values[1]);
    }
    if (cross_block.singular_values.size() >= 3) {
      summary.max_cross_block_third_singular = std::max(
          summary.max_cross_block_third_singular,
          cross_block.singular_values[2]);
    }
  }

  return summary;
}

Eigen::MatrixXd build_block_diagonalized_support_overlap(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components) {
  Eigen::MatrixXd block_diagonal_overlap = support_overlap;
  for (std::size_t left_index = 0; left_index < components.size(); ++left_index) {
    for (std::size_t right_index = 0; right_index < components.size(); ++right_index) {
      if (left_index == right_index) {
        continue;
      }
      for (const int row : components[left_index].local_vertices) {
        for (const int column : components[right_index].local_vertices) {
          block_diagonal_overlap(row, column) = 0.0;
        }
      }
    }
  }
  return block_diagonal_overlap;
}

Eigen::MatrixXd build_offblock_support_overlap(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components) {
  Eigen::MatrixXd offblock_overlap = support_overlap;
  for (const auto& component : components) {
    for (const int row : component.local_vertices) {
      for (const int column : component.local_vertices) {
        offblock_overlap(row, column) = 0.0;
      }
    }
  }
  return offblock_overlap;
}

Eigen::MatrixXd build_blockwise_truncated_offblock(
    const Eigen::MatrixXd& support_overlap,
    const std::vector<UnionGraphComponent>& components,
    int rank_cap) {
  Eigen::MatrixXd truncated = Eigen::MatrixXd::Zero(support_overlap.rows(), support_overlap.cols());
  for (std::size_t left_index = 0; left_index < components.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1; right_index < components.size(); ++right_index) {
      const auto upper_block = extract_cross_block(
          support_overlap,
          components[left_index],
          components[right_index]);
      const auto lower_block = extract_cross_block(
          support_overlap,
          components[right_index],
          components[left_index]);
      const auto truncated_upper = truncate_block_rank(upper_block, rank_cap);
      const auto truncated_lower = truncate_block_rank(lower_block, rank_cap);
      assign_directed_cross_block(
          truncated_upper,
          components[left_index],
          components[right_index],
          &truncated);
      assign_directed_cross_block(
          truncated_lower,
          components[right_index],
          components[left_index],
          &truncated);
    }
  }
  return truncated;
}

MetricAwareComponentGraph build_metric_aware_component_graph(
    const std::vector<UnionGraphComponent>& components,
    const std::vector<UnionGraphCrossBlockSummary>& cross_blocks,
    MetricAwareComponentGraphOptions options) {
  if (options.edge_max_abs_threshold < 0.0) {
    throw std::invalid_argument("edge_max_abs_threshold must be non-negative");
  }

  MetricAwareComponentGraph graph;
  graph.node_count = static_cast<int>(components.size());
  graph.component_covalent_label_counts.reserve(components.size());
  for (const auto& component : components) {
    graph.component_covalent_label_counts.push_back(count_component_covalent_labels(component));
  }

  // Each graph edge represents one non-negligible cross-block support-overlap
  // coupling between two union-graph components.
  for (const auto& cross_block : cross_blocks) {
    if (cross_block.max_abs <= options.edge_max_abs_threshold) {
      continue;
    }
    MetricAwareComponentGraphEdge edge;
    edge.left_component = cross_block.left_component;
    edge.right_component = cross_block.right_component;
    edge.frobenius_norm = cross_block.frobenius_norm;
    edge.spectral_norm = cross_block.spectral_norm;
    edge.max_abs = cross_block.max_abs;
    edge.numerical_rank = cross_block.numerical_rank;
    graph.edges.push_back(edge);
  }

  std::sort(
      graph.edges.begin(),
      graph.edges.end(),
      [](const MetricAwareComponentGraphEdge& left,
         const MetricAwareComponentGraphEdge& right) {
        if (left.spectral_norm != right.spectral_norm) {
          return left.spectral_norm > right.spectral_norm;
        }
        if (left.max_abs != right.max_abs) {
          return left.max_abs > right.max_abs;
        }
        if (left.left_component != right.left_component) {
          return left.left_component < right.left_component;
        }
        return left.right_component < right.right_component;
      });
  graph.adjacency = build_metric_aware_adjacency(graph.node_count, graph.edges);
  return graph;
}

MetricAwareGraphSummary summarize_metric_aware_component_graph(
    const MetricAwareComponentGraph& graph,
    const std::vector<UnionGraphComponent>& components) {
  if (graph.node_count != static_cast<int>(components.size())) {
    throw std::invalid_argument("graph and union components must have the same node count");
  }
  if (graph.component_covalent_label_counts.size() != components.size() ||
      graph.adjacency.size() != components.size()) {
    throw std::invalid_argument("graph arrays must match the component count");
  }

  MetricAwareGraphSummary summary;
  summary.node_count = graph.node_count;
  summary.edge_count = static_cast<int>(graph.edges.size());
  for (const auto& neighbors : graph.adjacency) {
    summary.max_degree = std::max(summary.max_degree, static_cast<int>(neighbors.size()));
  }

  summary.connected_components =
      build_metric_connected_components(graph, components, -1);
  summary.connected_component_count =
      static_cast<int>(summary.connected_components.size());
  for (const auto& connected_component : summary.connected_components) {
    summary.max_component_node_count = std::max(
        summary.max_component_node_count,
        static_cast<int>(connected_component.graph_nodes.size()));
    summary.max_component_covalent_labels = std::max(
        summary.max_component_covalent_labels,
        connected_component.total_covalent_labels);
  }

  const auto width_and_order = build_weighted_min_degree_order(graph);
  summary.weighted_min_degree_width_upper_bound = width_and_order.first;
  summary.weighted_min_degree_order = width_and_order.second;

  const int original_component_count = summary.connected_component_count;
  for (int node = 0; node < graph.node_count; ++node) {
    const auto reduced_components =
        build_metric_connected_components(graph, components, node);
    if (static_cast<int>(reduced_components.size()) <= original_component_count) {
      continue;
    }
    summary.articulation_candidates.push_back(
        summarize_articulation_candidate(node, reduced_components));
  }
  std::sort(
      summary.articulation_candidates.begin(),
      summary.articulation_candidates.end(),
      [](const MetricAwareSeparatorCandidate& left,
         const MetricAwareSeparatorCandidate& right) {
        if (left.largest_piece_covalent_labels != right.largest_piece_covalent_labels) {
          return left.largest_piece_covalent_labels <
              right.largest_piece_covalent_labels;
        }
        if (left.largest_piece_node_count != right.largest_piece_node_count) {
          return left.largest_piece_node_count < right.largest_piece_node_count;
        }
        return left.articulation_component < right.articulation_component;
      });
  summary.articulation_count =
      static_cast<int>(summary.articulation_candidates.size());
  return summary;
}

}  // namespace xmvb::vb
