#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/exact_separator/component_tree.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::OrbitalPair;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentTree;
using xmvb::vb::exact_separator::ComponentTreeHamiltonianGradientResult;
using xmvb::vb::exact_separator::OrientationTerm;

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    terms.push_back(OrientationTerm{
        .alpha_occ = legacy_term.alpha_occ,
        .beta_occ = legacy_term.beta_occ,
        .coefficient = legacy_term.coefficient,
    });
  }
  return terms;
}

std::vector<double> flatten_column_major(
    const std::vector<std::vector<double>>& row_major_matrix) {
  const int n_rows = static_cast<int>(row_major_matrix.size());
  const int n_cols = static_cast<int>(row_major_matrix.front().size());
  std::vector<double> storage(
      xmvb::to_size(n_rows * n_cols),
      0.0);
  for (int row = 0; row < n_rows; ++row) {
    if (static_cast<int>(row_major_matrix[xmvb::to_size(row)].size()) != n_cols) {
      throw std::invalid_argument("row_major_matrix must be rectangular");
    }
    for (int col = 0; col < n_cols; ++col) {
      storage[xmvb::to_size(col * n_rows + row)] =
          row_major_matrix[xmvb::to_size(row)][xmvb::to_size(col)];
    }
  }
  return storage;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch in max_abs_difference");
  }
  double max_error = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_error = std::max(max_error, std::abs(left[index] - right[index]));
  }
  return max_error;
}

std::size_t max_abs_difference_index(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch in max_abs_difference_index");
  }
  std::size_t max_index = 0U;
  double max_error = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double error = std::abs(left[index] - right[index]);
    if (error > max_error) {
      max_error = error;
      max_index = index;
    }
  }
  return max_index;
}

double max_scalar_channel_error(
    const ComponentTreeHamiltonianGradientResult& exact,
    const ComponentTreeHamiltonianGradientResult& boundary) {
  return std::max(
      std::max(
          std::abs(exact.hamiltonian.overlap - boundary.hamiltonian.overlap),
          std::abs(exact.hamiltonian.one_electron - boundary.hamiltonian.one_electron)),
      std::max(
          std::max(
              std::abs(
                  exact.hamiltonian.same_spin_alpha_two_electron -
                  boundary.hamiltonian.same_spin_alpha_two_electron),
              std::abs(
                  exact.hamiltonian.same_spin_beta_two_electron -
                  boundary.hamiltonian.same_spin_beta_two_electron)),
          std::max(
              std::abs(
                  exact.hamiltonian.opposite_spin_two_electron -
                  boundary.hamiltonian.opposite_spin_two_electron),
              std::abs(
                  exact.hamiltonian.total_electronic_hamiltonian -
                  boundary.hamiltonian.total_electronic_hamiltonian))));
}

bool should_run_case_label(const std::string& label) {
  const char* filter = std::getenv("XMVB_COMPONENT_TREE_GRAD_CASE_FILTER");
  if (filter == nullptr || filter[0] == '\0') {
    return true;
  }
  return label.find(filter) != std::string::npos;
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void check_case(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    double hamiltonian_weight,
    double overlap_weight,
    double tolerance) {
  if (!should_run_case_label(label)) {
    return;
  }
  const DeterminantOverlapResolver overlap_resolver;
  const ComponentTreeHamiltonianGradientResult exact =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_active_space_gradient_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver,
          hamiltonian_weight,
          overlap_weight);
  const ComponentTreeHamiltonianGradientResult boundary =
      xmvb::vb::exact_separator::evaluate_rooted_component_tree_active_space_gradient_boundary_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver,
          hamiltonian_weight,
          overlap_weight);

  const double scalar_error = max_scalar_channel_error(exact, boundary);
  const double overlap_gradient_error = max_abs_difference(
      exact.active_orbital_overlap_gradient,
      boundary.active_orbital_overlap_gradient);
  const std::size_t overlap_gradient_error_index = max_abs_difference_index(
      exact.active_orbital_overlap_gradient,
      boundary.active_orbital_overlap_gradient);
  const double h1e_gradient_error = max_abs_difference(
      exact.active_one_electron_gradient,
      boundary.active_one_electron_gradient);
  const std::size_t h1e_gradient_error_index = max_abs_difference_index(
      exact.active_one_electron_gradient,
      boundary.active_one_electron_gradient);
  const double eri_gradient_error = max_abs_difference(
      exact.packed_active_two_electron_gradient,
      boundary.packed_active_two_electron_gradient);
  const std::size_t eri_gradient_error_index = max_abs_difference_index(
      exact.packed_active_two_electron_gradient,
      boundary.packed_active_two_electron_gradient);

  std::cout << "case = " << label << '\n';
  std::cout << "hamiltonian_weight = " << hamiltonian_weight << '\n';
  std::cout << "overlap_weight = " << overlap_weight << '\n';
  std::cout << "scalar_max_abs_error = " << scalar_error << '\n';
  std::cout << "overlap_gradient_max_abs_error = " << overlap_gradient_error << '\n';
  std::cout << "overlap_gradient_error_index = " << overlap_gradient_error_index << '\n';
  std::cout << "exact_overlap_gradient_at_index = "
            << exact.active_orbital_overlap_gradient[overlap_gradient_error_index] << '\n';
  std::cout << "boundary_overlap_gradient_at_index = "
            << boundary.active_orbital_overlap_gradient[overlap_gradient_error_index] << '\n';
  std::cout << "h1e_gradient_max_abs_error = " << h1e_gradient_error << '\n';
  std::cout << "h1e_gradient_error_index = " << h1e_gradient_error_index << '\n';
  std::cout << "exact_h1e_gradient_at_index = "
            << exact.active_one_electron_gradient[h1e_gradient_error_index] << '\n';
  std::cout << "boundary_h1e_gradient_at_index = "
            << boundary.active_one_electron_gradient[h1e_gradient_error_index] << '\n';
  std::cout << "eri_gradient_max_abs_error = " << eri_gradient_error << '\n';
  std::cout << "eri_gradient_error_index = " << eri_gradient_error_index << '\n';
  std::cout << "exact_eri_gradient_at_index = "
            << exact.packed_active_two_electron_gradient[eri_gradient_error_index] << '\n';
  std::cout << "boundary_eri_gradient_at_index = "
            << boundary.packed_active_two_electron_gradient[eri_gradient_error_index] << '\n';
  std::cout << "boundary_message_state_count = "
            << boundary.hamiltonian.subtree_message_state_count << '\n';
  std::cout << "boundary_term_pair_count = "
            << boundary.hamiltonian.subtree_term_pair_count << '\n';
  std::cout << "boundary_subdeterminant_evaluations = "
            << boundary.hamiltonian.subdeterminant_evaluations << '\n';
  std::cout << "boundary_dp_transition_count = "
            << boundary.hamiltonian.dp_transition_count << '\n';
  std::cout << '\n';

  if (scalar_error > tolerance ||
      overlap_gradient_error > tolerance ||
      h1e_gradient_error > tolerance ||
      eri_gradient_error > tolerance) {
    throw std::runtime_error("component-tree active-space gradient check failed for case: " + label);
  }
}

void check_case_suite(
    const std::string& label,
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    double tolerance) {
  check_case(
      label + "/hamiltonian_only",
      tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      1.0,
      0.0,
      tolerance);
  check_case(
      label + "/overlap_only",
      tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      0.0,
      1.0,
      tolerance);
  check_case(
      label + "/mixed",
      tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      0.75,
      -0.35,
      tolerance);
}

int main_impl() {
  std::vector<ComponentData> components(4);
  const std::vector<std::vector<OrbitalPair>> component_pairs{
      {{0, 1}},
      {{2, 3}},
      {{4, 5}},
      {{6, 7}},
  };
  for (int node = 0; node < 4; ++node) {
    components[xmvb::to_size(node)].graph_node = node;
    components[xmvb::to_size(node)].left_pairs =
        component_pairs[xmvb::to_size(node)];
    components[xmvb::to_size(node)].right_pairs =
        component_pairs[xmvb::to_size(node)];
    components[xmvb::to_size(node)].left_orientation_terms =
        enumerate_orientation_terms(component_pairs[xmvb::to_size(node)]);
    components[xmvb::to_size(node)].right_orientation_terms =
        enumerate_orientation_terms(component_pairs[xmvb::to_size(node)]);
  }

  const ComponentTree chain_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
          components[3],
      },
      .children = {
          {1},
          {2},
          {},
      },
  };
  const ComponentTree branched_tree{
      .root_index = 0,
      .components = components,
      .children = {
          {1, 2},
          {3},
          {},
          {},
      },
  };
  const ComponentTree one_leaf_star_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
      },
      .children = {
          {1},
          {},
      },
  };
  const ComponentTree star_tree{
      .root_index = 0,
      .components = {
          components[0],
          components[1],
          components[2],
      },
      .children = {
          {1, 2},
          {},
          {},
      },
  };

  const std::vector<double> support_overlap_storage = flatten_column_major({
      {1.00, 0.08, 0.12, 0.05, 0.09, 0.04, 0.00, 0.00},
      {0.11, 0.97, 0.07, 0.10, 0.03, 0.06, 0.00, 0.00},
      {0.14, 0.09, 0.95, 0.11, 0.00, 0.00, 0.13, 0.07},
      {0.06, 0.12, 0.10, 0.98, 0.00, 0.00, 0.08, 0.09},
      {0.07, 0.04, 0.00, 0.00, 0.96, 0.10, 0.00, 0.00},
      {0.05, 0.07, 0.00, 0.00, 0.09, 0.99, 0.00, 0.00},
      {0.00, 0.00, 0.11, 0.06, 0.00, 0.00, 0.94, 0.08},
      {0.00, 0.00, 0.09, 0.10, 0.00, 0.00, 0.07, 0.97},
  });
  const std::vector<double> support_one_electron_storage = flatten_column_major({
      {0.31, 0.05, 0.07, 0.03, 0.02, 0.01, 0.00, 0.00},
      {0.04, 0.28, 0.06, 0.08, 0.01, 0.03, 0.00, 0.00},
      {0.09, 0.07, 0.35, 0.04, 0.00, 0.00, 0.06, 0.02},
      {0.02, 0.08, 0.05, 0.33, 0.00, 0.00, 0.03, 0.07},
      {0.03, 0.02, 0.00, 0.00, 0.29, 0.05, 0.00, 0.00},
      {0.01, 0.04, 0.00, 0.00, 0.06, 0.32, 0.00, 0.00},
      {0.00, 0.00, 0.08, 0.03, 0.00, 0.00, 0.27, 0.04},
      {0.00, 0.00, 0.02, 0.09, 0.00, 0.00, 0.05, 0.30},
  });

  const int packed_eri_size =
      TwoElectronIndexer::two_electron_storage_index(7, 7, 7, 7) + 1;
  std::vector<double> packed_active_two_electron_integrals(
      xmvb::to_size(packed_eri_size),
      0.0);
  for (int p = 0; p < 8; ++p) {
    for (int q = 0; q < 8; ++q) {
      for (int r = 0; r < 8; ++r) {
        for (int s = 0; s < 8; ++s) {
          const double value =
              0.21 + 0.025 * (p + 1) + 0.02 * (q + 1) + 0.015 * (r + 1) +
              0.01 * (s + 1);
          packed_active_two_electron_integrals[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] = value;
        }
      }
    }
  }
  std::vector<double> filtered_one_electron_storage = support_one_electron_storage;
  std::vector<double> filtered_packed_active_two_electron_integrals =
      packed_active_two_electron_integrals;
  if (env_flag_enabled("XMVB_COMPONENT_TREE_GRAD_ZERO_H1E")) {
    std::fill(
        filtered_one_electron_storage.begin(),
        filtered_one_electron_storage.end(),
        0.0);
  }
  if (env_flag_enabled("XMVB_COMPONENT_TREE_GRAD_ZERO_ERI")) {
    std::fill(
        filtered_packed_active_two_electron_integrals.begin(),
        filtered_packed_active_two_electron_integrals.end(),
        0.0);
  }

  constexpr double tolerance = 1.0e-10;
  std::cout << std::setprecision(15);
  check_case_suite(
      "chain_rooted_subtree",
      chain_tree,
      support_overlap_storage,
      filtered_one_electron_storage,
      filtered_packed_active_two_electron_integrals,
      8,
      tolerance);
  check_case_suite(
      "branched_rooted_subtree",
      branched_tree,
      support_overlap_storage,
      filtered_one_electron_storage,
      filtered_packed_active_two_electron_integrals,
      8,
      tolerance);
  check_case_suite(
      "one_leaf_star",
      one_leaf_star_tree,
      support_overlap_storage,
      filtered_one_electron_storage,
      filtered_packed_active_two_electron_integrals,
      8,
      tolerance);
  check_case_suite(
      "multi_leaf_star",
      star_tree,
      support_overlap_storage,
      filtered_one_electron_storage,
      filtered_packed_active_two_electron_integrals,
      8,
      tolerance);
  return 0;
}

}  // namespace

int main() {
  try {
    return main_impl();
  } catch (const std::exception& error) {
    std::cerr << "check_exact_separator_component_tree_gradient failed: "
              << error.what() << '\n';
    return 1;
  }
}
