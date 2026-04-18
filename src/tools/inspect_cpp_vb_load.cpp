#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/cpp_block_guess_builder.hpp"
#include "runtime_c/cpp_runtime_extractor.h"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"

namespace {

struct MatrixStats {
  int off_diagonal_nnz = 0;
  double max_abs_off_diagonal = 0.0;
  double min_diagonal = 0.0;
  double max_diagonal = 0.0;
};

struct RectangularMatrixStats {
  int nnz = 0;
  double max_abs = 0.0;
};

std::vector<double> flatten_column_major(const Eigen::MatrixXd& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + xmvb::to_size(matrix.size()));
}

Eigen::MatrixXd build_dense_occupied_orbital_matrix(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  const int n_inactive =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_occupied =
      n_inactive + orbital_preparation_input.n_active_orbitals;
  if (n_occupied < 0 ||
      n_occupied > orbital_preparation_input.n_orbitals) {
    throw std::invalid_argument(
        "invalid occupied-space partition while building dense occupied orbitals");
  }

  Eigen::MatrixXd occupied_orbitals =
      Eigen::MatrixXd::Zero(
          orbital_preparation_input.n_basis_functions,
          n_occupied);
  for (int orbital_index = 0;
       orbital_index < n_occupied;
       ++orbital_index) {
    const int coefficient_count =
        xmvb::vb::get_orbital_basis_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) *
                   orbital_preparation_input.n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input.n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while building dense occupied orbitals");
      }
      occupied_orbitals(basis_function_index, orbital_index) =
          orbital_preparation_input.orbital_value_table
              [xmvb::to_size(orbital_index) *
                   orbital_preparation_input.n_basis_functions +
               coefficient_index];
    }
  }
  return occupied_orbitals;
}

RectangularMatrixStats analyze_rectangular_matrix(
    const Eigen::MatrixXd& matrix,
    double nnz_tolerance) {
  RectangularMatrixStats stats;
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double abs_value = std::abs(matrix(row, column));
      stats.max_abs = std::max(stats.max_abs, abs_value);
      if (abs_value > nnz_tolerance) {
        ++stats.nnz;
      }
    }
  }
  return stats;
}

void print_usage() {
  std::cerr << "usage: inspect_cpp_vb_load <input.xmi>"
               " [--raw-structure-selection full|covalent]"
               " [--expand true|false]"
               " [--print-block-details true|false]"
               " [--runtime-only true|false]"
               " [--skip-orbital-guess true|false]"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--ao-integral-source legacy|libcint_cpp]"
               " [--orbital-guess-source legacy|cpp]\n";
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

void apply_raw_structure_selection_argument(
    const std::string& value,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("options must not be null");
  }
  if (value == "full") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (value == "covalent") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid raw structure selection: " + value);
}

void apply_ao_integral_source_argument(
    const std::string& value,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("options must not be null");
  }
  if (value == "legacy") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
    return;
  }
  if (value == "libcint_cpp") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    return;
  }
  throw std::invalid_argument("invalid AO integral source: " + value);
}

void apply_orbital_guess_source_argument(
    const std::string& value,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("options must not be null");
  }
  if (value == "legacy") {
    options->orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    return;
  }
  if (value == "cpp") {
    options->orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    return;
  }
  throw std::invalid_argument("invalid orbital guess source: " + value);
}

void apply_standard_two_electron_mode_argument(
    const std::string& value,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("options must not be null");
  }
  if (value == "auto") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
    return;
  }
  if (value == "exact") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    return;
  }
  if (value == "ri") {
    options->standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    return;
  }
  throw std::invalid_argument("invalid standard two-electron mode: " + value);
}

struct RuntimeSnapshotOwner {
  RuntimeSnapshotOwner() {
    init_cpp_runtime_snapshot(&snapshot);
  }

  ~RuntimeSnapshotOwner() {
    free_cpp_runtime_snapshot(&snapshot);
  }

  CppRuntimeSnapshot snapshot{};
};

class DisjointSet {
public:
  explicit DisjointSet(int size)
      : parent_(xmvb::to_size(size)),
        rank_(xmvb::to_size(size), 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(int index) {
    if (parent_[xmvb::to_size(index)] != index) {
      parent_[xmvb::to_size(index)] = find(parent_[xmvb::to_size(index)]);
    }
    return parent_[xmvb::to_size(index)];
  }

  void unite(int left, int right) {
    int left_root = find(left);
    int right_root = find(right);
    if (left_root == right_root) {
      return;
    }
    if (rank_[xmvb::to_size(left_root)] < rank_[xmvb::to_size(right_root)]) {
      std::swap(left_root, right_root);
    }
    parent_[xmvb::to_size(right_root)] = left_root;
    if (rank_[xmvb::to_size(left_root)] == rank_[xmvb::to_size(right_root)]) {
      ++rank_[xmvb::to_size(left_root)];
    }
  }

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

int get_runtime_orbital_basis_count(
    const CppRuntimeSnapshot& snapshot,
    int orbital_index) {
  const int explicit_count = snapshot.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count != 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < snapshot.n_basis_functions) {
    const int basis_function_index =
        snapshot.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                               snapshot.n_basis_functions +
                                           coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<std::vector<int>> collect_runtime_orbital_supports(
    const CppRuntimeSnapshot& snapshot) {
  std::vector<std::vector<int>> orbital_supports(xmvb::to_size(snapshot.n_orbitals));
  for (int orbital_index = 0; orbital_index < snapshot.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_runtime_orbital_basis_count(snapshot, orbital_index);
    auto& support = orbital_supports[xmvb::to_size(orbital_index)];
    support.reserve(xmvb::to_size(coefficient_count));
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          snapshot.orbital_basis_index_table[xmvb::to_size(orbital_index) *
                                                 snapshot.n_basis_functions +
                                             coefficient_index] -
          1;
      if (basis_function_index < 0 || basis_function_index >= snapshot.n_basis_functions) {
        throw std::runtime_error("invalid runtime sparse orbital basis index");
      }
      support.push_back(basis_function_index);
    }
  }
  return orbital_supports;
}

std::vector<std::vector<int>> collect_runtime_legacy_blocks(
    const CppRuntimeSnapshot& snapshot) {
  std::vector<std::vector<int>> blocks;
  blocks.reserve(xmvb::to_size(snapshot.n_blocks));
  for (int block_index = 0; block_index < snapshot.n_blocks; ++block_index) {
    const int orbital_count =
        snapshot.block_orbital_counts[xmvb::to_size(block_index)];
    if (orbital_count <= 0) {
      continue;
    }
    std::vector<int> block;
    block.reserve(xmvb::to_size(orbital_count));
    for (int orbital_offset = 0; orbital_offset < orbital_count; ++orbital_offset) {
      const int orbital_index =
          snapshot.block_members[xmvb::to_size(block_index) *
                                     snapshot.block_storage_dimension +
                                 orbital_offset];
      if (orbital_index < 0 || orbital_index >= snapshot.n_orbitals) {
        throw std::runtime_error("invalid runtime legacy block member");
      }
      block.push_back(orbital_index);
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

std::vector<int> build_union_support(
    const std::vector<int>& orbitals,
    const std::vector<std::vector<int>>& orbital_supports,
    int n_basis_functions) {
  std::vector<int> union_support;
  union_support.reserve(xmvb::to_size(n_basis_functions));
  std::vector<char> seen_support(xmvb::to_size(n_basis_functions), 0);
  for (const int orbital_index : orbitals) {
    for (const int basis_function_index : orbital_supports[xmvb::to_size(orbital_index)]) {
      if (seen_support[xmvb::to_size(basis_function_index)] != 0) {
        continue;
      }
      seen_support[xmvb::to_size(basis_function_index)] = 1;
      union_support.push_back(basis_function_index);
    }
  }
  return union_support;
}

std::vector<std::vector<int>> build_overlap_connected_components(
    const std::vector<std::vector<int>>& orbital_supports,
    int n_basis_functions) {
  DisjointSet components(static_cast<int>(orbital_supports.size()));
  std::vector<std::vector<int>> basis_to_orbitals(xmvb::to_size(n_basis_functions));
  for (int orbital_index = 0;
       orbital_index < static_cast<int>(orbital_supports.size());
       ++orbital_index) {
    for (const int basis_function_index : orbital_supports[xmvb::to_size(orbital_index)]) {
      basis_to_orbitals[xmvb::to_size(basis_function_index)].push_back(orbital_index);
    }
  }

  for (const auto& orbitals_on_basis : basis_to_orbitals) {
    if (orbitals_on_basis.size() <= 1) {
      continue;
    }
    const int representative_orbital = orbitals_on_basis.front();
    for (std::size_t orbital_offset = 1;
         orbital_offset < orbitals_on_basis.size();
         ++orbital_offset) {
      components.unite(representative_orbital, orbitals_on_basis[orbital_offset]);
    }
  }

  std::unordered_map<int, std::vector<int>> components_by_root;
  components_by_root.reserve(orbital_supports.size());
  for (int orbital_index = 0;
       orbital_index < static_cast<int>(orbital_supports.size());
       ++orbital_index) {
    components_by_root[components.find(orbital_index)].push_back(orbital_index);
  }

  std::vector<std::vector<int>> overlap_components;
  overlap_components.reserve(components_by_root.size());
  for (auto& [root, component] : components_by_root) {
    (void)root;
    std::sort(component.begin(), component.end());
    overlap_components.push_back(std::move(component));
  }
  std::sort(
      overlap_components.begin(),
      overlap_components.end(),
      [](const std::vector<int>& left, const std::vector<int>& right) {
        if (left.empty() || right.empty()) {
          return left.size() < right.size();
        }
        return left.front() < right.front();
      });
  return overlap_components;
}

std::string format_index_list(const std::vector<int>& values, int offset = 0) {
  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      result += ", ";
    }
    result += std::to_string(values[index] + offset);
  }
  result += "]";
  return result;
}

void print_runtime_block_diagnostics(const CppRuntimeSnapshot& snapshot) {
  if (snapshot.orbital_basis_index_table == nullptr ||
      snapshot.orbital_basis_counts == nullptr) {
    throw std::runtime_error("runtime snapshot is missing sparse-orbital metadata");
  }
  const auto orbital_supports = collect_runtime_orbital_supports(snapshot);
  const auto legacy_blocks = collect_runtime_legacy_blocks(snapshot);
  const auto overlap_components =
      build_overlap_connected_components(orbital_supports, snapshot.n_basis_functions);
  const int n_inactive =
      (snapshot.n_total_electrons - snapshot.n_active_electrons) / 2;
  const int n_occupied = n_inactive + snapshot.n_active_orbitals;

  std::cout << "orbital_supports_begin\n";
  for (int orbital_index = 0; orbital_index < snapshot.n_orbitals; ++orbital_index) {
    const char* occupancy_class =
        orbital_index < n_inactive ? "inactive" :
        (orbital_index < n_occupied ? "active" : "virtual");
    const auto& support = orbital_supports[xmvb::to_size(orbital_index)];
    std::cout << "orbital[" << orbital_index << "]"
              << " occupancy=" << occupancy_class
              << " support_count=" << support.size()
              << " support_0based=" << format_index_list(support)
              << " support_1based=" << format_index_list(support, 1)
              << '\n';
  }
  std::cout << "orbital_supports_end\n";

  int legacy_direction_total = 0;
  std::cout << "legacy_blocks_begin\n";
  for (std::size_t block_index = 0; block_index < legacy_blocks.size(); ++block_index) {
    const auto& block = legacy_blocks[block_index];
    const auto union_support =
        build_union_support(block, orbital_supports, snapshot.n_basis_functions);
    int inactive_count = 0;
    int occupied_count = 0;
    for (const int orbital_index : block) {
      if (orbital_index < n_occupied) {
        ++occupied_count;
      }
      if (orbital_index < n_inactive) {
        ++inactive_count;
      }
    }
    const int active_count = occupied_count - inactive_count;
    const int local_virtual_count =
        static_cast<int>(union_support.size()) - occupied_count;
    const int candidate_direction_count =
        inactive_count * active_count + occupied_count * std::max(0, local_virtual_count);
    legacy_direction_total += candidate_direction_count;
    const int stored_basis_count =
        snapshot.block_basis_counts != nullptr &&
                block_index < xmvb::to_size(snapshot.n_blocks)
            ? snapshot.block_basis_counts[block_index]
            : 0;
    std::cout << "legacy_block[" << block_index << "]"
              << " orbitals=" << format_index_list(block)
              << " orbitals_1based=" << format_index_list(block, 1)
              << " stored_basis_count=" << stored_basis_count
              << " union_basis_count=" << union_support.size()
              << " inactive=" << inactive_count
              << " active=" << active_count
              << " occupied=" << occupied_count
              << " local_virtual=" << std::max(0, local_virtual_count)
              << " candidate_directions=" << candidate_direction_count
              << " union_support_1based=" << format_index_list(union_support, 1)
              << '\n';
    if (stored_basis_count > 0 &&
        stored_basis_count != static_cast<int>(union_support.size())) {
      std::cout << "legacy_block[" << block_index << "]"
                << "_basis_mismatch stored=" << stored_basis_count
                << " union=" << union_support.size()
                << '\n';
    }
  }
  std::cout << "legacy_blocks_end\n";
  std::cout << "legacy_candidate_direction_total = "
            << legacy_direction_total << '\n';

  int component_direction_total = 0;
  std::cout << "overlap_components_begin\n";
  for (std::size_t component_index = 0;
       component_index < overlap_components.size();
       ++component_index) {
    const auto& component = overlap_components[component_index];
    const auto union_support =
        build_union_support(component, orbital_supports, snapshot.n_basis_functions);
    int inactive_count = 0;
    int occupied_count = 0;
    for (const int orbital_index : component) {
      if (orbital_index < n_occupied) {
        ++occupied_count;
      }
      if (orbital_index < n_inactive) {
        ++inactive_count;
      }
    }
    const int active_count = occupied_count - inactive_count;
    const int local_virtual_count =
        static_cast<int>(union_support.size()) - occupied_count;
    const int candidate_direction_count =
        inactive_count * active_count + occupied_count * std::max(0, local_virtual_count);
    component_direction_total += candidate_direction_count;
    std::cout << "overlap_component[" << component_index << "]"
              << " orbitals=" << format_index_list(component)
              << " orbitals_1based=" << format_index_list(component, 1)
              << " union_basis_count=" << union_support.size()
              << " inactive=" << inactive_count
              << " active=" << active_count
              << " occupied=" << occupied_count
              << " local_virtual=" << std::max(0, local_virtual_count)
              << " candidate_directions=" << candidate_direction_count
              << " union_support_1based=" << format_index_list(union_support, 1)
              << '\n';
  }
  std::cout << "overlap_components_end\n";
  std::cout << "overlap_component_candidate_direction_total = "
            << component_direction_total << '\n';
}

void print_nonredundant_space_comparison(
    const xmvb::vb::CppVbInput& input,
    double nuclear_repulsion_energy) {
  xmvb::vb::SparseOrbitalParameterView original_parameter_view(
      input.orbital_preparation_input);
  const auto original_blocks =
      xmvb::vb::detect_orbital_blocks(input.orbital_preparation_input);

  const xmvb::vb::CppVbInput adapted_input =
      xmvb::vb::build_nonredundant_optimizer_input(input);
  xmvb::vb::SparseOrbitalParameterView adapted_parameter_view(
      adapted_input.orbital_preparation_input);
  const auto adapted_blocks =
      xmvb::vb::detect_orbital_blocks(adapted_input.orbital_preparation_input);
  xmvb::vb::CppOrbitalGradientEvaluator gradient_evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto original_gradient_result =
      gradient_evaluator.evaluate_without_reference_energy_gradient(
          input,
          nuclear_repulsion_energy);
  const Eigen::VectorXd original_packed_gradient =
      original_parameter_view.gather_from_full(
          original_gradient_result.sparse_orbital_energy_gradient);
  const int original_n_inactive =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const int original_n_occupied =
      original_n_inactive +
      input.orbital_preparation_input.n_active_orbitals;
  xmvb::vb::NonredundantOrbitalSpace original_space(
      input.orbital_preparation_input,
      original_parameter_view,
      original_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
          std::max(0, original_n_occupied)),
      original_gradient_result.orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix);
  const auto original_projection =
      original_space.project_vector(original_packed_gradient);
  const auto adapted_gradient_result =
      gradient_evaluator.evaluate_without_reference_energy_gradient(
          adapted_input,
          nuclear_repulsion_energy);
  const Eigen::VectorXd adapted_packed_gradient =
      adapted_parameter_view.gather_from_full(
          adapted_gradient_result.sparse_orbital_energy_gradient);
  const int adapted_n_inactive =
      (adapted_input.orbital_preparation_input.n_total_electrons -
       adapted_input.orbital_preparation_input.n_active_electrons) / 2;
  const int adapted_n_occupied =
      adapted_n_inactive +
      adapted_input.orbital_preparation_input.n_active_orbitals;
  xmvb::vb::NonredundantOrbitalSpace adapted_space(
      adapted_input.orbital_preparation_input,
      adapted_parameter_view,
      adapted_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
          std::max(0, adapted_n_occupied)),
      adapted_gradient_result.orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix);
  const auto adapted_projection =
      adapted_space.project_vector(adapted_packed_gradient);

  std::cout << "nonredundant_space_comparison_begin\n";
  std::cout << "original_detected_block_count = "
            << original_blocks.size() << '\n';
  std::cout << "original_packed_parameter_size = "
            << original_parameter_view.size() << '\n';
  std::cout << "original_reduced_size = "
            << original_space.reduced_size() << '\n';
  std::cout << "original_packed_gradient_inf_norm = "
            << original_packed_gradient.lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "original_projected_gradient_inf_norm = "
            << original_projection.reduced_gradient.lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "original_projected_gradient_l2_norm = "
            << original_projection.reduced_gradient.norm() << '\n';
  std::cout << "adapted_detected_block_count = "
            << adapted_blocks.size() << '\n';
  std::cout << "adapted_packed_parameter_size = "
            << adapted_parameter_view.size() << '\n';
  std::cout << "adapted_reduced_size = "
            << adapted_space.reduced_size() << '\n';
  std::cout << "adapted_packed_gradient_inf_norm = "
            << adapted_packed_gradient.lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "adapted_projected_gradient_inf_norm = "
            << adapted_projection.reduced_gradient.lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "adapted_projected_gradient_l2_norm = "
            << adapted_projection.reduced_gradient.norm() << '\n';
  const bool adapted_support_expansion_applied =
      input.orbital_preparation_input.block_partial_overlap != 0 &&
      (adapted_blocks.size() != original_blocks.size() ||
       adapted_parameter_view.size() != original_parameter_view.size() ||
       adapted_space.reduced_size() != original_space.reduced_size());
  std::cout << "adapted_support_expansion_applied = "
            << (adapted_support_expansion_applied ? "true" : "false")
            << '\n';
  std::cout << "nonredundant_space_comparison_end\n";
}

MatrixStats analyze_square_matrix(
    const std::vector<double>& matrix,
    int dimension,
    double nnz_tolerance) {
  if (dimension <= 0) {
    throw std::invalid_argument("matrix dimension must be positive");
  }
  if (static_cast<int>(matrix.size()) != dimension * dimension) {
    throw std::invalid_argument("matrix size does not match dimension");
  }

  MatrixStats stats;
  stats.min_diagonal = matrix[0];
  stats.max_diagonal = matrix[0];
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      const double value =
          matrix[xmvb::to_size(column) * dimension + row];
      if (row == column) {
        stats.min_diagonal = std::min(stats.min_diagonal, value);
        stats.max_diagonal = std::max(stats.max_diagonal, value);
        continue;
      }
      const double abs_value = std::abs(value);
      stats.max_abs_off_diagonal =
          std::max(stats.max_abs_off_diagonal, abs_value);
      if (abs_value > nnz_tolerance) {
        ++stats.off_diagonal_nnz;
      }
    }
  }
  return stats;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || ((argc - 2) % 2 != 0)) {
      print_usage();
      return 1;
    }

    const std::string input_path = argv[1];
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    bool runtime_only = false;
    bool print_block_details = false;
    for (int argument_index = 2; argument_index < argc; argument_index += 2) {
      const std::string argument_name = argv[argument_index];
      const std::string argument_value = argv[argument_index + 1];
      if (argument_name == "--raw-structure-selection") {
        apply_raw_structure_selection_argument(argument_value, &load_options);
      } else if (argument_name == "--expand") {
        load_options.expand_selected_raw_structures = parse_bool_argument(argument_value);
      } else if (argument_name == "--print-block-details") {
        print_block_details = parse_bool_argument(argument_value);
      } else if (argument_name == "--runtime-only") {
        runtime_only = parse_bool_argument(argument_value);
      } else if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else if (argument_name == "--standard-two-electron-mode") {
        apply_standard_two_electron_mode_argument(argument_value, &load_options);
      } else if (argument_name == "--ao-integral-source") {
        apply_ao_integral_source_argument(argument_value, &load_options);
      } else if (argument_name == "--orbital-guess-source") {
        apply_orbital_guess_source_argument(argument_value, &load_options);
      } else {
        throw std::invalid_argument("unknown argument: " + argument_name);
      }
    }

    std::cout << std::setprecision(12);
    if (runtime_only) {
      CppRuntimeExtractionOptions extraction_options{};
      init_cpp_runtime_extraction_options(&extraction_options);
      if (load_options.ao_integral_source == xmvb::vb::AoIntegralSource::LibcintMaterializedCpp) {
        extraction_options.skip_legacy_two_electron_integrals = 1;
      }
      if (load_options.skip_orbital_guess ||
          load_options.orbital_guess_source == xmvb::vb::OrbitalGuessSource::Cpp) {
        extraction_options.skip_legacy_vbguess = 1;
        extraction_options.skip_legacy_hf_setup = 1;
      }

      RuntimeSnapshotOwner runtime_snapshot_owner;
      CppRuntimeExtractionTimings runtime_timings{};
      char error_message[1024] = {0};
      const int status = extract_cpp_runtime_snapshot_with_options(
          input_path.c_str(),
          &extraction_options,
          &runtime_snapshot_owner.snapshot,
          &runtime_timings,
          error_message,
          sizeof(error_message));
      if (status != 0) {
        throw std::runtime_error(
            error_message[0] != '\0' ? error_message : "C++ runtime extraction failed");
      }

      const auto& snapshot = runtime_snapshot_owner.snapshot;
      std::cout << "mode = runtime_only\n";
      std::cout << "ao_integral_source = "
                << xmvb::vb::ao_integral_source_name(load_options.ao_integral_source) << '\n';
      std::cout << "orbital_guess_source = "
                << xmvb::vb::orbital_guess_source_name(load_options.orbital_guess_source) << '\n';
      std::cout << "standard_two_electron_mode = "
                << xmvb::vb::standard_two_electron_mode_name(
                       load_options.standard_two_electron_mode)
                << '\n';
      std::cout << "n_basis_functions = " << snapshot.n_basis_functions << '\n';
      std::cout << "n_orbitals = " << snapshot.n_orbitals << '\n';
      std::cout << "n_active_orbitals = " << snapshot.n_active_orbitals << '\n';
      std::cout << "n_total_electrons = " << snapshot.n_total_electrons << '\n';
      std::cout << "n_active_electrons = " << snapshot.n_active_electrons << '\n';
      std::cout << "n_blocks = " << snapshot.n_blocks << '\n';
      std::cout << "block_storage_dimension = "
                << snapshot.block_storage_dimension << '\n';
      std::cout << "block_partial_overlap = "
                << snapshot.block_partial_overlap << '\n';
      std::cout << "raw_structure_count = " << snapshot.n_structures << '\n';
      std::cout << "n_ao_two_electron_integrals = "
                << snapshot.n_ao_two_electron_integrals << '\n';
      std::cout << "runtime_total_wall_time_seconds = "
                << runtime_timings.total_seconds << '\n';
      std::cout << "runtime_read_input_seconds = "
                << runtime_timings.read_input_seconds << '\n';
      std::cout << "runtime_vb_input_seconds = "
                << runtime_timings.vb_input_seconds << '\n';
      std::cout << "runtime_hf_setup_seconds = "
                << runtime_timings.hf_setup_seconds << '\n';
      std::cout << "runtime_vbguess_seconds = "
                << runtime_timings.vbguess_seconds << '\n';
      std::cout << "runtime_one_electron_integrals_seconds = "
                << runtime_timings.one_electron_integrals_seconds << '\n';
      std::cout << "runtime_two_electron_integrals_seconds = "
                << runtime_timings.two_electron_integrals_seconds << '\n';
      if (print_block_details) {
        print_runtime_block_diagnostics(snapshot);
      }
      return 0;
    }

    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(input_path, load_options);
    const auto& input = load_result.input;

    std::cout << "mode = full_load\n";
    std::cout << "ao_integral_source = "
              << xmvb::vb::ao_integral_source_name(load_result.ao_integral_source) << '\n';
    std::cout << "orbital_guess_source = "
              << xmvb::vb::orbital_guess_source_name(load_result.orbital_guess_source) << '\n';
    std::cout << "standard_two_electron_mode = "
              << xmvb::vb::standard_two_electron_mode_name(
                     load_result.standard_two_electron_mode)
              << '\n';
    std::cout << "skip_orbital_guess = "
              << (load_options.skip_orbital_guess ? "true" : "false") << '\n';
    std::cout << "raw_structure_selection = "
              << xmvb::vb::raw_structure_selection_mode_name(
                     load_result.raw_structure_selection)
              << '\n';
    std::cout << "expand_selected_raw_structures = "
              << (load_options.expand_selected_raw_structures ? "true" : "false") << '\n';
    std::cout << "n_basis_functions = "
              << input.orbital_preparation_input.n_basis_functions << '\n';
    std::cout << "n_orbitals = "
              << input.orbital_preparation_input.n_orbitals << '\n';
    std::cout << "n_active_orbitals = "
              << input.orbital_preparation_input.n_active_orbitals << '\n';
    std::cout << "n_total_electrons = "
              << input.orbital_preparation_input.n_total_electrons << '\n';
    std::cout << "n_active_electrons = "
              << input.orbital_preparation_input.n_active_electrons << '\n';
    std::cout << "n_blocks = "
              << input.orbital_preparation_input.n_blocks << '\n';
    std::cout << "block_storage_dimension = "
              << input.orbital_preparation_input.block_storage_dimension << '\n';
    std::cout << "block_partial_overlap = "
              << input.orbital_preparation_input.block_partial_overlap << '\n';
    std::cout << "orbital_type = "
              << input.orbital_preparation_input.orbital_type << '\n';
    std::cout << "n_materialized_ao_two_electron_integrals = "
              << input.ao_integral_input.ao_two_electron_integral_values.size() << '\n';
    std::cout << "source_raw_structure_count = "
              << load_result.source_raw_structure_count << '\n';
    std::cout << "selected_raw_structure_count = "
              << load_result.raw_structure_data.n_structures << '\n';
    if (load_options.expand_selected_raw_structures) {
      std::cout << "expanded_determinant_count = "
                << input.structure_data.alpha_det.size() << '\n';
    } else {
      std::cout << "expanded_determinant_count = skipped\n";
    }
    std::cout << "runtime_total_wall_time_seconds = "
              << load_result.runtime_timings.total_seconds << '\n';
    std::cout << "ao_integral_provider_seconds = "
              << load_result.ao_integral_provider_seconds << '\n';
    std::cout << "ao_integral_input_build_seconds = "
              << load_result.ao_integral_input_build_seconds << '\n';
    std::cout << "orbital_guess_seconds = "
              << load_result.orbital_guess_seconds << '\n';
    std::cout << "raw_structure_selection_seconds = "
              << load_result.raw_structure_selection_seconds << '\n';
    std::cout << "structure_expansion_seconds = "
              << load_result.structure_expansion_seconds << '\n';
    if (print_block_details) {
      print_nonredundant_space_comparison(
          input,
          load_result.nuclear_repulsion_energy);
    }

    {
      const int n_inactive =
          (input.orbital_preparation_input.n_total_electrons -
           input.orbital_preparation_input.n_active_electrons) / 2;
      const int n_active = input.orbital_preparation_input.n_active_orbitals;
      const Eigen::MatrixXd occupied_orbitals =
          build_dense_occupied_orbital_matrix(
              input.orbital_preparation_input);
      const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
          input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
          input.orbital_preparation_input.n_basis_functions,
          input.orbital_preparation_input.n_basis_functions);
      const Eigen::MatrixXd occupied_s_overlap =
          occupied_orbitals.transpose() * basis_overlap_matrix * occupied_orbitals;
      const Eigen::MatrixXd occupied_euclidean_gram =
          occupied_orbitals.transpose() * occupied_orbitals;
      const MatrixStats occupied_s_overlap_stats =
          analyze_square_matrix(
              flatten_column_major(occupied_s_overlap),
              static_cast<int>(occupied_s_overlap.rows()),
              1.0e-12);
      const MatrixStats occupied_euclidean_gram_stats =
          analyze_square_matrix(
              flatten_column_major(occupied_euclidean_gram),
              static_cast<int>(occupied_euclidean_gram.rows()),
              1.0e-12);
      std::cout << "occupied_orbital_s_overlap_offdiag_nnz = "
                << occupied_s_overlap_stats.off_diagonal_nnz << '\n';
      std::cout << "occupied_orbital_s_overlap_max_abs_offdiag = "
                << occupied_s_overlap_stats.max_abs_off_diagonal << '\n';
      std::cout << "occupied_orbital_s_overlap_min_diag = "
                << occupied_s_overlap_stats.min_diagonal << '\n';
      std::cout << "occupied_orbital_s_overlap_max_diag = "
                << occupied_s_overlap_stats.max_diagonal << '\n';
      std::cout << "occupied_orbital_euclidean_gram_offdiag_nnz = "
                << occupied_euclidean_gram_stats.off_diagonal_nnz << '\n';
      std::cout << "occupied_orbital_euclidean_gram_max_abs_offdiag = "
                << occupied_euclidean_gram_stats.max_abs_off_diagonal << '\n';
      std::cout << "occupied_orbital_euclidean_gram_min_diag = "
                << occupied_euclidean_gram_stats.min_diagonal << '\n';
      std::cout << "occupied_orbital_euclidean_gram_max_diag = "
                << occupied_euclidean_gram_stats.max_diagonal << '\n';

      if (n_inactive > 0) {
        const Eigen::MatrixXd inactive_s_overlap =
            occupied_s_overlap.topLeftCorner(n_inactive, n_inactive);
        const MatrixStats inactive_s_overlap_stats =
            analyze_square_matrix(
                flatten_column_major(inactive_s_overlap),
                n_inactive,
                1.0e-12);
        std::cout << "inactive_orbital_s_overlap_offdiag_nnz = "
                  << inactive_s_overlap_stats.off_diagonal_nnz << '\n';
        std::cout << "inactive_orbital_s_overlap_max_abs_offdiag = "
                  << inactive_s_overlap_stats.max_abs_off_diagonal << '\n';
        std::cout << "inactive_orbital_s_overlap_min_diag = "
                  << inactive_s_overlap_stats.min_diagonal << '\n';
        std::cout << "inactive_orbital_s_overlap_max_diag = "
                  << inactive_s_overlap_stats.max_diagonal << '\n';
      }
      if (n_active > 0) {
        const Eigen::MatrixXd active_s_overlap =
            occupied_s_overlap.bottomRightCorner(n_active, n_active);
        const MatrixStats active_s_overlap_stats =
            analyze_square_matrix(
                flatten_column_major(active_s_overlap),
                n_active,
                1.0e-12);
        std::cout << "active_orbital_s_overlap_offdiag_nnz = "
                  << active_s_overlap_stats.off_diagonal_nnz << '\n';
        std::cout << "active_orbital_s_overlap_max_abs_offdiag = "
                  << active_s_overlap_stats.max_abs_off_diagonal << '\n';
        std::cout << "active_orbital_s_overlap_min_diag = "
                  << active_s_overlap_stats.min_diagonal << '\n';
        std::cout << "active_orbital_s_overlap_max_diag = "
                  << active_s_overlap_stats.max_diagonal << '\n';
      }
      if (n_inactive > 0 && n_active > 0) {
        const Eigen::MatrixXd inactive_active_s_overlap =
            occupied_s_overlap.topRightCorner(n_inactive, n_active);
        const RectangularMatrixStats inactive_active_s_overlap_stats =
            analyze_rectangular_matrix(
                inactive_active_s_overlap,
                1.0e-12);
        std::cout << "inactive_active_orbital_s_overlap_nnz = "
                  << inactive_active_s_overlap_stats.nnz << '\n';
        std::cout << "inactive_active_orbital_s_overlap_max_abs = "
                  << inactive_active_s_overlap_stats.max_abs << '\n';
      }
    }

    {
      const int n_inactive =
          (input.orbital_preparation_input.n_total_electrons -
           input.orbital_preparation_input.n_active_electrons) / 2;
      const int n_active = input.orbital_preparation_input.n_active_orbitals;
      const int n_occupied = n_inactive + n_active;
      const int n_virtual =
          input.orbital_preparation_input.n_basis_functions - n_occupied;
      xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
      const xmvb::vb::OrbitalPreparationResult orbital_result =
          orbital_preparer.prepare(input.orbital_preparation_input);
      const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
          input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
          input.orbital_preparation_input.n_basis_functions,
          input.orbital_preparation_input.n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> auxiliary_inverse(
          orbital_result.auxiliary_orbital_inverse_matrix.data(),
          input.orbital_preparation_input.n_basis_functions,
          input.orbital_preparation_input.n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> inactive_dual(
          orbital_result.inactive_auxiliary_transform.data(),
          input.orbital_preparation_input.n_basis_functions,
          input.orbital_preparation_input.n_basis_functions);
      const Eigen::MatrixXd inactive_dual_block =
          inactive_dual.leftCols(std::max(0, n_inactive));
      const Eigen::MatrixXd inactive_physical =
          build_dense_occupied_orbital_matrix(
              input.orbital_preparation_input).leftCols(std::max(0, n_inactive));
      const Eigen::MatrixXd active_auxiliary =
          orbital_result.auxiliary_orbital_matrix.middleCols(
              std::max(0, n_inactive),
              std::max(0, n_active));
      const Eigen::MatrixXd occupied_auxiliary =
          orbital_result.auxiliary_orbital_matrix.leftCols(std::max(0, n_occupied));

      if (n_inactive > 0) {
        const Eigen::MatrixXd inactive_dual_inactive_metric =
            inactive_dual_block.transpose() *
            basis_overlap_matrix *
            inactive_physical;
        const Eigen::MatrixXd inactive_dual_inactive_residual =
            inactive_dual_inactive_metric -
            Eigen::MatrixXd::Identity(n_inactive, n_inactive);
        const MatrixStats inactive_dual_inactive_residual_stats =
            analyze_square_matrix(
                flatten_column_major(inactive_dual_inactive_residual),
                n_inactive,
                1.0e-12);
        std::cout << "inactive_dual_inactive_metric_residual_offdiag_nnz = "
                  << inactive_dual_inactive_residual_stats.off_diagonal_nnz << '\n';
        std::cout << "inactive_dual_inactive_metric_residual_max_abs_offdiag = "
                  << inactive_dual_inactive_residual_stats.max_abs_off_diagonal << '\n';
        std::cout << "inactive_dual_inactive_metric_residual_min_diag = "
                  << inactive_dual_inactive_residual_stats.min_diagonal << '\n';
        std::cout << "inactive_dual_inactive_metric_residual_max_diag = "
                  << inactive_dual_inactive_residual_stats.max_diagonal << '\n';
      }
      if (n_inactive > 0 && n_active > 0) {
        const Eigen::MatrixXd inactive_dual_active_aux_metric =
            inactive_dual_block.transpose() *
            basis_overlap_matrix *
            active_auxiliary;
        const RectangularMatrixStats inactive_dual_active_aux_metric_stats =
            analyze_rectangular_matrix(
                inactive_dual_active_aux_metric,
                1.0e-12);
        std::cout << "inactive_dual_active_aux_metric_nnz = "
                  << inactive_dual_active_aux_metric_stats.nnz << '\n';
        std::cout << "inactive_dual_active_aux_metric_max_abs = "
                  << inactive_dual_active_aux_metric_stats.max_abs << '\n';
      }
      if (n_virtual > 0) {
        const Eigen::MatrixXd virtual_auxiliary =
            orbital_result.auxiliary_orbital_matrix.rightCols(n_virtual);
        const Eigen::MatrixXd occupied_virtual_metric =
            occupied_auxiliary.transpose() *
            basis_overlap_matrix *
            virtual_auxiliary;
        const RectangularMatrixStats occupied_virtual_metric_stats =
            analyze_rectangular_matrix(
                occupied_virtual_metric,
                1.0e-12);
        std::cout << "occupied_auxiliary_virtual_metric_nnz = "
                  << occupied_virtual_metric_stats.nnz << '\n';
        std::cout << "occupied_auxiliary_virtual_metric_max_abs = "
                  << occupied_virtual_metric_stats.max_abs << '\n';

        const Eigen::MatrixXd virtual_metric_residual =
            virtual_auxiliary.transpose() *
            basis_overlap_matrix *
            virtual_auxiliary -
            Eigen::MatrixXd::Identity(n_virtual, n_virtual);
        const MatrixStats virtual_metric_residual_stats =
            analyze_square_matrix(
                flatten_column_major(virtual_metric_residual),
                n_virtual,
                1.0e-12);
        std::cout << "virtual_auxiliary_metric_residual_offdiag_nnz = "
                  << virtual_metric_residual_stats.off_diagonal_nnz << '\n';
        std::cout << "virtual_auxiliary_metric_residual_max_abs_offdiag = "
                  << virtual_metric_residual_stats.max_abs_off_diagonal << '\n';
        std::cout << "virtual_auxiliary_metric_residual_min_diag = "
                  << virtual_metric_residual_stats.min_diagonal << '\n';
        std::cout << "virtual_auxiliary_metric_residual_max_diag = "
                  << virtual_metric_residual_stats.max_diagonal << '\n';
      }

      const Eigen::MatrixXd inverse_left_residual =
          auxiliary_inverse * orbital_result.auxiliary_orbital_matrix -
          Eigen::MatrixXd::Identity(
              input.orbital_preparation_input.n_basis_functions,
              input.orbital_preparation_input.n_basis_functions);
      const MatrixStats inverse_left_residual_stats =
          analyze_square_matrix(
              flatten_column_major(inverse_left_residual),
              input.orbital_preparation_input.n_basis_functions,
              1.0e-12);
      std::cout << "auxiliary_inverse_left_residual_offdiag_nnz = "
                << inverse_left_residual_stats.off_diagonal_nnz << '\n';
      std::cout << "auxiliary_inverse_left_residual_max_abs_offdiag = "
                << inverse_left_residual_stats.max_abs_off_diagonal << '\n';
      std::cout << "auxiliary_inverse_left_residual_min_diag = "
                << inverse_left_residual_stats.min_diagonal << '\n';
      std::cout << "auxiliary_inverse_left_residual_max_diag = "
                << inverse_left_residual_stats.max_diagonal << '\n';

      const Eigen::MatrixXd inverse_right_residual =
          orbital_result.auxiliary_orbital_matrix * auxiliary_inverse -
          Eigen::MatrixXd::Identity(
              input.orbital_preparation_input.n_basis_functions,
              input.orbital_preparation_input.n_basis_functions);
      const MatrixStats inverse_right_residual_stats =
          analyze_square_matrix(
              flatten_column_major(inverse_right_residual),
              input.orbital_preparation_input.n_basis_functions,
              1.0e-12);
      std::cout << "auxiliary_inverse_right_residual_offdiag_nnz = "
                << inverse_right_residual_stats.off_diagonal_nnz << '\n';
      std::cout << "auxiliary_inverse_right_residual_max_abs_offdiag = "
                << inverse_right_residual_stats.max_abs_off_diagonal << '\n';
      std::cout << "auxiliary_inverse_right_residual_min_diag = "
                << inverse_right_residual_stats.min_diagonal << '\n';
      std::cout << "auxiliary_inverse_right_residual_max_diag = "
                << inverse_right_residual_stats.max_diagonal << '\n';
    }

    if (load_options.expand_selected_raw_structures) {
      xmvb::vb::StructureMatrixEvaluator structure_evaluator;
      const auto prepared_active_space =
          structure_evaluator.prepare_active_space(input);
      const auto structure_matrices =
          structure_evaluator.evaluate(input, prepared_active_space);

      const MatrixStats active_overlap_stats = analyze_square_matrix(
          prepared_active_space.orbital_result.active_orbital_overlap_matrix,
          input.orbital_preparation_input.n_active_orbitals,
          1.0e-12);
      const MatrixStats active_h1e_stats = analyze_square_matrix(
          prepared_active_space.active_space_one_electron_result.h1e_act,
          input.orbital_preparation_input.n_active_orbitals,
          1.0e-12);
      const MatrixStats structure_overlap_stats = analyze_square_matrix(
          structure_matrices.overlap_matrix,
          input.structure_data.n_structures,
          1.0e-12);
      const MatrixStats structure_hamiltonian_stats = analyze_square_matrix(
          structure_matrices.hamiltonian_matrix,
          input.structure_data.n_structures,
          1.0e-12);

      std::cout << "prepared_one_electron_reference_energy = "
                << prepared_active_space.one_electron_reference_energy << '\n';
      std::cout << "prepared_active_overlap_offdiag_nnz = "
                << active_overlap_stats.off_diagonal_nnz << '\n';
      std::cout << "prepared_active_overlap_max_abs_offdiag = "
                << active_overlap_stats.max_abs_off_diagonal << '\n';
      std::cout << "prepared_active_overlap_min_diag = "
                << active_overlap_stats.min_diagonal << '\n';
      std::cout << "prepared_active_overlap_max_diag = "
                << active_overlap_stats.max_diagonal << '\n';
      std::cout << "prepared_active_h1e_offdiag_nnz = "
                << active_h1e_stats.off_diagonal_nnz << '\n';
      std::cout << "prepared_active_h1e_max_abs_offdiag = "
                << active_h1e_stats.max_abs_off_diagonal << '\n';
      std::cout << "structure_overlap_offdiag_nnz = "
                << structure_overlap_stats.off_diagonal_nnz << '\n';
      std::cout << "structure_overlap_max_abs_offdiag = "
                << structure_overlap_stats.max_abs_off_diagonal << '\n';
      std::cout << "structure_hamiltonian_offdiag_nnz = "
                << structure_hamiltonian_stats.off_diagonal_nnz << '\n';
      std::cout << "structure_hamiltonian_max_abs_offdiag = "
                << structure_hamiltonian_stats.max_abs_off_diagonal << '\n';
    }

    std::cout << "input_total_wall_time_seconds = "
              << load_result.total_seconds << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
