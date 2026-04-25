#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xmvb::vb {

namespace {

class DisjointSet {
public:
  explicit DisjointSet(std::size_t size)
      : parent_(size),
        rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(int index) {
    if (parent_[index] != index) {
      parent_[index] = find(parent_[index]);
    }
    return parent_[index];
  }

  void unite(int left, int right) {
    int left_root = find(left);
    int right_root = find(right);
    if (left_root == right_root) {
      return;
    }
    if (rank_[left_root] < rank_[right_root]) {
      std::swap(left_root, right_root);
    }
    parent_[right_root] = left_root;
    if (rank_[left_root] == rank_[right_root]) {
      ++rank_[left_root];
    }
  }

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

std::vector<std::vector<int>> collect_orbital_supports(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<std::vector<int>> orbital_supports(
      orbital_preparation_input.n_orbitals);
  for (std::size_t orbital_index = 0;
       orbital_index < orbital_preparation_input.n_orbitals;
       ++orbital_index) 
  {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            static_cast<int>(orbital_index));
    auto& support = orbital_supports[orbital_index];
    support.reserve(coefficient_count);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index *
                   orbital_preparation_input.n_basis_functions +
               coefficient_index] -
          1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input.n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while adapting nonredundant input");
      }
      support.push_back(basis_function_index);
    }
  }
  return orbital_supports;
}

std::vector<std::vector<int>> build_overlap_connected_components(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<std::vector<int>>& orbital_supports) {
  DisjointSet components(orbital_preparation_input.n_orbitals);
  std::vector<std::vector<int>> basis_to_orbitals(
      orbital_preparation_input.n_basis_functions);
  for (std::size_t orbital_index = 0;
       orbital_index < orbital_preparation_input.n_orbitals;
       ++orbital_index) {
    for (const int basis_function_index : orbital_supports[orbital_index]) {
      basis_to_orbitals[basis_function_index].push_back(
          static_cast<int>(orbital_index));
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
  components_by_root.reserve(orbital_preparation_input.n_orbitals);
  for (std::size_t orbital_index = 0;
       orbital_index < orbital_preparation_input.n_orbitals;
       ++orbital_index) {
    components_by_root[components.find(static_cast<int>(orbital_index))]
        .push_back(static_cast<int>(orbital_index));
  }

  std::vector<std::vector<int>> overlap_components;
  overlap_components.reserve(components_by_root.size());
  for (auto& [root, component] : components_by_root) {
    (void)root;
    overlap_components.push_back(std::move(component));
  }
  std::sort(
      overlap_components.begin(),
      overlap_components.end(),
      [](const std::vector<int>& left, const std::vector<int>& right) {
        return left.empty() ? false
                            : (right.empty() ? true : left.front() < right.front());
      });
  return overlap_components;
}

std::vector<int> build_component_union_support(
    const std::vector<int>& component_orbitals,
    const std::vector<std::vector<int>>& orbital_supports,
    int n_basis_functions) {
  std::vector<int> union_support;
  union_support.reserve(n_basis_functions);
  std::vector<char> seen_support(n_basis_functions, 0);
  // Preserve first appearance order across the component so the adapted sparse
  // rows stay close to the original legacy slot ordering whenever possible.
  for (const int orbital_index : component_orbitals) {
    for (const int basis_function_index :
         orbital_supports[orbital_index]) {
      if (seen_support[basis_function_index] != 0) {
        continue;
      }
      seen_support[basis_function_index] = 1;
      union_support.push_back(basis_function_index);
    }
  }
  return union_support;
}

bool orbital_supports_match(
    const std::vector<int>& left_support,
    const std::vector<int>& right_support,
    int n_basis_functions) {
  if (left_support.size() != right_support.size()) {
    return false;
  }
  std::vector<char> left_mask(n_basis_functions, 0);
  std::vector<char> right_mask(n_basis_functions, 0);
  for (const int basis_function_index : left_support) {
    left_mask[basis_function_index] = 1;
  }
  for (const int basis_function_index : right_support) {
    right_mask[basis_function_index] = 1;
  }
  return left_mask == right_mask;
}

bool orbital_supports_overlap(
    const std::vector<int>& left_support,
    const std::vector<int>& right_support,
    int n_basis_functions) {
  std::vector<char> right_mask(n_basis_functions, 0);
  for (const int basis_function_index : right_support) {
    right_mask[basis_function_index] = 1;
  }
  for (const int basis_function_index : left_support) {
    if (right_mask[basis_function_index] != 0) {
      return true;
    }
  }
  return false;
}

void expand_component_sparse_supports(
    const std::vector<int>& component_orbitals,
    const std::vector<int>& union_support,
    const std::vector<std::vector<int>>& orbital_supports,
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument(
        "orbital_preparation_input must not be null for support expansion");
  }
  if (union_support.size() >
      orbital_preparation_input->n_basis_functions) {
    throw std::runtime_error(
        "union sparse orbital support exceeds n_basis_functions");
  }

  const std::size_t n_basis_functions = orbital_preparation_input->n_basis_functions;
  std::vector<double> updated_orbital_values =
      orbital_preparation_input->orbital_value_table;
  std::vector<int> updated_basis_index_table =
      orbital_preparation_input->orbital_basis_index_table;
  std::vector<int> updated_basis_counts =
      orbital_preparation_input->orbital_basis_counts;
  std::vector<int> updated_original_basis_counts =
      orbital_preparation_input->original_orbital_basis_counts;

  for (const int orbital_index : component_orbitals) {
    const auto& orbital_support =
        orbital_supports[orbital_index];
    // After support expansion, every downstream consumer must see a parameter
    // count that matches the adapted sparse slots, even if it still keys off
    // the legacy `original_orbital_basis_counts` metadata.
    updated_original_basis_counts[orbital_index] =
        static_cast<int>(union_support.size());
    if (orbital_supports_match(
            orbital_support,
            union_support,
            n_basis_functions)) {
      continue;
    }

    std::unordered_map<int, double> value_by_basis;
    value_by_basis.reserve(orbital_support.size());
    for (int coefficient_index = 0;
         coefficient_index < static_cast<int>(orbital_support.size());
         ++coefficient_index) {
      value_by_basis.emplace(
          orbital_support[coefficient_index],
          orbital_preparation_input->orbital_value_table
              [orbital_index * n_basis_functions +
               coefficient_index]);
    }

    for (std::size_t coefficient_index = 0;
         coefficient_index < n_basis_functions;
         ++coefficient_index) {
      updated_orbital_values[orbital_index * n_basis_functions +
                             coefficient_index] = 0.0;
      updated_basis_index_table[orbital_index * n_basis_functions +
                                coefficient_index] = 0;
    }

    for (int coefficient_index = 0;
         coefficient_index < static_cast<int>(union_support.size());
         ++coefficient_index) {
      const int basis_function_index =
          union_support[coefficient_index];
      updated_basis_index_table[orbital_index * n_basis_functions +
                                coefficient_index] =
          basis_function_index + 1;
      const auto value_iterator =
          value_by_basis.find(basis_function_index);
      if (value_iterator != value_by_basis.end()) {
        updated_orbital_values[orbital_index * n_basis_functions +
                               coefficient_index] =
            value_iterator->second;
      }
    }
    updated_basis_counts[orbital_index] =
        static_cast<int>(union_support.size());
  }

  orbital_preparation_input->orbital_value_table =
      std::move(updated_orbital_values);
  orbital_preparation_input->orbital_basis_index_table =
      std::move(updated_basis_index_table);
  orbital_preparation_input->orbital_basis_counts =
      std::move(updated_basis_counts);
  orbital_preparation_input->original_orbital_basis_counts =
      std::move(updated_original_basis_counts);
}

void rebuild_exact_support_block_metadata(
    const std::vector<std::vector<int>>& orbital_supports,
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }

  std::vector<std::vector<int>> blocks;
  std::vector<int> block_basis_counts;
  blocks.reserve(orbital_preparation_input->n_orbitals);
  block_basis_counts.reserve(orbital_preparation_input->n_orbitals);
  for (std::size_t orbital_index = 0;
       orbital_index < orbital_preparation_input->n_orbitals;
       ++orbital_index) {
    bool appended_to_existing_block = false;
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      const int representative_orbital = blocks[block_index].front();
      if (!orbital_supports_match(
              orbital_supports[orbital_index],
              orbital_supports[representative_orbital],
              orbital_preparation_input->n_basis_functions)) {
        continue;
      }
      blocks[block_index].push_back(static_cast<int>(orbital_index));
      appended_to_existing_block = true;
      break;
    }
    if (!appended_to_existing_block) {
      blocks.push_back({static_cast<int>(orbital_index)});
      block_basis_counts.push_back(
          static_cast<int>(
              orbital_supports[orbital_index].size()));
    }
  }

  std::size_t block_storage_dimension = 0;
  for (const auto& block : blocks) {
    block_storage_dimension =
        std::max(block_storage_dimension, block.size());
  }
  const std::size_t block_storage_stride = std::max<std::size_t>(1, block_storage_dimension);

  std::vector<int> block_members(
      blocks.size() * block_storage_stride,
      0);
  std::vector<int> block_orbital_counts(blocks.size(), 0);
  for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const auto& block = blocks[block_index];
    block_orbital_counts[block_index] = static_cast<int>(block.size());
    for (std::size_t orbital_offset = 0; orbital_offset < block.size(); ++orbital_offset) {
      block_members[block_index * block_storage_stride +
                    orbital_offset] = block[orbital_offset];
    }
  }

  orbital_preparation_input->n_blocks = blocks.size();
  orbital_preparation_input->block_storage_dimension = block_storage_dimension;
  orbital_preparation_input->block_members = std::move(block_members);
  orbital_preparation_input->block_orbital_counts = std::move(block_orbital_counts);
  orbital_preparation_input->block_basis_counts = std::move(block_basis_counts);
}

void rebuild_overlap_connected_block_metadata(
    const std::vector<std::vector<int>>& overlap_components,
    const std::vector<std::vector<int>>& orbital_supports,
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }

  std::size_t block_storage_dimension = 0;
  for (const auto& component : overlap_components) {
    block_storage_dimension =
        std::max(block_storage_dimension, component.size());
  }
  const std::size_t block_storage_stride = std::max<std::size_t>(1, block_storage_dimension);

  std::vector<int> block_members(
      overlap_components.size() * block_storage_stride,
      0);
  std::vector<int> block_orbital_counts(
      overlap_components.size(),
      0);
  std::vector<int> block_basis_counts;
  block_basis_counts.reserve(overlap_components.size());

  for (std::size_t block_index = 0;
       block_index < overlap_components.size();
       ++block_index) {
    const auto& component = overlap_components[block_index];
    block_orbital_counts[block_index] = static_cast<int>(component.size());
    for (std::size_t orbital_offset = 0;
         orbital_offset < component.size();
         ++orbital_offset) {
      block_members[block_index * block_storage_stride +
                    orbital_offset] = component[orbital_offset];
    }
    block_basis_counts.push_back(
        static_cast<int>(
            build_component_union_support(
                component,
                orbital_supports,
                orbital_preparation_input->n_basis_functions)
                .size()));
  }

  orbital_preparation_input->n_blocks = overlap_components.size();
  orbital_preparation_input->block_storage_dimension =
      block_storage_dimension;
  orbital_preparation_input->block_members = std::move(block_members);
  orbital_preparation_input->block_orbital_counts =
      std::move(block_orbital_counts);
  orbital_preparation_input->block_basis_counts =
      std::move(block_basis_counts);
}

bool support_layout_has_partial_overlap(
    const std::vector<std::vector<int>>& orbital_supports,
    int n_basis_functions) {
  const int n_orbitals = static_cast<int>(orbital_supports.size());
  for (int left_orbital = 0; left_orbital < n_orbitals; ++left_orbital) {
    for (int right_orbital = left_orbital + 1;
         right_orbital < n_orbitals;
         ++right_orbital) {
      if (!orbital_supports_overlap(
              orbital_supports[left_orbital],
              orbital_supports[right_orbital],
              n_basis_functions)) {
        continue;
      }
      if (orbital_supports_match(
              orbital_supports[left_orbital],
              orbital_supports[right_orbital],
              n_basis_functions)) {
        continue;
      }
      return true;
    }
  }
  return false;
}

}  // namespace

bool orbital_input_requires_partial_overlap_support_expansion(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.block_partial_overlap != 0;
}

OrbitalPreparationInput build_partial_overlap_support_expanded_input(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<char> expand_orbital_mask(
      orbital_preparation_input.n_orbitals,
      1);
  return build_partial_overlap_support_expanded_input(
      orbital_preparation_input,
      expand_orbital_mask);
}

OrbitalPreparationInput build_partial_overlap_support_expanded_input(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<char>& expand_orbital_mask) {
  if (!orbital_input_requires_partial_overlap_support_expansion(
          orbital_preparation_input)) {
    return orbital_preparation_input;
  }
  if (expand_orbital_mask.size() !=
      orbital_preparation_input.n_orbitals) {
    throw std::invalid_argument(
        "expand_orbital_mask size does not match n_orbitals");
  }

  OrbitalPreparationInput adapted_input = orbital_preparation_input;
  const auto orbital_supports =
      collect_orbital_supports(adapted_input);
  const auto overlap_components =
      build_overlap_connected_components(
          adapted_input,
          orbital_supports);

  // Expand each overlap-connected component to a common sparse support before
  // the optimizer or guess builder interprets the sparse rows as an exact block
  // structure. This is an exact change of sparse layout: the represented
  // orbitals are unchanged, but future MO guesses can now populate the full
  // component support instead of being truncated back to the legacy per-orbital
  // support rows.
  for (const auto& component_orbitals : overlap_components) {
    std::vector<int> selected_component_orbitals;
    selected_component_orbitals.reserve(component_orbitals.size());
    for (const int orbital_index : component_orbitals) {
      if (expand_orbital_mask[orbital_index] == 0) {
        continue;
      }
      selected_component_orbitals.push_back(orbital_index);
    }
    if (selected_component_orbitals.empty()) {
      continue;
    }
    const auto union_support =
        build_component_union_support(
            component_orbitals,
            orbital_supports,
            adapted_input.n_basis_functions);
    expand_component_sparse_supports(
        selected_component_orbitals,
        union_support,
        orbital_supports,
        &adapted_input);
  }

  const auto adapted_orbital_supports =
      collect_orbital_supports(adapted_input);
  rebuild_exact_support_block_metadata(adapted_orbital_supports, &adapted_input);
  adapted_input.block_partial_overlap =
      support_layout_has_partial_overlap(
          adapted_orbital_supports,
          adapted_input.n_basis_functions)
      ? 1
      : 0;
  return adapted_input;
}

CppVbInput build_nonredundant_optimizer_input(
    const CppVbInput& input) {
  // The nonredundant optimizer must start from the same physical sparse-orbital
  // chart that the VB objective, restart artifacts, and Molden export use.
  // Pre-orthonormalizing the full-support OEO inactive block changes the
  // accepted-point gauge in a way that is not an exact legacy chart transform
  // and measurably distorts TiCl active orbitals relative to XMVB.
  return input;
}

}  // namespace xmvb::vb
