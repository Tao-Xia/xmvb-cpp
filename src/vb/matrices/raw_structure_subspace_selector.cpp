#include "vb/matrices/raw_structure_subspace_selector.hpp"

#include <stdexcept>

namespace xmvb::vb {

namespace {

void validate_raw_structure_data(
    const RawStructureData& raw_structure_data) {
  if (raw_structure_data.n_structures <= 0) {
    throw std::invalid_argument("raw structure data must contain at least one structure");
  }
  if (raw_structure_data.n_total_electrons <= 0 ||
      raw_structure_data.n_active_electrons <= 0) {
    throw std::invalid_argument("raw structure data must define positive electron counts");
  }
  if (raw_structure_data.raw_structure_orbitals.size() !=
      raw_structure_data.flat_orbital_count()) {
    throw std::invalid_argument("raw structure count does not match stored structures");
  }
}

std::vector<int> build_structure_index_remap(
    int n_structures,
    const std::vector<int>& selected_structure_indices) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (selected_structure_indices.empty()) {
    throw std::invalid_argument("selected_structure_indices must not be empty");
  }

  std::vector<int> structure_index_remap(
      xmvb::to_size(n_structures),
      -1);
  for (std::size_t selected_offset = 0;
       selected_offset < selected_structure_indices.size();
       ++selected_offset) {
    const int structure_index = selected_structure_indices[selected_offset];
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (structure_index_remap[xmvb::to_size(structure_index)] >= 0) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    structure_index_remap[xmvb::to_size(structure_index)] =
        static_cast<int>(selected_offset);
  }
  return structure_index_remap;
}

bool is_covalent_structure(
    const RawStructureData& raw_structure_data,
    int structure_index) {
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  const int active_stop = active_start + raw_structure_data.n_active_electrons;
  if (active_start < 0 || active_stop > raw_structure_data.n_total_electrons) {
    throw std::invalid_argument("active-electron window is out of range for raw structures");
  }

  const int* structure_orbitals =
      raw_structure_data.structure_orbitals_data(structure_index);
  for (int left_index = active_start; left_index < active_stop; ++left_index) {
    const int left_orbital = structure_orbitals[left_index];
    for (int right_index = left_index + 1; right_index < active_stop; ++right_index) {
      if (left_orbital == structure_orbitals[right_index]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

const char* raw_structure_selection_mode_name(
    RawStructureSelectionMode mode) {
  switch (mode) {
    case RawStructureSelectionMode::Full:
      return "full";
    case RawStructureSelectionMode::Covalent:
      return "covalent";
  }
  return "unknown";
}

std::vector<int> select_raw_structure_indices(
    const RawStructureData& raw_structure_data,
    RawStructureSelectionMode mode) {
  validate_raw_structure_data(raw_structure_data);

  std::vector<int> selected_structure_indices;
  selected_structure_indices.reserve(xmvb::to_size(raw_structure_data.n_structures));
  if (mode == RawStructureSelectionMode::Full) {
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      selected_structure_indices.push_back(structure_index);
    }
    return selected_structure_indices;
  }

  if (mode == RawStructureSelectionMode::Covalent) {
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      if (is_covalent_structure(raw_structure_data, structure_index)) {
        selected_structure_indices.push_back(structure_index);
      }
    }
    if (selected_structure_indices.empty()) {
      throw std::runtime_error("raw structure selection produced an empty covalent subspace");
    }
    return selected_structure_indices;
  }

  throw std::invalid_argument("unsupported raw structure selection mode");
}

RawStructureData build_raw_structure_subset(
    const RawStructureData& raw_structure_data,
    const std::vector<int>& selected_structure_indices) {
  validate_raw_structure_data(raw_structure_data);
  const auto structure_index_remap = build_structure_index_remap(
      raw_structure_data.n_structures,
      selected_structure_indices);
  (void)structure_index_remap;

  RawStructureData subset_data = raw_structure_data;
  subset_data.n_structures = static_cast<int>(selected_structure_indices.size());
  subset_data.raw_structure_orbitals.clear();
  subset_data.raw_structure_orbitals.reserve(subset_data.flat_orbital_count());
  for (const int structure_index : selected_structure_indices) {
    const int* structure_orbitals =
        raw_structure_data.structure_orbitals_data(structure_index);
    subset_data.raw_structure_orbitals.insert(
        subset_data.raw_structure_orbitals.end(),
        structure_orbitals,
        structure_orbitals + raw_structure_data.n_total_electrons);
  }
  return subset_data;
}

}  // namespace xmvb::vb
