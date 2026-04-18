#pragma once

#include <vector>

#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

enum class RawStructureSelectionMode {
  Full,
  Covalent,
};

const char* raw_structure_selection_mode_name(
    RawStructureSelectionMode mode);

std::vector<int> select_raw_structure_indices(
    const RawStructureData& raw_structure_data,
    RawStructureSelectionMode mode);

RawStructureData build_raw_structure_subset(
    const RawStructureData& raw_structure_data,
    const std::vector<int>& selected_structure_indices);

}  // namespace xmvb::vb
