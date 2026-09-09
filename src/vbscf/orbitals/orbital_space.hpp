#pragma once

#include <vector>

#include "vb/model/vb_dimensions.hpp"

namespace xmvb::vb {

struct OrbitalSpace {
  VbDimensions dims;

  std::vector<double> active_orbital_overlap_matrix;
  std::vector<double> orbital_transform_matrix;
  std::vector<double> orbital_parameter_matrix;

  std::vector<int> orbital_block_sizes;
  std::vector<int> orbital_occupancy_table;
};

}  // namespace xmvb::vb
