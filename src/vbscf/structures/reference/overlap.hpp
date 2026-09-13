#pragma once

#include <map>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/structures/selection/union_graph/screening.hpp"

namespace xmvb::vb {

struct RawStructureDeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

std::vector<RawStructureDeterminantTerm> enumerate_raw_determinant_terms(
    const std::vector<OrbitalPair>& pairs);

std::vector<RawStructureDeterminantTerm> remap_raw_determinant_terms(
    const std::vector<RawStructureDeterminantTerm>& terms,
    const std::map<int, int>& orbital_index_remap);

double raw_structure_overlap(
    const std::vector<RawStructureDeterminantTerm>& left_terms,
    const std::vector<RawStructureDeterminantTerm>& right_terms,
    const Eigen::MatrixXd& overlap_matrix,
    const DeterminantOverlapResolver& overlap_resolver);

}  // namespace xmvb::vb
