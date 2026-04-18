#pragma once

#include <map>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace xmvb::vb {

struct LegacyStructureDeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

std::vector<LegacyStructureDeterminantTerm> enumerate_legacy_determinant_terms(
    const std::vector<OrbitalPair>& pairs);

std::vector<LegacyStructureDeterminantTerm> remap_legacy_determinant_terms(
    const std::vector<LegacyStructureDeterminantTerm>& terms,
    const std::map<int, int>& orbital_index_remap);

double legacy_structure_overlap(
    const std::vector<LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<LegacyStructureDeterminantTerm>& right_terms,
    const Eigen::MatrixXd& overlap_matrix,
    const DeterminantOverlapResolver& overlap_resolver);

}  // namespace xmvb::vb
