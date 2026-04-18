#pragma once

#include <utility>
#include <vector>

#include "vb/matrices/union_graph_screening.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief One spin-resolved determinant orientation term inside one union-graph component.
 *
 * The occupied-orbital lists use support-local orbital labels in ascending
 * canonical order. `coefficient` is the signed weight of this orientation term
 * in the component-local raw-structure expansion.
 */
struct OrientationTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

/**
 * @brief One component of the component-ordered star decomposition.
 *
 * `left_pairs` and `right_pairs` describe the local raw-VB pair pattern on the
 * bra and ket sides. The orientation terms enumerate the component-local
 * spin-determinant states that survive after fixing the pair pattern.
 */
struct ComponentData {
  int graph_node = -1;
  std::vector<OrbitalPair> left_pairs;
  std::vector<OrbitalPair> right_pairs;
  std::vector<OrientationTerm> left_orientation_terms;
  std::vector<OrientationTerm> right_orientation_terms;
};

/**
 * @brief Canonical occupied-orbital key for one spin-adapted determinant.
 *
 * The first vector stores the alpha occupied orbitals and the second vector
 * stores the beta occupied orbitals, both in ascending canonical order.
 */
using CanonicalDeterminantKey = std::pair<std::vector<int>, std::vector<int>>;

/**
 * @brief One fully assembled global determinant term across all components.
 *
 * `alpha_occ` and `beta_occ` are canonical global occupied-orbital lists. The
 * coefficient already includes the component-local signs and the additional
 * inter-component canonicalization parity from concatenating component terms.
 */
struct GlobalOrientationTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

}  // namespace xmvb::vb::exact_separator
