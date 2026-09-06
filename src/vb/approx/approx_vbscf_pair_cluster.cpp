#include "vb/approx/approx_vbscf_pair_cluster.hpp"

#include <utility>
#include <vector>

#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"

namespace xmvb::vb {

namespace {

std::vector<OrbitalPair> make_orbital_pairs(
    const std::vector<std::pair<int, int>>& pairs) {
  std::vector<OrbitalPair> orbital_pairs;
  orbital_pairs.reserve(pairs.size());
  for (const auto& pair : pairs) {
    orbital_pairs.emplace_back(pair.first, pair.second);
  }
  return orbital_pairs;
}

}  // namespace

int approx_vbscf_active_pair_count(int n_active_orbitals) {
  return n_active_orbitals * (n_active_orbitals + 1) / 2;
}

std::pair<int, int> unpack_approx_vbscf_active_pair(int pair_index) {
  int second = 0;
  while ((second + 1) * (second + 2) / 2 <= pair_index) {
    ++second;
  }
  const int first = pair_index - second * (second + 1) / 2;
  return {first, second};
}

bool approx_vbscf_pair_cluster_is_compatible(
    const std::vector<std::pair<int, int>>& pairs) {
  std::vector<int> orbital_indices;
  orbital_indices.reserve(2 * pairs.size());
  for (const auto& pair : pairs) {
    orbital_indices.push_back(pair.first);
    orbital_indices.push_back(pair.second);
  }

  for (int probe_index = 0;
       probe_index < static_cast<int>(orbital_indices.size());
       ++probe_index) {
    int count = 0;
    for (int orbital_index = 0;
         orbital_index < static_cast<int>(orbital_indices.size());
         ++orbital_index) {
      if (orbital_indices[orbital_index] == orbital_indices[probe_index]) {
        ++count;
      }
    }
    if (count > 2) {
      return false;
    }
  }
  return true;
}

double evaluate_approx_vbscf_pair_cluster_overlap(
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<double>& active_orbital_overlap_matrix,
    int n_active_orbitals) {
  const auto determinant_terms =
      enumerate_legacy_determinant_terms(make_orbital_pairs(pairs));
  const Eigen::Map<const Eigen::MatrixXd> active_overlap(
      active_orbital_overlap_matrix.data(),
      n_active_orbitals,
      n_active_orbitals);
  const DeterminantOverlapResolver overlap_resolver;
  return legacy_structure_overlap(
      determinant_terms,
      determinant_terms,
      active_overlap,
      overlap_resolver);
}

ApproxVbScfPairClusterMatrixElement
evaluate_approx_vbscf_pair_cluster_matrix_element(
    const std::vector<std::pair<int, int>>& pairs,
    const ApproxVbScfPairClusterHamiltonianInput& input) {
  ApproxVbScfPairClusterMatrixElement result;
  if (!approx_vbscf_pair_cluster_is_compatible(pairs)) {
    return result;
  }

  const auto determinant_terms =
      enumerate_legacy_determinant_terms(make_orbital_pairs(pairs));
  const FullDeterminantPairEvaluator determinant_pair_evaluator;

  for (const auto& left_term : determinant_terms) {
    for (const auto& right_term : determinant_terms) {
      const auto determinant_pair =
          determinant_pair_evaluator.evaluate(
              left_term.alpha_occ,
              right_term.alpha_occ,
              left_term.beta_occ,
              right_term.beta_occ,
              *input.active_orbital_overlap_matrix,
              *input.active_one_electron_integrals,
              input.n_active_orbitals,
              *input.active_two_electron_result,
              false);
      const double coefficient =
          left_term.coefficient * right_term.coefficient;
      result.overlap +=
          coefficient * determinant_pair.overlap_determinant;
      result.one_electron_hamiltonian +=
          coefficient * determinant_pair.one_electron_hamiltonian;
      result.total_hamiltonian +=
          coefficient * determinant_pair.total_hamiltonian;
    }
  }

  return result;
}

}  // namespace xmvb::vb
