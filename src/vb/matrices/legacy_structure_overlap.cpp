#include "vb/matrices/legacy_structure_overlap.hpp"

#include <map>
#include <stdexcept>
#include <utility>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb {

namespace {

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals->size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap((*occupied_orbitals)[left_index], (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

double determinant_overlap(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& overlap_matrix,
    const DeterminantOverlapResolver& overlap_resolver) {
  const auto overlap_submatrix = build_overlap_submatrix(
      left_occ,
      right_occ,
      overlap_matrix);
  return overlap_resolver.resolve_matrix(overlap_submatrix).overlap_determinant;
}

}  // namespace

std::vector<LegacyStructureDeterminantTerm> enumerate_legacy_determinant_terms(
    const std::vector<OrbitalPair>& pairs) {
  const int n_pairs = static_cast<int>(pairs.size());
  std::vector<std::vector<int>> determinants;
  std::vector<int> initial_determinant(2 * n_pairs, 0);
  for (int pair_index = 0; pair_index < n_pairs; ++pair_index) {
    initial_determinant[pair_index] =
        pairs[pair_index].first;
    initial_determinant[pair_index + n_pairs] =
        pairs[pair_index].second;
  }
  determinants.push_back(initial_determinant);

  for (int pair_index = 0; pair_index < n_pairs; ++pair_index) {
    if (initial_determinant[pair_index] ==
        initial_determinant[pair_index + n_pairs]) {
      continue;
    }
    const std::size_t current_size = determinants.size();
    determinants.reserve(current_size * 2);
    for (std::size_t determinant_index = 0; determinant_index < current_size; ++determinant_index) {
      auto swapped_determinant = determinants[determinant_index];
      std::swap(
          swapped_determinant[pair_index],
          swapped_determinant[pair_index + n_pairs]);
      determinants.push_back(std::move(swapped_determinant));
    }
  }

  std::map<std::pair<std::vector<int>, std::vector<int>>, double> accumulated_terms;
  for (const auto& determinant : determinants) {
    std::vector<int> alpha_occ(
        determinant.begin(),
        determinant.begin() + n_pairs);
    std::vector<int> beta_occ(
        determinant.begin() + n_pairs,
        determinant.end());
    const int alpha_sign = canonicalize_spin_string(&alpha_occ);
    const int beta_sign = canonicalize_spin_string(&beta_occ);
    accumulated_terms[{alpha_occ, beta_occ}] +=
        static_cast<double>(alpha_sign * beta_sign);
  }

  std::vector<LegacyStructureDeterminantTerm> result;
  result.reserve(accumulated_terms.size());
  for (const auto& [key, coefficient] : accumulated_terms) {
    if (std::abs(coefficient) <= 1.0e-12) {
      continue;
    }
    LegacyStructureDeterminantTerm term;
    term.alpha_occ = key.first;
    term.beta_occ = key.second;
    term.coefficient = coefficient;
    result.push_back(std::move(term));
  }
  return result;
}

std::vector<LegacyStructureDeterminantTerm> remap_legacy_determinant_terms(
    const std::vector<LegacyStructureDeterminantTerm>& terms,
    const std::map<int, int>& orbital_index_remap) {
  std::vector<LegacyStructureDeterminantTerm> remapped_terms;
  remapped_terms.reserve(terms.size());
  for (const auto& term : terms) {
    LegacyStructureDeterminantTerm remapped_term;
    remapped_term.coefficient = term.coefficient;
    remapped_term.alpha_occ.reserve(term.alpha_occ.size());
    remapped_term.beta_occ.reserve(term.beta_occ.size());
    for (const int orbital_index : term.alpha_occ) {
      const auto iterator = orbital_index_remap.find(orbital_index);
      if (iterator == orbital_index_remap.end()) {
        throw std::runtime_error("missing alpha orbital remap for legacy structure term");
      }
      remapped_term.alpha_occ.push_back(iterator->second);
    }
    for (const int orbital_index : term.beta_occ) {
      const auto iterator = orbital_index_remap.find(orbital_index);
      if (iterator == orbital_index_remap.end()) {
        throw std::runtime_error("missing beta orbital remap for legacy structure term");
      }
      remapped_term.beta_occ.push_back(iterator->second);
    }
    remapped_terms.push_back(std::move(remapped_term));
  }
  return remapped_terms;
}

double legacy_structure_overlap(
    const std::vector<LegacyStructureDeterminantTerm>& left_terms,
    const std::vector<LegacyStructureDeterminantTerm>& right_terms,
    const Eigen::MatrixXd& overlap_matrix,
    const DeterminantOverlapResolver& overlap_resolver) {
  double overlap_value = 0.0;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const double alpha_overlap = determinant_overlap(
          left_term.alpha_occ,
          right_term.alpha_occ,
          overlap_matrix,
          overlap_resolver);
      const double beta_overlap = determinant_overlap(
          left_term.beta_occ,
          right_term.beta_occ,
          overlap_matrix,
          overlap_resolver);
      overlap_value +=
          left_term.coefficient * right_term.coefficient * alpha_overlap * beta_overlap;
    }
  }
  return overlap_value;
}

}  // namespace xmvb::vb
