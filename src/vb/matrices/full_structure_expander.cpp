#include "vb/matrices/full_structure_expander.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/vb_model_flags.hpp"

namespace xmvb::vb {

namespace {

/**
 * @brief Canonical determinant key used for deduplication.
 */
struct DeterminantKey {
  std::vector<int> alpha_orbitals;
  std::vector<int> beta_orbitals;

  bool operator==(const DeterminantKey& other) const {
    return alpha_orbitals == other.alpha_orbitals &&
           beta_orbitals == other.beta_orbitals;
  }
};

struct DeterminantKeyHasher {
  std::size_t operator()(const DeterminantKey& determinant_key) const {
    std::size_t hash_value = 0;
    for (const int orbital_index : determinant_key.alpha_orbitals) {
      hash_value = hash_value * 1315423911u + orbital_index + 257;
    }
    hash_value = hash_value * 2654435761u + 17u;
    for (const int orbital_index : determinant_key.beta_orbitals) {
      hash_value = hash_value * 1315423911u + orbital_index + 257;
    }
    return hash_value;
  }
};

/**
 * @brief Sorts occupied orbitals in ascending order and returns the parity sign.
 */
int canonicalize_spin_string(std::vector<int>& occupied_orbitals) {
  int permutation_sign = 1;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1; right_index < occupied_orbitals.size();
         ++right_index) {
      if (occupied_orbitals[left_index] > occupied_orbitals[right_index]) {
        std::swap(occupied_orbitals[left_index], occupied_orbitals[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

/**
 * @brief Enumerates paired-electron determinants for a single structure.
 *
 * This is the C++ translation of the legacy `str2det` logic used on the
 * paired active orbitals of a VB structure.
 */
std::vector<std::vector<int>> expand_paired_active_orbitals(
    const std::vector<int>& paired_active_orbitals,
    int n_active_beta_electrons,
    int wavefunction_type) {
  if (static_cast<int>(paired_active_orbitals.size()) != 2 * n_active_beta_electrons) {
    throw std::invalid_argument("paired active orbital count does not match beta electron count");
  }

  if (n_active_beta_electrons == 0) {
    return {{}};
  }

  std::vector<std::vector<int>> determinants;
  if (wavefunction_type ==
      wavefunction_type_code(WavefunctionType::Determinant)) {
    determinants.push_back(paired_active_orbitals);
    return determinants;
  }

  std::vector<int> initial_determinant(2 * n_active_beta_electrons, 0);
  for (int electron_index = 0; electron_index < n_active_beta_electrons; ++electron_index) {
    initial_determinant[electron_index] =
        paired_active_orbitals[2 * electron_index];
    initial_determinant[electron_index + n_active_beta_electrons] =
        paired_active_orbitals[2 * electron_index + 1];
  }
  determinants.push_back(initial_determinant);

  for (int electron_index = 0; electron_index < n_active_beta_electrons; ++electron_index) {
    if (initial_determinant[electron_index] ==
        initial_determinant[electron_index + n_active_beta_electrons]) {
      continue;
    }

    const std::size_t current_size = determinants.size();
    determinants.reserve(current_size * 2);
    for (std::size_t determinant_index = 0; determinant_index < current_size; ++determinant_index) {
      auto swapped_determinant = determinants[determinant_index];
      std::swap(
          swapped_determinant[electron_index],
          swapped_determinant[electron_index + n_active_beta_electrons]);
      determinants.push_back(std::move(swapped_determinant));
    }
  }

  return determinants;
}

}  // namespace

FullDeterminantStructureData FullDeterminantStructureExpander::expand(
    const RawStructureData& raw_structure_data) const {
  if (raw_structure_data.n_total_electrons <= 0 || raw_structure_data.n_active_electrons <= 0) {
    throw std::invalid_argument("raw structure data must define positive electron counts");
  }
  if (raw_structure_data.n_structures <= 0 ||
      raw_structure_data.raw_structure_orbitals.size() !=
          raw_structure_data.flat_orbital_count()) {
    throw std::invalid_argument("raw structure count does not match stored structures");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_spin_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_spin_open_shell_electrons) / 2;
  const int n_active_alpha_electrons =
      raw_structure_data.n_active_electrons - n_active_beta_electrons;

  if (raw_structure_data.vb_function_type !=
          vb_function_type_code(VbFunctionType::Determinant) &&
      raw_structure_data.wavefunction_type !=
          wavefunction_type_code(WavefunctionType::Determinant)) {
    throw std::invalid_argument(
        "full determinant expansion currently supports determinant VB or determinant wavefunctions only");
  }

  FullDeterminantStructureData result;
  result.n_structures = raw_structure_data.n_structures;

  std::unordered_map<DeterminantKey, int, DeterminantKeyHasher> determinant_to_index;

  for (int structure_index = 0; structure_index < raw_structure_data.n_structures; ++structure_index) {
    const int* raw_structure = raw_structure_data.structure_orbitals_data(structure_index);

    std::vector<int> active_structure_orbitals(
        raw_structure_data.n_active_electrons,
        0);
    for (int active_index = 0; active_index < raw_structure_data.n_active_electrons; ++active_index) {
      active_structure_orbitals[active_index] =
          raw_structure[
              active_index + 2 * n_inactive_doubly_occupied_orbitals] -
          n_inactive_doubly_occupied_orbitals - 1;
    }

    std::vector<int> paired_active_orbitals(
        active_structure_orbitals.begin(),
        active_structure_orbitals.begin() + 2 * n_active_beta_electrons);
    const auto paired_determinants = expand_paired_active_orbitals(
        paired_active_orbitals,
        n_active_beta_electrons,
        raw_structure_data.wavefunction_type);

    for (const auto& paired_determinant : paired_determinants) {
      std::vector<int> alpha_orbitals(
          paired_determinant.begin(),
          paired_determinant.begin() + n_active_beta_electrons);
      alpha_orbitals.insert(
          alpha_orbitals.end(),
          active_structure_orbitals.begin() + 2 * n_active_beta_electrons,
          active_structure_orbitals.end());

      std::vector<int> beta_orbitals(
          paired_determinant.begin() + n_active_beta_electrons,
          paired_determinant.end());

      const int alpha_sign = canonicalize_spin_string(alpha_orbitals);
      const int beta_sign = canonicalize_spin_string(beta_orbitals);
      const double expansion_coefficient = static_cast<double>(alpha_sign * beta_sign);

      DeterminantKey determinant_key{alpha_orbitals, beta_orbitals};
      const auto [determinant_iterator, inserted] =
          determinant_to_index.emplace(determinant_key, static_cast<int>(result.alpha_det.size()));
      if (inserted) {
        result.alpha_det.push_back(alpha_orbitals);
        result.beta_det.push_back(beta_orbitals);
        result.determinant_to_structure_terms.push_back({});
      }

      result.determinant_to_structure_terms[determinant_iterator->second]
          .push_back({
              .structure_index = structure_index,
              .coefficient = expansion_coefficient,
          });
    }
  }

  return result;
}

FullDeterminantStructureData FullDeterminantStructureExpander::expand_subset(
    const RawStructureData& raw_structure_data,
    const std::vector<int>& selected_structure_indices) const {
  return expand(
      build_raw_structure_subset(raw_structure_data, selected_structure_indices));
}

}  // namespace xmvb::vb
