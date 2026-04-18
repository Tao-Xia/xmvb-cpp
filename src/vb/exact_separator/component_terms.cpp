#include "vb/exact_separator/component_terms.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace xmvb::vb::exact_separator {

namespace {

std::vector<int> concatenate_occ_lists(
    const std::vector<const std::vector<int>*>& occ_lists) {
  std::size_t total_size = 0;
  for (const auto* occ : occ_lists) {
    if (occ == nullptr) {
      throw std::invalid_argument("occupied-orbital list pointer must not be null");
    }
    total_size += occ->size();
  }

  std::vector<int> concatenated;
  concatenated.reserve(total_size);
  for (const auto* occ : occ_lists) {
    concatenated.insert(concatenated.end(), occ->begin(), occ->end());
  }
  return concatenated;
}

void enumerate_global_terms_recursive(
    const std::vector<ComponentData>& ordered_components,
    bool use_left_terms,
    int component_index,
    double running_coefficient,
    std::vector<const std::vector<int>*>* alpha_occ_lists,
    std::vector<const std::vector<int>*>* beta_occ_lists,
    std::map<CanonicalDeterminantKey, double>* merged_terms) {
  // This recursion enumerates one side of the exact raw-VB structure-space
  // expansion in component order. At the leaves, the component-local occupied
  // lists are concatenated, canonicalized, and merged into a canonical global
  // determinant map.
  if (alpha_occ_lists == nullptr || beta_occ_lists == nullptr || merged_terms == nullptr) {
    throw std::invalid_argument("global-term enumeration outputs must not be null");
  }
  if (component_index == static_cast<int>(ordered_components.size())) {
    std::vector<int> alpha_occ = concatenate_occ_lists(*alpha_occ_lists);
    std::vector<int> beta_occ = concatenate_occ_lists(*beta_occ_lists);
    const int inter_component_parity =
        canonicalization_parity(alpha_occ) ^ canonicalization_parity(beta_occ);
    std::sort(alpha_occ.begin(), alpha_occ.end());
    std::sort(beta_occ.begin(), beta_occ.end());
    (*merged_terms)[{alpha_occ, beta_occ}] +=
        running_coefficient * parity_sign(inter_component_parity);
    return;
  }

  const ComponentData& component =
      ordered_components[xmvb::to_size(component_index)];
  const std::vector<OrientationTerm>& orientation_terms =
      use_left_terms ? component.left_orientation_terms : component.right_orientation_terms;
  for (const auto& term : orientation_terms) {
    if (std::abs(term.coefficient) <= 1.0e-15) {
      continue;
    }
    alpha_occ_lists->push_back(&term.alpha_occ);
    beta_occ_lists->push_back(&term.beta_occ);
    enumerate_global_terms_recursive(
        ordered_components,
        use_left_terms,
        component_index + 1,
        running_coefficient * term.coefficient,
        alpha_occ_lists,
        beta_occ_lists,
        merged_terms);
    alpha_occ_lists->pop_back();
    beta_occ_lists->pop_back();
  }
}

}  // namespace

double parity_sign(int parity) {
  return (parity % 2 == 0) ? 1.0 : -1.0;
}

int canonicalization_parity(const std::vector<int>& occupied_orbitals) {
  // The component-local orientation terms are concatenated in component order,
  // while the determinant kernels expect canonical ascending occupied-orbital
  // order. This parity counts the sign needed to sort the concatenated block
  // order into canonical order.
  int parity = 0;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals.size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals.size();
         ++right_index) {
      if (occupied_orbitals[left_index] > occupied_orbitals[right_index]) {
        parity ^= 1;
      }
    }
  }
  return parity;
}

std::vector<GlobalOrientationTerm> build_global_orientation_terms(
    const std::vector<ComponentData>& ordered_components,
    bool use_left_terms) {
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }

  std::map<CanonicalDeterminantKey, double> merged_terms;
  std::vector<const std::vector<int>*> alpha_occ_lists;
  std::vector<const std::vector<int>*> beta_occ_lists;
  alpha_occ_lists.reserve(ordered_components.size());
  beta_occ_lists.reserve(ordered_components.size());
  enumerate_global_terms_recursive(
      ordered_components,
      use_left_terms,
      0,
      1.0,
      &alpha_occ_lists,
      &beta_occ_lists,
      &merged_terms);

  std::vector<GlobalOrientationTerm> global_terms;
  global_terms.reserve(merged_terms.size());
  for (const auto& [key, coefficient] : merged_terms) {
    if (std::abs(coefficient) <= 1.0e-15) {
      continue;
    }
    global_terms.push_back({key.first, key.second, coefficient});
  }
  return global_terms;
}

}  // namespace xmvb::vb::exact_separator
