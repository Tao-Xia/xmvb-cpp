#pragma once

#include <cstdint>
#include <vector>

#include "vb/exact_separator/component_data.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Unique one-spin paired state for one component-side pair.
 *
 * The lists keep the exact occupied-orbital ordering already used by the
 * one-leaf separator kernel: left/bra orbitals in component block order and
 * right/ket orbitals in component block order.
 */
struct SpinPairStateKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
};

bool operator==(const SpinPairStateKey& left, const SpinPairStateKey& right);

/**
 * @brief Sparse coefficient entry of one component operator.
 *
 * `alpha_state_index` and `beta_state_index` point into the corresponding
 * unique state tables of the same `ComponentSpinCoefficientOperator`.
 */
struct ComponentSpinCoefficientEntry {
  int alpha_state_index = -1;
  int beta_state_index = -1;
  double coefficient = 0.0;
};

/**
 * @brief Exact compressed coefficient operator for one component.
 *
 * The operator collects all nonzero left/right orientation-term products and
 * merges duplicates that project to the same paired alpha/beta state labels.
 */
struct ComponentSpinCoefficientOperator {
  std::vector<SpinPairStateKey> alpha_states;
  std::vector<SpinPairStateKey> beta_states;
  std::vector<ComponentSpinCoefficientEntry> entries;
  std::uint64_t raw_nonzero_pair_count = 0;
};

/**
 * @brief Builds the exact sparse coefficient operator for one component.
 */
ComponentSpinCoefficientOperator build_component_spin_coefficient_operator(
    const ComponentData& component);

}  // namespace xmvb::vb::exact_separator
