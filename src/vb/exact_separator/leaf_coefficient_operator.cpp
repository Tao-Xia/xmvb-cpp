#include "vb/exact_separator/leaf_coefficient_operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace xmvb::vb::exact_separator {

namespace {

void hash_combine(std::size_t* seed, std::size_t value) {
  if (seed == nullptr) {
    throw std::invalid_argument("hash seed must not be null");
  }
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

std::size_t hash_occ_list(const std::vector<int>& occ) {
  std::size_t seed = occ.size();
  for (const int orbital : occ) {
    hash_combine(&seed, xmvb::to_size(orbital + 0x10000));
  }
  return seed;
}

struct SpinPairStateKeyHasher {
  std::size_t operator()(const SpinPairStateKey& key) const {
    std::size_t seed = 0U;
    hash_combine(&seed, hash_occ_list(key.left_occ));
    hash_combine(&seed, hash_occ_list(key.right_occ));
    return seed;
  }
};

std::uint64_t pack_state_pair_key(int alpha_state_index, int beta_state_index) {
  if (alpha_state_index < 0 || beta_state_index < 0) {
    throw std::invalid_argument("state indices must be non-negative");
  }
  return
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(alpha_state_index)) << 32) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(beta_state_index));
}

int intern_spin_pair_state(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    std::vector<SpinPairStateKey>* states,
    std::unordered_map<SpinPairStateKey, int, SpinPairStateKeyHasher>* state_index) {
  if (states == nullptr || state_index == nullptr) {
    throw std::invalid_argument("state tables must not be null");
  }
  const SpinPairStateKey key{left_occ, right_occ};
  const auto iterator = state_index->find(key);
  if (iterator != state_index->end()) {
    return iterator->second;
  }

  const int index = static_cast<int>(states->size());
  states->push_back(key);
  state_index->emplace(states->back(), index);
  return index;
}

}  // namespace

bool operator==(const SpinPairStateKey& left, const SpinPairStateKey& right) {
  return left.left_occ == right.left_occ &&
      left.right_occ == right.right_occ;
}

ComponentSpinCoefficientOperator build_component_spin_coefficient_operator(
    const ComponentData& component) {
  ComponentSpinCoefficientOperator result;

  std::unordered_map<SpinPairStateKey, int, SpinPairStateKeyHasher> alpha_state_index;
  std::unordered_map<SpinPairStateKey, int, SpinPairStateKeyHasher> beta_state_index;
  std::unordered_map<std::uint64_t, double> coefficient_by_state_pair;

  result.alpha_states.reserve(
      component.left_orientation_terms.size() *
      component.right_orientation_terms.size());
  result.beta_states.reserve(
      component.left_orientation_terms.size() *
      component.right_orientation_terms.size());
  coefficient_by_state_pair.reserve(
      component.left_orientation_terms.size() *
      component.right_orientation_terms.size());

  for (const auto& left_term : component.left_orientation_terms) {
    for (const auto& right_term : component.right_orientation_terms) {
      const double coefficient = left_term.coefficient * right_term.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }
      ++result.raw_nonzero_pair_count;

      const int alpha_index = intern_spin_pair_state(
          left_term.alpha_occ,
          right_term.alpha_occ,
          &result.alpha_states,
          &alpha_state_index);
      const int beta_index = intern_spin_pair_state(
          left_term.beta_occ,
          right_term.beta_occ,
          &result.beta_states,
          &beta_state_index);
      coefficient_by_state_pair[pack_state_pair_key(alpha_index, beta_index)] += coefficient;
    }
  }

  result.entries.reserve(coefficient_by_state_pair.size());
  for (const auto& [packed_key, coefficient] : coefficient_by_state_pair) {
    if (std::abs(coefficient) <= 1.0e-15) {
      continue;
    }
    result.entries.push_back({
        .alpha_state_index = static_cast<int>(packed_key >> 32U),
        .beta_state_index = static_cast<int>(packed_key & 0xffffffffU),
        .coefficient = coefficient,
    });
  }

  std::sort(
      result.entries.begin(),
      result.entries.end(),
      [](const auto& left, const auto& right) {
        return std::tie(left.alpha_state_index, left.beta_state_index) <
            std::tie(right.alpha_state_index, right.beta_state_index);
      });
  return result;
}

}  // namespace xmvb::vb::exact_separator
