#include "vbscf/determinants/pairs/traversal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace xmvb::vb {
namespace {

int positional_distance(
    const std::vector<int>& left,
    const std::vector<int>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument(
        "pair traversal strings have different electron counts");
  }
  int distance = 0;
  for (std::size_t position = 0; position < left.size(); ++position) {
    distance += left[position] != right[position] ? 1 : 0;
  }
  return distance;
}

int replacement_distance(
    const std::vector<int>& left,
    const std::vector<int>& right) {
  int common = 0;
  for (const int left_orbital : left) {
    common += std::find(right.begin(), right.end(), left_orbital) != right.end()
        ? 1
        : 0;
  }
  return static_cast<int>(left.size()) - common;
}

}  // namespace

std::vector<int> build_pair_update_traversal(
    const std::vector<std::vector<int>>& unique_spin_strings,
    int left_index,
    int right_begin,
    int right_end) {
  const int n_strings = static_cast<int>(unique_spin_strings.size());
  if (left_index < 0 || left_index >= n_strings || right_begin < 0 ||
      right_end <= right_begin || right_end > n_strings) {
    throw std::invalid_argument("pair traversal bounds are invalid");
  }

  const int tile_size = right_end - right_begin;
  std::vector<int> traversal;
  traversal.reserve(tile_size);
  std::vector<char> visited(tile_size, 0);

  int current = left_index >= right_begin && left_index < right_end
      ? left_index
      : right_begin;
  traversal.push_back(current);
  visited[current - right_begin] = 1;

  while (static_cast<int>(traversal.size()) < tile_size) {
    int best = -1;
    int best_rank = std::numeric_limits<int>::max();
    int best_replacements = std::numeric_limits<int>::max();
    for (int candidate = right_begin; candidate < right_end; ++candidate) {
      if (visited[candidate - right_begin]) {
        continue;
      }
      const int rank = positional_distance(
          unique_spin_strings[current], unique_spin_strings[candidate]);
      const int replacements = replacement_distance(
          unique_spin_strings[current], unique_spin_strings[candidate]);
      if (rank < best_rank ||
          (rank == best_rank && replacements < best_replacements) ||
          (rank == best_rank && replacements == best_replacements &&
           candidate < best)) {
        best = candidate;
        best_rank = rank;
        best_replacements = replacements;
      }
    }
    visited[best - right_begin] = 1;
    traversal.push_back(best);
    current = best;
  }
  return traversal;
}

}  // namespace xmvb::vb
