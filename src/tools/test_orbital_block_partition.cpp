#include <stdexcept>
#include <string>
#include <vector>

#include "vbscf/orbitals/charts/orbital_block_partition.hpp"

namespace {

void require_equal(
    const std::vector<std::vector<int>>& actual,
    const std::vector<std::vector<int>>& expected,
    const char* label) {
  if (actual != expected) {
    throw std::runtime_error(std::string(label) + " block partition mismatch");
  }
}

xmvb::vb::OrbitalPreparationInput make_support_input() {
  xmvb::vb::OrbitalPreparationInput input;
  input.n_basis_functions = 5;
  input.n_orbitals = 4;
  input.orbital_basis_counts = {2, 2, 2, 3};
  input.orbital_basis_index_table = {
      1, 2, 0, 0, 0,
      1, 2, 0, 0, 0,
      2, 3, 0, 0, 0,
      1, 2, 3, 0, 0,
  };
  return input;
}

}  // namespace

int main() {
  auto input = make_support_input();
  require_equal(
      xmvb::vb::detect_orbital_blocks(input),
      {{0, 1}, {2}, {3}},
      "support-inferred");

  input.n_blocks = 2;
  input.block_storage_dimension = 3;
  input.block_orbital_counts = {2, 2};
  input.block_members = {2, 0, -1, 1, 3, -1};
  require_equal(
      xmvb::vb::detect_orbital_blocks(input),
      {{2, 0}, {1, 3}},
      "legacy-metadata");
  return 0;
}
