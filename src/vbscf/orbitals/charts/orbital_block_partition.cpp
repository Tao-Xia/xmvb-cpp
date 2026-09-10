#include "vbscf/orbitals/charts/orbital_block_partition.hpp"

#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

bool has_block_metadata(
    const OrbitalPreparationInput& input) {
  return input.n_blocks > 0 && input.block_storage_dimension > 0 &&
      input.block_members.size() ==
          input.n_blocks * input.block_storage_dimension &&
      input.block_orbital_counts.size() == input.n_blocks;
}

std::vector<std::vector<int>> build_metadata_orbital_blocks(
    const OrbitalPreparationInput& input) {
  std::vector<std::vector<int>> blocks;
  blocks.reserve(input.n_blocks);
  for (std::size_t block_index = 0; block_index < input.n_blocks;
       ++block_index) {
    const int orbital_count = input.block_orbital_counts[block_index];
    if (orbital_count <= 0) {
      continue;
    }
    std::vector<int> block;
    block.reserve(orbital_count);
    for (int offset = 0; offset < orbital_count; ++offset) {
      const int orbital = input.block_members[
          block_index * input.block_storage_dimension + offset];
      if (orbital < 0 || orbital >= input.n_orbitals) {
        throw std::runtime_error(
            "stored block metadata contains an out-of-range orbital index");
      }
      block.push_back(orbital);
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

}  // namespace

std::vector<std::vector<int>> detect_orbital_blocks(
    const OrbitalPreparationInput& input) {
  if (has_block_metadata(input)) {
    return build_metadata_orbital_blocks(input);
  }

  const int n_orbitals = input.n_orbitals;
  const int n_basis_functions = input.n_basis_functions;
  std::vector<std::vector<int>> blocks;
  blocks.reserve(n_orbitals);
  std::vector<int> block_basis_counts;
  block_basis_counts.reserve(n_orbitals);

  for (int orbital = 0; orbital < n_orbitals; ++orbital) {
    const int basis_count =
        stored_sparse_orbital_coefficient_count(input, orbital);
    bool appended = false;
    for (std::size_t block_index = 0; block_index < blocks.size();
         ++block_index) {
      const int representative = blocks[block_index].front();
      const int representative_count = block_basis_counts[block_index];
      int overlap_count = 0;
      for (int coefficient = 0; coefficient < basis_count; ++coefficient) {
        const int basis = input.orbital_basis_index_table[
            orbital * n_basis_functions + coefficient];
        for (int representative_coefficient = 0;
             representative_coefficient < representative_count;
             ++representative_coefficient) {
          const int representative_basis = input.orbital_basis_index_table[
              representative * n_basis_functions +
              representative_coefficient];
          if (basis == representative_basis) {
            ++overlap_count;
            break;
          }
        }
      }
      if (overlap_count == basis_count &&
          basis_count == representative_count) {
        blocks[block_index].push_back(orbital);
        appended = true;
        break;
      }
    }
    if (!appended) {
      blocks.push_back({orbital});
      block_basis_counts.push_back(basis_count);
    }
  }
  return blocks;
}

}  // namespace xmvb::vb
