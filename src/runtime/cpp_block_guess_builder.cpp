#include "runtime/cpp_block_guess_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "lapacke.h"

namespace xmvb::vb {

namespace {

bool has_legacy_block_metadata(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.n_blocks > 0 &&
         orbital_preparation_input.block_storage_dimension > 0 &&
         orbital_preparation_input.block_members.size() ==
             xmvb::to_size(orbital_preparation_input.n_blocks) *
                 orbital_preparation_input.block_storage_dimension &&
         orbital_preparation_input.block_orbital_counts.size() ==
             xmvb::to_size(orbital_preparation_input.n_blocks);
}

std::vector<std::vector<int>> build_legacy_orbital_blocks(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<std::vector<int>> blocks;
  blocks.reserve(xmvb::to_size(orbital_preparation_input.n_blocks));

  for (int block_index = 0; block_index < orbital_preparation_input.n_blocks; ++block_index) {
    const int orbital_count =
        orbital_preparation_input.block_orbital_counts[xmvb::to_size(block_index)];
    if (orbital_count <= 0) {
      continue;
    }

    std::vector<int> block;
    block.reserve(xmvb::to_size(orbital_count));
    for (int orbital_offset = 0; orbital_offset < orbital_count; ++orbital_offset) {
      const int orbital_index =
          orbital_preparation_input.block_members
              [xmvb::to_size(block_index) *
                   orbital_preparation_input.block_storage_dimension +
               orbital_offset];
      if (orbital_index < 0 || orbital_index >= orbital_preparation_input.n_orbitals) {
        throw std::runtime_error("legacy block metadata contains an out-of-range orbital index");
      }
      block.push_back(orbital_index);
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
}

int get_block_basis_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int block_index,
    int representative_orbital) {
  if (orbital_preparation_input.block_basis_counts.size() ==
      xmvb::to_size(orbital_preparation_input.n_blocks)) {
    const int stored_count =
        orbital_preparation_input.block_basis_counts[xmvb::to_size(block_index)];
    if (stored_count > 0) {
      return stored_count;
    }
  }
  return get_orbital_basis_count(orbital_preparation_input, representative_orbital);
}

}  // namespace

int get_orbital_basis_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count != 1) {
    return explicit_count;
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<std::vector<int>> detect_orbital_blocks(
    const OrbitalPreparationInput& orbital_preparation_input) {
  if (has_legacy_block_metadata(orbital_preparation_input)) {
    return build_legacy_orbital_blocks(orbital_preparation_input);
  }

  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;

  std::vector<std::vector<int>> blocks;
  blocks.reserve(xmvb::to_size(n_orbitals));
  std::vector<int> block_max_basis_counts;
  block_max_basis_counts.reserve(xmvb::to_size(n_orbitals));

  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int orbital_basis_count =
        get_orbital_basis_count(orbital_preparation_input, orbital_index);
    bool appended_to_existing_block = false;
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      const int representative_orbital = blocks[block_index].front();
      const int representative_basis_count = block_max_basis_counts[block_index];
      int overlap_basis_count = 0;
      for (int coefficient_index = 0; coefficient_index < orbital_basis_count; ++coefficient_index) {
        const int basis_function_index =
            orbital_preparation_input.orbital_basis_index_table
                [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index];
        for (int representative_index = 0;
             representative_index < representative_basis_count;
             ++representative_index) {
          const int representative_basis_function =
              orbital_preparation_input.orbital_basis_index_table
                  [xmvb::to_size(representative_orbital) * n_basis_functions +
                   representative_index];
          if (basis_function_index == representative_basis_function) {
            ++overlap_basis_count;
            break;
          }
        }
      }

      if (overlap_basis_count == orbital_basis_count &&
          orbital_basis_count == representative_basis_count) {
        blocks[block_index].push_back(orbital_index);
        appended_to_existing_block = true;
        break;
      }
    }

    if (!appended_to_existing_block) {
      blocks.push_back({orbital_index});
      block_max_basis_counts.push_back(orbital_basis_count);
    }
  }

  return blocks;
}

std::vector<double> build_ao_normalization(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return build_ao_normalization(
      orbital_preparation_input.active_orbital_overlap_matrix.vector(),
      orbital_preparation_input.n_basis_functions);
}

std::vector<double> build_ao_normalization(
    const std::vector<double>& overlap_matrix,
    int n_basis_functions) {
  if (overlap_matrix.size() !=
      xmvb::to_size(n_basis_functions) * n_basis_functions) {
    throw std::invalid_argument("active_orbital_overlap_matrix size does not match n_basis_functions");
  }

  std::vector<double> ao_normalization(xmvb::to_size(n_basis_functions), 1.0);
  for (int basis_index = 0; basis_index < n_basis_functions; ++basis_index) {
    const double diagonal_value =
        overlap_matrix[xmvb::to_size(basis_index) * n_basis_functions + basis_index];
    if (!(diagonal_value > 0.0)) {
      throw std::runtime_error("encountered non-positive AO overlap diagonal during guess scaling");
    }
    ao_normalization[xmvb::to_size(basis_index)] = std::sqrt(diagonal_value);
  }
  return ao_normalization;
}

void scale_guess_back_to_original_basis(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& ao_normalization,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    for (int coefficient_index = 0; coefficient_index < n_basis_functions; ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index] -
          1;
      if (basis_function_index < 0) {
        break;
      }
      (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                             coefficient_index] *=
          ao_normalization[xmvb::to_size(basis_function_index)];
    }
  }
}

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (orbital_driving_matrix.size() !=
      xmvb::to_size(n_basis_functions) * n_basis_functions) {
    throw std::invalid_argument("orbital_driving_matrix size does not match n_basis_functions");
  }
  if (overlap_matrix.size() !=
      xmvb::to_size(n_basis_functions) * n_basis_functions) {
    throw std::invalid_argument("overlap_matrix size does not match n_basis_functions");
  }
  const auto blocks = detect_orbital_blocks(orbital_preparation_input);
  (void)libcint_input;
  const auto ao_normalization = build_ao_normalization(overlap_matrix, n_basis_functions);
  std::fill(orbital_value_table->begin(), orbital_value_table->end(), 0.0);

  for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const auto& block = blocks[block_index];
    if (block.empty()) {
      continue;
    }
    const int representative_orbital = block.front();
    const int block_basis_count = has_legacy_block_metadata(orbital_preparation_input)
                                      ? get_block_basis_count(
                                            orbital_preparation_input,
                                            static_cast<int>(block_index),
                                            representative_orbital)
                                      : get_orbital_basis_count(
                                            orbital_preparation_input,
                                            representative_orbital);
    if (block_basis_count <= 0) {
      continue;
    }

    std::vector<double> overlap_block(
        xmvb::to_size(block_basis_count) * block_basis_count,
        0.0);
    std::vector<double> fock_block(
        xmvb::to_size(block_basis_count) * block_basis_count,
        0.0);
    std::vector<double> eigenvalues(xmvb::to_size(block_basis_count), 0.0);
    for (int local_i = 0; local_i < block_basis_count; ++local_i) {
      const int global_i =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(representative_orbital) * n_basis_functions + local_i] -
          1;
      for (int local_j = 0; local_j < block_basis_count; ++local_j) {
        const int global_j =
            orbital_preparation_input.orbital_basis_index_table
                [xmvb::to_size(representative_orbital) * n_basis_functions + local_j] -
            1;
        overlap_block[xmvb::to_size(local_i) * block_basis_count + local_j] =
            overlap_matrix[xmvb::to_size(global_i) * n_basis_functions + global_j];
        fock_block[xmvb::to_size(local_i) * block_basis_count + local_j] =
            orbital_driving_matrix[xmvb::to_size(global_i) * n_basis_functions +
                                   global_j];
      }
    }

    if (LAPACKE_dsygv(
            LAPACK_COL_MAJOR,
            1,
            'V',
            'U',
            block_basis_count,
            fock_block.data(),
            block_basis_count,
            overlap_block.data(),
            block_basis_count,
            eigenvalues.data()) != 0) {
      throw std::runtime_error("LAPACKE_dsygv failed while building C++ block guess");
    }

    for (std::size_t orbital_offset = 0; orbital_offset < block.size(); ++orbital_offset) {
      const int orbital_index = block[orbital_offset];
      std::memcpy(
          orbital_value_table->data() +
              xmvb::to_size(orbital_index) * n_basis_functions,
          fock_block.data() + orbital_offset * xmvb::to_size(block_basis_count),
          sizeof(double) * xmvb::to_size(block_basis_count));
    }
  }

  scale_guess_back_to_original_basis(
      orbital_preparation_input,
      ao_normalization,
      orbital_value_table);
}

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  build_block_matrix_guess(
      libcint_input,
      orbital_driving_matrix,
      orbital_preparation_input.active_orbital_overlap_matrix.vector(),
      orbital_preparation_input,
      orbital_value_table);
}

std::vector<double> build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<double> orbital_value_table(
      xmvb::to_size(orbital_preparation_input.n_basis_functions) *
          orbital_preparation_input.n_orbitals,
      0.0);
  build_block_matrix_guess(
      libcint_input,
      orbital_driving_matrix,
      overlap_matrix,
      orbital_preparation_input,
      &orbital_value_table);
  return orbital_value_table;
}

std::vector<double> build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<double> orbital_value_table(
      xmvb::to_size(orbital_preparation_input.n_basis_functions) *
          orbital_preparation_input.n_orbitals,
      0.0);
  build_block_matrix_guess(
      libcint_input,
      orbital_driving_matrix,
      orbital_preparation_input,
      &orbital_value_table);
  return orbital_value_table;
}

}  // namespace xmvb::vb
