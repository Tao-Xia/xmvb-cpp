#include "runtime/cpp_initial_guess_builder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lapacke.h"
#include "runtime/cpp_block_guess_builder.hpp"
#include "runtime/cpp_restricted_hartree_fock.hpp"
#include "runtime_c/local_runtime/cint_compat.h"

#ifdef atm
#undef atm
#endif

#ifdef bas
#undef bas
#endif

#include "vb/vb.h"

namespace xmvb::vb {

namespace {

std::string uppercase_copy(const std::string& text) {
  std::string upper = text;
  for (char& character : upper) {
    character = static_cast<char>(
        std::toupper(static_cast<unsigned char>(character)));
  }
  return upper;
}

std::string strip_inline_comment(const std::string& line) {
  const std::size_t comment_position = line.find('#');
  if (comment_position == std::string::npos) {
    return line;
  }
  return line.substr(0, comment_position);
}

std::vector<std::string> split_whitespace_tokens(const std::string& line) {
  std::istringstream token_stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (token_stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

int parse_guess_integer(const std::string& token) {
  std::size_t consumed_characters = 0;
  const int value = std::stoi(token, &consumed_characters);
  if (consumed_characters != token.size()) {
    throw std::runtime_error("invalid integer token in $GUS section: " + token);
  }
  return value;
}

double parse_guess_double(const std::string& token) {
  std::string normalized_token = token;
  for (char& character : normalized_token) {
    if (character == 'd' || character == 'D') {
      character = 'E';
    }
  }
  std::size_t consumed_characters = 0;
  const double value = std::stod(normalized_token, &consumed_characters);
  if (consumed_characters != normalized_token.size()) {
    throw std::runtime_error("invalid floating-point token in $GUS section: " + token);
  }
  return value;
}

void expand_orbital_tokens(
    const std::vector<std::string>& tokens,
    std::vector<int>* expanded_values) {
  if (expanded_values == nullptr) {
    throw std::invalid_argument("expanded_values must not be null");
  }

  for (const std::string& token : tokens) {
    const std::size_t colon_position = token.find(':');
    const std::size_t hyphen_position = token.find('-');
    const std::size_t star_position = token.find('*');
    if (colon_position != std::string::npos) {
      const int first_value = parse_guess_integer(token.substr(0, colon_position));
      const int last_value = parse_guess_integer(token.substr(colon_position + 1));
      if (last_value < first_value) {
        throw std::runtime_error("descending ':' range is not supported in $GUS section: " + token);
      }
      for (int value = first_value; value <= last_value; ++value) {
        expanded_values->push_back(value);
        expanded_values->push_back(value);
      }
      continue;
    }
    if (hyphen_position != std::string::npos) {
      const int first_value = parse_guess_integer(token.substr(0, hyphen_position));
      const int last_value = parse_guess_integer(token.substr(hyphen_position + 1));
      if (last_value < first_value) {
        throw std::runtime_error("descending '-' range is not supported in $GUS section: " + token);
      }
      for (int value = first_value; value <= last_value; ++value) {
        expanded_values->push_back(value);
      }
      continue;
    }
    if (star_position != std::string::npos) {
      const int repeated_value = parse_guess_integer(token.substr(0, star_position));
      const int repeat_count = parse_guess_integer(token.substr(star_position + 1));
      if (repeat_count < 0) {
        throw std::runtime_error("negative repetition count is not supported in $GUS section: " + token);
      }
      for (int repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
        expanded_values->push_back(repeated_value);
      }
      continue;
    }
    expanded_values->push_back(parse_guess_integer(token));
  }
}

struct ReadGuessSection {
  std::vector<int> orbital_basis_counts;
  std::vector<double> dense_orbital_coefficients;
};

std::vector<double> build_support_gathered_dense_guess(
    const ReadGuessSection& read_guess,
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int coefficient_count =
      get_orbital_basis_count(orbital_preparation_input, orbital_index);
  std::vector<double> gathered_coefficients(
      xmvb::to_size(coefficient_count),
      0.0);
  for (int coefficient_index = 0;
       coefficient_index < coefficient_count;
       ++coefficient_index) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [xmvb::to_size(orbital_index) * n_basis_functions +
             coefficient_index] -
        1;
    if (basis_function_index < 0 || basis_function_index >= n_basis_functions) {
      throw std::runtime_error(
          "invalid sparse orbital basis index while gathering C++ GUESS=READ/RDCI");
    }
    gathered_coefficients[xmvb::to_size(coefficient_index)] =
        read_guess.dense_orbital_coefficients
            [xmvb::to_size(orbital_index) * n_basis_functions +
             basis_function_index];
  }
  return gathered_coefficients;
}

ReadGuessSection parse_read_guess_section(
    const std::string& input_file_path,
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::ifstream input_stream(input_file_path);
  if (!input_stream) {
    throw std::runtime_error(
        "failed to open input file for C++ GUESS=READ/RDCI: " + input_file_path);
  }

  std::vector<std::string> guess_section_lines;
  std::string line;
  bool inside_guess_section = false;
  while (std::getline(input_stream, line)) {
    const std::string upper_line = uppercase_copy(line);
    if (!inside_guess_section) {
      if (upper_line.find("$GUS") != std::string::npos) {
        inside_guess_section = true;
      }
      continue;
    }
    if (upper_line.find("$END") != std::string::npos) {
      break;
    }
    guess_section_lines.push_back(line);
  }
  if (!inside_guess_section) {
    throw std::runtime_error("C++ GUESS=READ/RDCI requires a $GUS section in the input file");
  }

  std::vector<int> file_orbital_basis_counts;
  std::size_t first_coefficient_line = guess_section_lines.size();
  for (std::size_t line_index = 0; line_index < guess_section_lines.size(); ++line_index) {
    const std::string uncommented_line =
        strip_inline_comment(guess_section_lines[line_index]);
    if (uncommented_line.find('.') != std::string::npos) {
      first_coefficient_line = line_index;
      break;
    }
    const auto tokens = split_whitespace_tokens(uncommented_line);
    if (tokens.empty()) {
      continue;
    }
    expand_orbital_tokens(tokens, &file_orbital_basis_counts);
  }

  if (first_coefficient_line == guess_section_lines.size()) {
    throw std::runtime_error(
        "C++ GUESS=READ/RDCI could not find coefficient data after the $GUS header");
  }
  if (static_cast<int>(file_orbital_basis_counts.size()) <
      orbital_preparation_input.n_orbitals) {
    throw std::runtime_error(
        "C++ GUESS=READ/RDCI currently supports only non-BOVB guesses with one $GUS header "
        "entry per VB orbital");
  }

  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<int> parsed_orbital_basis_counts(
      file_orbital_basis_counts.begin(),
      file_orbital_basis_counts.begin() + n_orbitals);
  std::vector<double> dense_coefficients(
      xmvb::to_size(n_orbitals) * n_basis_functions,
      0.0);
  std::vector<std::string> buffered_tokens;
  std::size_t next_token_offset = 0;
  int orbital_index = 0;

  // Legacy vb_readguess accumulates multi-line (coef, basis) pairs for each
  // orbital, scatters them into a dense AO coefficient row with cvitra, and
  // only afterwards projects onto the current VB sparse orbital layout. Mirror
  // that two-stage mapping here so GUESS=READ/RDCI no longer depends on the C
  // runtime reader.
  for (std::size_t line_index = first_coefficient_line;
       line_index < guess_section_lines.size() && orbital_index < n_orbitals;
       ++line_index) {
    const auto tokens = split_whitespace_tokens(
        strip_inline_comment(guess_section_lines[line_index]));
    if (tokens.empty()) {
      continue;
    }
    buffered_tokens.insert(buffered_tokens.end(), tokens.begin(), tokens.end());

    while (orbital_index < n_orbitals) {
      const int expected_pair_count = parsed_orbital_basis_counts[xmvb::to_size(orbital_index)];
      const std::size_t required_token_count =
          xmvb::to_size(expected_pair_count) * 2;
      const std::size_t available_token_count =
          buffered_tokens.size() - next_token_offset;
      if (available_token_count < required_token_count) {
        break;
      }

      for (int coefficient_index = 0; coefficient_index < expected_pair_count; ++coefficient_index) {
        const std::size_t token_offset =
            next_token_offset + xmvb::to_size(coefficient_index) * 2;
        const double coefficient_value =
            parse_guess_double(buffered_tokens[token_offset]);
        const int basis_function_index =
            parse_guess_integer(buffered_tokens[token_offset + 1]);
        if (basis_function_index <= 0 ||
            basis_function_index > n_basis_functions) {
          throw std::runtime_error(
              "basis index in $GUS section is out of range for GUESS=READ/RDCI");
        }
        dense_coefficients[xmvb::to_size(orbital_index) * n_basis_functions +
                           (basis_function_index - 1)] = coefficient_value;
      }

      next_token_offset += required_token_count;
      if (next_token_offset == buffered_tokens.size()) {
        buffered_tokens.clear();
        next_token_offset = 0;
      }
      ++orbital_index;
    }
  }

  if (orbital_index != n_orbitals) {
    throw std::runtime_error(
        "C++ GUESS=READ/RDCI did not find enough orbital coefficient data in $GUS");
  }

  return {
      std::move(parsed_orbital_basis_counts),
      std::move(dense_coefficients),
  };
}

void expand_read_guess_layout_if_needed(
    const ReadGuessSection& read_guess,
    OrbitalPreparationInput* orbital_preparation_input) {
  (void)read_guess;
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument(
        "orbital_preparation_input must not be null while expanding read guess layout");
  }

  // Legacy `vb_readguess` treats `$GUS` as a dense AO coefficient source and
  // then gathers those coefficients back onto the already prepared optimizer
  // chart carried by `vb_str->nv/ma`. The `$GUS` header counts therefore do
  // not define the variational manifold. Expanding the runtime support here
  // incorrectly turns localized HAO/BDO decks into full-AO optimizations and
  // shifts the converged energy away from the legacy `.xmo` reference.
}

void build_read_or_rdci_guess(
    const std::string& input_file_path,
    OrbitalPreparationInput* orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const ReadGuessSection read_guess = parse_read_guess_section(
      input_file_path,
      *orbital_preparation_input);
  expand_read_guess_layout_if_needed(
      read_guess,
      orbital_preparation_input);

  const int n_orbitals = orbital_preparation_input->n_orbitals;
  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  std::fill(orbital_value_table->begin(), orbital_value_table->end(), 0.0);

  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_orbital_basis_count(*orbital_preparation_input, orbital_index);
    // Match legacy `vb_readguess`: `$GUS` is first expanded to a dense AO
    // vector and then gathered back onto the target runtime support chart.
    // The `$GUS` header does not redefine the variational manifold.
    const std::vector<double> support_coefficients =
        build_support_gathered_dense_guess(
            read_guess,
            *orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                             coefficient_index] =
          support_coefficients[xmvb::to_size(coefficient_index)];
    }
  }
}

void normalize_sparse_guess(
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const auto& overlap_matrix =
      orbital_preparation_input.active_orbital_overlap_matrix.vector();
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int basis_count = get_orbital_basis_count(orbital_preparation_input, orbital_index);
    double norm = 0.0;
    for (int left_index = 0; left_index < basis_count; ++left_index) {
      const int left_basis =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) * n_basis_functions + left_index] -
          1;
      const double left_coefficient =
          (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                                 left_index];
      for (int right_index = 0; right_index < basis_count; ++right_index) {
        const int right_basis =
            orbital_preparation_input.orbital_basis_index_table
                [xmvb::to_size(orbital_index) * n_basis_functions + right_index] -
            1;
        const double right_coefficient =
            (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                                   right_index];
        norm += left_coefficient * right_coefficient *
                overlap_matrix[xmvb::to_size(left_basis) * n_basis_functions +
                               right_basis];
      }
    }
    if (!(norm > 0.0)) {
      throw std::runtime_error("encountered non-positive orbital norm during C++ guess build");
    }
    const double scale = std::sqrt(1.0 / norm);
    for (int coefficient_index = 0; coefficient_index < basis_count; ++coefficient_index) {
      (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                             coefficient_index] *= scale;
    }
  }
}

bool supports_cpp_rhf_auto_guess(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.spin_multiplicity == 1 &&
         orbital_preparation_input.n_total_electrons >= 0 &&
         orbital_preparation_input.n_total_electrons % 2 == 0 &&
         orbital_preparation_input.n_total_electrons <=
             orbital_preparation_input.n_basis_functions * 2;
}

bool has_hf_overlap_matrix(
    const OrbitalPreparationInput& orbital_preparation_input) {
  return orbital_preparation_input.hf_overlap_matrix.size() ==
         xmvb::to_size(orbital_preparation_input.n_basis_functions) *
             orbital_preparation_input.n_basis_functions;
}

bool has_materialized_ao_two_electron_integrals(
    const AoIntegralInput& ao_integral_input) {
  return !ao_integral_input.ao_two_electron_integral_values.empty();
}

void build_hcore_block_guess(
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  build_block_matrix_guess(
      libcint_input,
      ao_integral_input.ao_core_hamiltonian_matrix.vector(),
      orbital_preparation_input,
      orbital_value_table);
}

CppRestrictedHartreeFockResult solve_rhf_guess(
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<std::vector<int>> atom_ao_indices(
      xmvb::to_size(libcint_input.n_atoms));
  for (int shell_index = 0; shell_index < libcint_input.n_shells; ++shell_index) {
    const int atom_index =
        libcint_input.bas[xmvb::to_size(shell_index) * BAS_SLOTS + ATOM_OF];
    const int ao_offset = libcint_input.basidx[xmvb::to_size(shell_index) * 2];
    const int ao_count = libcint_input.basidx[xmvb::to_size(shell_index) * 2 + 1];
    if (atom_index < 0 || atom_index >= libcint_input.n_atoms) {
      throw std::runtime_error("invalid atom index in LibcintInput shell table");
    }
    auto& atom_indices = atom_ao_indices[xmvb::to_size(atom_index)];
    for (int local_ao = 0; local_ao < ao_count; ++local_ao) {
      atom_indices.push_back(ao_offset + local_ao);
    }
  }

  const auto& overlap_matrix =
      orbital_preparation_input.active_orbital_overlap_matrix.vector();
  const auto& hcore_matrix = ao_integral_input.ao_core_hamiltonian_matrix.vector();
  std::vector<double> initial_density_projector(
      xmvb::to_size(n_basis_functions) * n_basis_functions,
      0.0);
  constexpr double kDegeneracyTolerance = 1.0e-6;
  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    const auto& ao_indices = atom_ao_indices[xmvb::to_size(atom_index)];
    if (ao_indices.empty()) {
      continue;
    }

    const int block_size = static_cast<int>(ao_indices.size());
    std::vector<double> local_overlap(
        xmvb::to_size(block_size) * block_size,
        0.0);
    std::vector<double> local_hcore(
        xmvb::to_size(block_size) * block_size,
        0.0);
    std::vector<double> local_eigenvalues(xmvb::to_size(block_size), 0.0);
    for (int local_column = 0; local_column < block_size; ++local_column) {
      const int global_column = ao_indices[xmvb::to_size(local_column)];
      for (int local_row = 0; local_row < block_size; ++local_row) {
        const int global_row = ao_indices[xmvb::to_size(local_row)];
        local_overlap[xmvb::to_size(local_column) * block_size + local_row] =
            overlap_matrix[xmvb::to_size(global_column) * n_basis_functions + global_row];
        local_hcore[xmvb::to_size(local_column) * block_size + local_row] =
            hcore_matrix[xmvb::to_size(global_column) * n_basis_functions + global_row];
      }
    }

    if (LAPACKE_dsygv(
            LAPACK_COL_MAJOR,
            1,
            'V',
            'U',
            block_size,
            local_hcore.data(),
            block_size,
            local_overlap.data(),
            block_size,
            local_eigenvalues.data()) != 0) {
      throw std::runtime_error("LAPACKE_dsygv failed while building SAD-like atomic density");
    }

    double remaining_spinless_occupancy = 0.5 * static_cast<double>(
        libcint_input.atm[xmvb::to_size(atom_index) * ATM_SLOTS + CHARGE_OF]);
    int group_begin = 0;
    while (group_begin < block_size && remaining_spinless_occupancy > 1.0e-12) {
      int group_end = group_begin + 1;
      while (group_end < block_size &&
             std::abs(local_eigenvalues[xmvb::to_size(group_end)] -
                      local_eigenvalues[xmvb::to_size(group_begin)]) <=
                 kDegeneracyTolerance) {
        ++group_end;
      }
      const int group_size = group_end - group_begin;
      const double occupied_in_group =
          std::min(remaining_spinless_occupancy, static_cast<double>(group_size));
      const double occupation = occupied_in_group / static_cast<double>(group_size);
      for (int group_orbital = group_begin; group_orbital < group_end; ++group_orbital) {
        const double* local_orbital =
            local_hcore.data() + xmvb::to_size(group_orbital) * block_size;
        for (int local_column = 0; local_column < block_size; ++local_column) {
          const int global_column = ao_indices[xmvb::to_size(local_column)];
          const double column_value = local_orbital[local_column];
          for (int local_row = 0; local_row < block_size; ++local_row) {
            const int global_row = ao_indices[xmvb::to_size(local_row)];
            initial_density_projector[xmvb::to_size(global_column) *
                                          n_basis_functions +
                                      global_row] +=
                occupation * local_orbital[local_row] * column_value;
          }
        }
      }
      remaining_spinless_occupancy -= occupied_in_group;
      group_begin = group_end;
    }
  }

  CppRestrictedHartreeFockSolver solver;
  return solver.solve(
      orbital_preparation_input.n_total_electrons,
      orbital_preparation_input.active_orbital_overlap_matrix.vector(),
      ao_integral_input,
      initial_density_projector);
}

void build_rhf_block_guess(
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  const auto rhf_result = solve_rhf_guess(
      libcint_input,
      ao_integral_input,
      orbital_preparation_input);
  if (has_hf_overlap_matrix(orbital_preparation_input)) {
    build_block_matrix_guess(
        libcint_input,
        rhf_result.fock_matrix,
        orbital_preparation_input.hf_overlap_matrix.vector(),
        orbital_preparation_input,
        orbital_value_table);
  } else {
    build_block_matrix_guess(
        libcint_input,
        rhf_result.fock_matrix,
        orbital_preparation_input,
        orbital_value_table);
  }
}

std::vector<int> read_mo_indices(
    const std::string& input_file_path,
    int n_orbitals) {
  std::ifstream input_stream(input_file_path);
  if (!input_stream) {
    throw std::runtime_error("failed to open input file for C++ MO guess: " + input_file_path);
  }

  std::vector<int> mo_indices(xmvb::to_size(n_orbitals), 0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    mo_indices[xmvb::to_size(orbital_index)] = orbital_index + 1;
  }

  bool inside_guess_section = false;
  bool found_guess_section = false;
  std::string line;
  while (std::getline(input_stream, line)) {
    if (!inside_guess_section) {
      if (line.find("$GUS") != std::string::npos) {
        inside_guess_section = true;
        found_guess_section = true;
      }
      continue;
    }
    if (line.find("$END") != std::string::npos) {
      break;
    }
    if (line.find('#') != std::string::npos) {
      continue;
    }

    int vb_orbital = 0;
    int mo_orbital = 0;
    if (std::sscanf(line.c_str(), "%d %d", &vb_orbital, &mo_orbital) == 2 &&
        vb_orbital >= 1 && vb_orbital <= n_orbitals) {
      mo_indices[xmvb::to_size(vb_orbital - 1)] = mo_orbital;
    }
  }

  if (!found_guess_section) {
    throw std::runtime_error("C++ MO guess requires a $GUS section in the input file");
  }

  return mo_indices;
}

void build_hcore_mo_guess(
    const std::string& input_file_path,
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  (void)libcint_input;
  const auto ao_normalization = build_ao_normalization(orbital_preparation_input);
  std::vector<double> overlap_matrix =
      orbital_preparation_input.active_orbital_overlap_matrix.vector();
  std::vector<double> orbital_matrix =
      ao_integral_input.ao_core_hamiltonian_matrix.vector();
  std::vector<double> eigenvalues(xmvb::to_size(n_basis_functions), 0.0);

  if (LAPACKE_dsygv(
          LAPACK_COL_MAJOR,
          1,
          'V',
          'U',
          n_basis_functions,
          orbital_matrix.data(),
          n_basis_functions,
          overlap_matrix.data(),
          n_basis_functions,
          eigenvalues.data()) != 0) {
    throw std::runtime_error("LAPACKE_dsygv failed while building C++ MO guess");
  }

  for (int basis_index = 0; basis_index < n_basis_functions; ++basis_index) {
    for (int mo_index = 0; mo_index < n_basis_functions; ++mo_index) {
      orbital_matrix[xmvb::to_size(mo_index) * n_basis_functions + basis_index] *=
          ao_normalization[xmvb::to_size(basis_index)];
    }
  }

  const auto mo_indices = read_mo_indices(input_file_path, n_orbitals);
  std::fill(orbital_value_table->begin(), orbital_value_table->end(), 0.0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int selected_mo = std::abs(mo_indices[xmvb::to_size(orbital_index)]) - 1;
    if (selected_mo < 0 || selected_mo >= n_basis_functions) {
      throw std::runtime_error("MO guess index is out of range for C++ MO guess");
    }
    const double sign =
        mo_indices[xmvb::to_size(orbital_index)] >= 0 ? 1.0 : -1.0;
    const int basis_count =
        get_orbital_basis_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < basis_count; ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index] -
          1;
      (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                             coefficient_index] =
          sign * orbital_matrix[xmvb::to_size(selected_mo) * n_basis_functions +
                                basis_function_index];
    }
  }

  normalize_sparse_guess(orbital_preparation_input, orbital_value_table);
}

void build_rhf_mo_guess(
    const std::string& input_file_path,
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const auto rhf_result = solve_rhf_guess(
      libcint_input,
      ao_integral_input,
      orbital_preparation_input);
  const auto ao_normalization = build_ao_normalization(orbital_preparation_input);
  std::vector<double> orbital_matrix = rhf_result.molecular_orbital_matrix;
  for (int basis_index = 0; basis_index < n_basis_functions; ++basis_index) {
    for (int mo_index = 0; mo_index < n_basis_functions; ++mo_index) {
      orbital_matrix[xmvb::to_size(mo_index) * n_basis_functions + basis_index] *=
          ao_normalization[xmvb::to_size(basis_index)];
    }
  }

  const auto mo_indices = read_mo_indices(input_file_path, n_orbitals);
  std::fill(orbital_value_table->begin(), orbital_value_table->end(), 0.0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int selected_mo = std::abs(mo_indices[xmvb::to_size(orbital_index)]) - 1;
    if (selected_mo < 0 || selected_mo >= n_basis_functions) {
      throw std::runtime_error("MO guess index is out of range for C++ RHF MO guess");
    }
    const double sign =
        mo_indices[xmvb::to_size(orbital_index)] >= 0 ? 1.0 : -1.0;
    const int basis_count =
        get_orbital_basis_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < basis_count; ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index] -
          1;
      (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                             coefficient_index] =
          sign * orbital_matrix[xmvb::to_size(selected_mo) * n_basis_functions +
                                basis_function_index];
    }
  }

  normalize_sparse_guess(orbital_preparation_input, orbital_value_table);
}

void build_unit_guess(
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table) {
  if (orbital_value_table == nullptr) {
    throw std::invalid_argument("orbital_value_table must not be null");
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::fill(orbital_value_table->begin(), orbital_value_table->end(), 0.0);
  std::vector<int> basis_used(xmvb::to_size(n_basis_functions), 0);
  const auto blocks = detect_orbital_blocks(orbital_preparation_input);
  for (const auto& block : blocks) {
    for (int orbital_index : block) {
      const int stored_basis_count =
          orbital_preparation_input.orbital_basis_counts[xmvb::to_size(orbital_index)];
      for (int coefficient_index = 0; coefficient_index < stored_basis_count; ++coefficient_index) {
        const int basis_function_index =
            orbital_preparation_input.orbital_basis_index_table
                [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index] -
            1;
        if (basis_function_index < 0) {
          break;
        }
        if (basis_used[xmvb::to_size(basis_function_index)] == 0) {
          (*orbital_value_table)[xmvb::to_size(orbital_index) * n_basis_functions +
                                 coefficient_index] = 1.0;
          basis_used[xmvb::to_size(basis_function_index)] = 1;
          break;
        }
      }
    }
  }
}

}  // namespace

const char* orbital_guess_source_name(OrbitalGuessSource source) {
  switch (source) {
    case OrbitalGuessSource::LegacyRuntime:
      return "legacy";
    case OrbitalGuessSource::Cpp:
      return "cpp";
  }
  return "unknown";
}

bool cpp_initial_guess_supported(int guess_type) noexcept {
  return guess_type == GUS_AUTO ||
         guess_type == GUS_UNIT ||
         guess_type == GUS_MO ||
         guess_type == GUS_READ ||
         guess_type == GUS_RDCI;
}

void build_cpp_initial_guess(
    const std::string& input_file_path,
    int guess_type,
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }
  const std::size_t expected_orbital_value_count =
      xmvb::to_size(orbital_preparation_input->n_basis_functions) *
      orbital_preparation_input->n_orbitals;

  std::vector<double> orbital_value_table(
      expected_orbital_value_count,
      0.0);
  switch (guess_type) {
    case GUS_AUTO:
      // Prefer the RHF-like guess when exact AO ERIs are available. In RI or
      // hcore-only load modes the C++ path no longer has a materialized AO 2e
      // tensor, so fall back to the one-electron block guess instead of
      // requiring the legacy HF/vbguess runtime.
      if (supports_cpp_rhf_auto_guess(*orbital_preparation_input) &&
          has_materialized_ao_two_electron_integrals(ao_integral_input)) {
        build_rhf_block_guess(
            libcint_input,
            ao_integral_input,
            *orbital_preparation_input,
            &orbital_value_table);
      } else {
        build_hcore_block_guess(
            libcint_input,
            ao_integral_input,
            *orbital_preparation_input,
            &orbital_value_table);
      }
      break;
    case GUS_UNIT:
      build_unit_guess(*orbital_preparation_input, &orbital_value_table);
      break;
    case GUS_MO:
      // Legacy `vb_moguess` uses HF canonical orbitals for `GUESS=MO`, not the
      // one-electron hcore eigensystem. Match that behavior whenever the C++
      // path can form an RHF reference; otherwise keep the old hcore fallback.
      if (supports_cpp_rhf_auto_guess(*orbital_preparation_input) &&
          has_materialized_ao_two_electron_integrals(ao_integral_input)) {
        build_rhf_mo_guess(
            input_file_path,
            libcint_input,
            ao_integral_input,
            *orbital_preparation_input,
            &orbital_value_table);
      } else {
        build_hcore_mo_guess(
            input_file_path,
            libcint_input,
            ao_integral_input,
            *orbital_preparation_input,
            &orbital_value_table);
      }
      break;
    case GUS_READ:
    case GUS_RDCI:
      build_read_or_rdci_guess(
          input_file_path,
          orbital_preparation_input,
          &orbital_value_table);
      break;
    case GUS_NBO:
      throw std::runtime_error(
          "C++ guess builder does not support GUESS=NBO; "
          "rerun with --orbital-guess-source legacy");
    default:
      throw std::runtime_error("unsupported guess type in C++ guess builder");
  }

  orbital_preparation_input->orbital_value_table = std::move(orbital_value_table);
}

}  // namespace xmvb::vb
