#include "runtime/molden_file_writer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/libcint_compat.hpp"
#include "runtime/legacy_shell_utils.hpp"

namespace xmvb::vb {

namespace {

namespace fs = std::filesystem;

constexpr std::array<char, 6> kShellTags = {'s', 'p', 'd', 'f', 'g', 'h'};
constexpr std::array<const char*, 87> kElementSymbols = {{
    "X",
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
}};

std::string trim_ascii_whitespace(std::string value) {
  const auto first = std::find_if_not(
      value.begin(),
      value.end(),
      [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(
      value.rbegin(),
      value.rend(),
      [](unsigned char c) { return std::isspace(c) != 0; })
                        .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::string atomic_symbol(int atomic_number) {
  if (atomic_number <= 0 ||
      atomic_number >= static_cast<int>(kElementSymbols.size())) {
    return "X";
  }
  const std::string symbol = trim_ascii_whitespace(kElementSymbols[
      atomic_number]);
  return symbol.empty() ? "X" : symbol;
}

void validate_libcint_input(const LibcintInput& libcint_input) {
  if (libcint_input.n_atoms <= 0) {
    throw std::invalid_argument("Molden export requires at least one atom");
  }
  if (libcint_input.n_shells <= 0) {
    throw std::invalid_argument("Molden export requires at least one shell");
  }
  if (libcint_input.atm.size() != libcint_input.n_atoms * ATM_SLOTS) {
    throw std::invalid_argument("Libcint atom table size mismatch during Molden export");
  }
  if (libcint_input.bas.size() != libcint_input.n_shells * BAS_SLOTS) {
    throw std::invalid_argument("Libcint basis table size mismatch during Molden export");
  }
  if (libcint_input.basidx.size() != libcint_input.n_shells * 2) {
    throw std::invalid_argument("Libcint shell index table size mismatch during Molden export");
  }
  if (libcint_input.env.empty()) {
    throw std::invalid_argument("Libcint environment table is empty during Molden export");
  }
}

void validate_orbital_input(const OrbitalPreparationInput& orbital_input) {
  if (orbital_input.n_basis_functions <= 0) {
    throw std::invalid_argument("Molden export requires a positive basis-function count");
  }
  if (orbital_input.n_orbitals <= 0) {
    throw std::invalid_argument("Molden export requires at least one orbital");
  }
  const std::size_t expected_table_size =
      orbital_input.n_basis_functions *
      orbital_input.n_orbitals;
  if (orbital_input.orbital_value_table.size() != expected_table_size) {
    throw std::invalid_argument("orbital_value_table size mismatch during Molden export");
  }
  if (orbital_input.orbital_basis_index_table.size() != expected_table_size) {
    throw std::invalid_argument(
        "orbital_basis_index_table size mismatch during Molden export");
  }
  if (orbital_input.orbital_basis_counts.size() !=
      orbital_input.n_orbitals) {
    throw std::invalid_argument("orbital_basis_counts size mismatch during Molden export");
  }
}

std::vector<double> expand_dense_orbital_coefficients(
    const OrbitalPreparationInput& orbital_input,
    int orbital_index) {
  const int n_basis_functions = orbital_input.n_basis_functions;
  std::vector<double> dense_coefficients(n_basis_functions, 0.0);

  // The VB optimizer stores one sparse support list per orbital inside a
  // padded `(n_orbitals, n_basis_functions)` table. Molden expects a dense AO
  // coefficient column, so rebuild that dense vector here.
  const int coefficient_count =
      stored_sparse_orbital_coefficient_count(orbital_input, orbital_index);
  for (int coefficient_index = 0; coefficient_index < coefficient_count;
       ++coefficient_index) {
    const std::size_t storage_index =
        orbital_index * n_basis_functions + coefficient_index;
    const int one_based_basis_index =
        orbital_input.orbital_basis_index_table[storage_index];
    if (one_based_basis_index <= 0 || one_based_basis_index > n_basis_functions) {
      throw std::runtime_error(
          "orbital basis index is out of range during Molden export");
    }
    dense_coefficients[one_based_basis_index - 1] =
        orbital_input.orbital_value_table[storage_index];
  }
  return dense_coefficients;
}

char shell_tag(int angular_momentum) {
  if (angular_momentum < 0 ||
      angular_momentum >= static_cast<int>(kShellTags.size())) {
    throw std::runtime_error(
        "unsupported angular momentum in Molden export: " +
        std::to_string(angular_momentum));
  }
  return kShellTags[angular_momentum];
}

std::vector<double> expand_dense_orbital_coefficients_in_molden_order(
    const OrbitalPreparationInput& orbital_input,
    const LibcintInput& libcint_input,
    int orbital_index) {
  const std::vector<double> dense_coefficients =
      expand_dense_orbital_coefficients(orbital_input, orbital_index);
  std::vector<double> molden_coefficients;
  molden_coefficients.reserve(dense_coefficients.size());

  // The standalone orbital chart stores Cartesian shell-local AO coefficients
  // in XMVB's historical ordering. Molden expects a different d/f ordering, so
  // reorder those shell blocks here to keep visualization consistent.
  for (int shell_index = 0; shell_index < libcint_input.n_shells; ++shell_index) {
    const std::size_t shell_offset = shell_index * BAS_SLOTS;
    const int angular_momentum = libcint_input.bas[shell_offset + ANG_OF];
    const int ao_offset = libcint_input.basidx[shell_index * 2];
    const int ao_count = libcint_input.basidx[shell_index * 2 + 1];
    if (ao_offset < 0 ||
        ao_count != cartesian_ao_count(angular_momentum) ||
        ao_offset + ao_count > orbital_input.n_basis_functions) {
      throw std::runtime_error("shell AO layout is inconsistent during Molden export");
    }

    if (angular_momentum == 2) {
      constexpr std::array<int, 6> kDOrder = {0, 3, 5, 1, 2, 4};
      for (const int local_index : kDOrder) {
        molden_coefficients.push_back(
            dense_coefficients[ao_offset + local_index]);
      }
      continue;
    }
    if (angular_momentum == 3) {
      constexpr std::array<int, 10> kFOrder = {0, 6, 9, 3, 1, 2, 5, 8, 7, 4};
      for (const int local_index : kFOrder) {
        molden_coefficients.push_back(
            dense_coefficients[ao_offset + local_index]);
      }
      continue;
    }
    for (int local_index = 0; local_index < ao_count; ++local_index) {
      molden_coefficients.push_back(
          dense_coefficients[ao_offset + local_index]);
    }
  }

  if (molden_coefficients.size() != dense_coefficients.size()) {
    throw std::runtime_error("Molden AO reorder size mismatch");
  }
  return molden_coefficients;
}

void write_atoms_section(
    std::ostream& output,
    const LibcintInput& libcint_input) {
  output << "[Molden Format]\n"
         << "[N_Atoms]\n"
         << std::setw(21) << libcint_input.n_atoms << '\n'
         << "[Atoms] (AU)\n";

  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    const std::size_t atom_offset = atom_index * ATM_SLOTS;
    const int atomic_number = libcint_input.atm[atom_offset + CHARGE_OF];
    const int coordinate_offset = libcint_input.atm[atom_offset + PTR_COORD];
    if (coordinate_offset < 0 ||
        coordinate_offset + 2 >= static_cast<int>(libcint_input.env.size())) {
      throw std::runtime_error("atom coordinate pointer is out of range during Molden export");
    }
    output << std::left << std::setw(10) << atomic_symbol(atomic_number)
           << std::right << std::setw(6) << atom_index + 1
           << std::setw(7) << atomic_number
           << std::fixed << std::setprecision(8)
           << std::setw(20) << libcint_input.env[coordinate_offset]
           << std::setw(16) << libcint_input.env[coordinate_offset + 1]
           << std::setw(16) << libcint_input.env[coordinate_offset + 2]
           << '\n';
  }
}

void write_charge_section(
    std::ostream& output,
    int n_atoms) {
  output << "[Charge] (Mulliken)\n"
         << std::fixed << std::setprecision(16);
  for (int atom_index = 0; atom_index < n_atoms; ++atom_index) {
    output << std::setw(21) << 0.0 << '\n';
  }
}

void write_gto_section(
    std::ostream& output,
    const LibcintInput& libcint_input) {
  output << "[GTO] (AU)\n";
  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    output << std::setw(4) << atom_index + 1 << '\n';
    for (int shell_index = 0; shell_index < libcint_input.n_shells; ++shell_index) {
      const std::size_t shell_offset = shell_index * BAS_SLOTS;
      if (libcint_input.bas[shell_offset + ATOM_OF] != atom_index) {
        continue;
      }

      const int angular_momentum = libcint_input.bas[shell_offset + ANG_OF];
      const int n_primitives = libcint_input.bas[shell_offset + NPRIM_OF];
      const int n_contractions = libcint_input.bas[shell_offset + NCTR_OF];
      const int exponent_offset = libcint_input.bas[shell_offset + PTR_EXP];
      const int coefficient_offset = libcint_input.bas[shell_offset + PTR_COEFF];
      if (n_primitives <= 0) {
        throw std::runtime_error("shell primitive count is invalid during Molden export");
      }
      if (n_contractions != 1) {
        throw std::runtime_error(
            "Molden export only supports single-contraction shells in the standalone basis data");
      }
      if (exponent_offset < 0 || coefficient_offset < 0 ||
          exponent_offset + n_primitives > static_cast<int>(libcint_input.env.size()) ||
          coefficient_offset + n_primitives > static_cast<int>(libcint_input.env.size())) {
        throw std::runtime_error("shell env pointer is out of range during Molden export");
      }

      output << "   " << shell_tag(angular_momentum)
             << "   " << n_primitives << '\n'
             << std::scientific << std::uppercase << std::setprecision(9);
      for (int primitive_index = 0; primitive_index < n_primitives; ++primitive_index) {
        const double exponent =
            libcint_input.env[exponent_offset + primitive_index];
        const double normalized_coefficient =
            libcint_input.env[coefficient_offset + primitive_index];
        output << std::setw(18)
               << exponent
               << std::setw(18)
               // The standalone basis loader stores primitive coefficients multiplied
               // by the historical XMVB/cartesian normalization factor.
               // Molden viewers expect the corresponding de-normalized values.
               << normalized_coefficient /
                      legacy_shell_normalization(angular_momentum, exponent)
               << '\n';
      }
    }
    output << " \n";
  }
}

void write_mo_section(
    std::ostream& output,
    const LibcintInput& libcint_input,
    const OrbitalPreparationInput& orbital_input) {
  output << "[MO]\n";
  for (int orbital_index = 0; orbital_index < orbital_input.n_orbitals; ++orbital_index) {
    const std::vector<double> dense_coefficients =
        expand_dense_orbital_coefficients_in_molden_order(
            orbital_input,
            libcint_input,
            orbital_index);
    output << "Sym=     " << orbital_index + 1 << "a\n"
           << "Ene=     0.0000\n"
           << "Spin= Alpha\n"
           << "Occup=    0.00000\n"
           << std::fixed << std::setprecision(8);
    for (int basis_function_index = 0;
         basis_function_index < orbital_input.n_basis_functions;
         ++basis_function_index) {
      output << std::setw(4) << basis_function_index + 1
             << "  "
             << std::setw(16)
             << dense_coefficients[basis_function_index]
             << '\n';
    }
  }
}

}  // namespace

fs::path write_molden_file(
    const fs::path& input_file_path,
    const CppVbInput& input) {
  validate_libcint_input(input.libcint_input);
  validate_orbital_input(input.orbital_preparation_input);

  fs::path output_path = input_file_path;
  output_path.replace_extension(".molden");
  output_path = fs::absolute(output_path);
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error(
        "failed to open Molden output file: " + output_path.string());
  }

  // Keep the Molden sections close to the historical XMVB layout so existing
  // visualization workflows continue to recognize the file without new knobs.
  write_atoms_section(output, input.libcint_input);
  write_charge_section(output, input.libcint_input.n_atoms);
  write_gto_section(output, input.libcint_input);
  write_mo_section(output, input.libcint_input, input.orbital_preparation_input);
  if (!output) {
    throw std::runtime_error(
        "failed while writing Molden output file: " + output_path.string());
  }
  return output_path;
}

}  // namespace xmvb::vb
