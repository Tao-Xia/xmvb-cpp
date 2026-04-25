#include "runtime/input_deck_primary_basis_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include "cint.h"
}

#include "runtime/legacy_shell_utils.hpp"

namespace xmvb::vb {
namespace {

constexpr double kBohrPerAngstrom = 1.0 / 0.529177249;

struct BasisShellTemplate {
  int angular_momentum = 0;
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

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

std::string to_ascii_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string canonicalize_element_symbol(std::string symbol) {
  symbol = trim_ascii_whitespace(std::move(symbol));
  if (symbol.empty()) {
    return {};
  }
  symbol[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));
  for (std::size_t character_index = 1;
       character_index < symbol.size();
       ++character_index) {
    symbol[character_index] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(symbol[character_index])));
  }
  return symbol;
}

int atomic_number_from_element_symbol(const std::string& element_symbol) {
  static const std::array<const char*, 87> kElementSymbols = {
      "", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
      "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
      "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
      "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
      "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
      "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
      "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
      "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
  const std::string canonical_symbol =
      canonicalize_element_symbol(element_symbol);
  for (int atomic_number = 1;
       atomic_number < static_cast<int>(kElementSymbols.size());
       ++atomic_number) {
    if (canonical_symbol == kElementSymbols[atomic_number]) {
      return atomic_number;
    }
  }
  throw std::runtime_error("unsupported element symbol in input deck geometry: " + element_symbol);
}

double parse_fortran_double(std::string token) {
  for (char& character : token) {
    if (character == 'd' || character == 'D') {
      character = 'E';
    }
  }
  std::size_t consumed_characters = 0;
  const double value = std::stod(token, &consumed_characters);
  if (consumed_characters != token.size()) {
    throw std::runtime_error("invalid floating-point token in basis file: " + token);
  }
  return value;
}

std::vector<std::string> split_ascii_whitespace(const std::string& line) {
  std::vector<std::string> tokens;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

bool is_comment_or_empty_line(const std::string& line) {
  const std::string trimmed = trim_ascii_whitespace(line);
  return trimmed.empty() || trimmed[0] == '!';
}

bool is_ecp_section_header_line(const std::vector<std::string>& tokens) {
  if (tokens.empty()) {
    return false;
  }
  return to_ascii_lower(tokens.front()).find("-ecp") != std::string::npos;
}

int angular_momentum_from_label(const std::string& raw_label) {
  const std::string label = to_ascii_lower(raw_label);
  if (label == "s") {
    return 0;
  }
  if (label == "p") {
    return 1;
  }
  if (label == "d") {
    return 2;
  }
  if (label == "f") {
    return 3;
  }
  if (label == "g") {
    return 4;
  }
  if (label == "h") {
    return 5;
  }
  throw std::runtime_error("unsupported angular-momentum label in basis file: " + raw_label);
}

bool is_legacy_pople_basis_name(const std::string& basis_name) {
  return basis_name.find("sto-") != std::string::npos ||
      basis_name.find("3-21") != std::string::npos ||
      basis_name.find("6-31") != std::string::npos;
}

std::string normalize_legacy_pople_basis_name(std::string basis_name) {
  const std::size_t g_position = basis_name.find('g');
  if (g_position == std::string::npos) {
    throw std::runtime_error("invalid Pople basis name: " + basis_name);
  }

  int n_plus = 0;
  int n_star = 0;
  for (char character : basis_name) {
    if (character == '+') {
      ++n_plus;
    } else if (character == '*') {
      ++n_star;
    }
  }
  if (n_plus > 2 || n_star > 2) {
    throw std::runtime_error("unsupported Pople basis variant: " + basis_name);
  }

  const std::size_t open_parenthesis = basis_name.find('(', g_position);
  const std::size_t comma = basis_name.find(',', g_position);
  const std::size_t close_parenthesis = basis_name.find(')', g_position);
  if ((open_parenthesis != std::string::npos ||
       comma != std::string::npos ||
       close_parenthesis != std::string::npos) &&
      (open_parenthesis == std::string::npos ||
       close_parenthesis == std::string::npos ||
       close_parenthesis <= open_parenthesis + 1 ||
       (comma != std::string::npos &&
        (comma <= open_parenthesis + 1 || close_parenthesis <= comma + 1)))) {
    throw std::runtime_error("failed to parse Pople basis name: " + basis_name);
  }

  std::string normalized;
  if (n_plus == 0) {
    normalized = basis_name.substr(0, g_position);
  } else {
    normalized = basis_name.substr(0, g_position - n_plus);
    for (int plus_index = 0; plus_index < n_plus; ++plus_index) {
      normalized += 'p';
    }
  }
  normalized += 'g';

  if (n_star == 1) {
    normalized += "_d_";
  } else if (n_star == 2) {
    normalized += "_d_p_";
  } else if (open_parenthesis != std::string::npos) {
    normalized += "_";
    if (comma != std::string::npos) {
      normalized += basis_name.substr(
          open_parenthesis + 1,
          comma - open_parenthesis - 1);
      normalized += "_";
      normalized += basis_name.substr(
          comma + 1,
          close_parenthesis - comma - 1);
      normalized += "_";
    } else {
      normalized += basis_name.substr(
          open_parenthesis + 1,
          close_parenthesis - open_parenthesis - 1);
      normalized += "_";
    }
  }
  return normalized;
}

std::string normalize_basis_leaf_name(const std::string& raw_basis_name) {
  if (raw_basis_name.empty()) {
    throw std::runtime_error("input deck is missing BASIS=");
  }
  std::string normalized = to_ascii_lower(raw_basis_name);
  if (is_legacy_pople_basis_name(normalized)) {
    normalized = normalize_legacy_pople_basis_name(normalized);
  }
  if (normalized.size() < 4 ||
      normalized.substr(normalized.size() - 4) != ".gbs") {
    normalized += ".gbs";
  }
  return normalized;
}

std::filesystem::path resolve_basis_file_path(
    const std::string& basis_leaf_name) {
  namespace fs = std::filesystem;

  const fs::path direct_path(basis_leaf_name);
  if (fs::exists(direct_path)) {
    return fs::absolute(direct_path);
  }

  fs::path current_path = fs::current_path();
  while (true) {
    const fs::path basis_candidate = current_path / "basis" / basis_leaf_name;
    if (fs::exists(basis_candidate)) {
      return fs::absolute(basis_candidate);
    }
    const fs::path leaf_candidate = current_path / basis_leaf_name;
    if (fs::exists(leaf_candidate)) {
      return fs::absolute(leaf_candidate);
    }
    if (!current_path.has_parent_path() ||
        current_path.parent_path() == current_path) {
      break;
    }
    current_path = current_path.parent_path();
  }

  throw std::runtime_error("failed to locate basis file: " + basis_leaf_name);
}

std::unordered_map<int, std::vector<BasisShellTemplate>> parse_basis_templates(
    const std::filesystem::path& basis_file_path,
    const std::set<int>& required_atomic_numbers) {
  std::ifstream input_stream(basis_file_path);
  if (!input_stream) {
    throw std::runtime_error("failed to open basis file: " + basis_file_path.string());
  }

  std::unordered_map<int, std::vector<BasisShellTemplate>> shells_by_atomic_number;
  std::string line;
  bool found_first_block_separator = false;
  while (std::getline(input_stream, line)) {
    if (trim_ascii_whitespace(line) == "****") {
      found_first_block_separator = true;
      break;
    }
  }

  // Gaussian basis files use `****` as both the current block terminator and
  // the next block separator. After one block consumes a separator, the next
  // element header follows immediately on the next line, so the parser must
  // continue from that header rather than scanning for another leading `****`.
  while (found_first_block_separator) {
    std::string header_line;
    while (std::getline(input_stream, header_line)) {
      if (!is_comment_or_empty_line(header_line)) {
        break;
      }
    }
    if (input_stream.fail()) {
      break;
    }

    const std::string header_trimmed = trim_ascii_whitespace(header_line);
    if (header_trimmed.empty() || header_trimmed == "****") {
      continue;
    }
    const std::vector<std::string> header_tokens =
        split_ascii_whitespace(header_trimmed);
    if (header_tokens.empty()) {
      continue;
    }
    const int atomic_number =
        atomic_number_from_element_symbol(header_tokens.front());
    const bool keep_element =
        required_atomic_numbers.find(atomic_number) != required_atomic_numbers.end();

    bool reached_block_separator = false;
    while (std::getline(input_stream, line)) {
      const std::string shell_trimmed = trim_ascii_whitespace(line);
      if (shell_trimmed == "****") {
        reached_block_separator = true;
        break;
      }
      if (is_comment_or_empty_line(shell_trimmed)) {
        continue;
      }

      const std::vector<std::string> shell_tokens =
          split_ascii_whitespace(shell_trimmed);
      // Gaussian-format basis files append optional ECP data after the AO
      // shells. Those blocks start with labels such as `RB-ECP` and do not use
      // shell labels, so stop the AO parser before those tails are mistaken for
      // orbital angular-momentum headers.
      if (is_ecp_section_header_line(shell_tokens)) {
        return shells_by_atomic_number;
      }
      if (shell_tokens.size() < 2) {
        throw std::runtime_error(
            "invalid shell header in basis file: " + shell_trimmed);
      }
      const std::string shell_label = to_ascii_lower(shell_tokens[0]);
      const int n_primitives = std::stoi(shell_tokens[1]);
      if (n_primitives <= 0) {
        throw std::runtime_error("basis shell must contain at least one primitive");
      }

      const bool is_sp_shell = shell_label == "sp";
      BasisShellTemplate first_shell;
      BasisShellTemplate second_shell;
      if (is_sp_shell) {
        first_shell.angular_momentum = 0;
        second_shell.angular_momentum = 1;
        first_shell.exponents.reserve(n_primitives);
        first_shell.coefficients.reserve(n_primitives);
        second_shell.exponents.reserve(n_primitives);
        second_shell.coefficients.reserve(n_primitives);
      } else {
        first_shell.angular_momentum = angular_momentum_from_label(shell_label);
        first_shell.exponents.reserve(n_primitives);
        first_shell.coefficients.reserve(n_primitives);
      }

      for (int primitive_index = 0; primitive_index < n_primitives; ++primitive_index) {
        std::string primitive_line;
        if (!std::getline(input_stream, primitive_line)) {
          throw std::runtime_error("basis file ended while reading primitive coefficients");
        }
        const std::vector<std::string> primitive_tokens =
            split_ascii_whitespace(trim_ascii_whitespace(primitive_line));
        if ((!is_sp_shell && primitive_tokens.size() < 2) ||
            (is_sp_shell && primitive_tokens.size() < 3)) {
          throw std::runtime_error(
              "invalid primitive line in basis file: " + primitive_line);
        }

        const double exponent = parse_fortran_double(primitive_tokens[0]);
        if (!(exponent > 0.0)) {
          throw std::runtime_error("basis exponent must be positive");
        }
        if (is_sp_shell) {
          first_shell.exponents.push_back(exponent);
          first_shell.coefficients.push_back(
              parse_fortran_double(primitive_tokens[1]) *
              legacy_shell_normalization(0, exponent));
          second_shell.exponents.push_back(exponent);
          second_shell.coefficients.push_back(
              parse_fortran_double(primitive_tokens[2]) *
              legacy_shell_normalization(1, exponent));
        } else {
          first_shell.exponents.push_back(exponent);
          first_shell.coefficients.push_back(
              parse_fortran_double(primitive_tokens[1]) *
              legacy_shell_normalization(first_shell.angular_momentum, exponent));
        }
      }

      if (!keep_element) {
        continue;
      }
      auto& element_shells =
          shells_by_atomic_number[atomic_number];
      element_shells.push_back(std::move(first_shell));
      if (is_sp_shell) {
        element_shells.push_back(std::move(second_shell));
      }
    }

    if (!input_stream && !reached_block_separator) {
      break;
    }
  }

  for (int atomic_number : required_atomic_numbers) {
    if (shells_by_atomic_number.find(atomic_number) == shells_by_atomic_number.end()) {
      throw std::runtime_error(
          "basis file does not contain shells for atomic number " +
          std::to_string(atomic_number));
    }
  }
  return shells_by_atomic_number;
}

LibcintInput build_libcint_input(
    const InputDeck& input_deck,
    const std::unordered_map<int, std::vector<BasisShellTemplate>>& shells_by_atomic_number) {
  if (input_deck.geometry_atoms.empty()) {
    throw std::runtime_error("pure C++ standalone basis builder requires a non-empty $GEO block");
  }

  const int n_atoms = static_cast<int>(input_deck.geometry_atoms.size());
  int n_shells = 0;
  int n_gaussian_primitives = 0;
  for (const InputDeckGeometryAtom& atom : input_deck.geometry_atoms) {
    const int atomic_number =
        atomic_number_from_element_symbol(atom.element_symbol);
    const auto iterator = shells_by_atomic_number.find(atomic_number);
    if (iterator == shells_by_atomic_number.end()) {
      throw std::runtime_error("missing basis templates for one geometry atom");
    }
    n_shells += static_cast<int>(iterator->second.size());
    for (const BasisShellTemplate& shell_template : iterator->second) {
      n_gaussian_primitives += static_cast<int>(shell_template.exponents.size());
    }
  }

  LibcintInput input;
  input.n_atoms = n_atoms;
  input.n_shells = n_shells;
  input.n_gaussian_primitives = n_gaussian_primitives;
  input.atm.assign(n_atoms * ATM_SLOTS, 0);
  input.bas.assign(n_shells * BAS_SLOTS, 0);
  input.basidx.assign(n_shells * 2, 0);
  input.env.assign(
      PTR_ENV_START + 3 * n_atoms + 2 * n_gaussian_primitives,
      0.0);

  int env_offset = PTR_ENV_START;
  for (int atom_index = 0; atom_index < n_atoms; ++atom_index) {
    const InputDeckGeometryAtom& atom = input_deck.geometry_atoms[atom_index];
    input.atm[atom_index * ATM_SLOTS + CHARGE_OF] =
        atomic_number_from_element_symbol(atom.element_symbol);
    input.atm[atom_index * ATM_SLOTS + PTR_COORD] = env_offset;
    const double coordinate_scale =
        input_deck.metadata.geometry_coordinates_in_bohr ? 1.0 : kBohrPerAngstrom;
    for (int coordinate_index = 0; coordinate_index < 3; ++coordinate_index) {
      input.env[env_offset + coordinate_index] =
          atom.coordinates[coordinate_index] * coordinate_scale;
    }
    env_offset += 3;
  }

  int shell_index = 0;
  int ao_offset = 0;
  for (int atom_index = 0; atom_index < n_atoms; ++atom_index) {
    const int atomic_number = input.atm[atom_index * ATM_SLOTS + CHARGE_OF];
    const auto iterator = shells_by_atomic_number.find(atomic_number);
    if (iterator == shells_by_atomic_number.end()) {
      throw std::runtime_error("basis template lookup failed during shell expansion");
    }
    for (const BasisShellTemplate& shell_template : iterator->second) {
      const int n_primitives =
          static_cast<int>(shell_template.exponents.size());
      const std::size_t bas_offset = shell_index * BAS_SLOTS;
      input.bas[bas_offset + ATOM_OF] = atom_index;
      input.bas[bas_offset + ANG_OF] = shell_template.angular_momentum;
      input.bas[bas_offset + NPRIM_OF] = n_primitives;
      input.bas[bas_offset + NCTR_OF] = 1;
      input.bas[bas_offset + PTR_EXP] = env_offset;
      input.bas[bas_offset + PTR_COEFF] = env_offset + n_primitives;

      for (int primitive_index = 0; primitive_index < n_primitives; ++primitive_index) {
        input.env[env_offset + primitive_index] =
            shell_template.exponents[primitive_index];
        input.env[env_offset + n_primitives + primitive_index] =
            shell_template.coefficients[primitive_index];
      }
      env_offset += 2 * n_primitives;

      input.basidx[shell_index * 2] = ao_offset;
      const int ao_count = cartesian_ao_count(shell_template.angular_momentum);
      input.basidx[shell_index * 2 + 1] = ao_count;
      ao_offset += ao_count;
      ++shell_index;
    }
  }

  return input;
}

}  // namespace

InputDeckPrimaryBasisBuildResult build_input_deck_primary_basis(
    const InputDeck& input_deck) {
  if (input_deck.geometry_atoms.empty()) {
    throw std::runtime_error("pure C++ standalone input loader requires $GEO");
  }

  const std::string basis_display_name =
      normalize_basis_leaf_name(input_deck.metadata.basis_name);
  const std::filesystem::path basis_file_path =
      resolve_basis_file_path(basis_display_name);

  std::set<int> required_atomic_numbers;
  for (const InputDeckGeometryAtom& atom : input_deck.geometry_atoms) {
    required_atomic_numbers.insert(
        atomic_number_from_element_symbol(atom.element_symbol));
  }

  const auto shells_by_atomic_number =
      parse_basis_templates(
          basis_file_path,
          required_atomic_numbers);

  InputDeckPrimaryBasisBuildResult result;
  result.basis_file_path = basis_file_path;
  result.basis_display_name = basis_display_name;
  result.libcint_input =
      build_libcint_input(input_deck, shells_by_atomic_number);
  return result;
}

}  // namespace xmvb::vb
