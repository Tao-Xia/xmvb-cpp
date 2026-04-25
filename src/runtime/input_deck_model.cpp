#include "runtime/input_deck_model.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xmvb::vb {
namespace {

enum class InputDeckBlock {
  None,
  Control,
  Geometry,
  Fragment,
  OrbitalSupport,
  Guess,
  RawStructures,
};

std::string trim_ascii_whitespace(std::string value) {
  const auto first = std::find_if_not(
      value.begin(),
      value.end(),
      [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(
      value.rbegin(),
      value.rend(),
      [](unsigned char c) { return std::isspace(c) != 0; }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::string strip_inline_comment(const std::string& line) {
  const std::size_t comment_position = line.find('#');
  if (comment_position == std::string::npos) {
    return line;
  }
  return line.substr(0, comment_position);
}

std::string to_ascii_upper(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string to_ascii_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::vector<std::string> split_ascii_whitespace(
    const std::string& line) {
  std::vector<std::string> tokens;
  std::string token;
  for (char character : line) {
    if (std::isspace(static_cast<unsigned char>(character)) != 0) {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
      continue;
    }
    token.push_back(character);
  }
  if (!token.empty()) {
    tokens.push_back(token);
  }
  return tokens;
}

bool token_looks_like_floating_point(
    const std::string& token) {
  return token.find('.') != std::string::npos ||
      token.find('e') != std::string::npos ||
      token.find('E') != std::string::npos ||
      token.find('d') != std::string::npos ||
      token.find('D') != std::string::npos;
}

double parse_fortran_double(
    std::string token) {
  for (char& character : token) {
    if (character == 'd' || character == 'D') {
      character = 'E';
    }
  }
  std::size_t consumed_characters = 0;
  const double value = std::stod(token, &consumed_characters);
  if (consumed_characters != token.size()) {
    throw std::runtime_error("invalid floating-point token in input deck: " + token);
  }
  return value;
}

void append_expanded_integer_token(
    const std::string& token,
    bool duplicate_colon_range,
    std::vector<int>* expanded_values) {
  if (expanded_values == nullptr) {
    throw std::invalid_argument("expanded_values must not be null");
  }
  const std::size_t colon_position = token.find(':');
  const std::size_t hyphen_position = token.find('-');
  const std::size_t star_position = token.find('*');
  if (colon_position != std::string::npos) {
    const int first_value = std::stoi(token.substr(0, colon_position));
    const int last_value = std::stoi(token.substr(colon_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending ':' range is not supported in input deck");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
      if (duplicate_colon_range) {
        expanded_values->push_back(value);
      }
    }
    return;
  }
  if (hyphen_position != std::string::npos) {
    const int first_value = std::stoi(token.substr(0, hyphen_position));
    const int last_value = std::stoi(token.substr(hyphen_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending '-' range is not supported in input deck");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
    }
    return;
  }
  if (star_position != std::string::npos) {
    const int repeated_value = std::stoi(token.substr(0, star_position));
    const int repeat_count = std::stoi(token.substr(star_position + 1));
    if (repeat_count < 0) {
      throw std::runtime_error("negative repetition count is not supported in input deck");
    }
    for (int repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
      expanded_values->push_back(repeated_value);
    }
    return;
  }
  expanded_values->push_back(std::stoi(token));
}

std::vector<int> expand_integer_tokens(
    const std::vector<std::string>& tokens,
    bool duplicate_colon_range) {
  std::vector<int> expanded_values;
  for (const std::string& token : tokens) {
    append_expanded_integer_token(
        token,
        duplicate_colon_range,
        &expanded_values);
  }
  return expanded_values;
}

void apply_ctrl_assignment(
    const std::string& raw_key,
    const std::string& raw_value,
    InputDeckMetadata* metadata) {
  if (metadata == nullptr) {
    throw std::invalid_argument("metadata must not be null");
  }
  const std::string key = to_ascii_upper(raw_key);
  const std::string value_upper = to_ascii_upper(raw_value);
  if (key == "ITMAX") {
    metadata->requested_scf_max_iterations = std::stoi(raw_value);
    return;
  }
  if (key == "BASIS") {
    metadata->basis_name = to_ascii_lower(raw_value);
    return;
  }
  if (key == "STR") {
    metadata->structure_class_keyword = value_upper;
    return;
  }
  if (key == "NAO") {
    metadata->declared_active_orbitals = std::stoi(raw_value);
    return;
  }
  if (key == "NAE") {
    metadata->declared_active_electrons = std::stoi(raw_value);
    return;
  }
  if (key == "NMUL") {
    metadata->declared_spin_multiplicity = std::stoi(raw_value);
    return;
  }
  if (key == "NCHARGE") {
    metadata->total_charge = std::stoi(raw_value);
    return;
  }
  if (key == "UNIT") {
    if (value_upper == "BOHR") {
      metadata->geometry_coordinates_in_bohr = true;
    } else if (value_upper == "ANGS") {
      metadata->geometry_coordinates_in_bohr = false;
    }
    return;
  }
  if (key == "GUESS") {
    if (value_upper == "AUTO") {
      metadata->guess_type = kGuessTypeAuto;
    } else if (value_upper == "UNIT") {
      metadata->guess_type = kGuessTypeUnit;
    } else if (value_upper == "READ") {
      metadata->guess_type = kGuessTypeRead;
    } else if (value_upper == "RDCI") {
      metadata->guess_type = kGuessTypeRdci;
    } else if (value_upper == "MO") {
      metadata->guess_type = kGuessTypeMo;
    } else if (value_upper == "NBO") {
      metadata->guess_type = kGuessTypeNbo;
    }
    return;
  }
  if (key == "ORBTYP") {
    if (value_upper == "GEN") {
      metadata->orbital_type = kOrbitalTypeGen;
    } else if (value_upper == "HAO") {
      metadata->orbital_type = kOrbitalTypeHao;
    } else if (value_upper == "BDO") {
      metadata->orbital_type = kOrbitalTypeBdo;
    } else if (value_upper == "OEO") {
      metadata->orbital_type = kOrbitalTypeOeo;
    }
    return;
  }
  if (key == "FRGTYP") {
    if (value_upper == "ATOM") {
      metadata->fragment_type = kFragmentTypeAtom;
    } else if (value_upper == "SAO") {
      metadata->fragment_type = kFragmentTypeSao;
    }
    return;
  }
  if (key == "WFNTYP") {
    if (value_upper == "STR") {
      metadata->wavefunction_type = kWavefunctionTypeStructure;
    } else if (value_upper == "DET") {
      metadata->wavefunction_type = kWavefunctionTypeDeterminant;
    }
    return;
  }
  if (key == "VBFTYP") {
    if (value_upper == "DET") {
      metadata->vb_function_type = kVbFunctionTypeDeterminant;
    } else if (value_upper == "PPD") {
      metadata->vb_function_type = kVbFunctionTypePerfectPairingDeterminant;
    }
    return;
  }
  if (key == "INT") {
    metadata->request_ri_two_electron_mode = (value_upper == "RI");
  }
}

void apply_ctrl_flag(
    const std::string& raw_token,
    InputDeckMetadata* metadata) {
  if (metadata == nullptr) {
    throw std::invalid_argument("metadata must not be null");
  }
  const std::string token = to_ascii_upper(raw_token);
  if (token == "TBVBSCF") {
    throw std::invalid_argument(
        "TBVBSCF is no longer supported; only the standard VBSCF path remains");
  } else if (token == "MOLDEN") {
    metadata->request_molden_output = true;
  } else if (token == "VBPDFT") {
    throw std::invalid_argument(
        "VBPDFT is no longer supported; only the standard VBSCF path remains");
  }
}

InputDeckTokenLine make_token_line(
    const std::string& raw_line) {
  InputDeckTokenLine token_line;
  token_line.raw_line = raw_line;
  token_line.tokens =
      split_ascii_whitespace(trim_ascii_whitespace(strip_inline_comment(raw_line)));
  return token_line;
}

RawStructureData parse_explicit_raw_structures(
    const std::vector<InputDeckTokenLine>& structure_lines,
    const InputDeckMetadata& metadata) {
  RawStructureData raw_structure_data;
  raw_structure_data.n_active_electrons = metadata.declared_active_electrons;
  raw_structure_data.spin_multiplicity = metadata.declared_spin_multiplicity;
  raw_structure_data.wavefunction_type = metadata.wavefunction_type;
  raw_structure_data.vb_function_type = metadata.vb_function_type;

  for (const InputDeckTokenLine& token_line : structure_lines) {
    if (token_line.tokens.empty()) {
      continue;
    }
    std::vector<std::string> tokens = token_line.tokens;
    if (!tokens.empty() &&
        token_looks_like_floating_point(tokens.back())) {
      tokens.pop_back();
    }
    std::vector<int> current_structure = expand_integer_tokens(tokens, true);
    // `$STR` already lists one complete VB structure per logical line after
    // the legacy token expansion rules are applied. Infer the total-electron
    // count from the first expanded structure so explicit `$STR` decks no
    // longer need to borrow that dimension from the C runtime.
    if (raw_structure_data.n_total_electrons == 0) {
      raw_structure_data.n_total_electrons =
          static_cast<int>(current_structure.size());
    }
    if (static_cast<int>(current_structure.size()) !=
        raw_structure_data.n_total_electrons) {
      throw std::runtime_error(
          "explicit $STR line does not match the declared total electron count");
    }
    raw_structure_data.raw_structure_orbitals.insert(
        raw_structure_data.raw_structure_orbitals.end(),
        current_structure.begin(),
        current_structure.end());
    ++raw_structure_data.n_structures;
  }
  if (raw_structure_data.n_structures <= 0) {
    throw std::runtime_error("explicit $STR block does not contain any VB structures");
  }
  return raw_structure_data;
}

}  // namespace

InputDeck parse_input_deck_model(
    const std::string& input_file_path) {
  std::ifstream input_stream(input_file_path);
  if (!input_stream) {
    throw std::runtime_error("failed to open input file: " + input_file_path);
  }

  InputDeck input_deck;
  std::vector<InputDeckTokenLine> explicit_structure_lines;
  InputDeckBlock current_block = InputDeckBlock::None;
  std::string raw_line;
  bool fragment_header_consumed = false;
  bool orbital_header_consumed = false;
  while (std::getline(input_stream, raw_line)) {
    const std::string normalized_line =
        to_ascii_upper(trim_ascii_whitespace(strip_inline_comment(raw_line)));
    if (normalized_line.empty()) {
      if (current_block == InputDeckBlock::Guess) {
        input_deck.guess_block.raw_lines.push_back(raw_line);
      }
      continue;
    }

    if (normalized_line == "$CTRL") {
      current_block = InputDeckBlock::Control;
      continue;
    }
    if (normalized_line == "$GEO") {
      current_block = InputDeckBlock::Geometry;
      continue;
    }
    if (normalized_line == "$FRAG") {
      current_block = InputDeckBlock::Fragment;
      input_deck.fragment_block.present = true;
      fragment_header_consumed = false;
      continue;
    }
    if (normalized_line == "$ORB") {
      if (input_deck.orbital_support_block.block_type !=
          OrbitalSupportBlockType::None) {
        throw std::runtime_error("$ORB and $ACTORB cannot both appear in one input deck");
      }
      current_block = InputDeckBlock::OrbitalSupport;
      input_deck.orbital_support_block.block_type = OrbitalSupportBlockType::Orb;
      orbital_header_consumed = false;
      continue;
    }
    if (normalized_line == "$ACTORB") {
      if (input_deck.orbital_support_block.block_type !=
          OrbitalSupportBlockType::None) {
        throw std::runtime_error("$ORB and $ACTORB cannot both appear in one input deck");
      }
      current_block = InputDeckBlock::OrbitalSupport;
      input_deck.orbital_support_block.block_type = OrbitalSupportBlockType::ActOrb;
      orbital_header_consumed = false;
      continue;
    }
    if (normalized_line == "$GUS") {
      current_block = InputDeckBlock::Guess;
      input_deck.guess_block.present = true;
      continue;
    }
    if (normalized_line == "$STR") {
      current_block = InputDeckBlock::RawStructures;
      input_deck.has_explicit_raw_structures = true;
      continue;
    }
    if (normalized_line == "$END") {
      current_block = InputDeckBlock::None;
      continue;
    }

    if (current_block == InputDeckBlock::Control) {
      // `$CTRL` is tokenized as a flat stream because legacy decks freely mix
      // flags like `VBSCF` with assignments such as `NAE=5` on the same line.
      const std::vector<std::string> tokens =
          split_ascii_whitespace(trim_ascii_whitespace(strip_inline_comment(raw_line)));
      for (const std::string& token : tokens) {
        const std::size_t equals_position = token.find('=');
        if (equals_position == std::string::npos) {
          apply_ctrl_flag(token, &input_deck.metadata);
          continue;
        }
        apply_ctrl_assignment(
            token.substr(0, equals_position),
            token.substr(equals_position + 1),
            &input_deck.metadata);
      }
      continue;
    }

    if (current_block == InputDeckBlock::Geometry) {
      const std::vector<std::string> tokens =
          split_ascii_whitespace(trim_ascii_whitespace(strip_inline_comment(raw_line)));
      if (tokens.size() < 4) {
        throw std::runtime_error("invalid $GEO line: " + raw_line);
      }
      InputDeckGeometryAtom atom;
      atom.element_symbol = tokens[0];
      atom.coordinates[0] = parse_fortran_double(tokens[1]);
      atom.coordinates[1] = parse_fortran_double(tokens[2]);
      atom.coordinates[2] = parse_fortran_double(tokens[3]);
      input_deck.geometry_atoms.push_back(std::move(atom));
      continue;
    }

    if (current_block == InputDeckBlock::Fragment) {
      InputDeckTokenLine token_line = make_token_line(raw_line);
      if (token_line.tokens.empty()) {
        continue;
      }
      if (!fragment_header_consumed) {
        // The first `$FRAG` body line declares how many atoms/SAO clauses
        // belong to each fragment after the same legacy integer-expansion
        // syntax used elsewhere in VB input decks.
        input_deck.fragment_block.declared_fragment_sizes =
            expand_integer_tokens(token_line.tokens, true);
        fragment_header_consumed = true;
      } else {
        input_deck.fragment_block.entries.push_back(std::move(token_line));
      }
      continue;
    }

    if (current_block == InputDeckBlock::OrbitalSupport) {
      InputDeckTokenLine token_line = make_token_line(raw_line);
      if (token_line.tokens.empty()) {
        continue;
      }
      if (input_deck.orbital_support_block.block_type ==
          OrbitalSupportBlockType::ActOrb) {
        // Legacy `$ACTORB` does not carry a support-count header. Each body
        // line is one active-orbital support declaration, and the inactive
        // orbitals keep the historical default layout.
        input_deck.orbital_support_block.entries.push_back(std::move(token_line));
        continue;
      }
      if (!orbital_header_consumed) {
        // The support header records one declared support length per orbital.
        // Later pure C++ orbital builders will interpret the body either as
        // raw fragment labels or explicit AO supports according to ORBTYP.
        input_deck.orbital_support_block.declared_support_sizes =
            expand_integer_tokens(token_line.tokens, true);
        orbital_header_consumed = true;
      } else {
        input_deck.orbital_support_block.entries.push_back(std::move(token_line));
      }
      continue;
    }

    if (current_block == InputDeckBlock::Guess) {
      input_deck.guess_block.raw_lines.push_back(raw_line);
      continue;
    }

    if (current_block == InputDeckBlock::RawStructures) {
      InputDeckTokenLine token_line = make_token_line(raw_line);
      if (!token_line.tokens.empty()) {
        explicit_structure_lines.push_back(std::move(token_line));
      }
      continue;
    }
  }

  if (input_deck.has_explicit_raw_structures) {
    input_deck.explicit_raw_structures =
        parse_explicit_raw_structures(
            explicit_structure_lines,
            input_deck.metadata);
  }

  return input_deck;
}

}  // namespace xmvb::vb
