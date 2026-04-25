#include "runtime/input_deck_orbital_support_builder.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/input_deck_keywords.hpp"

namespace xmvb::vb {
namespace {

bool is_angular_prefix(char character) {
  switch (character) {
    case 'S':
    case 'P':
    case 'D':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
      return true;
    default:
      return false;
  }
}

std::string to_ascii_upper(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

int parse_integer_token(const std::string& token) {
  std::size_t consumed_characters = 0;
  const int value = std::stoi(token, &consumed_characters);
  if (consumed_characters != token.size()) {
    throw std::runtime_error("invalid integer token in orbital support block: " + token);
  }
  return value;
}

void append_expanded_integer_token(
    const std::string& token,
    std::vector<int>* expanded_values) {
  if (expanded_values == nullptr) {
    throw std::invalid_argument("expanded_values must not be null");
  }

  const std::size_t colon_position = token.find(':');
  const std::size_t hyphen_position = token.find('-');
  const std::size_t star_position = token.find('*');
  if (colon_position != std::string::npos) {
    const int first_value = parse_integer_token(token.substr(0, colon_position));
    const int last_value = parse_integer_token(token.substr(colon_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending ':' range is not supported in orbital support block");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
    }
    return;
  }
  if (hyphen_position != std::string::npos) {
    const int first_value = parse_integer_token(token.substr(0, hyphen_position));
    const int last_value = parse_integer_token(token.substr(hyphen_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending '-' range is not supported in orbital support block");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
    }
    return;
  }
  if (star_position != std::string::npos) {
    const int repeated_value = parse_integer_token(token.substr(0, star_position));
    const int repeat_count = parse_integer_token(token.substr(star_position + 1));
    if (repeat_count < 0) {
      throw std::runtime_error("negative repetition count is not supported in orbital support block");
    }
    for (int repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
      expanded_values->push_back(repeated_value);
    }
    return;
  }
  expanded_values->push_back(parse_integer_token(token));
}

std::vector<int> expand_integer_tokens(
    const std::vector<std::string>& tokens) {
  std::vector<int> expanded_values;
  for (const std::string& token : tokens) {
    append_expanded_integer_token(token, &expanded_values);
  }
  return expanded_values;
}

std::vector<std::string> split_sao_label_sequence(
    const std::string& compact_label_text) {
  const std::string label_text = to_ascii_upper(compact_label_text);
  std::vector<std::string> labels;
  std::size_t current_begin = std::string::npos;
  for (std::size_t character_index = 0;
       character_index < label_text.size();
       ++character_index) {
    if (!is_angular_prefix(label_text[character_index])) {
      continue;
    }
    if (current_begin != std::string::npos) {
      labels.push_back(
          label_text.substr(current_begin, character_index - current_begin));
    }
    current_begin = character_index;
  }
  if (current_begin != std::string::npos) {
    labels.push_back(label_text.substr(current_begin));
  }
  if (labels.empty()) {
    throw std::runtime_error("SAO fragment line does not contain any AO labels");
  }
  return labels;
}

char angular_letter_from_l(int angular_momentum) {
  switch (angular_momentum) {
    case 0:
      return 'S';
    case 1:
      return 'P';
    case 2:
      return 'D';
    case 3:
      return 'F';
    case 4:
      return 'G';
    case 5:
      return 'H';
    case 6:
      return 'I';
    default:
      throw std::runtime_error("unsupported AO angular momentum in support builder");
  }
}

std::string build_cartesian_ao_label(
    const InputDeckOrbitalSupportBuildInput& build_input,
    int basis_function_index) {
  const int exponent_offset = basis_function_index * 3;
  const int lx = build_input.ao_cartesian_exponents[exponent_offset];
  const int ly = build_input.ao_cartesian_exponents[exponent_offset + 1];
  const int lz = build_input.ao_cartesian_exponents[exponent_offset + 2];
  const int angular_momentum = lx + ly + lz;

  std::string label(1, angular_letter_from_l(angular_momentum));
  label.append(lx, 'X');
  label.append(ly, 'Y');
  label.append(lz, 'Z');
  return label;
}

std::vector<std::vector<int>> build_default_atom_fragments(
    const InputDeckOrbitalSupportBuildInput& build_input) {
  std::vector<std::vector<int>> atom_fragments(build_input.n_atoms);
  for (int basis_function_index = 0;
       basis_function_index < build_input.n_basis_functions;
       ++basis_function_index) {
    const int atom_index = build_input.ao_to_atom[basis_function_index];
    if (atom_index < 0 || atom_index >= build_input.n_atoms) {
      throw std::runtime_error("AO-to-atom map contains an out-of-range atom index");
    }
    atom_fragments[atom_index].push_back(basis_function_index + 1);
  }
  return atom_fragments;
}

std::vector<int> build_atom_fragment(
    const std::vector<int>& atoms,
    const std::vector<std::vector<int>>& atom_fragments,
    int n_atoms) {
  std::vector<int> fragment_basis_indices;
  for (int atom_label : atoms) {
    if (atom_label <= 0 || atom_label > n_atoms) {
      throw std::runtime_error("fragment atom index is out of range");
    }
    const auto& atom_basis_indices =
        atom_fragments[atom_label - 1];
    fragment_basis_indices.insert(
        fragment_basis_indices.end(),
        atom_basis_indices.begin(),
        atom_basis_indices.end());
  }
  return fragment_basis_indices;
}

std::vector<int> build_sao_fragment(
    const InputDeckTokenLine& fragment_line,
    int declared_atom_count,
    const InputDeckOrbitalSupportBuildInput& build_input) {
  if (fragment_line.tokens.empty()) {
    throw std::runtime_error("empty SAO fragment line");
  }
  std::vector<int> atoms;
  for (std::size_t token_index = 1;
       token_index < fragment_line.tokens.size();
       ++token_index) {
    append_expanded_integer_token(fragment_line.tokens[token_index], &atoms);
  }
  if (static_cast<int>(atoms.size()) != declared_atom_count) {
    throw std::runtime_error("SAO fragment atom count does not match the declared header size");
  }

  const std::vector<std::string> labels =
      split_sao_label_sequence(fragment_line.tokens.front());
  std::vector<int> fragment_basis_indices;
  for (int atom_label : atoms) {
    if (atom_label <= 0 || atom_label > build_input.n_atoms) {
      throw std::runtime_error("SAO fragment atom index is out of range");
    }
    const int atom_index = atom_label - 1;
    for (const std::string& label : labels) {
      for (int basis_function_index = 0;
           basis_function_index < build_input.n_basis_functions;
           ++basis_function_index) {
        if (build_input.ao_to_atom[basis_function_index] != atom_index) {
          continue;
        }
        const std::string ao_label =
            build_cartesian_ao_label(build_input, basis_function_index);
        if (ao_label.find(label) != std::string::npos) {
          fragment_basis_indices.push_back(basis_function_index + 1);
        }
      }
    }
  }
  std::sort(fragment_basis_indices.begin(), fragment_basis_indices.end());
  return fragment_basis_indices;
}

std::vector<std::vector<int>> build_fragments(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input) {
  const auto atom_fragments = build_default_atom_fragments(build_input);
  if (!input_deck.fragment_block.present) {
    if (build_input.fragment_type != kFragmentTypeAtom) {
      throw std::runtime_error(
          "pure C++ support builder requires $FRAG when FRGTYP is not ATOM");
    }
    return atom_fragments;
  }

  const int n_fragments =
      static_cast<int>(input_deck.fragment_block.declared_fragment_sizes.size());
  if (n_fragments <= 0) {
    throw std::runtime_error("fragment block header does not declare any fragments");
  }

  std::vector<std::vector<int>> fragments;
  fragments.reserve(n_fragments);
  if (build_input.fragment_type == kFragmentTypeAtom) {
    std::vector<int> buffered_atoms;
    std::size_t next_entry_index = 0;
    for (int fragment_index = 0; fragment_index < n_fragments; ++fragment_index) {
      const int expected_atom_count =
          input_deck.fragment_block.declared_fragment_sizes[fragment_index];
      while (static_cast<int>(buffered_atoms.size()) < expected_atom_count &&
             next_entry_index < input_deck.fragment_block.entries.size()) {
        const auto expanded_atoms =
            expand_integer_tokens(
                input_deck.fragment_block.entries[next_entry_index].tokens);
        buffered_atoms.insert(
            buffered_atoms.end(),
            expanded_atoms.begin(),
            expanded_atoms.end());
        ++next_entry_index;
      }
      if (static_cast<int>(buffered_atoms.size()) < expected_atom_count) {
        throw std::runtime_error("not enough atom labels in $FRAG block");
      }
      std::vector<int> fragment_atoms(
          buffered_atoms.begin(),
          buffered_atoms.begin() + expected_atom_count);
      buffered_atoms.erase(
          buffered_atoms.begin(),
          buffered_atoms.begin() + expected_atom_count);
      fragments.push_back(
          build_atom_fragment(fragment_atoms, atom_fragments, build_input.n_atoms));
    }
    return fragments;
  }

  if (build_input.fragment_type != kFragmentTypeSao) {
    throw std::runtime_error("unsupported fragment type in pure C++ support builder");
  }
  if (input_deck.fragment_block.entries.size() != n_fragments) {
    throw std::runtime_error("SAO fragment block does not contain one line per fragment");
  }
  for (int fragment_index = 0; fragment_index < n_fragments; ++fragment_index) {
    fragments.push_back(
        build_sao_fragment(
            input_deck.fragment_block.entries[fragment_index],
            input_deck.fragment_block.declared_fragment_sizes[fragment_index],
            build_input));
  }
  return fragments;
}

std::vector<std::vector<int>> build_historical_full_ao_chart(
    int n_orbitals,
    int n_basis_functions) {
  std::vector<std::vector<int>> orbital_basis_indices(
      n_orbitals);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    auto& orbital_indices = orbital_basis_indices[orbital_index];
    orbital_indices.reserve(n_basis_functions);
    for (int basis_function_index = 0;
         basis_function_index < n_basis_functions;
         ++basis_function_index) {
      orbital_indices.push_back(basis_function_index + 1);
    }
  }
  return orbital_basis_indices;
}

std::vector<std::vector<int>> build_orbital_fragment_chart(
    const InputDeck& input_deck,
    const std::vector<std::vector<int>>& fragments,
    int n_orbitals) {
  if (input_deck.orbital_support_block.block_type != OrbitalSupportBlockType::Orb) {
    throw std::runtime_error("pure C++ support builder currently expects a $ORB block");
  }
  if (static_cast<int>(input_deck.orbital_support_block.declared_support_sizes.size()) <
      n_orbitals) {
    throw std::runtime_error("orbital support header does not declare enough orbitals");
  }

  std::vector<std::vector<int>> orbital_fragment_indices;
  orbital_fragment_indices.reserve(n_orbitals);
  std::vector<int> buffered_fragment_indices;
  std::size_t next_entry_index = 0;
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int expected_fragment_count =
        input_deck.orbital_support_block.declared_support_sizes[orbital_index];
    while (static_cast<int>(buffered_fragment_indices.size()) < expected_fragment_count &&
           next_entry_index < input_deck.orbital_support_block.entries.size()) {
      const auto expanded_fragment_indices =
          expand_integer_tokens(
              input_deck.orbital_support_block.entries[next_entry_index].tokens);
      buffered_fragment_indices.insert(
          buffered_fragment_indices.end(),
          expanded_fragment_indices.begin(),
          expanded_fragment_indices.end());
      ++next_entry_index;
    }
    if (static_cast<int>(buffered_fragment_indices.size()) < expected_fragment_count) {
      throw std::runtime_error("not enough fragment labels in $ORB block");
    }
    orbital_fragment_indices.emplace_back(
        buffered_fragment_indices.begin(),
        buffered_fragment_indices.begin() + expected_fragment_count);
    buffered_fragment_indices.erase(
        buffered_fragment_indices.begin(),
        buffered_fragment_indices.begin() + expected_fragment_count);
  }
  return orbital_fragment_indices;
}

std::vector<std::vector<int>> build_actorb_basis_chart(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input) {
  if (input_deck.orbital_support_block.block_type !=
      OrbitalSupportBlockType::ActOrb) {
    throw std::runtime_error("pure C++ active-orbital builder expects a $ACTORB block");
  }
  if (build_input.orbital_type != kOrbitalTypeHao ||
      build_input.fragment_type != kFragmentTypeAtom) {
    throw std::runtime_error("$ACTORB is only valid for ORBTYP=HAO and FRGTYP=ATOM");
  }
  if (build_input.n_active_orbitals <= 0 ||
      build_input.n_active_orbitals > build_input.n_orbitals) {
    throw std::runtime_error("invalid active-orbital count for $ACTORB support builder");
  }

  const int n_inactive_orbitals =
      build_input.n_orbitals - build_input.n_active_orbitals;
  if (static_cast<int>(input_deck.orbital_support_block.entries.size()) !=
      build_input.n_active_orbitals) {
    throw std::runtime_error(
        "$ACTORB must provide exactly one support line for each active orbital");
  }

  const auto atom_fragments = build_default_atom_fragments(build_input);
  std::vector<std::vector<int>> orbital_basis_indices(
      build_input.n_orbitals);

  // Legacy readorb.c initializes all inactive orbitals to the default HAO
  // support before reading the explicit active-orbital lines from `$ACTORB`.
  // For `FRGTYP=ATOM`, the default support is the full list of atomic
  // fragments, which expands to the full AO span of the molecule.
  std::vector<int> all_atom_labels;
  all_atom_labels.reserve(build_input.n_atoms);
  for (int atom_index = 0; atom_index < build_input.n_atoms; ++atom_index) {
    all_atom_labels.push_back(atom_index + 1);
  }
  for (int orbital_index = 0; orbital_index < n_inactive_orbitals; ++orbital_index) {
    orbital_basis_indices[orbital_index] =
        build_atom_fragment(
            all_atom_labels,
            atom_fragments,
            build_input.n_atoms);
  }

  // Each `$ACTORB` line names the atom support of one active orbital directly.
  // There is no separate support-count header in this format.
  for (int active_index = 0;
       active_index < build_input.n_active_orbitals;
       ++active_index) {
    std::vector<int> atom_labels;
    const auto& entry =
        input_deck.orbital_support_block.entries[active_index];
    for (const std::string& token : entry.tokens) {
      append_expanded_integer_token(token, &atom_labels);
    }
    orbital_basis_indices[n_inactive_orbitals + active_index] =
        build_atom_fragment(
            atom_labels,
            atom_fragments,
            build_input.n_atoms);
  }

  return orbital_basis_indices;
}

InputDeckOrbitalSupportChart pack_support_chart(
    const std::vector<std::vector<int>>& orbital_basis_indices,
    int n_orbitals,
    int n_basis_functions) {
  InputDeckOrbitalSupportChart chart;
  chart.orbital_basis_index_table.assign(
      n_orbitals * n_basis_functions,
      0);
  chart.orbital_basis_counts.assign(n_orbitals, 0);
  chart.original_orbital_basis_counts.assign(n_orbitals, 0);

  // The optimizer still consumes the historical fixed-width orbital table, but
  // the pure C++ builder keeps the intermediate representation as vectors of AO
  // indices so fragment expansion, sorting, and validation are all explicit.
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const auto& basis_indices =
        orbital_basis_indices[orbital_index];
    const int basis_count = static_cast<int>(basis_indices.size());
    if (basis_count > n_basis_functions) {
      throw std::runtime_error("orbital support exceeds the total number of basis functions");
    }
    chart.orbital_basis_counts[orbital_index] = basis_count;
    chart.original_orbital_basis_counts[orbital_index] = basis_count;
    for (int coefficient_index = 0; coefficient_index < basis_count; ++coefficient_index) {
      const int basis_function_index = basis_indices[coefficient_index];
      if (basis_function_index <= 0 || basis_function_index > n_basis_functions) {
        throw std::runtime_error("orbital support contains an out-of-range AO index");
      }
      chart.orbital_basis_index_table[orbital_index * n_basis_functions +
                                      coefficient_index] = basis_function_index;
    }
  }
  return chart;
}

}  // namespace

bool can_build_input_deck_orbital_support_chart(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input) noexcept {
  if (build_input.n_atoms <= 0 ||
      build_input.n_basis_functions <= 0 ||
      build_input.n_orbitals <= 0) {
    return false;
  }
  if (static_cast<int>(build_input.ao_to_atom.size()) != build_input.n_basis_functions ||
      static_cast<int>(build_input.ao_cartesian_exponents.size()) !=
          build_input.n_basis_functions * 3) {
    return false;
  }
  if (build_input.orbital_type == kOrbitalTypeOeo) {
    return true;
  }
  if (build_input.orbital_type != kOrbitalTypeHao) {
    return false;
  }
  if (input_deck.orbital_support_block.block_type == OrbitalSupportBlockType::Orb) {
    return true;
  }
  if (input_deck.orbital_support_block.block_type == OrbitalSupportBlockType::ActOrb) {
    return build_input.fragment_type == kFragmentTypeAtom &&
        build_input.n_active_orbitals > 0 &&
        build_input.n_active_orbitals <= build_input.n_orbitals;
  }
  return false;
}

InputDeckOrbitalSupportChart build_input_deck_orbital_support_chart(
    const InputDeck& input_deck,
    const InputDeckOrbitalSupportBuildInput& build_input) {
  if (!can_build_input_deck_orbital_support_chart(input_deck, build_input)) {
    throw std::runtime_error(
        "pure C++ orbital support builder does not support this input-deck path yet");
  }

  if (build_input.orbital_type == kOrbitalTypeOeo) {
    return pack_support_chart(
        build_historical_full_ao_chart(
            build_input.n_orbitals,
            build_input.n_basis_functions),
        build_input.n_orbitals,
        build_input.n_basis_functions);
  }

  std::vector<std::vector<int>> orbital_basis_indices;
  if (input_deck.orbital_support_block.block_type == OrbitalSupportBlockType::ActOrb) {
    orbital_basis_indices =
        build_actorb_basis_chart(
            input_deck,
            build_input);
  } else {
    const auto fragments = build_fragments(input_deck, build_input);
    const auto orbital_fragment_chart =
        build_orbital_fragment_chart(
            input_deck,
            fragments,
            build_input.n_orbitals);
    orbital_basis_indices.assign(build_input.n_orbitals, {});
    for (int orbital_index = 0;
         orbital_index < build_input.n_orbitals;
         ++orbital_index) {
      auto& basis_indices = orbital_basis_indices[orbital_index];
      for (int fragment_label :
           orbital_fragment_chart[orbital_index]) {
        if (fragment_label <= 0 ||
            fragment_label > static_cast<int>(fragments.size())) {
          throw std::runtime_error("orbital support references an out-of-range fragment");
        }
        const auto& fragment_basis_indices =
            fragments[fragment_label - 1];
        basis_indices.insert(
            basis_indices.end(),
            fragment_basis_indices.begin(),
            fragment_basis_indices.end());
      }
    }
  }
  return pack_support_chart(
      orbital_basis_indices,
      build_input.n_orbitals,
      build_input.n_basis_functions);
}

}  // namespace xmvb::vb
