#include "runtime/input_deck_raw_structures.hpp"

#include <stdexcept>

#include "runtime/input_deck_model.hpp"

namespace xmvb::vb {

bool input_deck_contains_raw_structure_block(
    const std::string& input_file_path) {
  return parse_input_deck_model(input_file_path).has_explicit_raw_structures;
}

RawStructureData parse_input_deck_raw_structures(
    const std::string& input_file_path,
    int n_total_electrons,
    int n_active_electrons,
    int spin_multiplicity,
    int wavefunction_type,
    int vb_function_type) {
  InputDeck input_deck = parse_input_deck_model(input_file_path);
  if (!input_deck.has_explicit_raw_structures) {
    throw std::runtime_error("input file does not contain a $STR block");
  }

  RawStructureData raw_structure_data = input_deck.explicit_raw_structures;
  if (raw_structure_data.n_total_electrons <= 0) {
    raw_structure_data.n_total_electrons = n_total_electrons;
  }
  if (raw_structure_data.n_active_electrons <= 0) {
    raw_structure_data.n_active_electrons = n_active_electrons;
  }
  if (raw_structure_data.spin_multiplicity <= 0) {
    raw_structure_data.spin_multiplicity = spin_multiplicity;
  }
  raw_structure_data.wavefunction_type = wavefunction_type;
  raw_structure_data.vb_function_type = vb_function_type;
  return raw_structure_data;
}

}  // namespace xmvb::vb
