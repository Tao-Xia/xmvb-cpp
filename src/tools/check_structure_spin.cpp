#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "input/deck/model.hpp"
#include "input/loading/generated_structures.hpp"
#include "vbscf/structures/analysis/spin.hpp"
#include "vbscf/structures/expansion/expander.hpp"

namespace {

xmvb::vb::RawStructureData build_active_structure_space(
    const xmvb::vb::InputDeck& deck) {
  const auto& metadata = deck.metadata;
  if (deck.has_explicit_raw_structures) {
    auto raw = deck.explicit_raw_structures;
    if (raw.n_active_electrons <= 0) {
      raw.n_active_electrons = metadata.declared_active_electrons;
    }
    if (raw.spin_multiplicity <= 0) {
      raw.spin_multiplicity = metadata.declared_spin_multiplicity;
    }
    if (raw.n_total_electrons <= 0) {
      raw.n_total_electrons = raw.n_active_electrons;
    }
    return raw;
  }

  const int active_electrons = metadata.declared_active_electrons;
  if (!xmvb::vb::can_build_generated_raw_structures_from_structure_class(
          metadata,
          active_electrons)) {
    throw std::runtime_error(
        "spin audit requires an explicit structure block or supported STR class");
  }
  return xmvb::vb::build_generated_raw_structures_from_structure_class(
      metadata,
      active_electrons);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: check_structure_spin <input.xmi> [input.xmi ...]\n";
    return EXIT_FAILURE;
  }

  bool all_passed = true;
  std::cout << "input,state,multiplicity,spin_squared,max_splus_residual,status\n";
  std::cout << std::setprecision(16);
  for (int argument = 1; argument < argc; ++argument) {
    try {
      const std::string input_path = argv[argument];
      const xmvb::vb::InputDeck deck =
          xmvb::vb::parse_input_deck_model(input_path);
      const auto raw = build_active_structure_space(deck);
      const auto structures =
          xmvb::vb::FullDeterminantStructureExpander().expand(raw);
      const auto spin = xmvb::vb::analyze_structure_spin(
          structures,
          raw.spin_multiplicity);
      const bool passed = spin.spin_adapted();
      all_passed = all_passed && passed;
      for (int state = 0; state < deck.metadata.state_average_count; ++state) {
        std::cout << input_path << ','
                  << state + 1 << ','
                  << raw.spin_multiplicity << ','
                  << spin.spin_squared << ','
                  << spin.maximum_raising_residual << ','
                  << (passed ? "spin-adapted" : "failed") << '\n';
      }
    } catch (const std::exception& error) {
      all_passed = false;
      std::cerr << argv[argument] << ": " << error.what() << '\n';
    }
  }
  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
