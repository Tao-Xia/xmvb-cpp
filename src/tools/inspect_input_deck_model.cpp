#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/input_deck_model.hpp"

namespace {

const char* orbital_support_block_name(
    xmvb::vb::OrbitalSupportBlockType block_type) {
  switch (block_type) {
    case xmvb::vb::OrbitalSupportBlockType::None:
      return "none";
    case xmvb::vb::OrbitalSupportBlockType::Orb:
      return "orb";
    case xmvb::vb::OrbitalSupportBlockType::ActOrb:
      return "actorb";
  }
  return "unknown";
}

}  // namespace

int main(
    int argc,
    char** argv) {
  if (argc != 2) {
    std::cerr << "usage: inspect_input_deck_model <input.xmi>\n";
    return EXIT_FAILURE;
  }

  try {
    const xmvb::vb::InputDeck input_deck =
        xmvb::vb::parse_input_deck_model(argv[1]);
    std::cout << "basis_name = " << input_deck.metadata.basis_name << '\n';
    std::cout << "structure_class = " << input_deck.metadata.structure_class_keyword << '\n';
    std::cout << "declared_active_orbitals = "
              << input_deck.metadata.declared_active_orbitals << '\n';
    std::cout << "declared_active_electrons = "
              << input_deck.metadata.declared_active_electrons << '\n';
    std::cout << "declared_spin_multiplicity = "
              << input_deck.metadata.declared_spin_multiplicity << '\n';
    std::cout << "geometry_atoms = " << input_deck.geometry_atoms.size() << '\n';
    std::cout << "fragment_block_present = "
              << (input_deck.fragment_block.present ? "true" : "false") << '\n';
    std::cout << "fragment_declared_count = "
              << input_deck.fragment_block.declared_fragment_sizes.size() << '\n';
    std::cout << "fragment_entry_count = "
              << input_deck.fragment_block.entries.size() << '\n';
    std::cout << "orbital_support_block = "
              << orbital_support_block_name(input_deck.orbital_support_block.block_type) << '\n';
    std::cout << "orbital_declared_count = "
              << input_deck.orbital_support_block.declared_support_sizes.size() << '\n';
    std::cout << "orbital_entry_count = "
              << input_deck.orbital_support_block.entries.size() << '\n';
    std::cout << "guess_block_present = "
              << (input_deck.guess_block.present ? "true" : "false") << '\n';
    std::cout << "guess_line_count = "
              << input_deck.guess_block.raw_lines.size() << '\n';
    std::cout << "explicit_str_present = "
              << (input_deck.has_explicit_raw_structures ? "true" : "false") << '\n';
    std::cout << "explicit_str_count = "
              << input_deck.explicit_raw_structures.n_structures << '\n';
  } catch (const std::exception& exception) {
    std::cerr << "inspect_input_deck_model failed: "
              << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
