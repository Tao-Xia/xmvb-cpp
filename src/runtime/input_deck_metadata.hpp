#pragma once

#include <string>

#include "runtime/input_deck_keywords.hpp"

namespace xmvb::vb {

/**
 * @brief Pure C++ summary of the `$CTRL` metadata needed by the standalone loader.
 *
 * This parser intentionally targets the subset of historical XMVB deck syntax
 * that the standalone C++ front-end consumes directly: basis selection,
 * guess/orbital chart types, charge/unit metadata, SCF iteration limits, and
 * a few standalone flags. Keeping these fields in a dedicated C++ object
 * avoids routing the pure loader through the old monolithic C runtime headers.
 */
struct InputDeckMetadata {
  std::string basis_name;
  std::string structure_class_keyword;
  int guess_type = kGuessTypeAuto;
  int orbital_type = kOrbitalTypeGen;
  int fragment_type = kFragmentTypeAtom;
  int wavefunction_type = kWavefunctionTypeStructure;
  int vb_function_type = kVbFunctionTypeDeterminant;
  int declared_active_orbitals = 0;
  int declared_active_electrons = 0;
  int declared_spin_multiplicity = 1;
  int total_charge = 0;
  int requested_scf_max_iterations = 0;
  bool geometry_coordinates_in_bohr = false;
  bool request_ri_two_electron_mode = false;
  bool request_molden_output = false;
};

InputDeckMetadata parse_input_deck_metadata(
    const std::string& input_file_path);

}  // namespace xmvb::vb
