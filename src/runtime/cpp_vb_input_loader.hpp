#pragma once

#include <string>
#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/raw_structure_data.hpp"
#include "runtime_c/cpp_runtime_extractor.h"

namespace xmvb::vb {

struct CppVbStaticMoleculeMetadata {
  int n_atoms = 0;
  int n_shells = 0;
  std::vector<int> atomic_numbers;
  std::vector<double> atomic_coordinates;
  std::vector<int> shell_to_atom;
  std::vector<int> shell_angular_momenta;
  std::vector<int> shell_n_primitives;
  std::vector<int> shell_ao_starts;
  std::vector<int> shell_ao_counts;
  std::vector<int> ao_to_atom;
  std::vector<int> ao_to_shell;
  std::vector<int> ao_angular_momenta;
  std::vector<int> ao_shell_local_indices;
  std::vector<int> ao_cartesian_exponents;
};

struct CppVbInputLoadResult {
  CppVbInput input;
  RawStructureData raw_structure_data;
  CppVbStaticMoleculeMetadata static_molecule_metadata;
  CppRuntimeExtractionTimings runtime_timings{};
  double nuclear_repulsion_energy = 0.0;
  double structure_expansion_seconds = 0.0;
  double total_seconds = 0.0;
};

/**
 * @brief Loads the full C++ VB input bundle from the mixed C/C++ runtime.
 *
 * The C runtime prepares the raw molecular, VB-structure, orbital, and AO-integral data.
 * The determinant expansion and subsequent VBSCF numerical kernels remain in C++.
 *
 * @param input_file_path Input deck used to initialize the runtime bundle.
 * @return CppVbInput Fully populated matrix-builder input.
 */
CppVbInput load_cpp_vb_input(
    const std::string& input_file_path);

CppVbInputLoadResult load_cpp_vb_input_with_timings(
    const std::string& input_file_path);

}  // namespace xmvb::vb
