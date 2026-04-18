#pragma once

#include <filesystem>

#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::vb {

/**
 * @brief Writes the current orbital state to a Molden file next to an input deck.
 *
 * The exported file reuses the input path stem with a `.molden` suffix and
 * contains the cartesian AO basis stored in `input.libcint_input` together
 * with the current VB orbital coefficients from
 * `input.orbital_preparation_input`. The Molden orbital energies and
 * occupations are written as zeros because the VB orbital parameterization
 * does not expose a canonical MO spectrum.
 *
 * @param input_file_path Path to the source `.xmi` deck whose stem names the
 *        Molden output file.
 * @param input Current C++ VB input bundle after loading or optimization.
 * @return Absolute path to the written `.molden` file.
 */
std::filesystem::path write_molden_file(
    const std::filesystem::path& input_file_path,
    const CppVbInput& input);

}  // namespace xmvb::vb
