#pragma once

#include <filesystem>

#include "integrals/ao/libcint_input.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

/**
 * @brief Writes the current orbital state to a Molden file next to an input deck.
 *
 * The exported file reuses the input path stem with a `.molden` suffix and
 * contains the supplied Cartesian AO basis and orbital coefficients. The
 * Molden orbital energies and occupations are written as zeros because this
 * output interface does not expose a canonical MO spectrum.
 *
 * @param input_file_path Path to the source `.xmi` deck whose stem names the
 *        Molden output file.
 * @param libcint_input Primary AO basis and molecular geometry.
 * @param orbitals Current orbital coefficients in that AO basis.
 * @return Absolute path to the written `.molden` file.
 */
std::filesystem::path write_molden_file(const std::filesystem::path& input_file_path,
                                        const integrals::LibcintInput& libcint_input,
                                        const OrbitalPreparationInput& orbitals);

} // namespace xmvb::vb
