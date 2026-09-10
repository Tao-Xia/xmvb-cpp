#pragma once

#include <string>
#include <vector>

#include "vbscf/integrals/ao/ao_integral_input.hpp"
#include "vbscf/integrals/ao/libcint_input.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"

namespace xmvb::vb {

bool initial_orbital_guess_supported(int guess_type) noexcept;

/**
 * @brief Builds the orbital guess for the requested GUESS mode.
 *
 * GUESS=AUTO/UNIT/MO are constructed from the C++ AO integral inputs. For
 * GUESS=READ/RDCI the builder consumes the already parsed `$GUS` body lines
 * from `InputDeck` and remaps those coefficients onto the current VB sparse-
 * orbital layout directly from the parsed input and AO integrals.
 */
void build_initial_orbital_guess(
    const std::string& input_file_path,
    int guess_type,
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const std::vector<std::string>* read_guess_lines,
    OrbitalPreparationInput* orbital_preparation_input);

}  // namespace xmvb::vb
