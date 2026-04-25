#pragma once

#include <string>
#include <vector>

#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/libcint_input.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

enum class OrbitalGuessSource {
  Cpp,
};

const char* orbital_guess_source_name(OrbitalGuessSource source);

bool cpp_initial_guess_supported(int guess_type) noexcept;

/**
 * @brief Builds the standalone C++ orbital guess for the requested GUESS mode.
 *
 * GUESS=AUTO/UNIT/MO are constructed from the C++ AO integral inputs. For
 * GUESS=READ/RDCI the builder consumes the already parsed `$GUS` body lines
 * from `InputDeck` and remaps those coefficients onto the current VB sparse-
 * orbital layout without calling the legacy C runtime `vbguess` path.
 */
void build_cpp_initial_guess(
    const std::string& input_file_path,
    int guess_type,
    const LibcintInput& libcint_input,
    const AoIntegralInput& ao_integral_input,
    const std::vector<std::string>* read_guess_lines,
    OrbitalPreparationInput* orbital_preparation_input);

}  // namespace xmvb::vb
