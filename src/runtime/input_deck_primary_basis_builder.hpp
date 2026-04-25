#pragma once

#include <filesystem>
#include <string>

#include "runtime/input_deck_model.hpp"
#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

/**
 * @brief Primary basis resolution result for the standalone pure C++ loader.
 *
 * `basis_file_path` is the concrete `.gbs` file opened by the parser after
 * applying the legacy basis-name normalization rules. `basis_display_name`
 * keeps the normalized leaf name used for logging so the caller can report the
 * actual basis asset consumed by the standalone path.
 */
struct InputDeckPrimaryBasisBuildResult {
  std::filesystem::path basis_file_path;
  std::string basis_display_name;
  LibcintInput libcint_input;
};

/**
 * @brief Builds the primary `LibcintInput` directly from `$GEO` and `BASIS=`.
 *
 * This pure C++ builder replaces the legacy `init_bas + int_data_trans` path
 * for the standalone executable. The implementation preserves the historical
 * shell ordering, Cartesian AO counts, and per-primitive normalization factors
 * expected by the existing AO integral and orbital-preparation code.
 */
InputDeckPrimaryBasisBuildResult build_input_deck_primary_basis(
    const InputDeck& input_deck);

}  // namespace xmvb::vb
