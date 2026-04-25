#pragma once

#include <array>
#include <string>
#include <vector>

#include "runtime/input_deck_metadata.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

enum class OrbitalSupportBlockType {
  None,
  Orb,
  ActOrb,
};

/**
 * @brief One raw, non-empty line from an input-deck block.
 *
 * `tokens` are split from the comment-stripped line, while `raw_line` keeps the
 * original text without the trailing newline so later parsers can still inspect
 * legacy formatting when needed.
 */
struct InputDeckTokenLine {
  std::string raw_line;
  std::vector<std::string> tokens;
};

/**
 * @brief Parsed `$GEO` atom record.
 *
 * `coordinates` stores the three Cartesian components exactly as written in
 * the deck. A fixed-size array keeps this lightweight parser independent of
 * Eigen/LAPACK header order while still expressing the fact that geometry
 * points are always length-3 objects rather than dense matrices.
 */
struct InputDeckGeometryAtom {
  std::string element_symbol;
  std::array<double, 3> coordinates{0.0, 0.0, 0.0};
};

/**
 * @brief Raw fragment block content before AO expansion.
 *
 * `declared_fragment_sizes` is the expanded header from the first `$FRAG` body
 * line. Each entry line is preserved in tokenized form because `FRGTYP=ATOM`
 * and `FRGTYP=SAO` interpret the body differently.
 */
struct InputDeckFragmentBlock {
  bool present = false;
  std::vector<int> declared_fragment_sizes;
  std::vector<InputDeckTokenLine> entries;
};

/**
 * @brief Raw orbital-support block content before fragment-to-AO expansion.
 *
 * `$ORB` and `$ACTORB` share the same container because both preserve the raw
 * token lines from the deck, but their legacy formats differ:
 * `$ORB` starts with a support-count header while `$ACTORB` lists only one
 * active-orbital support line per row. The downstream support builder must
 * therefore branch on `block_type` instead of assuming a single format.
 */
struct InputDeckOrbitalSupportBlock {
  OrbitalSupportBlockType block_type = OrbitalSupportBlockType::None;
  std::vector<int> declared_support_sizes;
  std::vector<InputDeckTokenLine> entries;
};

/**
 * @brief Raw `$GUS` lines used by `GUESS=READ/RDCI`.
 *
 * The standalone guess builder still consumes legacy line-oriented syntax, so
 * the deck model preserves the body verbatim instead of forcing an early dense
 * coefficient representation.
 */
struct InputDeckGuessBlock {
  bool present = false;
  std::vector<std::string> raw_lines;
};

/**
 * @brief Pure C++ semantic summary of one `.xmi` input deck.
 *
 * This object is the future replacement for the legacy `readinp.c` surface in
 * the standalone path. It centralizes deck parsing so downstream loaders no
 * longer need to rescan the same file for `$CTRL`, `$STR`, `$GEO`, `$FRAG`,
 * `$ORB/$ACTORB`, and `$GUS`.
 */
struct InputDeck {
  InputDeckMetadata metadata;
  std::vector<InputDeckGeometryAtom> geometry_atoms;
  InputDeckFragmentBlock fragment_block;
  InputDeckOrbitalSupportBlock orbital_support_block;
  InputDeckGuessBlock guess_block;
  bool has_explicit_raw_structures = false;
  RawStructureData explicit_raw_structures;
};

InputDeck parse_input_deck_model(
    const std::string& input_file_path);

}  // namespace xmvb::vb
