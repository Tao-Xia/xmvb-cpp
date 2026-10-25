#pragma once

#include <filesystem>
#include <Eigen/Core>
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::output {

/** @brief Write converged orbitals beside the input as .orb, in GUS body format.
 * Preserves stored AO indices, including allowed zeros; coefficients are
 * rounded to ten decimal places in fixed-point notation.
 * No GUS/END delimiters, normalization or orbital transformation is applied.
 */
std::filesystem::path write_orbital_file(const std::filesystem::path& input_path,
                                         const vb::OrbitalPreparationInput& orbitals);

/** @brief Write a dense normalized-AO orbital frame without a VB container. */
std::filesystem::path write_orbital_file(const std::filesystem::path& input_path,
                                         const Eigen::MatrixXd& orbitals);

} // namespace xmvb::output
