#pragma once

#include <vector>

#include "vb/orbital/libcint_input.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

int get_orbital_basis_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index);

std::vector<std::vector<int>> detect_orbital_blocks(
    const OrbitalPreparationInput& orbital_preparation_input);

std::vector<double> build_ao_normalization(
    const OrbitalPreparationInput& orbital_preparation_input);

std::vector<double> build_ao_normalization(
    const std::vector<double>& overlap_matrix,
    int n_basis_functions);

void scale_guess_back_to_original_basis(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& ao_normalization,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

std::vector<double> build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input);

std::vector<double> build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const OrbitalPreparationInput& orbital_preparation_input);

}  // namespace xmvb::vb
