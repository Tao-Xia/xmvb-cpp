#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/ao/libcint/input.hpp"
#include "vbscf/orbitals/preparation/input.hpp"
#include "vbscf/orbitals/charts/partition.hpp"

namespace xmvb::vb {

std::vector<double> build_ao_normalization(
    const OrbitalPreparationInput& orbital_preparation_input);

std::vector<double> build_ao_normalization(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix);

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
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_driving_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_driving_matrix,
    const std::vector<double>& overlap_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const std::vector<double>& orbital_driving_matrix,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* orbital_value_table);

void build_block_matrix_guess(
    const LibcintInput& libcint_input,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_driving_matrix,
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
