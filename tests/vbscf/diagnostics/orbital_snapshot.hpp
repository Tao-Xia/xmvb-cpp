#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include <Eigen/Core>

#include "core/index.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::test {

inline void read_matrix(std::ifstream& file, Eigen::MatrixXd& matrix) {
    const auto bytes =
        core::checked_multiply(static_cast<std::size_t>(matrix.size()), sizeof(double));
    file.read(reinterpret_cast<char*>(matrix.data()), core::checked_index<std::streamsize>(bytes));
    if (!file)
        throw std::runtime_error("invalid recorded orbital/CI snapshot");
}

inline Eigen::MatrixXd read_orbitals(const char* path, const vb::OrbitalPreparationInput& input) {
    Eigen::MatrixXd result(input.n_basis_functions, input.n_orbitals);
    std::ifstream file(path, std::ios::binary);
    read_matrix(file, result);
    return result;
}

inline void set_orbitals(vb::OrbitalPreparationInput& input, const Eigen::MatrixXd& values) {
    for (std::size_t i = 0; i < input.n_orbitals; ++i)
        for (int j = 0; j < vb::stored_sparse_orbital_coefficient_count(input, i); ++j) {
            const auto slot = i * input.n_basis_functions + j;
            input.orbital_value_table[slot] = values(input.orbital_basis_index_table[slot] - 1, i);
        }
}

inline Eigen::MatrixXd read_coefficients(const char* path,
                                         const vb::OrbitalPreparationInput& input,
                                         std::size_t n_structures) {
    std::ifstream file(path, std::ios::binary);
    std::array<std::uint64_t, 4> shape;
    file.read(reinterpret_cast<char*>(shape.data()), sizeof(shape));
    if (!file || shape[0] != input.n_basis_functions || shape[1] != input.n_orbitals ||
        shape[2] != n_structures || shape[3] == 0)
        throw std::runtime_error("CI snapshot dimensions differ from the input");
    const auto skip = core::checked_multiply(
        core::checked_multiply(input.n_basis_functions, input.n_orbitals), sizeof(double));
    file.seekg(core::checked_index<std::streamoff>(skip), std::ios::cur);
    Eigen::MatrixXd coefficients(core::checked_index<Eigen::Index>(shape[2]),
                                 core::checked_index<Eigen::Index>(shape[3]));
    read_matrix(file, coefficients);
    return coefficients;
}

} // namespace xmvb::test
