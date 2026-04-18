#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_frame.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::ActiveSpaceTwoElectronResult;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalDeterminant;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalFrame;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalIntegrals;
using xmvb::vb::biorthogonal_vbscf::DenseMatrix;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_frame;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals;

void require_close(
    double actual,
    double expected,
    double tolerance,
    const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(
        std::string(label) + " mismatch: actual=" + std::to_string(actual) +
        " expected=" + std::to_string(expected));
  }
}

std::vector<double> make_zero_packed_two_electron_integrals(int n_orbitals) {
  const int last_pair_index =
      xmvb::vb::TwoElectronIndexer::packed_pair_index(
          n_orbitals - 1,
          n_orbitals - 1);
  const int last_storage_index =
      xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
          last_pair_index,
          last_pair_index);
  return std::vector<double>(static_cast<std::size_t>(last_storage_index + 1), 0.0);
}

void test_one_electron_single_alpha_matrix() {
  DenseMatrix ao_overlap = DenseMatrix::Identity(2, 2);
  DenseMatrix right_orbitals(2, 2);
  right_orbitals << 1.0, 0.2,
                    0.0, 1.0;
  const BiorthogonalOrbitalFrame orbital_frame =
      build_biorthogonal_orbital_frame(ao_overlap, right_orbitals);

  const std::vector<double> right_right_one_electron = {
      1.4, 0.3,
      0.3, 0.8,
  };
  const BiorthogonalOrbitalIntegrals orbital_integrals =
      build_biorthogonal_orbital_integrals(
          orbital_frame,
          right_right_one_electron);

  ActiveSpaceTwoElectronResult two_electron_result;
  two_electron_result.packed_active_two_electron_integrals =
      make_zero_packed_two_electron_integrals(2);

  const std::vector<BiorthogonalDeterminant> determinants = {
      {{0}, {}},
      {{1}, {}},
  };
  const DenseMatrix determinant_hamiltonian =
      build_biorthogonal_determinant_hamiltonian_matrix(
          determinants,
          orbital_integrals,
          two_electron_result);

  require_close(
      determinant_hamiltonian(0, 0),
      orbital_integrals.left_right_one_electron(0, 0),
      1.0e-12,
      "single-alpha H(0,0)");
  require_close(
      determinant_hamiltonian(0, 1),
      orbital_integrals.left_right_one_electron(0, 1),
      1.0e-12,
      "single-alpha H(0,1)");
  require_close(
      determinant_hamiltonian(1, 0),
      orbital_integrals.left_right_one_electron(1, 0),
      1.0e-12,
      "single-alpha H(1,0)");
  require_close(
      determinant_hamiltonian(1, 1),
      orbital_integrals.left_right_one_electron(1, 1),
      1.0e-12,
      "single-alpha H(1,1)");
}

void test_same_spin_diagonal_direct_exchange() {
  DenseMatrix ao_overlap = DenseMatrix::Identity(2, 2);
  DenseMatrix right_orbitals = DenseMatrix::Identity(2, 2);
  const BiorthogonalOrbitalFrame orbital_frame =
      build_biorthogonal_orbital_frame(ao_overlap, right_orbitals);

  const std::vector<double> right_right_one_electron = {
      1.0, 0.0,
      0.0, 2.0,
  };
  const BiorthogonalOrbitalIntegrals orbital_integrals =
      build_biorthogonal_orbital_integrals(
          orbital_frame,
          right_right_one_electron);

  ActiveSpaceTwoElectronResult two_electron_result;
  two_electron_result.packed_active_two_electron_integrals =
      make_zero_packed_two_electron_integrals(2);
  two_electron_result.packed_active_two_electron_integrals[static_cast<std::size_t>(
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(0, 0, 1, 1))] = 0.7;
  two_electron_result.packed_active_two_electron_integrals[static_cast<std::size_t>(
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(0, 1, 0, 1))] = 0.2;

  const std::vector<BiorthogonalDeterminant> determinants = {
      {{0, 1}, {}},
  };
  const DenseMatrix determinant_hamiltonian =
      build_biorthogonal_determinant_hamiltonian_matrix(
          determinants,
          orbital_integrals,
          two_electron_result);

  require_close(
      determinant_hamiltonian(0, 0),
      1.0 + 2.0 + 0.7 - 0.2,
      1.0e-12,
      "same-spin diagonal");
}

void test_opposite_spin_diagonal_direct_term() {
  DenseMatrix ao_overlap = DenseMatrix::Identity(2, 2);
  DenseMatrix right_orbitals = DenseMatrix::Identity(2, 2);
  const BiorthogonalOrbitalFrame orbital_frame =
      build_biorthogonal_orbital_frame(ao_overlap, right_orbitals);

  const std::vector<double> right_right_one_electron = {
      0.5, 0.0,
      0.0, 1.5,
  };
  const BiorthogonalOrbitalIntegrals orbital_integrals =
      build_biorthogonal_orbital_integrals(
          orbital_frame,
          right_right_one_electron);

  ActiveSpaceTwoElectronResult two_electron_result;
  two_electron_result.packed_active_two_electron_integrals =
      make_zero_packed_two_electron_integrals(2);
  two_electron_result.packed_active_two_electron_integrals[static_cast<std::size_t>(
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(0, 0, 1, 1))] = 0.4;

  const std::vector<BiorthogonalDeterminant> determinants = {
      {{0}, {1}},
  };
  const DenseMatrix determinant_hamiltonian =
      build_biorthogonal_determinant_hamiltonian_matrix(
          determinants,
          orbital_integrals,
          two_electron_result);

  require_close(
      determinant_hamiltonian(0, 0),
      0.5 + 1.5 + 0.4,
      1.0e-12,
      "opposite-spin diagonal");
}

}  // namespace

int main() {
  try {
    test_one_electron_single_alpha_matrix();
    test_same_spin_diagonal_direct_exchange();
    test_opposite_spin_diagonal_direct_term();
  } catch (const std::exception& error) {
    std::cerr << "check_biorthogonal_determinant_hamiltonian failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "check_biorthogonal_determinant_hamiltonian passed\n";
  return 0;
}
