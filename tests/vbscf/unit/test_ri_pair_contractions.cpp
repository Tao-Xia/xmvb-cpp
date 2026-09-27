#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/hamiltonian.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace {

void require_close(double left, double right, double tolerance, const char* label) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  if (std::abs(left - right) > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

void require_matrix_close(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right,
    double tolerance,
    const char* label) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::runtime_error(std::string(label) + " shape mismatch");
  }
  const double scale = std::max({1.0, left.norm(), right.norm()});
  if ((left - right).norm() > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " value mismatch");
  }
}

}  // namespace

int main() {
  try {
    constexpr int n_active = 5;
    constexpr int n_electrons = 3;
    constexpr int n_auxiliary = 7;
    const std::vector<int> occ_left{0, 2, 4};
    const std::vector<int> occ_right{1, 2, 3};

    Eigen::MatrixXd overlap_block(n_electrons, n_electrons);
    overlap_block <<
        1.10, 0.08, -0.03,
        0.04, 0.93, 0.06,
        -0.02, 0.05, 1.07;
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    const auto overlap = overlap_resolver.resolve_matrix(overlap_block);

    Eigen::MatrixXd h1e(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        h1e(row, column) =
            0.03 * (row + 1) * (column + 2) - 0.01 * (row - column);
      }
    }

    xmvb::vb::ActiveSpaceTwoElectronResult direct_ri;
    direct_ri.representation =
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    direct_ri.n_auxiliary_functions = n_auxiliary;
    direct_ri.ri_active_pair_factors.resize(
        n_auxiliary,
        xmvb::vb::packed_active_pair_count(n_active));
    for (int pair = 0; pair < direct_ri.ri_active_pair_factors.cols(); ++pair) {
      for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
        direct_ri.ri_active_pair_factors(auxiliary, pair) =
            0.017 * (auxiliary + 1) - 0.006 * (pair + 2) +
            0.001 * ((auxiliary + 2) * (pair + 3) % 5);
      }
    }

    xmvb::vb::ActiveSpaceTwoElectronResult packed_ri = direct_ri;
    packed_ri.packed_active_two_electron_integrals =
        xmvb::vb::reconstruct_packed_active_two_electron_integrals(
            xmvb::vb::make_active_space_two_electron_view(direct_ri),
            n_active);

    xmvb::vb::DeterminantHamiltonianResolver hamiltonian_resolver;
    const auto direct_hamiltonian = hamiltonian_resolver.resolve(
        occ_left,
        occ_right,
        overlap,
        h1e,
        n_active,
        direct_ri);
    const auto packed_hamiltonian = hamiltonian_resolver.resolve(
        occ_left,
        occ_right,
        overlap,
        h1e,
        n_active,
        packed_ri);
    require_close(
        direct_hamiltonian.one_electron_hamiltonian,
        packed_hamiltonian.one_electron_hamiltonian,
        2.0e-13,
        "one-electron Hamiltonian");
    require_close(
        direct_hamiltonian.total_hamiltonian,
        packed_hamiltonian.total_hamiltonian,
        2.0e-13,
        "total Hamiltonian");

    Eigen::MatrixXd direct_inverse_gradient;
    Eigen::MatrixXd packed_inverse_gradient;
    const auto direct_phi = xmvb::vb::compute_same_spin_original_phi(
        occ_left,
        occ_right,
        h1e,
        n_active,
        direct_ri,
        overlap,
        &direct_inverse_gradient);
    const auto packed_phi = xmvb::vb::compute_same_spin_original_phi(
        occ_left,
        occ_right,
        h1e,
        n_active,
        packed_ri,
        overlap,
        &packed_inverse_gradient);
    require_close(
        direct_phi.one_electron_phi,
        packed_phi.one_electron_phi,
        2.0e-13,
        "one-electron phi");
    require_close(
        direct_phi.total_phi,
        packed_phi.total_phi,
        2.0e-13,
        "total phi");
    require_matrix_close(
        direct_inverse_gradient,
        packed_inverse_gradient,
        5.0e-13,
        "inverse-overlap gradient");

    std::cout << "RI pair contractions agree with the packed reference\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
