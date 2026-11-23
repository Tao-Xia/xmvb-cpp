#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>

#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"

namespace {
void check_virtual_complement(double separation) {
    constexpr Eigen::Index size = 8, occupied = 3;
    Eigen::MatrixXd seed(size, size);
    for (Eigen::Index j = 0; j < size; ++j)
        for (Eigen::Index i = 0; i < size; ++i)
            seed(i, j) = std::sin(0.31 * (i + 1) * (j + 2)) + (i == j ? 2.0 : 0.0);
    Eigen::HouseholderQR<Eigen::MatrixXd> rotation(seed);
    const Eigen::MatrixXd q = rotation.householderQ() * Eigen::MatrixXd::Identity(size, size);
    const Eigen::MatrixXd overlap =
        q * Eigen::VectorXd::LinSpaced(size, 0.3, 2.0).asDiagonal() * q.transpose();
    Eigen::MatrixXd columns(size, occupied);
    columns.col(0) = q.col(0) + 0.2 * q.col(2);
    columns.col(1) = columns.col(0) + separation * q.col(1);
    columns.col(2) = q.col(3) + 0.3 * q.col(0);
    xmvb::vb::OrbitalPreparationInput input;
    input.n_basis_functions = size;
    input.n_orbitals = occupied;
    input.n_active_orbitals = 1;
    input.n_active_electrons = 1;
    input.n_total_electrons = 5;
    input.orbital_type = xmvb::vb::kOrbitalTypeHao;
    input.ao_overlap_matrix = overlap;
    input.orbital_basis_counts.assign(occupied, size);
    input.orbital_basis_index_table.resize(size * occupied);
    input.orbital_value_table.resize(size * occupied);
    for (Eigen::Index j = 0; j < occupied; ++j)
        for (Eigen::Index i = 0; i < size; ++i) {
            input.orbital_basis_index_table[j * size + i] = i + 1;
            input.orbital_value_table[j * size + i] = columns(i, j);
        }
    const auto result = xmvb::vb::ActiveSpaceOrbitalPreparer{}.prepare(input);
    const Eigen::MatrixXd virtuals = result.auxiliary_orbital_matrix.rightCols(size - occupied);
    // Independent thin QR in whitened AO coordinates verifies the occupied
    // span, without reusing its inverse Gram or the production projector.
    Eigen::LLT<Eigen::MatrixXd> ao(overlap);
    const Eigen::MatrixXd whitened = ao.matrixU() * columns;
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(whitened);
    const Eigen::MatrixXd span = qr.householderQ() * Eigen::MatrixXd::Identity(size, occupied);
    const double cross = (span.transpose() * ao.matrixU() * virtuals).cwiseAbs().maxCoeff();
    const double norm = (virtuals.transpose() * overlap * virtuals -
                         Eigen::MatrixXd::Identity(size - occupied, size - occupied))
                            .cwiseAbs()
                            .maxCoeff();
    std::cout << "separation=" << separation << ",cross=" << cross << ",norm=" << norm << '\n';
    if (cross > 2e-10 || norm > 2e-10)
        throw std::runtime_error("virtual complement lost its occupied-span orthogonality");
    const Eigen::MatrixXd reference_core = ao.matrixU().solve(Eigen::MatrixXd(span.leftCols(2)));
    const double density_error =
        (result.inactive_density_matrix - reference_core * reference_core.transpose()).norm();
    const Eigen::Map<const Eigen::MatrixXd> inverse(
        result.auxiliary_orbital_inverse_matrix.data(), size, size);
    const double inverse_error =
        (inverse * result.auxiliary_orbital_matrix - Eigen::MatrixXd::Identity(size, size))
            .cwiseAbs()
            .maxCoeff();
    if (density_error > 2e-9 || inverse_error > 2e-9)
        throw std::runtime_error("inactive factors or auxiliary duals lost their span identities");
}
} // namespace

int main() {
    try {
        check_virtual_complement(0.4);
        check_virtual_complement(2e-4);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
