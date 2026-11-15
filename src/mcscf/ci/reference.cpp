#include "mcscf/ci/reference.hpp"

#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "core/pair_index.hpp"

namespace xmvb::mcscf {

Eigen::MatrixXd mean_field_orbitals(const ActiveIntegrals& integrals, int electrons) {
    const auto n = integrals.one_electron.rows();
    if (n <= 0 || integrals.one_electron.cols() != n ||
        integrals.pair_kernel.rows() != n * (n + 1) / 2 ||
        integrals.pair_kernel.cols() != integrals.pair_kernel.rows())
        throw std::invalid_argument("mean-field CI reference requires complete spin strings");
    using core::PairIndex;
    const double occupation = static_cast<double>(electrons) / n;
    Eigen::MatrixXd fock = integrals.one_electron;
    for (Eigen::Index p = 0; p < n; ++p)
        for (Eigen::Index q = 0; q <= p; ++q) {
            double value = 0;
            for (Eigen::Index r = 0; r < n; ++r)
                value += integrals.pair_kernel(PairIndex::packed_pair_index(p, q),
                                               PairIndex::packed_pair_index(r, r)) -
                         0.5 * integrals.pair_kernel(PairIndex::packed_pair_index(p, r),
                                                     PairIndex::packed_pair_index(q, r));
            fock(p, q) += occupation * value;
            fock(q, p) = fock(p, q);
        }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(fock);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("active mean-field CI reference diagonalization failed");
    return solver.eigenvectors();
}

} // namespace xmvb::mcscf
