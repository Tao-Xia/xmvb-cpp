#include "scf/hf/potential.hpp"

#include <stdexcept>

#include "integrals/jk/contraction.hpp"

namespace xmvb::scf {

namespace {

double electronic_energy(const Eigen::Ref<const Eigen::MatrixXd>& density,
                         const Eigen::Ref<const Eigen::MatrixXd>& core,
                         const Eigen::Ref<const Eigen::MatrixXd>& fock) {
    return 0.5 * (density.array() * (core + fock).array()).sum();
}

void check_shapes(const Eigen::Ref<const Eigen::MatrixXd>& density,
                  const Eigen::Ref<const Eigen::MatrixXd>& core,
                  int n_basis) {
    if (n_basis <= 0 || density.rows() != n_basis || density.cols() != n_basis ||
        core.rows() != n_basis || core.cols() != n_basis) {
        throw std::invalid_argument("invalid restricted HF matrix dimensions");
    }
}

} // namespace

Evaluation evaluate_hf(const Eigen::Ref<const Eigen::MatrixXd>& density,
                       const integrals::AoIntegralInput& ao) {
    check_shapes(density, ao.ao_core_hamiltonian_matrix, ao.n_basis_functions);
    Evaluation result;
    result.fock = ao.ao_core_hamiltonian_matrix +
                  integrals::contract_jk(density, ao, integrals::JkWeights{1.0, -0.5});
    result.electronic_energy =
        electronic_energy(density, ao.ao_core_hamiltonian_matrix, result.fock);
    return result;
}

Evaluation evaluate_hf(const Eigen::Ref<const Eigen::MatrixXd>& density,
                       const Eigen::Ref<const Eigen::MatrixXd>& core,
                       const integrals::RiAoFactorization& ri) {
    check_shapes(density, core, ri.n_basis_functions);
    Evaluation result;
    result.fock = core + integrals::contract_jk(density, ri, integrals::JkWeights{1.0, -0.5});
    result.electronic_energy = electronic_energy(density, core, result.fock);
    return result;
}

} // namespace xmvb::scf
