#include "dft/driver/potential.hpp"

#include <stdexcept>

#include "dft/xc/functional.hpp"
#include "integrals/jk/contraction.hpp"

namespace xmvb::dft {

KsPotential::KsPotential(const integrals::LibcintInput& basis,
                         const Grid& grid,
                         const Functional& functional,
                         const integrals::AoIntegralInput& ao)
    : ao_(ao), n_spin_(functional.n_spin()), exact_exchange_(functional.exact_exchange()),
      xc_(basis, grid, functional) {}

scf::SpinMatrices
KsPotential::interaction(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density) const {
    if (density.size() != static_cast<std::size_t>(n_spin_))
        throw std::invalid_argument("KS density channel count mismatch");
    scf::SpinMatrices result;
    if (n_spin_ == 1) {
        result[0] = integrals::contract_jk(density[0], ao_, {1.0, -0.5 * exact_exchange_});
    } else {
        const Eigen::MatrixXd total = density[0] + density[1];
        const auto coulomb = integrals::contract_jk(total, ao_);
        for (int spin = 0; spin < 2; ++spin) {
            result[spin] = coulomb;
            if (exact_exchange_ != 0.0)
                result[spin] += integrals::contract_jk(density[spin], ao_, {0.0, -exact_exchange_});
        }
    }
    return result;
}

scf::UnrestrictedEvaluation
KsPotential::evaluate(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density) {
    const auto xc = xc_.evaluate(density);
    scf::UnrestrictedEvaluation result;
    result.fock = interaction(density);
    const auto& core = ao_.ao_core_hamiltonian_matrix;
    if (n_spin_ == 1) {
        result.electronic_energy = (density[0].array() * core.array()).sum() +
                                   0.5 * (density[0].array() * result.fock[0].array()).sum() +
                                   xc.energy;
    } else if (exact_exchange_ == 0.0) {
        // Preserve the shared-J total-density contraction for pure UKS.
        const Eigen::MatrixXd total = density[0] + density[1];
        result.electronic_energy = (total.array() * core.array()).sum() +
                                   0.5 * (total.array() * result.fock[0].array()).sum() + xc.energy;
    } else {
        result.electronic_energy = xc.energy;
        for (int spin = 0; spin < n_spin_; ++spin)
            result.electronic_energy +=
                (density[spin].array() * core.array()).sum() +
                0.5 * (density[spin].array() * result.fock[spin].array()).sum();
    }
    for (int spin = 0; spin < n_spin_; ++spin) {
        result.fock[spin] = core + result.fock[spin] + xc.potential[spin];
    }
    xc_energy_ = xc.energy;
    electron_count_ = xc.electron_count;
    return result;
}

scf::SpinMatrices
KsPotential::response(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density,
                      std::span<const Eigen::Ref<const Eigen::MatrixXd>> direction) const {
    auto result = xc_.response(density, direction);
    const auto mean_field = interaction(direction);
    for (int spin = 0; spin < n_spin_; ++spin)
        result[spin] += mean_field[spin];
    return result;
}

} // namespace xmvb::dft
