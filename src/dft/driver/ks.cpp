#include "dft/driver/ks.hpp"

#include <stdexcept>
#include <utility>

#include "dft/driver/potential.hpp"
#include "dft/xc/functional.hpp"

namespace xmvb::dft {

KsResult solve_ks(int n_alpha,
                  int n_beta,
                  const integrals::LibcintInput& basis,
                  const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                  const integrals::AoIntegralInput& ao,
                  const KsOptions& options) {
    if (!options.unrestricted && n_alpha != n_beta)
        throw std::invalid_argument("RKS requires equal alpha and beta populations");
    const Functional functional(options.functional, options.unrestricted ? 2 : 1);
    const auto grid = make_grid(basis, options.grid);
    KsPotential potential(basis, grid, functional, ao);
    KsResult result;
    result.exact_exchange = functional.exact_exchange();
    result.grid_points = static_cast<std::size_t>(grid.weights.size());
    bool first = true;
    const auto retain = [&](double energy) {
        if (first) {
            result.initial_electronic_energy = energy;
            first = false;
        }
    };
    const auto& core = ao.ao_core_hamiltonian_matrix;
    if (options.unrestricted) {
        const auto evaluate = [&](const scf::SpinMatrices& density) {
            const std::array<Eigen::Ref<const Eigen::MatrixXd>, 2> channels{density[0], density[1]};
            auto physical = potential.evaluate(channels);
            retain(physical.electronic_energy);
            return physical;
        };
        const auto response = [&](const scf::SpinMatrices& density,
                                  const scf::SpinMatrices& direction) {
            const std::array<Eigen::Ref<const Eigen::MatrixXd>, 2> reference{density[0],
                                                                             density[1]};
            const std::array<Eigen::Ref<const Eigen::MatrixXd>, 2> delta{direction[0],
                                                                         direction[1]};
            return potential.response(reference, delta);
        };
        result.state = scf::solve_unrestricted(
            n_alpha, n_beta, overlap, core, evaluate, options.scf, {}, response);
    } else {
        const auto evaluate = [&](const Eigen::Ref<const Eigen::MatrixXd>& density) {
            const std::array<Eigen::Ref<const Eigen::MatrixXd>, 1> channels{density};
            auto physical = potential.evaluate(channels);
            retain(physical.electronic_energy);
            return scf::Evaluation{std::move(physical.fock[0]), physical.electronic_energy};
        };
        const auto response = [&](const Eigen::Ref<const Eigen::MatrixXd>& density,
                                  const Eigen::Ref<const Eigen::MatrixXd>& direction) {
            const std::array<Eigen::Ref<const Eigen::MatrixXd>, 1> reference{density},
                delta{direction};
            auto image = potential.response(reference, delta);
            return std::move(image[0]);
        };
        result.state = scf::solve_restricted(
            n_alpha + n_beta, overlap, core, evaluate, options.scf, {}, response);
    }
    result.xc_energy = potential.xc_energy();
    result.integrated_electrons = potential.electron_count();
    return result;
}

} // namespace xmvb::dft
