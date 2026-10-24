#pragma once

#include <span>

#include "dft/integration/potential.hpp"
#include "integrals/ao/input.hpp"
#include "scf/driver/solver.hpp"

namespace xmvb::dft {

/** @brief Physical KS energy/Fock and its analytical density response.
 * RKS uses one total-density channel; UKS uses alpha/beta channels. Both use
 * the same J/K contraction in evaluation and response. Borrowed basis, grid,
 * functional and AO integrals must outlive this serialized SCF workspace.
 */
class KsPotential {
  public:
    KsPotential(const integrals::LibcintInput& basis,
                const Grid& grid,
                const Functional& functional,
                const integrals::AoIntegralInput& ao);

    scf::UnrestrictedEvaluation
    evaluate(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density);
    scf::SpinMatrices response(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density,
                               std::span<const Eigen::Ref<const Eigen::MatrixXd>> direction) const;

    double xc_energy() const {
        return xc_energy_;
    }
    const std::array<double, 2>& electron_count() const {
        return electron_count_;
    }

  private:
    scf::SpinMatrices interaction(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density) const;

    const integrals::AoIntegralInput& ao_;
    int n_spin_;
    double exact_exchange_;
    XcIntegrator xc_;
    double xc_energy_ = 0.0;
    std::array<double, 2> electron_count_{};
};

} // namespace xmvb::dft
