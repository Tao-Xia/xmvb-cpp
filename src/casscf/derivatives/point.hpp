#pragma once

#include <Eigen/Core>
#include "casscf/integrals/active.hpp"
#include "integrals/active/two_electron_result.hpp"
#include "integrals/response/types.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/spin.hpp"

namespace xmvb::casscf {

struct OrbitalResponse {
    mcscf::ActiveIntegrals integrals;
    Eigen::MatrixXd orbital_hessian;
};

/** One accepted real, orthonormal, complete-CAS state. The AO input is borrowed. */
class FixedCiPoint {
  public:
    FixedCiPoint(const integrals::AoIntegralInput& ao,
                 const Eigen::MatrixXd& orbitals,
                 int ncore,
                 int nactive,
                 double nuclear_energy,
                 int active_electrons,
                 int twice_spin,
                 const Eigen::MatrixXd& coefficients);
    FixedCiPoint(const FixedCiPoint&) = delete;
    FixedCiPoint& operator=(const FixedCiPoint&) = delete;
    FixedCiPoint(FixedCiPoint&&) = delete;
    FixedCiPoint& operator=(FixedCiPoint&&) = delete;

    double energy() const;
    double core_energy() const noexcept {
        return prepared_.integrals.core_energy;
    }
    double active_energy() const {
        (void)energy();
        return active_energy_;
    }
    const CasIntegrals& integrals() const noexcept {
        return prepared_.integrals;
    }
    const mcscf::DirectCiSigmaAction& sigma() const noexcept {
        return sigma_;
    }
    const Eigen::MatrixXd& coefficients() const noexcept {
        return coefficients_;
    }
    const Eigen::MatrixXd& orbitals() const noexcept {
        return orbitals_;
    }
    const Eigen::MatrixXd& core_fock_ao() const noexcept {
        return prepared_.fock;
    }
    const mcscf::SpinSpace& spin_space() const noexcept {
        return spin_space_;
    }
    Eigen::MatrixXd project_ci(const Eigen::MatrixXd& direction) const;
    const mcscf::DirectCiIntegralAdjoint& integral_adjoint() const;
    /** Updates only the CI state; accepted integrals and spin links are reused. */
    void set_coefficients(const Eigen::MatrixXd& coefficients);

    /** Active CI stationarity residual, independent of the core energy origin. */
    Eigen::MatrixXd ci_residual() const;
    /** Euclidean AO-by-MO orbital gradient at fixed CI coefficients. */
    const Eigen::MatrixXd& orbital_gradient_raw() const;
    /** Shared exact integral direction and fixed-CI Euclidean orbital HVP. */
    OrbitalResponse direction(const Eigen::MatrixXd& direction) const;
    /** Pullback of left^T H_act right; excludes the inactive/core scalar. */
    Eigen::MatrixXd transition_orbital_gradient(const Eigen::MatrixXd& left,
                                                const Eigen::MatrixXd& right) const;
    /** Pullback of an arbitrary active-integral cotangent; excludes core scalar. */
    Eigen::MatrixXd
    orbital_gradient_from_adjoint(const mcscf::DirectCiIntegralAdjoint& cotangent) const;

  private:
    const integrals::AoIntegralInput* ao_;
    Eigen::MatrixXd orbitals_;
    int ncore_, nactive_;
    int twice_spin_;
    Eigen::MatrixXd coefficients_;
    mcscf::SpinSpace spin_space_;
    PreparedCasIntegrals prepared_;
    mcscf::DirectCiSigmaAction sigma_;
    mutable mcscf::DirectCiIntegralAdjoint adjoint_;
    mutable integrals::ExactPackedActiveTwoElectronAdjointCache two_electron_cache_;
    mutable Eigen::MatrixXd active_density_response_;
    mutable Eigen::MatrixXd sigma_image_, orbital_gradient_;
    mutable double energy_ = 0.0;
    mutable double active_energy_ = 0.0;
    mutable bool energy_prepared_ = false;
    mutable bool state_prepared_ = false;
    mutable bool orbital_gradient_prepared_ = false;
    mutable bool hessian_cache_prepared_ = false;

    void prepare_state() const;
    void prepare_hessian_cache() const;
    Eigen::MatrixXd apply_ao_response(const Eigen::MatrixXd& density) const;
    Eigen::MatrixXd two_electron_pullback(const Eigen::MatrixXd& pair_adjoint) const;
};

} // namespace xmvb::casscf
