#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/integrals/ao/contracts/input.hpp"
#include "vbscf/integrals/ao/one_electron/result.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

/**
 * @brief Builds AO effective one-electron matrices from the inactive density.
 *
 * Exact four-center and RI inputs are distinct operator contracts.
 */
class AoEffectiveOneElectronBuilder {
public:
  /**
   * @brief Evaluates `G11` and `F11` from AO-side metric-whitened RI factors.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const RiAoFactorization& ri_factorization,
      int n_basis_functions) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const RiAoFactorization& ri_factorization,
      int n_basis_functions) const;

  /**
   * @brief Evaluates `G11` and `F11` from explicit inactive-density factors.
   *
   * The RI path can reuse the low-rank factorization of `P11` produced during
   * orbital preparation, so the AO RI operator no longer has to diagonalize the
   * full AO density matrix to rediscover the same occupied-space structure.
   */
  AoEffectiveOneElectronResult build(
      const OrbitalPreparationResult& orbital_result,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const RiAoFactorization& ri_factorization,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals) const;

  /**
   * @brief Evaluates exact `G11` with an explicit AO core matrix.
   *
   * This form supports directional calculations with a zero core matrix.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const AoIntegralInput& ao_integral_input) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const AoIntegralInput& ao_integral_input) const;

  /**
   * @brief Evaluates exact `G11` and `F11` from the canonical AO operator.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const AoIntegralInput& ao_integral_input) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const AoIntegralInput& ao_integral_input) const;

};

}  // namespace xmvb::vb
