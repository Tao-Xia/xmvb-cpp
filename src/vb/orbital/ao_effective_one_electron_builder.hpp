#pragma once

#include <Eigen/Core>

#include <vector>

#include "runtime/libcint_ri_integral_provider.hpp"
#include "vb/orbital/ao_integral_input.hpp"
#include "vb/orbital/ao_effective_one_electron_result.hpp"
#include "vb/orbital/orbital_preparation_result.hpp"

namespace xmvb::vb {

/**
 * @brief Builds AO effective one-electron matrices from the inactive density.
 *
 * This class is the pure C++ replacement for the `Cal_G11` and `Cal_F11`
 * stages in the legacy VBSCF implementation for the standard non-RI path.
 */
class AoEffectiveOneElectronBuilder {
public:
  /**
   * @brief Evaluates `G11` and `F11 = HHF + G11`.
   *
   * @param inactive_density_matrix Column-major inactive density matrix `P11`.
   * @param ao_core_hamiltonian_matrix Column-major AO core Hamiltonian `HHF`.
   * @param ao_two_electron_integral_values Sparse AO two-electron values `ggf`.
   * @param ao_two_electron_integral_indices Flattened AO index table `g2eidx`.
   * @param n_basis_functions Number of AO basis functions.
   * @return AoEffectiveOneElectronResult AO `G11` and `F11`.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      int n_basis_functions) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const std::vector<double>& ao_two_electron_integral_values,
      const std::vector<int>& ao_two_electron_integral_indices,
      int n_basis_functions) const;

  /**
   * @brief Evaluates `G11` and `F11` from AO-side metric-whitened RI factors.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
      int n_basis_functions) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
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
      const LibcintRiIntegralProviderResult& ri_integral_provider_result,
      int n_basis_functions,
      int n_inactive_doubly_occupied_orbitals) const;

  /**
   * @brief Evaluates `G11` and `F11` from validated AO integral input caches with an explicit AO core matrix.
   *
   * This overload is intended for probe paths that need the exact AO two-
   * electron caches carried by `AoIntegralInput`, but want to substitute a
   * custom AO core matrix such as the zero-core directional build used by the
   * analytic exact-context HVP.
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
   * @brief Evaluates `G11` and `F11` from validated AO integral input caches.
   *
   * This overload is intended for the normal C++ runtime path, where the AO
   * integral index table has already been validated during input loading.
   */
  AoEffectiveOneElectronResult build(
      const std::vector<double>& inactive_density_matrix,
      const AoIntegralInput& ao_integral_input) const;

  AoEffectiveOneElectronResult build(
      const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
      const AoIntegralInput& ao_integral_input) const;

};

}  // namespace xmvb::vb
