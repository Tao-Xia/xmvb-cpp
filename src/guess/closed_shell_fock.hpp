#pragma once

#include <Eigen/Core>

#include "vbscf/integrals/ao/contracts/input.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"

namespace xmvb::vb {

/**
 * @brief Builds a closed-shell AO Fock matrix `F = H + 2J - K`.
 *
 * The input density is the spinless occupied-orbital projector used by the
 * spin-free convention, i.e. `D = C_occ C_occ^T` without an extra factor of 2.
 */
class ClosedShellFockBuilder {
public:
  Eigen::MatrixXd build(
      const Eigen::Ref<const Eigen::MatrixXd>& density_projector,
      const AoIntegralInput& ao_integral_input) const;

  Eigen::MatrixXd build(
      const Eigen::Ref<const Eigen::MatrixXd>& density_projector,
      const Eigen::Ref<const Eigen::MatrixXd>& core_hamiltonian_matrix,
      const RiAoFactorization& ri_factorization) const;
};

}  // namespace xmvb::vb
