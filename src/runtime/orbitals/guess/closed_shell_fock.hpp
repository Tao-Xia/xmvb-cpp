#pragma once

#include <Eigen/Core>

#include "vbscf/integrals/ao/ao_integral_input.hpp"

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
};

}  // namespace xmvb::vb
