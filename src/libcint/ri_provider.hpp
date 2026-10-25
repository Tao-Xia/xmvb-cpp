#pragma once

#include "integrals/ao/ri_provider.hpp"

namespace xmvb::libcint {

struct LibcintRiIntegralProviderOptions {
    /** Absolute eigenvalue cutoff in unit-radial Cartesian auxiliary AOs.
   * Applied only when the Coulomb metric is not Cholesky positive definite,
   * as in PySCF DF. This is not an orbital/SCF convergence threshold.
   */
    double metric_eigenvalue_cutoff = 1.0e-7;
};

/**
 * @brief Builds molecule-static AO-side RI factors on the clean C++/libcint path.
 *
 * The provider consumes an explicit Coulomb-fitting auxiliary basis,
 * materializes the auxiliary metric and three-center tensors through libcint,
 * and returns metric-whitened AO-pair factors.
 */
class LibcintRiIntegralProvider final : public xmvb::integrals::RiAoFactorizationProvider {
  public:
    xmvb::integrals::RiAoFactorization
    build(const xmvb::integrals::LibcintInput& primary_input,
          const xmvb::integrals::LibcintInput& auxiliary_input,
          const LibcintRiIntegralProviderOptions& options = {}) const;

    xmvb::integrals::RiAoFactorization
    build(const xmvb::integrals::LibcintInput& primary_input,
          const xmvb::integrals::LibcintInput& auxiliary_input) const override;
};

} // namespace xmvb::libcint
