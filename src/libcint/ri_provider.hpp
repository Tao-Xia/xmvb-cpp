#pragma once

#include "vbscf/integrals/ao/ri/provider.hpp"

namespace xmvb::vb {

struct LibcintRiIntegralProviderOptions {
  double metric_eigenvalue_cutoff = 1.0e-10;
};

/**
 * @brief Builds molecule-static AO-side RI factors on the clean C++/libcint path.
 *
 * The provider consumes an explicit Coulomb-fitting auxiliary basis,
 * materializes the auxiliary metric and three-center tensors through libcint,
 * and returns metric-whitened AO-pair factors.
 */
class LibcintRiIntegralProvider final : public RiAoFactorizationProvider {
public:
  RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintInput& auxiliary_input,
      const LibcintRiIntegralProviderOptions& options = {}) const;

  RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintInput& auxiliary_input) const override;
};

}  // namespace xmvb::vb
