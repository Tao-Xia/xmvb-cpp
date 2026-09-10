#pragma once

#include "libcint/auxiliary_basis.hpp"
#include "vbscf/integrals/ao/ri/provider.hpp"

namespace xmvb::vb {

struct LibcintRiIntegralProviderOptions {
  LibcintAuxiliaryBasisBuilderOptions auxiliary_basis_options;
  double metric_eigenvalue_cutoff = 1.0e-10;
};

/**
 * @brief Builds molecule-static AO-side RI factors on the clean C++/libcint path.
 *
 * The provider generates a Coulomb-fitting auxiliary basis, materializes the
 * auxiliary metric and three-center tensors directly through libcint, and
 * returns metric-whitened AO-pair factors suitable for active-space RI
 * contraction.
 */
class LibcintRiIntegralProvider final : public RiAoFactorizationProvider {
public:
  RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintRiIntegralProviderOptions& options = {}) const;

  RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintInput& auxiliary_input,
      const LibcintRiIntegralProviderOptions& options) const;

  RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintInput* explicit_auxiliary_input) const override;
};

}  // namespace xmvb::vb
