#pragma once

#include "libcint/materialized_input.hpp"
#include "integrals/ao/libcint_input.hpp"

namespace xmvb::libcint {

struct LibcintMaterializedIntegralProviderOptions {
  /** @brief Absolute discard threshold in unit-normalized AO integrals. */
  double integral_tolerance = 1.0e-10;
  bool use_shell_pair_prescreen = true;
  /** Allocation contract only; screening and integral values are unchanged. */
  integrals::AoPairLayout ao_pair_layout=integrals::AoPairLayout::Automatic;
};

/**
 * @brief Builds normalized-AO four-center integrals with Schwarz screening.
 *
 * Returns canonical unique AO-pair indices and values without retaining a
 * second representation of the integral tensor.
 */
class LibcintMaterializedIntegralProvider {
public:
  MaterializedAoIntegralBuffers build(
      const xmvb::integrals::LibcintInput& input,
      const LibcintMaterializedIntegralProviderOptions& options = {}) const;
};

}  // namespace xmvb::libcint
