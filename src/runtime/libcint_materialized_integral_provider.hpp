#pragma once

#include "runtime/materialized_ao_integral_input_builder.hpp"
#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

struct LibcintMaterializedIntegralProviderOptions {
  double integral_tolerance = 1.0e-10;
  bool use_historical_shell_pair_prescreen = true;
};

/**
 * @brief Correctness-first C++ materialized AO integral provider backed by libcint.
 *
 * This provider rebuilds the historical `HHF / ggf / g2eidx` buffer contract
 * directly from `LibcintInput` so numerical
 * kernels can be compared without changing their downstream interfaces.
 */
class LibcintMaterializedIntegralProvider {
public:
  MaterializedAoIntegralBuffers build(
      const LibcintInput& input,
      const LibcintMaterializedIntegralProviderOptions& options = {}) const;
};

}  // namespace xmvb::vb
