#pragma once

#include "runtime/materialized_ao_integral_input_builder.hpp"
#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

struct LibcintMaterializedIntegralProviderOptions {
  double integral_tolerance = 1.0e-10;
  bool use_legacy_shell_pair_prescreen = true;
};

/**
 * @brief Correctness-first C++ materialized AO integral provider backed by libcint.
 *
 * This provider is intended as the first migration step away from legacy
 * `runtime_c`-owned integral materialization. It rebuilds the legacy-style
 * `HHF / ggf / g2eidx` buffers directly from `LibcintInput` so numerical
 * kernels can be compared without changing their downstream interfaces.
 */
class LibcintMaterializedIntegralProvider {
public:
  MaterializedAoIntegralBuffers build(
      const LibcintInput& input,
      const LibcintMaterializedIntegralProviderOptions& options = {}) const;
};

}  // namespace xmvb::vb
