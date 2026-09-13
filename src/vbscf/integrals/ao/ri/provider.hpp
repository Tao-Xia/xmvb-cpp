#pragma once

#include "vbscf/integrals/ao/libcint/input.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"

namespace xmvb::vb {

/**
 * @brief Backend boundary for lazily constructing AO-side RI factors.
 *
 * The VBSCF numerical layer depends only on this interface.  Runtime-specific
 * libcint setup is injected by the input loader.
 */
class RiAoFactorizationProvider {
public:
  virtual ~RiAoFactorizationProvider() = default;

  virtual RiAoFactorization build(
      const LibcintInput& primary_input,
      const LibcintInput* explicit_auxiliary_input) const = 0;
};

}  // namespace xmvb::vb
