#pragma once

#include "integrals/ao/libcint_input.hpp"
#include "integrals/ao/ri.hpp"

namespace xmvb::integrals {

/**
 * @brief Backend boundary for lazily constructing AO-side RI factors.
 *
 * The VBSCF numerical layer depends only on this interface.  Runtime-specific
 * libcint setup is injected by the input loader.
 */
class RiAoFactorizationProvider {
  public:
    virtual ~RiAoFactorizationProvider() = default;

    virtual RiAoFactorization build(const LibcintInput& primary_input,
                                    const LibcintInput& auxiliary_input) const = 0;
};

} // namespace xmvb::integrals
