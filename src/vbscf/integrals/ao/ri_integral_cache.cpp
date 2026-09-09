#include "vbscf/integrals/ao/ri_integral_cache.hpp"

#include <memory>
#include <stdexcept>

#include "vbscf/integrals/ao/ri_factorization_provider.hpp"

namespace xmvb::vb {

const RiAoFactorization& ensure_vbscf_input_ri_cache(
    const VbScfInput& input) {
  if (input.ri_factorization == nullptr) {
    if (input.libcint_input.n_atoms <= 0 || input.libcint_input.n_shells <= 0) {
      throw std::invalid_argument("VbScfInput does not contain a valid libcint input");
    }
    if (input.ri_factorization_provider == nullptr) {
      throw std::invalid_argument(
          "VbScfInput does not contain an RI factorization provider");
    }
    const LibcintInput* auxiliary_input =
        input.auxiliary_libcint_input.n_shells > 0
            ? &input.auxiliary_libcint_input
            : nullptr;
    input.ri_factorization = std::make_shared<const RiAoFactorization>(
        input.ri_factorization_provider->build(
            input.libcint_input,
            auxiliary_input));
  }
  return *input.ri_factorization;
}

}  // namespace xmvb::vb
