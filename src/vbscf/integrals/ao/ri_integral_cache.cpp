#include "vbscf/integrals/ao/ri_integral_cache.hpp"

#include <memory>
#include <stdexcept>

namespace xmvb::vb {

const LibcintRiIntegralProviderResult& ensure_cpp_vb_input_ri_cache(
    const VbScfInput& input,
    const LibcintRiIntegralProviderOptions& options) {
  if (input.ri_integral_provider_result == nullptr) {
    if (input.libcint_input.n_atoms <= 0 || input.libcint_input.n_shells <= 0) {
      throw std::invalid_argument("VbScfInput does not contain a valid libcint input");
    }
    LibcintRiIntegralProvider provider;
    input.ri_integral_provider_result =
        std::make_shared<const LibcintRiIntegralProviderResult>(
            input.auxiliary_libcint_input.n_shells > 0
                ? provider.build(
                      input.libcint_input,
                      input.auxiliary_libcint_input,
                      options)
                : provider.build(input.libcint_input, options));
  }
  return *input.ri_integral_provider_result;
}

}  // namespace xmvb::vb
