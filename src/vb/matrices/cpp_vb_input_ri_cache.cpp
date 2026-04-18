#include "vb/matrices/cpp_vb_input_ri_cache.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace xmvb::vb {

namespace {

bool try_parse_env_bool(const char* value, bool* parsed_value) {
  if (value == nullptr || value[0] == '\0' || parsed_value == nullptr) {
    return false;
  }
  const std::string text(value);
  if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
    *parsed_value = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF") {
    *parsed_value = false;
    return true;
  }
  throw std::invalid_argument("XMVB_CPP_RI_AUX_STAR must be a boolean");
}

LibcintRiIntegralProviderOptions apply_environment_overrides(
    const LibcintRiIntegralProviderOptions& options) {
  LibcintRiIntegralProviderOptions resolved_options = options;

  if (const char* level_text = std::getenv("XMVB_CPP_RI_AUX_LEVEL")) {
    resolved_options.auxiliary_basis_options.level = std::stoi(level_text);
  }

  if (const char* star_text = std::getenv("XMVB_CPP_RI_AUX_STAR")) {
    bool use_star = resolved_options.auxiliary_basis_options.use_star;
    if (!try_parse_env_bool(star_text, &use_star)) {
      throw std::invalid_argument("XMVB_CPP_RI_AUX_STAR must not be empty");
    }
    resolved_options.auxiliary_basis_options.use_star = use_star;
  }

  if (const char* cutoff_text = std::getenv("XMVB_CPP_RI_METRIC_CUTOFF")) {
    resolved_options.metric_eigenvalue_cutoff = std::stod(cutoff_text);
  }

  return resolved_options;
}

}  // namespace

const LibcintRiIntegralProviderResult& ensure_cpp_vb_input_ri_cache(
    const CppVbInput& input,
    const LibcintRiIntegralProviderOptions& options) {
  if (input.ri_integral_provider_result == nullptr) {
    if (input.libcint_input.n_atoms <= 0 || input.libcint_input.n_shells <= 0) {
      throw std::invalid_argument("CppVbInput does not contain a valid libcint input");
    }
    LibcintRiIntegralProvider provider;
    const LibcintRiIntegralProviderOptions resolved_options =
        apply_environment_overrides(options);
    input.ri_integral_provider_result =
        std::make_shared<const LibcintRiIntegralProviderResult>(
            input.auxiliary_libcint_input.n_shells > 0
                ? provider.build(
                      input.libcint_input,
                      input.auxiliary_libcint_input,
                      resolved_options)
                : provider.build(input.libcint_input, resolved_options));
  }
  return *input.ri_integral_provider_result;
}

}  // namespace xmvb::vb
