#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/options.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::optional<xmvb::cli::Options> parse(
    std::initializer_list<const char*> arguments) {
  std::vector<std::string> storage;
  storage.reserve(arguments.size());
  for (const char* argument : arguments) {
    storage.emplace_back(argument);
  }
  std::vector<char*> argv;
  argv.reserve(storage.size());
  for (std::string& argument : storage) {
    argv.push_back(argument.data());
  }
  return xmvb::cli::parse_options(
      static_cast<int>(argv.size()), argv.data());
}

}  // namespace

int main() {
  const xmvb::vb::VbScfOptimizerOptions api_defaults;
  require(
      api_defaults.backend == xmvb::vb::VbScfOptimizerBackend::Lbfgs &&
          api_defaults.lbfgs_initial_inverse ==
              xmvb::vb::LbfgsInitialInverse::OrbitalBlock,
      "optimizer API must default to orbital-block L-BFGS");
  const auto defaults = parse({"xmvb-cpp.exe", "unused.xmi"});
  require(defaults.has_value(), "default CLI options must parse");
  require(
      defaults->optimizer.backend == xmvb::vb::VbScfOptimizerBackend::Lbfgs,
      "CLI must default to the L-BFGS backend");
  require(
      defaults->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::OrbitalBlock,
      "standalone L-BFGS must default to the orbital-block inverse");

  const auto orbital_block = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "orbital-block"});
  require(orbital_block.has_value(), "orbital-block option must parse");
  require(
      orbital_block->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::OrbitalBlock,
      "orbital-block option must explicitly select the default inverse");

  const auto scalar = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "scalar"});
  require(scalar.has_value(), "scalar option must parse");
  require(
      scalar->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::ScaledIdentity,
      "scalar option must explicitly select conventional L-BFGS scaling");

  const auto invalid = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "unknown"});
  require(!invalid.has_value(), "invalid initial inverse must be rejected");
}
