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
  const auto defaults = parse({"xmvb-cpp.exe", "unused.xmi"});
  require(defaults.has_value(), "default CLI options must parse");
  require(
      defaults->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::ScaledIdentity,
      "standalone L-BFGS must retain the scalar default");

  const auto orbital_block = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "orbital-block"});
  require(orbital_block.has_value(), "orbital-block option must parse");
  require(
      orbital_block->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::OrbitalBlock,
      "orbital-block option must select the TNHVP ablation inverse");

  const auto scalar = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "scalar"});
  require(scalar.has_value(), "scalar option must parse");
  require(
      scalar->optimizer.lbfgs_initial_inverse ==
          xmvb::vb::LbfgsInitialInverse::ScaledIdentity,
      "scalar option must explicitly select the production default");

  const auto invalid = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--lbfgs-initial-inverse",
      "unknown"});
  require(!invalid.has_value(), "invalid initial inverse must be rejected");
}
