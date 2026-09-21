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
      api_defaults.backend == xmvb::vb::VbScfOptimizerBackend::BlockLbfgs,
      "optimizer API must default to block-LBFGS");
  const auto defaults = parse({"xmvb-cpp.exe", "unused.xmi"});
  require(defaults.has_value(), "default CLI options must parse");
  require(
      defaults->optimizer.backend ==
          xmvb::vb::VbScfOptimizerBackend::BlockLbfgs,
      "CLI must default to block-LBFGS");

  const auto block = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "block_lbfgs"});
  require(block.has_value(), "block-LBFGS option must parse");
  require(
      block->optimizer.backend ==
          xmvb::vb::VbScfOptimizerBackend::BlockLbfgs,
      "block-LBFGS option selected the wrong backend");

  const auto xmvb_lbfgs = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "lbfgs"});
  require(xmvb_lbfgs.has_value(), "XMVB L-BFGS option must parse");
  require(
      xmvb_lbfgs->optimizer.backend ==
          xmvb::vb::VbScfOptimizerBackend::Lbfgs,
      "XMVB L-BFGS option selected the wrong backend");

  const auto invalid = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "unknown"});
  require(!invalid.has_value(), "invalid optimizer backend must be rejected");
}
