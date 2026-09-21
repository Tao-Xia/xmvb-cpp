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

  const auto neo = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "neo"});
  require(neo.has_value(), "NEO option must parse");
  require(
      neo->optimizer.backend == xmvb::vb::VbScfOptimizerBackend::Neo,
      "NEO option selected the wrong backend");
  require(
      std::string(xmvb::vb::vbscf_optimizer_backend_name(
          neo->optimizer.backend)) == "neo",
      "NEO backend must have the canonical name neo");

  const auto hessian_preconditioner = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--orbital-preconditioner",
      "hessian-diagonal"});
  require(hessian_preconditioner.has_value(),
          "Hessian diagonal preconditioner option must parse");
  require(
      hessian_preconditioner->optimizer.orbital_preconditioner ==
          xmvb::vb::OrbitalPreconditioner::HessianDiagonal,
      "Hessian diagonal preconditioner option selected the wrong model");

  const auto neo_trace = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--neo-trace",
      "neo.tsv"});
  require(neo_trace.has_value(), "NEO trace option must parse");
  require(
      neo_trace->neo_trace_path == "neo.tsv",
      "NEO trace option selected the wrong path");
  require(
      neo_trace->tnhvp_trace_path.empty(),
      "NEO trace option must not set the TNHVP trace path");

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

  const auto invalid_preconditioner = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--orbital-preconditioner",
      "unknown"});
  require(!invalid_preconditioner.has_value(),
          "invalid orbital preconditioner must be rejected");
}
