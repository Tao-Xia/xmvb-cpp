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
      api_defaults.backend == xmvb::vb::VbScfOptimizerBackend::Plbfgs,
      "optimizer API must default to P-LBFGS");
  const auto defaults = parse({"xmvb-cpp.exe", "unused.xmi"});
  require(defaults.has_value(), "default CLI options must parse");
  require(!defaults->ci_workspace_bytes, "CPU workspace must default to automatic resources");
  const auto workspace = parse({"xmvb-cpp.exe", "unused.xmi", "--ci-workspace-mib", "8192"});
  require(workspace && workspace->ci_workspace_bytes == 8192ULL*1024*1024,
      "explicit workspace upper bound was lost");
  require(!parse({"xmvb-cpp.exe", "unused.xmi", "--ci-workspace-mib", "0"}),
      "zero workspace upper bound must be rejected");
  require(api_defaults.structure_eigensolver == xmvb::vb::StructureEigensolver::Davidson &&
          defaults->optimizer.structure_eigensolver == xmvb::vb::StructureEigensolver::Davidson,
      "API and CLI must default to matrix-free Davidson");
  require(
      defaults->optimizer.backend ==
          xmvb::vb::VbScfOptimizerBackend::Plbfgs,
      "CLI must default to P-LBFGS");

  const auto plbfgs = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "plbfgs"});
  require(plbfgs.has_value(), "P-LBFGS option must parse");
  require(
      plbfgs->optimizer.backend ==
          xmvb::vb::VbScfOptimizerBackend::Plbfgs,
      "P-LBFGS option selected the wrong backend");

  const auto ciah = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "ciah"});
  require(ciah.has_value(), "CIAH option must parse");
  require(
      ciah->optimizer.backend == xmvb::vb::VbScfOptimizerBackend::Ciah,
      "CIAH option selected the wrong backend");
  require(
      std::string(xmvb::vb::vbscf_optimizer_backend_name(
          ciah->optimizer.backend)) == "ciah",
      "CIAH backend must have the canonical name ciah");

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

  const auto ciah_trace = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--ciah-trace",
      "ciah.tsv"});
  require(ciah_trace.has_value(), "CIAH trace option must parse");
  require(
      ciah_trace->ciah_trace_path == "ciah.tsv",
      "CIAH trace option selected the wrong path");
  require(
      ciah_trace->tnhvp_trace_path.empty(),
      "CIAH trace option must not set the TNHVP trace path");

  const auto invalid = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--optimizer-backend",
      "unknown"});
  require(!invalid.has_value(), "invalid optimizer backend must be rejected");

  require(!parse({"xmvb-cpp.exe", "unused.xmi", "--optimizer-backend", "lbfgs"}),
      "removed native L-BFGS backend must not remain as an alias");

  require(!parse({"xmvb-cpp.exe", "unused.xmi", "--optimizer-backend", "neo"}),
      "retired neo CLI alias must not remain");
  require(!parse({"xmvb-cpp.exe", "unused.xmi", "--optimizer-backend", "block_lbfgs"}),
      "retired block_lbfgs CLI alias must not remain");
  require(!parse({"xmvb-cpp.exe", "unused.xmi", "--neo-trace", "trace.tsv"}),
      "retired neo trace option must not remain");

  const auto invalid_preconditioner = parse({
      "xmvb-cpp.exe",
      "unused.xmi",
      "--orbital-preconditioner",
      "unknown"});
  require(!invalid_preconditioner.has_value(),
          "invalid orbital preconditioner must be rejected");
}
