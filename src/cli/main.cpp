#include <utility>

#include "cli/options.hpp"
#include "cli/run.hpp"

int main(int argc, char** argv) {
  auto options = xmvb::cli::parse_options(argc, argv);
  if (!options.has_value()) {
    return 1;
  }
  return xmvb::cli::run(std::move(*options));
}
