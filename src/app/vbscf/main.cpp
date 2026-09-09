#include <utility>

#include "app/vbscf/options.hpp"
#include "app/vbscf/run.hpp"

int main(int argc, char** argv) {
  xmvb::app::vbscf::Options options;
  if (!xmvb::app::vbscf::parse_options(argc, argv, &options)) {
    return 1;
  }
  return xmvb::app::vbscf::run(std::move(options));
}
