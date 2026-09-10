#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/vbscf_input_loader.hpp"
#include "runtime/molden_file_writer.hpp"

namespace fs = std::filesystem;

namespace {

void print_usage() {
  std::cerr
      << "usage: dump_loaded_molden <input.xmi> <output_stem.xmi>"
      << " [--skip-orbital-guess true|false]\n";
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  }
  if (value == "false" || value == "0" || value == "no") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || ((argc - 3) % 2 != 0)) {
      print_usage();
      return 1;
    }

    const std::string input_path = argv[1];
    const fs::path output_stem_path = argv[2];
    xmvb::vb::VbScfInputLoadOptions load_options;
    for (int argument_index = 3; argument_index < argc; argument_index += 2) {
      const std::string argument_name = argv[argument_index];
      const std::string argument_value = argv[argument_index + 1];
      if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else {
        throw std::invalid_argument("unknown argument: " + argument_name);
      }
    }

    const auto load_result =
        xmvb::vb::load_vbscf_input_with_timings(input_path, load_options);
    const fs::path molden_path =
        xmvb::vb::write_molden_file(output_stem_path, load_result.input);
    std::cout << molden_path.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dump_loaded_molden: " << error.what() << '\n';
    return 1;
  }
}
