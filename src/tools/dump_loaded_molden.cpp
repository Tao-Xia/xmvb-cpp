#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/molden_file_writer.hpp"

namespace fs = std::filesystem;

namespace {

void print_usage() {
  std::cerr
      << "usage: dump_loaded_molden <input.xmi> <output_stem.xmi>"
      << " [--orbital-guess-source cpp|legacy]"
      << " [--skip-orbital-guess true|false]"
      << " [--ao-integral-source auto|legacy|libcint_cpp|runtime_hcore]\n";
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

void apply_orbital_guess_source_argument(
    const std::string& source_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (source_name == "cpp") {
    options->orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    return;
  }
  throw std::invalid_argument("invalid orbital guess source: " + source_name);
}

void apply_ao_integral_source_argument(
    const std::string& source_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (source_name == "auto") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
    return;
  }
  if (source_name == "libcint_cpp") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    return;
  }
  if (source_name == "runtime_hcore") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
    return;
  }
  throw std::invalid_argument("invalid AO integral source: " + source_name);
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
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source =
        xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;

    for (int argument_index = 3; argument_index < argc; argument_index += 2) {
      const std::string argument_name = argv[argument_index];
      const std::string argument_value = argv[argument_index + 1];
      if (argument_name == "--orbital-guess-source") {
        apply_orbital_guess_source_argument(argument_value, &load_options);
      } else if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else if (argument_name == "--ao-integral-source") {
        apply_ao_integral_source_argument(argument_value, &load_options);
      } else {
        throw std::invalid_argument("unknown argument: " + argument_name);
      }
    }

    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(input_path, load_options);
    const fs::path molden_path =
        xmvb::vb::write_molden_file(output_stem_path, load_result.input);
    std::cout << molden_path.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dump_loaded_molden: " << error.what() << '\n';
    return 1;
  }
}
