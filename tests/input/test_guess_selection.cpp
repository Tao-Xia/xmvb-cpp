#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "input/deck/keywords.hpp"
#include "input/deck/model.hpp"

namespace {

void write_deck(
    const std::filesystem::path& path,
    const std::string& control_guess) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to create input-deck test fixture");
  }
  output << "$ctrl\n"
         << "vbscf basis=cc-pvdz " << control_guess << "\n"
         << "$end\n"
         << "$gus\n"
         << "1\n"
         << "1.0 1\n"
         << "$end\n";
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "xmvb-cpp-guess-selection-test";
  std::filesystem::create_directories(root);

  try {
    const auto implicit_path = root / "implicit.xmi";
    write_deck(implicit_path, "");
    const auto implicit = xmvb::vb::parse_input_deck_model(
        implicit_path.string());
    require(
        implicit.metadata.guess_type == xmvb::vb::kGuessTypeRead,
        "an implicit $GUS block must select GUESS=READ");
    require(
        !implicit.metadata.has_explicit_guess_type,
        "an implicit $GUS block must not be marked as explicit GUESS=");

    const auto explicit_auto_path = root / "explicit-auto.xmi";
    write_deck(explicit_auto_path, "guess=auto");
    const auto explicit_auto = xmvb::vb::parse_input_deck_model(
        explicit_auto_path.string());
    require(
        explicit_auto.metadata.guess_type == xmvb::vb::kGuessTypeAuto,
        "explicit GUESS=AUTO must override the $GUS default");
    require(
        explicit_auto.metadata.has_explicit_guess_type,
        "explicit GUESS=AUTO must be recorded as explicit");

    const auto explicit_unit_path = root / "explicit-unit.xmi";
    write_deck(explicit_unit_path, "guess=unit");
    const auto explicit_unit = xmvb::vb::parse_input_deck_model(
        explicit_unit_path.string());
    require(
        explicit_unit.metadata.guess_type == xmvb::vb::kGuessTypeUnit,
        "explicit GUESS=UNIT must override the $GUS default");
  } catch (...) {
    std::filesystem::remove_all(root);
    throw;
  }

  std::filesystem::remove_all(root);
  std::cout << "input guess selection checks passed\n";
  return 0;
}
