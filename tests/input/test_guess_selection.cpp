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

    const auto state_average_path = root / "state-average.xmi";
    write_deck(state_average_path, "nstate=3");
    const auto state_average = xmvb::vb::parse_input_deck_model(
        state_average_path.string());
    require(
        state_average.metadata.state_average_count == 3,
        "NSTATE must select the equal-weight state count");

    const auto historical_state_average_path =
        root / "historical-state-average.xmi";
    write_deck(historical_state_average_path, "wstate(1)=1,1");
    const auto historical_state_average = xmvb::vb::parse_input_deck_model(
        historical_state_average_path.string());
    require(
        historical_state_average.metadata.state_average_count == 2,
        "equal historical WSTATE weights must select the state count");

    const auto unequal_state_average_path =
        root / "unequal-state-average.xmi";
    write_deck(unequal_state_average_path, "wstate(1)=1,2");
    bool rejected_unequal_weights = false;
    try {
      (void)xmvb::vb::parse_input_deck_model(
          unequal_state_average_path.string());
    } catch (const std::invalid_argument&) {
      rejected_unequal_weights = true;
    }
    require(
        rejected_unequal_weights,
        "unequal WSTATE weights must be rejected by the equal-weight implementation");

    const auto shifted_state_average_path =
        root / "shifted-state-average.xmi";
    write_deck(shifted_state_average_path, "wstate(2)=1,1");
    bool rejected_shifted_state_range = false;
    try {
      (void)xmvb::vb::parse_input_deck_model(
          shifted_state_average_path.string());
    } catch (const std::invalid_argument&) {
      rejected_shifted_state_range = true;
    }
    require(
        rejected_shifted_state_range,
        "WSTATE ranges not starting at the lowest state must be rejected");

    const auto invalid_state_average_path = root / "invalid-state-average.xmi";
    write_deck(invalid_state_average_path, "nstate=0");
    bool rejected_invalid_state_count = false;
    try {
      (void)xmvb::vb::parse_input_deck_model(
          invalid_state_average_path.string());
    } catch (const std::invalid_argument&) {
      rejected_invalid_state_count = true;
    }
    require(
        rejected_invalid_state_count,
        "non-positive NSTATE must be rejected");
  } catch (...) {
    std::filesystem::remove_all(root);
    throw;
  }

  std::filesystem::remove_all(root);
  std::cout << "input guess selection checks passed\n";
  return 0;
}
