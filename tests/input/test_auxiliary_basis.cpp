#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "input/deck/model.hpp"
#include "input/deck/primary_basis.hpp"
#include "libcint/c_api.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("expected one input-deck path");
    }
    const xmvb::vb::InputDeck deck =
        xmvb::vb::parse_input_deck_model(argv[1]);
    const auto auxiliary = xmvb::vb::build_input_deck_basis(
        deck,
        "def2-universal-jkfit").libcint_input;

    int max_angular_momentum = -1;
    for (int shell = 0; shell < auxiliary.n_shells; ++shell) {
      max_angular_momentum = std::max(
          max_angular_momentum,
          auxiliary.bas[shell * BAS_SLOTS + ANG_OF]);
    }
    if (max_angular_momentum != 6) {
      throw std::runtime_error(
          "def2-universal-jkfit did not preserve its i-type shells");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
