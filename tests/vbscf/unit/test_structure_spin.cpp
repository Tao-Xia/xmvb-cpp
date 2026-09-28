#include <cmath>
#include <iostream>
#include <stdexcept>

#include "vbscf/structures/analysis/spin.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

xmvb::vb::FullDeterminantStructureData singlet_pair() {
  xmvb::vb::FullDeterminantStructureData data;
  data.n_structures = 1;
  data.alpha_det = {{0}, {1}};
  data.beta_det = {{1}, {0}};
  data.determinant_to_structure_terms = {
      {{0, 1.0}},
      {{0, 1.0}},
  };
  return data;
}

xmvb::vb::FullDeterminantStructureData doublet_pair() {
  xmvb::vb::FullDeterminantStructureData data;
  data.n_structures = 1;
  data.alpha_det = {{0, 2}, {1, 2}};
  data.beta_det = {{1}, {0}};
  data.determinant_to_structure_terms = {
      {{0, 1.0}},
      {{0, 1.0}},
  };
  return data;
}

}  // namespace

int main() {
  try {
    const auto singlet = xmvb::vb::analyze_structure_spin(singlet_pair(), 1);
    require(singlet.spin_adapted(), "singlet pair is not spin adapted");
    require(std::abs(singlet.spin_squared) < 1.0e-15,
            "singlet spin square is wrong");

    const auto doublet = xmvb::vb::analyze_structure_spin(doublet_pair(), 2);
    require(doublet.spin_adapted(), "doublet pair is not spin adapted");
    require(std::abs(doublet.spin_squared - 0.75) < 1.0e-15,
            "doublet spin square is wrong");

    auto contaminated = singlet_pair();
    contaminated.alpha_det.resize(1);
    contaminated.beta_det.resize(1);
    contaminated.determinant_to_structure_terms.resize(1);
    const auto impure = xmvb::vb::analyze_structure_spin(contaminated, 1);
    require(!impure.spin_adapted(),
            "spin-contaminated determinant passed the raising test");

    std::cout << "structure spin analysis: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "structure spin analysis failed: " << error.what() << '\n';
    return 1;
  }
}
