#include <iostream>
#include <stdexcept>
#include "vbscf/orbitals/orbital_preparer.hpp"
#include "vbscf/orbitals/orbital_pullback.hpp"

int main() {
  try {
    using namespace xmvb::vb;
    OrbitalPreparationInput input;
    input.n_basis_functions = 5; input.n_orbitals = 3;
    input.n_active_orbitals = 2; input.n_active_electrons = 2;
    input.n_total_electrons = 4; input.orbital_type = 3;
    input.ao_overlap_matrix = Eigen::MatrixXd::Identity(5, 5);
    input.ao_overlap_matrix.array() += 0.07;
    input.orbital_basis_counts = {5, 5, 5};
    input.orbital_value_table = {1.4, .2, -.1, .3, .1, .1, .8, .3, -.2, .1, -.2, .1, .1, 1.3, .4};
    input.orbital_basis_index_table = {1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5};
    const Eigen::MatrixXd active = Eigen::MatrixXd::Random(5, 2);
    const Eigen::MatrixXd density = Eigen::MatrixXd::Random(5, 5);
    ActiveSpaceOrbitalPreparer prepare;
    ActiveSpaceOrbitalBackpropagator backward;
    auto energy = [&](const OrbitalPreparationInput& x) {
      const auto result = prepare.prepare(x);
      return (active.array() * result.auxiliary_orbital_matrix.middleCols(1, 2).array()).sum() +
             (density.array() * result.inactive_density_matrix.array()).sum();
    };
    const auto result = prepare.prepare(input);
    const auto gradient = backward.backpropagate(active, density, input, result).orbital_value_gradient;
    constexpr double step = 1e-5;
    for (int j = 0; j < 15; ++j) {
      auto plus = input, minus = input;
      plus.orbital_value_table[j] += step; minus.orbital_value_table[j] -= step;
      const double fd = (energy(plus)-energy(minus))/(2.0*step);
      if (std::abs(fd-gradient[j]) > 1e-8)
        throw std::runtime_error("OEO raw-coefficient gradient fails energy difference");
    }
    auto generic = input;
    generic.orbital_type = 1; // Identical explicit full-AO support, not a new input deck.
    const auto reference = backward.backpropagate(active, density, generic, prepare.prepare(generic)).orbital_value_gradient;
    for (int j = 0; j < 15; ++j)
      if (std::abs(gradient[j]-reference[j]) > 1e-13)
        throw std::runtime_error("full-AO pullback depends on the orbital-type label");
    for (int p = 0; p < 3; ++p) {
      double radial = 0.0;
      for (int j = 0; j < 5; ++j) radial += input.orbital_value_table[5*p+j]*gradient[5*p+j];
      if (std::abs(radial) > 1e-12) throw std::runtime_error("normalization scale gauge not annihilated");
    }
    std::cout << "Full-AO OEO energy derivatives, identical-support type equivalence and scaling gauge: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
