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

xmvb::vb::StructureExpansion singlet_pair() {
    xmvb::vb::StructureExpansion data;
    data.n_structures = 1;
    data.alpha_strings = {{0}, {1}};
    data.beta_strings = {{1}, {0}};
    data.pair_alpha = {0, 1};
    data.pair_beta = {0, 1};
    data.offsets = {0, 1, 2};
    data.terms = {{0, 1.0}, {0, 1.0}};
    return data;
}

xmvb::vb::StructureExpansion doublet_pair() {
    xmvb::vb::StructureExpansion data;
    data.n_structures = 1;
    data.alpha_strings = {{0, 2}, {1, 2}};
    data.beta_strings = {{1}, {0}};
    data.pair_alpha = {0, 1};
    data.pair_beta = {0, 1};
    data.offsets = {0, 1, 2};
    data.terms = {{0, 1.0}, {0, 1.0}};
    return data;
}

} // namespace

int main() {
    try {
        const auto singlet = xmvb::vb::analyze_structure_spin(singlet_pair(), 1);
        require(singlet.spin_adapted(), "singlet pair is not spin adapted");
        require(std::abs(singlet.spin_squared) < 1.0e-15, "singlet spin square is wrong");

        const auto doublet = xmvb::vb::analyze_structure_spin(doublet_pair(), 2);
        require(doublet.spin_adapted(), "doublet pair is not spin adapted");
        require(std::abs(doublet.spin_squared - 0.75) < 1.0e-15, "doublet spin square is wrong");

        auto contaminated = singlet_pair();
        contaminated.pair_alpha.resize(1);
        contaminated.pair_beta.resize(1);
        contaminated.offsets.resize(2);
        contaminated.terms.resize(1);
        const auto impure = xmvb::vb::analyze_structure_spin(contaminated, 1);
        require(!impure.spin_adapted(), "spin-contaminated determinant passed the raising test");

        std::cout << "structure spin analysis: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "structure spin analysis failed: " << error.what() << '\n';
        return 1;
    }
}
