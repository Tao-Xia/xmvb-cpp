#pragma once

#include <string>
#include <Eigen/Core>
#include "integrals/ao/input.hpp"
#include "integrals/ao/libcint_input.hpp"

namespace xmvb::vb {
struct InputDeck;
}

namespace xmvb::input {

/** @brief Molecular exact-AO data, with no active-space or VB structure setup. */
struct MoleculeInput {
    integrals::LibcintInput libcint_input;
    integrals::AoIntegralInput ao;
    Eigen::MatrixXd overlap;
    int n_electrons = 0;
    double nuclear_energy = 0.0;
    std::string basis;
};

MoleculeInput load_molecule(const vb::InputDeck& deck);

} // namespace xmvb::input
