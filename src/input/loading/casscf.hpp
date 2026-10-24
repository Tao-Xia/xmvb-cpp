#pragma once

#include <string>
#include <Eigen/Core>

#include "integrals/ao/input.hpp"
#include "integrals/ao/libcint_input.hpp"

namespace xmvb::vb {
struct InputDeck;
}

namespace xmvb::input {

/** Complete exact-CASSCF input prepared from one parsed .xmi deck. */
struct CasscfInput {
    integrals::AoIntegralInput ao;
    Eigen::MatrixXd orbitals;
    Eigen::MatrixXd overlap;
    integrals::LibcintInput libcint_input;
    int ncore = 0;
    int nactive = 0;
    int n_electrons = 0; ///< Total molecular electrons, including inactive pairs.
    int twice_spin = 0;
    int max_iterations = 2000;
    double nuclear_energy = 0.0;
    std::string basis;
    bool request_molden_output = false;
};

CasscfInput load_casscf_input(const vb::InputDeck& deck, const std::string& input_path);

} // namespace xmvb::input
