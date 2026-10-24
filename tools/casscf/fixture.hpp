#pragma once

#include <filesystem>

#include <Eigen/Core>

#include "integrals/ao/input.hpp"

namespace xmvb::casscf::fixture {

struct Input {
    integrals::AoIntegralInput ao;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd initial_orbitals;
    int ncore = 0;
    int nactive = 0;
    int active_electrons = 0;
    int twice_spin = 0;
    double nuclear_energy = 0;
};

// XMCASAO1 fixture format is described in benchmarks/casscf-20261017/README.md.
Input read(const std::filesystem::path& path);

} // namespace xmvb::casscf::fixture
