#pragma once

#include <array>
#include <string>
#include <variant>

#include "dft/grid/atomic.hpp"
#include "integrals/ao/input.hpp"
#include "scf/driver/solver.hpp"

namespace xmvb::dft {

struct KsOptions {
    GridOptions grid;
    std::string functional = "LDA_X,LDA_C_VWN";
    scf::Options scf;
    bool unrestricted = false;
};

struct KsResult {
    std::variant<scf::Result, scf::UnrestrictedResult> state;
    std::size_t grid_points = 0;
    double initial_electronic_energy = 0.0;
    /** @brief Semilocal XC energy; exact exchange is included in state energy. */
    double xc_energy = 0.0;
    /** @brief Global Hartree-Fock exchange fraction. */
    double exact_exchange = 0.0;
    std::array<double, 2> integrated_electrons{};
};

/** @brief LDA/GGA/global-hybrid KS model using the common RKS/UKS SCF driver.
 * Spin populations fix M_S, not a pure total-spin eigenstate. The default guess
 * diagonalizes the core Hamiltonian, with the requested occupations per spin.
 */
KsResult solve_ks(int n_alpha,
                  int n_beta,
                  const integrals::LibcintInput& basis,
                  const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                  const integrals::AoIntegralInput& ao,
                  const KsOptions& options = {});

} // namespace xmvb::dft
