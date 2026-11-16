#pragma once

#include <Eigen/Core>

#include "mcscf/ci/integrals.hpp"

namespace xmvb::mcscf {

/** Active ensemble-Fock eigenvectors, without changing physical orbitals. */
Eigen::MatrixXd mean_field_orbitals(const ActiveIntegrals& integrals, int electrons);

} // namespace xmvb::mcscf
