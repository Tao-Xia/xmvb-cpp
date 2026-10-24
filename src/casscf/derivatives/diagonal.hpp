#pragma once

#include <Eigen/Core>

namespace xmvb::integrals {
struct AoIntegralInput;
}
namespace xmvb::casscf {
class FixedCiPoint;
class RotationChart;

/** Analytic fixed-CI orbital Hessian diagonal in nonredundant chart order. */
Eigen::VectorXd orbital_hessian_diagonal(const FixedCiPoint& point,
                                         const RotationChart& chart,
                                         const integrals::AoIntegralInput& ao,
                                         int ncore,
                                         int nactive);

} // namespace xmvb::casscf
