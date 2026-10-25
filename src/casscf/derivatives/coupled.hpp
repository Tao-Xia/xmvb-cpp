#pragma once

#include <Eigen/Core>
#include "casscf/derivatives/point.hpp"
#include "casscf/orbitals/rotation.hpp"
#include "mcscf/optimization/ciah/driver.hpp"

namespace xmvb::casscf {

/** @brief Full-coordinate Hessian images from orbital and CI directions. */
struct CoupledImages {
    Eigen::VectorXd orbital;
    Eigen::VectorXd ci;
};

/** @brief Joint orbital/CI actions in one immutable origin chart.
 * CI directions use dc=q/sqrt(2), are target-spin pure and orthogonal to c.
 * The redundant coefficient directions are projected without a dense basis.
 */
class CoupledModel {
  public:
    CoupledModel(const FixedCiPoint& point,
                 const integrals::AoIntegralInput& ao,
                 int ncore,
                 int nactive);
    Eigen::Index orbital_size() const noexcept {
        return chart_.size();
    }
    Eigen::Index size() const noexcept {
        return chart_.size() + ci_size_;
    }
    const Eigen::VectorXd& gradient() const noexcept {
        return gradient_;
    }
    CoupledImages apply_split(const Eigen::VectorXd& direction) const;
    Eigen::VectorXd apply(const Eigen::VectorXd& direction) const;
    /** Project a recycled initial guess into this point's pure-spin CI tangent.
   * Orbital indices retain their labels across macros; no old H image is reused.
   */
    Eigen::VectorXd project_direction(const Eigen::VectorXd& direction) const;
    Eigen::VectorXd precondition(const Eigen::VectorXd& residual, double shift) const;
    Eigen::MatrixXd orbitals(const Eigen::VectorXd& displacement) const;
    Eigen::MatrixXd coefficients(const Eigen::VectorXd& displacement) const;
    /** True finite-state gradient pulled back into this same origin chart. */
    mcscf::CiahKeyframe keyframe(const FixedCiPoint& trial,
                                 const Eigen::VectorXd& displacement) const;
    double maximum_component(const Eigen::VectorXd& direction) const;

  private:
    const FixedCiPoint& point_;
    RotationChart chart_;
    Eigen::Index ci_size_;
    Eigen::VectorXd gradient_, diagonal_;
    Eigen::MatrixXd raw_gradient_;
    Eigen::MatrixXd ci_direction(const Eigen::VectorXd& vector) const;
    Eigen::VectorXd pack(const Eigen::VectorXd& orbital, const Eigen::MatrixXd& ci) const;
};

} // namespace xmvb::casscf
