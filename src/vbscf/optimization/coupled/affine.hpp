#pragma once

#include <utility>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"

namespace xmvb::vb {

/** The four affine-chart images, also used separately by the CIAH coarse model. */
struct AffineCouplingAction {
    OrbitalCouplingAction orbital;
    StructureCoordinateCouplingAction ci;
    Eigen::VectorXd forcing;
    Eigen::VectorXd adjoint;
};

/**
 * CIAH adapter from normalized eigenvector response to the fixed affine chart.
 *
 * The general response retains A_g=-G^-1 D/2. At finite Ritz residual R,
 * the affine chart needs one additional R A_g in Bp and its exact adjoint.
 * Both corrections reuse the existing directional overlap / adjoint sweep.
 */
class AffineCoupling {
  public:
    AffineCoupling(const AcceptedPointContext& context,
                   const ExactHvpOperator& hessian,
                   const StructureTangentOperator& structure,
                   bool fixed_active_overlap = false);

    std::pair<Eigen::VectorXd, Eigen::VectorXd> fixed_overlap_orbital_images(
        const Eigen::VectorXd& orbital, bool forcing_only) const;
    Eigen::VectorXd fixed_overlap_forcing(const Eigen::VectorXd& orbital) const {
        return fixed_overlap_orbital_images(orbital, true).second;
    }
    AffineCouplingAction apply(const Eigen::VectorXd& orbital, const Eigen::VectorXd& ci) const;

  private:
    const ExactHvpOperator& hessian_;
    const StructureTangentOperator& structure_;
    Eigen::MatrixXd residual_;
    Eigen::MatrixXd metric_inverse_;
    Eigen::VectorXd scales_;
    bool fixed_active_overlap_;
};

} // namespace xmvb::vb
