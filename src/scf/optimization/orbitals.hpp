#pragma once

#include <array>
#include "scf/driver/solver.hpp"

namespace xmvb::scf {

/** @brief Frozen HF/KS orbital model with fixed integer occupations.
 * Coordinates contain virtual-by-occupied rotations for each spin, column major.
 * The gradient and Hessian use the physical energy convention (2*f_occ*F_vo).
 * References must outlive the model; no AO matrices are copied per direction.
 */
class OrbitalModel {
  public:
    OrbitalModel(const SpinMatrices& orbitals,
                 const std::array<int, 2>& occupied,
                 int channels,
                 double occupation,
                 const SpinMatrices& density,
                 const SpinMatrices& fock,
                 const ApplyUnrestrictedResponse& response);

    const Eigen::VectorXd& gradient() const noexcept {
        return gradient_;
    }
    Eigen::VectorXd apply(const Eigen::VectorXd& direction) const;
    Eigen::VectorXd precondition(const Eigen::VectorXd& residual, double shift) const;
    /** @brief Positive gap inverse for nonlinear descent, not the signed AH solve. */
    Eigen::VectorXd positive_precondition(const Eigen::VectorXd& residual,
                                          double shift = 0.0) const;
    SpinMatrices rotate(const Eigen::VectorXd& displacement) const;
    SpinMatrices density(const SpinMatrices& orbitals) const;
    Eigen::VectorXd pullback(const SpinMatrices& fock,
                             const Eigen::VectorXd& displacement,
                             const SpinMatrices& rotated) const;
    /** @brief Project an observed AO density change into the current tangent. */
    Eigen::VectorXd displacement(const SpinMatrices& change,
                                 const Eigen::Ref<const Eigen::MatrixXd>& overlap) const;

  private:
    Eigen::MatrixXd generator(const Eigen::VectorXd& displacement, int spin) const;
    const SpinMatrices& orbitals_;
    const SpinMatrices& density_;
    const ApplyUnrestrictedResponse& response_;
    std::array<int, 2> occupied_, sizes_{}, offsets_{};
    int channels_;
    double occupation_;
    SpinMatrices foo_, fvv_;
    Eigen::VectorXd gradient_, diagonal_;
};

} // namespace xmvb::scf
