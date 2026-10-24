#pragma once

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"

namespace xmvb::vb {

/**
 * @brief Derivatives of the prepared active overlap in packed raw coefficients.
 *
 * The referenced input, preparation result, and parameter layout must outlive
 * this object. Geometry is cached once; upstream weights are formed per call.
 */
class OrbitalOverlapPullback {
  public:
    OrbitalOverlapPullback(const OrbitalPreparationInput& input,
                           const OrbitalPreparationResult& prepared,
                           const SparseParameterLayout& layout);
    OrbitalOverlapPullback(const OrbitalOverlapPullback&) = delete;
    OrbitalOverlapPullback& operator=(const OrbitalOverlapPullback&) = delete;
    OrbitalOverlapPullback(OrbitalOverlapPullback&&) = default;

    /** Directional derivative of A_aux^T S A_aux. */
    Eigen::MatrixXd overlap_direction(const Eigen::VectorXd& packed_raw_direction) const;

    /** Raw packed adjoint of the active overlap. */
    Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const;

    /** Full directional pullback, including the fixed-adjoint orbital curvature. */
    Eigen::VectorXd
    gradient_direction(const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint,
                       const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint_direction,
                       const Eigen::VectorXd& packed_raw_direction) const;

  private:
    void validate_overlap_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const;
    void validate_direction(const Eigen::VectorXd& packed_raw_direction) const;
    Eigen::MatrixXd
    active_auxiliary_gradient(const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const;

    const OrbitalPreparationInput& input_;
    const OrbitalPreparationResult& prepared_;
    const SparseParameterLayout& layout_;
    AcceptedOrbitalPreparationCache geometry_;
    Eigen::MatrixXd zero_inactive_density_gradient_;
};

} // namespace xmvb::vb
