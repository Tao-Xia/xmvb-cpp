#pragma once

#include <Eigen/Core>

namespace xmvb::vb {
struct OrbitalPreparationInput;
class OrbitalChart;
namespace diagnostics {

/** @brief AO-metric geometry of one actual raw additive displacement.
 * With x=b/||b||_S and d=delta_b/||b||_S, alpha=x^T S d and
 * theta=||d-alpha*x||_S. All angles are in radians, before core projection.
 */
struct OrbitalStepGeometry {
    double alpha = 0.0;
    double theta = 0.0;
    double step_norm = 0.0;
    double actual_angle = 0.0;
    double norm_ratio = 1.0;
};

OrbitalStepGeometry measure_orbital_step(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                         const Eigen::Ref<const Eigen::VectorXd>& displacement,
                                         const Eigen::Ref<const Eigen::MatrixXd>& overlap);

/** @brief Explicit geometry logging only; never an optimizer policy. */
bool orbital_step_audit_enabled();

/** @brief Log occupied-orbital geometry without an objective or HVP call. */
void audit_orbital_step(const OrbitalPreparationInput& input,
                        const OrbitalChart& chart,
                        const Eigen::VectorXd& reduced_step,
                        const char* stage,
                        int micro,
                        const Eigen::VectorXd& base_reduced = Eigen::VectorXd{});

/** @brief Opt-in raw orbital/CI state snapshot; no physical evaluation is performed. */
void audit_ciah_state(const OrbitalPreparationInput& input,
                      const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                      const char* stage,
                      double energy);

} // namespace diagnostics
} // namespace xmvb::vb
