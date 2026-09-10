#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/types.hpp"

namespace xmvb::vb {

struct ActiveSpaceTwoElectronResult;
struct VbScfInput;

/**
 * @brief Non-owning active-space integral direction for one HVP application.
 *
 * The three buffers form one mathematical perturbation and must always be
 * propagated together. This view keeps that coupling explicit without copying
 * the matrix-free workspaces owned by the HVP operator.
 */
struct ActiveSpaceIntegralDirectionView {
  const std::vector<double>& overlap;
  const std::vector<double>& one_electron;
  const std::vector<double>& packed_two_electron;
};

/**
 * @brief Accepted-point data shared by every active-integral direction.
 */
struct ActiveSpaceIntegralDirectionContext {
  const VbScfInput& input;
  const ActiveSpaceTwoElectronResult& accepted_two_electron_integrals;
  const Eigen::MatrixXd& active_auxiliary_orbitals;
  const Eigen::MatrixXd& overlap_times_active_auxiliary_orbitals;
  const Eigen::MatrixXd& effective_h1e_times_active_auxiliary_orbitals;
  const Eigen::MatrixXd& effective_h1e_transpose_times_active_auxiliary_orbitals;
  const Eigen::MatrixXd& dense_active_coefficients;
};

/**
 * @brief Direction-dependent inputs to the active-integral linearization.
 */
struct ActiveSpaceIntegralTangent {
  const Eigen::MatrixXd& active_auxiliary_orbitals;
  const Eigen::MatrixXd& dense_active_coefficients;
  const Eigen::MatrixXd& delta_effective_h1e_times_active_auxiliary_orbitals;
  const Eigen::VectorXd* precomputed_packed_two_electron = nullptr;
};

struct ActiveSpaceIntegralDirectionWorkspace {
  std::vector<double> overlap;
  std::vector<double> one_electron;
  std::vector<double> packed_two_electron;
  ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace two_electron;
};

ActiveSpaceIntegralDirectionView build_active_space_integral_direction(
    const ActiveSpaceIntegralDirectionContext& context,
    const ActiveSpaceIntegralTangent& tangent,
    ActiveSpaceIntegralDirectionWorkspace* workspace);

}  // namespace xmvb::vb
