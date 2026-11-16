#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "vbscf/core/contracts/result.hpp"

namespace xmvb::tnvb {

class StateDirectory;

/** Owned column-major numerical response; no determinant cache or rank-6 data. */
struct WorkerResponse {
  std::shared_ptr<const StateDirectory> retained_state;
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  Eigen::VectorXd energies;
  Eigen::MatrixXd coefficients;
  std::optional<vb::DavidsonDiagnostics> davidson_diagnostics;
  // Protocol v3 densities use the orthonormal active basis. Pi retains its
  // established index permutation; contracted_three is the orthogonal D3G.
  Eigen::MatrixXd density_one;
  std::vector<double> density_two_buffer;
  Eigen::MatrixXd contracted_three;
  double fit_energy_error = 0.0;
  double rdm_identity_error = 0.0;
  double hs_wall_seconds = 0.0;
  double rdm_wall_seconds = 0.0;
};

}  // namespace xmvb::tnvb
