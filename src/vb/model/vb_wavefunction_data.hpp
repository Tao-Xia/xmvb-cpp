#pragma once

#include <Eigen/Core>

#include "vb/model/vb_dimensions.hpp"

namespace xmvb::vb {

struct VbWavefunctionData {
  VbDimensions dims;

  double nuclear_repulsion_energy = 0.0;
  double one_electron_energy = 0.0;

  Eigen::MatrixXd hamiltonian_matrix;
  Eigen::MatrixXd overlap_matrix;
  Eigen::MatrixXd eigenvector_matrix;

  Eigen::VectorXd electronic_state_energies;
  Eigen::VectorXd state_average_weights;
};

}  // namespace xmvb::vb
