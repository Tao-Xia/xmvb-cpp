#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/orbital/ao_integral_input.hpp"

namespace xmvb::vb {

struct CppRestrictedHartreeFockOptions {
  int max_iterations = 96;
  double density_tolerance = 1.0e-8;
  double old_density_weight = 0.20;
  int diis_history = 6;
  int diis_start_iteration = 2;
};

struct CppRestrictedHartreeFockResult {
  bool converged = false;
  int iterations = 0;
  double electronic_energy = 0.0;
  Eigen::MatrixXd molecular_orbital_matrix;
  Eigen::VectorXd orbital_energies;
  Eigen::MatrixXd density_projector;
  Eigen::MatrixXd fock_matrix;
};

class CppRestrictedHartreeFockSolver {
public:
  explicit CppRestrictedHartreeFockSolver(
      CppRestrictedHartreeFockOptions options = {});

  CppRestrictedHartreeFockResult solve(
      int n_total_electrons,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
      const AoIntegralInput& ao_integral_input) const;

  CppRestrictedHartreeFockResult solve(
      int n_total_electrons,
      const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
      const AoIntegralInput& ao_integral_input,
      const std::vector<double>& initial_density_projector) const;

private:
  CppRestrictedHartreeFockOptions options_;
};

}  // namespace xmvb::vb
