#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"
#include "vbscf/derivatives/gradient/active_space/result.hpp"

namespace xmvb::vb {

/** Root-selection request shared by determinant and tensor electronic states. */
struct ElectronicStateRequest {
  std::vector<int> selected_states{0};
  std::vector<double> state_weights{1.0};
  double nuclear_repulsion_energy = 0.0;
  StructureEigensolver eigensolver = StructureEigensolver::Dense;
  StructureSolveAccuracy accuracy;
  Eigen::MatrixXd initial_eigenvectors;
};

/** A rejected energy trial can discard this without invoking any RDM kernel. */
class ElectronicStateForwardEvaluation {
 public:
  virtual ~ElectronicStateForwardEvaluation() = default;
  virtual double total_energy() const = 0;
};

/**
 * Active-space electronic-state boundary. Implementations own their forward
 * state; the caller owns the common AO/integral/orbital pullback. No tensor
 * implementation is required to create determinant response/HVP caches.
 */
class ElectronicStateBackend {
 public:
  virtual ~ElectronicStateBackend() = default;
  virtual std::unique_ptr<ElectronicStateForwardEvaluation> evaluate_energy(
      const VbScfInput&, const ElectronicStateRequest&) const = 0;
  virtual ActiveSpaceGradientResult complete_gradient(
      const VbScfInput&,
      std::unique_ptr<ElectronicStateForwardEvaluation>) const = 0;
  virtual ActiveSpaceGradientResult evaluate_energy_gradient(
      const VbScfInput&, const ElectronicStateRequest&) const;
  virtual bool provides_second_order_context() const { return false; }
};

}  // namespace xmvb::vb
