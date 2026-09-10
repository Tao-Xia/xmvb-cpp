#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/optimizer_types.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

class TransportedReducedLbfgsPreconditioner {
public:
  explicit TransportedReducedLbfgsPreconditioner(const OrbitalChart* space);

  bool try_add_pair(
      Eigen::VectorXd reduced_step,
      Eigen::VectorXd reduced_gradient_change);
  bool empty() const noexcept;
  int size() const noexcept;
  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_vector) const;

private:
  struct Pair {
    Eigen::VectorXd reduced_step;
    Eigen::VectorXd reduced_gradient_change;
    double inverse_curvature = 0.0;
  };

  const OrbitalChart* space_ = nullptr;
  std::vector<Pair> pairs_;
};

TransportedReducedLbfgsPreconditioner
build_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size);

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector);

void append_nonredundant_truncated_newton_secant_pair(
    Eigen::VectorXd packed_step,
    Eigen::VectorXd packed_projected_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history);

}  // namespace xmvb::vb
