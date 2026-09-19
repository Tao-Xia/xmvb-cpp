#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

struct TransportedReducedSecantPair {
  Eigen::VectorXd step;
  Eigen::VectorXd gradient_change;
};

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
build_transported_reduced_lbfgs_preconditioner(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size);

std::vector<TransportedReducedSecantPair>
transport_secant_pairs_to_chart(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size);

/**
 * @brief Transports every stored primal/dual secant into `current_space`.
 *
 * Steps use the chart's vector projection, whereas gradient changes use its
 * covector pullback.  Re-lifting both reduced quantities makes subsequent
 * transports compositional and removes obsolete gauge components.
 */
void transport_packed_secant_history_to_chart(
    const OrbitalChart& current_space,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history);

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector);

/** @brief Appends a curvature-valid secant expressed in one quotient chart. */
void append_reduced_secant_pair(
    const OrbitalChart& current_space,
    Eigen::VectorXd reduced_step,
    Eigen::VectorXd reduced_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history);

}  // namespace xmvb::vb
