#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

struct TransportedReducedSecantPair {
  Eigen::VectorXd step;
  Eigen::VectorXd gradient_change;
};

class TransportedReducedLbfgsPreconditioner {
public:
  TransportedReducedLbfgsPreconditioner(
      const OrbitalChart* space,
      LbfgsInitialInverse initial_inverse);

  bool try_add_pair(
      Eigen::VectorXd reduced_step,
      Eigen::VectorXd reduced_gradient_change);
  bool empty() const noexcept;
  int size() const noexcept;
  /**
   * @brief Maps a reduced gradient covector to a reduced tangent vector.
   *
   * Secant contractions use the natural vector-covector pairing. They do not
   * require the coupled physical trust metric to be assembled or whitened.
   */
  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_vector) const;

private:
  struct Pair {
    Eigen::VectorXd reduced_step;
    Eigen::VectorXd reduced_gradient_change;
    double inverse_curvature = 0.0;
  };

  const OrbitalChart* space_ = nullptr;
  LbfgsInitialInverse initial_inverse_ =
      LbfgsInitialInverse::ScaledIdentity;
  std::vector<Pair> pairs_;
};

TransportedReducedLbfgsPreconditioner
build_transported_reduced_lbfgs_preconditioner(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size,
    LbfgsInitialInverse initial_inverse);

std::vector<TransportedReducedSecantPair>
transport_secant_pairs_to_chart(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size);

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector);

/** @brief Appends an embedding secant with positive target-chart curvature. */
void append_projected_secant_pair(
    const OrbitalChart& current_space,
    Eigen::VectorXd packed_step,
    Eigen::VectorXd packed_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history);

}  // namespace xmvb::vb
