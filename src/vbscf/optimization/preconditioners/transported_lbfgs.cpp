#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {

TransportedReducedLbfgsPreconditioner::TransportedReducedLbfgsPreconditioner(
    const OrbitalChart* space)
    : space_(space) {}

bool TransportedReducedLbfgsPreconditioner::try_add_pair(
    Eigen::VectorXd reduced_step,
    Eigen::VectorXd reduced_gradient_change) {
  const double step_norm = reduced_step.norm();
  const double gradient_change_norm = reduced_gradient_change.norm();
  const double secant_curvature = reduced_step.dot(reduced_gradient_change);
  const double minimum_secant_alignment =
      std::sqrt(std::numeric_limits<double>::epsilon());
  if (!(step_norm > 0.0) ||
      !(gradient_change_norm > 0.0) ||
      !std::isfinite(step_norm) ||
      !std::isfinite(gradient_change_norm) ||
      !std::isfinite(secant_curvature) ||
      secant_curvature <=
          minimum_secant_alignment * step_norm * gradient_change_norm) {
    return false;
  }

  Pair pair;
  pair.reduced_step = std::move(reduced_step);
  pair.reduced_gradient_change = std::move(reduced_gradient_change);
  pair.inverse_curvature = 1.0 / secant_curvature;
  pairs_.push_back(std::move(pair));
  return true;
}

bool TransportedReducedLbfgsPreconditioner::empty() const noexcept {
  return pairs_.empty();
}

int TransportedReducedLbfgsPreconditioner::size() const noexcept {
  return static_cast<int>(pairs_.size());
}

Eigen::VectorXd TransportedReducedLbfgsPreconditioner::apply(
    const Eigen::VectorXd& reduced_vector) const {
  if (pairs_.empty()) {
    return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
  }

  Eigen::VectorXd q = reduced_vector;
  std::vector<double> alphas(pairs_.size(), 0.0);
  for (std::size_t pair_index = pairs_.size(); pair_index-- > 0;) {
    const auto& pair = pairs_[pair_index];
    const double alpha =
        pair.inverse_curvature * pair.reduced_step.dot(q);
    if (!std::isfinite(alpha)) {
      throw std::runtime_error(
          "transported L-BFGS preconditioner produced a non-finite first loop");
    }
    alphas[pair_index] = alpha;
    q.noalias() -= alpha * pair.reduced_gradient_change;
  }

  Eigen::VectorXd z = space_->apply_inverse_reduced_block_preconditioner(q);
  for (std::size_t pair_index = 0;
       pair_index < pairs_.size();
       ++pair_index) {
    const auto& pair = pairs_[pair_index];
    const double beta =
        pair.inverse_curvature * pair.reduced_gradient_change.dot(z);
    if (!std::isfinite(beta)) {
      throw std::runtime_error(
          "transported L-BFGS preconditioner produced a non-finite second loop");
    }
    z.noalias() += pair.reduced_step * (alphas[pair_index] - beta);
  }
  return z;
}

TransportedReducedLbfgsPreconditioner
build_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size) {
  TransportedReducedLbfgsPreconditioner preconditioner(&current_space);
  if (max_history_size <= 0 || packed_secant_history.empty()) {
    return preconditioner;
  }

  const std::size_t history_begin =
      packed_secant_history.size() > static_cast<std::size_t>(max_history_size)
          ? packed_secant_history.size() -
                static_cast<std::size_t>(max_history_size)
          : 0;
  for (std::size_t pair_index = history_begin;
       pair_index < packed_secant_history.size();
       ++pair_index) {
    const auto& packed_pair = packed_secant_history[pair_index];
    preconditioner.try_add_pair(
        current_space.project_vector(packed_pair.packed_step).reduced_gradient,
        current_space
            .project_gradient(packed_pair.packed_gradient_change)
            .reduced_gradient);
  }
  return preconditioner;
}

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const OrbitalChart& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector) {
  if (transported_preconditioner == nullptr ||
      transported_preconditioner->empty()) {
    return current_space.apply_inverse_reduced_block_preconditioner(
        reduced_vector);
  }

  const Eigen::VectorXd preconditioned =
      transported_preconditioner->apply(reduced_vector);
  const double curvature = reduced_vector.dot(preconditioned);
  if (!std::isfinite(curvature) || curvature <= 0.0) {
    const Eigen::VectorXd base =
        current_space.apply_inverse_reduced_block_preconditioner(reduced_vector);
    std::ostringstream message;
    message << "transported L-BFGS preconditioner lost positive definiteness"
            << ": pairs=" << transported_preconditioner->size()
            << " curvature=" << curvature
            << " base_curvature=" << reduced_vector.dot(base)
            << " input_norm=" << reduced_vector.norm()
            << " output_norm=" << preconditioned.norm();
    throw std::runtime_error(message.str());
  }
  return preconditioned;
}

void append_nonredundant_truncated_newton_secant_pair(
    Eigen::VectorXd packed_step,
    Eigen::VectorXd packed_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history) {
  if (max_history_size <= 0) {
    return;
  }
  const double step_norm = packed_step.norm();
  const double gradient_change_norm = packed_gradient_change.norm();
  const double secant_curvature =
      packed_step.dot(packed_gradient_change);
  const double minimum_secant_alignment =
      std::sqrt(std::numeric_limits<double>::epsilon());
  if (!(step_norm > 0.0) ||
      !(gradient_change_norm > 0.0) ||
      !std::isfinite(step_norm) ||
      !std::isfinite(gradient_change_norm) ||
      !std::isfinite(secant_curvature) ||
      secant_curvature <=
          minimum_secant_alignment * step_norm * gradient_change_norm) {
    return;
  }

  packed_secant_history->push_back(PackedSecantPair{
      std::move(packed_step),
      std::move(packed_gradient_change)});
  if (packed_secant_history->size() >
      static_cast<std::size_t>(max_history_size)) {
    packed_secant_history->erase(packed_secant_history->begin());
  }
}

}  // namespace xmvb::vb
