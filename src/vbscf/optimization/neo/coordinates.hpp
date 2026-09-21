#pragma once

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/coupled/operator.hpp"

namespace xmvb::vb {

class NonredundantRetractionMetric;
class OrbitalChart;
class StructureTangentOperator;

/**
 * @brief Independent coordinates for the coupled orbital--structure tangent.
 *
 * Orbital entries are the accepted-point quotient-chart coordinates. Structure
 * entries use an orthonormal basis of the horizontal coefficient space, so the
 * selected-state gauge null space never enters a NEO vector. The borrowed
 * accepted-point operators must outlive this object.
 */
class CoupledNeoCoordinates {
public:
  CoupledNeoCoordinates(
      const OrbitalChart& orbital_chart,
      const NonredundantRetractionMetric& orbital_metric,
      const StructureTangentOperator& structure);

  Eigen::Index orbital_size() const noexcept { return orbital_size_; }
  Eigen::Index structure_size() const noexcept { return structure_size_; }
  Eigen::Index size() const noexcept { return orbital_size_ + structure_size_; }

  /** @brief Packs one horizontal coupled direction into independent coordinates. */
  Eigen::VectorXd flatten(const CoupledDirection& direction) const;

  /** @brief Restores one horizontal coupled direction from independent coordinates. */
  CoupledDirection unflatten(const Eigen::VectorXd& coordinates) const;

  /** @brief Applies the block-diagonal coupled physical metric. */
  Eigen::VectorXd apply_metric(const Eigen::VectorXd& coordinates) const;

  /** @brief Returns the squared norm induced by the coupled physical metric. */
  double squared_norm(const Eigen::VectorXd& coordinates) const;

private:
  const NonredundantRetractionMetric* orbital_metric_ = nullptr;
  const StructureTangentOperator* structure_ = nullptr;
  Eigen::Index orbital_size_ = 0;
  Eigen::Index structure_size_ = 0;
};

}  // namespace xmvb::vb
