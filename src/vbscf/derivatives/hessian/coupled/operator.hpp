#pragma once

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/coupled/structure.hpp"

namespace xmvb::vb {

class ExactHvpOperator;

/** @brief One vector in the coupled orbital--structure tangent space. */
struct CoupledDirection {
  Eigen::VectorXd orbital;
  StructureTangent structure;
};

/** @brief Coupled Hessian image represented entirely in independent coordinates. */
struct CoupledCoordinateDirection {
  Eigen::VectorXd orbital;
  Eigen::VectorXd structure;
  /** @brief Structure metric image obtained by the same H/S contraction. */
  Eigen::VectorXd structure_metric;
};

/**
 * @brief Matrix-free coupled VBSCF Hessian at one accepted point.
 *
 * The borrowed orbital and structure operators must describe the same accepted
 * point and outlive this object. `apply` composes the four exact blocks without
 * forming a Hessian or solving a structure-response equation.
 */
class CoupledHessianOperator {
public:
  CoupledHessianOperator(
      const ExactHvpOperator& orbital,
      const StructureTangentOperator& structure);

  /** @brief Returns @f$[\bar A p+B^Tz,\;Bp+Cz]@f$. */
  CoupledDirection apply(const CoupledDirection& direction) const;

  /**
   * @brief Applies the coupled Hessian without ambient structure round trips.
   *
   * The returned structure metric image reuses the overlap contraction already
   * required for @f$Cz@f$; callers must not issue a second H/S action for it.
   */
  CoupledCoordinateDirection apply_coordinates(
      const Eigen::VectorXd& orbital,
      const Eigen::VectorXd& structure) const;

private:
  const ExactHvpOperator* orbital_ = nullptr;
  const StructureTangentOperator* structure_ = nullptr;
};

}  // namespace xmvb::vb
