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

private:
  const ExactHvpOperator* orbital_ = nullptr;
  const StructureTangentOperator* structure_ = nullptr;
};

}  // namespace xmvb::vb
