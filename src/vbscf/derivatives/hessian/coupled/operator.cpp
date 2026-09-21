#include "vbscf/derivatives/hessian/coupled/operator.hpp"

#include <stdexcept>

#include "vbscf/derivatives/hessian/exact/operator.hpp"

namespace xmvb::vb {

CoupledHessianOperator::CoupledHessianOperator(
    const ExactHvpOperator& orbital,
    const StructureTangentOperator& structure)
    : orbital_(&orbital), structure_(&structure) {}

CoupledDirection CoupledHessianOperator::apply(
    const CoupledDirection& direction) const {
  const OrbitalCouplingAction orbital_action =
      orbital_->apply_orbital_coupling(direction.orbital);
  const StructureCouplingAction structure_action =
      structure_->apply_coupling(direction.structure);
  if (orbital_action.scaled_structure_forcing.rows() !=
          structure_action.hessian.scaled_coefficients.rows() ||
      orbital_action.scaled_structure_forcing.cols() !=
          structure_action.hessian.scaled_coefficients.cols()) {
    throw std::logic_error(
        "coupled orbital and structure actions have incompatible dimensions");
  }

  CoupledDirection result;
  result.orbital = orbital_action.orbital_hessian +
      orbital_->apply_structure_coupling_adjoint(
          structure_action.coefficient_response,
          structure_action.adjoint_multipliers);
  result.structure.scaled_coefficients =
      orbital_action.scaled_structure_forcing +
      structure_action.hessian.scaled_coefficients;
  if (!result.orbital.allFinite() ||
      !result.structure.scaled_coefficients.allFinite()) {
    throw std::runtime_error("coupled Hessian action is not finite");
  }
  return result;
}

}  // namespace xmvb::vb
