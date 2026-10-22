#pragma once

#include "cuda/ci/sigma.hpp"
#include "cuda/integrals/exact.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/orbitals.hpp"
#include "cuda/vb/product.hpp"

namespace xmvb::cuda {

/** @brief Exact single-state VB energy and coupled raw-coordinate derivatives.
 * E=E_core(P)+<Y|H(F,P)|Y>/<Y|Y>, Y=wedge(R) W C. The selected VB expansion
 * W is never completed. The independent orbital/CI chart is a separate map;
 * these derivatives include finite CI residuals and all frame derivatives.
 */
class VbPoint {
 public:
  /** core and overlap are shared device views and must outlive this point. */
  VbPoint(Context& context,const vb::OrbitalPreparationInput& orbitals,
          StructureAction& action,const ExactAoPairOperator& integrals,
          linalg::ConstMatrixView core,linalg::ConstMatrixView overlap,double nuclear_repulsion);

  double evaluate(linalg::ConstMatrixView raw,linalg::ConstMatrixView coefficients);
  /** Prepare a new orbital point once before CI relaxation. */
  void prepare(linalg::ConstMatrixView raw);
  /** Evaluate the relaxed or fixed CI state using the prepared integrals. */
  double evaluate_coefficients(linalg::ConstMatrixView coefficients);
  void activate() const { select_frame(); }
  /** Analytic rotation diagonal and accepted response projections from the
   * same integral sweep. mo is ordered orthonormal core, active, virtual.
   */
  Matrix<double> hessian_diagonal(linalg::ConstMatrixView mo);
  /** Pull back only requested components; an empty output omits that work. */
  void gradient(linalg::MatrixView orbital,linalg::MatrixView structure = {}) const;
  /** Joint Hessian action, not a relaxed orbital response solve. Optional CI
   * images give H(0,q), reusing the same sigma, AO tiles and accepted responses;
   * H(p,0) is the total minus this image. No additional full action is taken.
   */
  void hessian(linalg::ConstMatrixView orbital,linalg::ConstMatrixView structure,
               linalg::MatrixView orbital_image,linalg::MatrixView structure_image,
               linalg::MatrixView ci_orbital_image = {},
               linalg::MatrixView ci_structure_image = {}) const;

  const OrbitalGeometry& geometry() const noexcept { return geometry_; }
  linalg::ConstMatrixView effective_one_electron() const noexcept { return effective_.view(); }
  linalg::ConstMatrixView one_electron_adjoint() const noexcept { return one_adjoint_.view(); }
  /** Upper orbital-generator cotangent of the normalized physical state. */
  linalg::ConstMatrixView frame_moment() const noexcept { return frame_moment_.view(); }
  /** Full Frobenius cotangent of the symmetric orthogonal active pair kernel. */
  linalg::ConstMatrixView pair_kernel_adjoint() const noexcept { return pair_adjoint_.view(); }
  linalg::ConstMatrixView one_electron() const noexcept { return one_.view(); }
  linalg::ConstMatrixView pair_kernel() const noexcept { return pair_.view(); }
  ConstPairPanel pair_products() const noexcept {
    return pair_products_.empty() ? ConstPairPanel{}:
        ConstPairPanel{pair_products_.data(),integrals_->pair_count(),pair_.rows(),pair_.rows()};
  }
  ConstPairPanel exchange_products() const noexcept {
    const auto n=one_.rows();
    return exchange_products_.empty() ? ConstPairPanel{}:
        ConstPairPanel{exchange_products_.data(),integrals_->pair_count(),n*n,n*n};
  }
  double energy() const noexcept { return total_energy_; }

 private:
  void select_frame() const;
  std::size_t product_workspace_bytes() const;

  Context* context_;
  StructureAction* action_;
  const ExactAoPairOperator* integrals_;
  linalg::ConstMatrixView core_;
  double nuclear_,active_energy_=0.0,total_energy_=0.0,norm_=0.0;
  OrbitalGeometry geometry_;
  mutable DirectCiSigma sigma_,direction_sigma_;
  Matrix<double> effective_,one_,pair_;
  mutable ProductEndpoint endpoint_;
  std::unique_ptr<ProductContraction> products_;
  Matrix<double> selected_h_,selected_m_,selected_gradient_;
  Matrix<double> one_adjoint_,pair_adjoint_,frame_gradient_,frame_moment_;
  Matrix<double> active_gradient_,density_gradient_;
  Buffer<double> pair_products_,exchange_products_;
  Buffer<double> inactive_response_;
  Matrix<double> active_response_,overlap_core_,core_basis_,core_to_dual_;
  mutable std::uint64_t selected_action_version_=0;
  bool ready_=false,prepared_=false;
  double core_energy_=0.0;
};

}  // namespace xmvb::cuda
