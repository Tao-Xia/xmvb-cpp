#pragma once

#include <memory>

#include "cuda/ci/sigma.hpp"
#include "cuda/vb/columns.hpp"
#include "cuda/vb/transform.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {

/** Borrowed sparse spin-product to selected-structure incidence. */
struct ProductIncidence {
  const std::int64_t* offsets;
  const std::int64_t* structures;
  const double* weights;
  int size;
};

/** CPU StructureResponseSweep in device selected coordinates. */
struct ProductJet {
  Matrix<double> hamiltonian,overlap,one,pair,generator;
};

/** Source-backed accepted endpoint; retained Y/residual panels are optional. */
struct ProductEndpoint {
  Matrix<double> coefficients;
  std::unique_ptr<ProductImages> images;
  int extent=0;
  double energy=0.0;
};

/** CPU broad-support product-column chain. Numerical state lives on the GPU;
 * symbolic support, immutable patterns and LRU lifetimes are shared contracts.
 * Endpoints are selected-structure coefficients. Product coordinates exist
 * only inside bounded transform/sigma panels and optional accepted images.
 */
class ProductContraction {
public:
  ProductContraction(Context&,const vb::StructureAction::OrthogonalView&,
      const ExteriorTransform& alpha,const ExteriorTransform& beta,
      const mcscf::ProductSupport* support);
  ~ProductContraction();
  void update_frame();
  /** Finish the accepted CPU-style Y/(H-E)Y panels after Rayleigh evaluation. */
  void accept(ProductEndpoint&,double energy) const;
  /** Selected-space H/M without a complete product source or group image. */
  void apply_selected(ProductIncidence,linalg::ConstMatrixView coefficients,
      const DirectCiSigma&,linalg::MatrixView hamiltonian,linalg::MatrixView overlap,
      std::size_t workspace_bytes,bool overlap_only=false) const;
  double maximum_orthogonal_component(ProductIncidence,
      linalg::ConstMatrixView coefficients,std::size_t workspace_bytes) const;
  /** CPU product_jet: primal, orbital and independent CI columns share the
   * same bounded source/read/sigma/reverse sweep. Outputs are selected-space
   * images and small integral/generator cotangents, never complete carriers.
   * The tangent result includes the supplied CI response. The optional ci
   * result isolates that response; the coarse solver subtracts it from the
   * total image to obtain the independent orbital image, as in VbPoint.
   */
  ProductJet jet(ProductIncidence,const DirectCiSigma&,ProductEndpoint&,
      linalg::ConstMatrixView upper,linalg::ConstMatrixView response,
      linalg::ConstMatrixView delta_upper,const DirectCiSigma* delta_sigma,
      std::size_t workspace_bytes,ProductJet* ci=nullptr,
      linalg::ConstMatrixView relative_direction={}) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
