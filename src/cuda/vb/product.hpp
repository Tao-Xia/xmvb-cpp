#pragma once

#include <memory>

#include "cuda/ci/sigma.hpp"
#include "cuda/vb/transform.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {

/** CPU broad-support product-column chain. Numerical state lives on the GPU;
 * symbolic support, immutable patterns and LRU lifetimes are shared contracts.
 * Input/output coordinates here are touched spin products, not a full carrier.
 */
class ProductContraction {
public:
  ProductContraction(Context&,const vb::StructureAction::OrthogonalView&,
      const ExteriorTransform& alpha,const ExteriorTransform& beta,
      std::shared_ptr<const mcscf::ProductSupport> support);
  ~ProductContraction();
  void update_frame();
  void apply(linalg::ConstMatrixView coefficients,linalg::ConstMatrixView upper,
      const DirectCiSigma&,linalg::MatrixView hamiltonian,linalg::MatrixView overlap,
      std::size_t workspace_bytes,bool overlap_only=false) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
