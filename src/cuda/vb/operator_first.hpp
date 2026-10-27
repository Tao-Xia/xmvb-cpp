#pragma once
#include <memory>
#include "cuda/vb/product.hpp"
namespace xmvb::cuda {
class StructureAction;
/** @brief Device source-order H/M and analytic adjoint/jet executor.
 * Borrows the selected incidence and exterior maps from the action. Topology is
 * shared by clones; accepted numerical state is orbital-sized and point-local.
 */
class OperatorFirstAction {
  public:
    OperatorFirstAction(Context&, const StructureAction&);
    ~OperatorFirstAction();
    std::unique_ptr<OperatorFirstAction> clone() const;
    void update(linalg::ConstMatrixView one,
                linalg::ConstMatrixView pair,
                linalg::ConstMatrixView upper);
    void update_frame(linalg::ConstMatrixView upper);
    void accept(double energy);
    void apply(linalg::ConstMatrixView coefficients,
               linalg::MatrixView h,
               linalg::MatrixView m,
               std::size_t workspace_bytes) const;
    /** Primal returns unnormalized energy cotangents. Tangent integral
     * cotangents are half derivatives, as required by VbPoint's two-sided
     * normalization; H/M images and the residual generator derivative are full.
     * accept() supplies the fixed Rayleigh multiplier for subsequent jets.
     */
    ProductJet jet(const DirectCiSigma&,
                   linalg::ConstMatrixView coefficients,
                   linalg::ConstMatrixView response,
                   linalg::ConstMatrixView delta_upper,
                   linalg::ConstMatrixView delta_one,
                   linalg::ConstMatrixView delta_pair,
                   std::size_t workspace_bytes,
                   ProductJet* ci = nullptr,
                   linalg::ConstMatrixView overlap_image = {});

  private:
    struct Impl;
    explicit OperatorFirstAction(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace xmvb::cuda
