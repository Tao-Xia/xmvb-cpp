#pragma once

#include <vector>

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Adjoint outputs associated with one forward kernel cache.
 */
struct PfAdjointResult {
  Matrix kernel_adjoint;
  Matrix sigma_adjoint;
  Matrix spatial_density;
  ScalarBuffer ggo_grad;
};

/**
 * @brief Reusable scratch storage for closed-shell exact Hamiltonian adjoints.
 *
 * Reusing these buffers across many pair evaluations avoids repeated heap
 * allocations inside the closed-shell exact reverse pass.
 */
struct PfClosedShellTwoElectronAdjointWorkspace {
  Matrix pair_term_matrix_adjoint;
  Matrix c_poly_h_n2_adjoint;
  Matrix c_adjoint;
  Matrix frechet_h_c_h_n2_adjoint;
  Matrix d_poly_h_n2_adjoint;
  Matrix d_adjoint;
  Matrix opposite_bridge_h_split_matrix_adjoint;
  Matrix pair_poly_adjoint;
  Matrix frechet_h_c_n2_adjoint;
  Matrix h_adjoint;
  Matrix sa_adjoint;
  ScalarBuffer pair_coefficient_adjoints;
  ScalarBuffer coeff_n2_adjoints;
  ScalarBuffer overlap_coefficient_adjoints;
};

/**
 * @brief Builds the trace-recursion adjoint objects for Pfaffian-VBSCF.
 */
class PfAdjointKernel {
public:
  /**
   * @brief Builds the full kernel adjoint, including both the trace-projection
   * path and tensor-operand backpropagation through `K`.
   *
   * @param cache Forward cache.
   * @param ggo Packed active-space two-electron tensor.
   * @param terms Tensor term descriptors.
   * @return Matrix Dense kernel adjoint.
   */
  static Matrix build_kernel_adjoint(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      const std::vector<PfTensorTerm>& terms = {},
      bool include_trace_path = true);

  /**
   * @brief Builds the full overlap-metric adjoint, combining direct tensor
   * operand contributions with the chain rule through `K`.
   *
   * @param cache Forward cache.
   * @param ggo Packed active-space two-electron tensor.
   * @param terms Tensor term descriptors.
   * @return Matrix Dense `Sigma` adjoint.
   */
  static Matrix build_sigma_adjoint(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      const std::vector<PfTensorTerm>& terms = {},
      bool include_trace_path = true);

  /**
   * @brief Extracts the spatial one-body primitive `Sigma_aa + Sigma_bb`.
   *
   * @param cache Forward cache.
   * @param ggo Packed active-space two-electron tensor.
   * @param terms Tensor term descriptors.
   * @return Matrix Spatial one-body primitive.
   */
  static Matrix build_spatial_density(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      const std::vector<PfTensorTerm>& terms = {},
      bool include_trace_path = true);

  /**
   * @brief Backpropagates one spatial 1-RDM adjoint to the overlap metric.
   *
   * This applies the exact reverse-mode chain rule through the cached
   * trace-projection density without explicitly forming any spatial Hessian.
   *
   * @param cache Forward cache.
   * @param one_rdm_adjoint Incoming spatial 1-RDM adjoint.
   * @return Matrix Dense adjoint of `Sigma`.
   */
  static Matrix build_one_rdm_source_sigma_adjoint(
      const PfKernelCache& cache,
      const ConstMatrixRef& one_rdm_adjoint);

  /**
   * @brief Accumulates the packed two-electron gradient associated with one
   * list of tensor terms.
   *
   * @param cache Forward cache.
   * @param terms Tensor term descriptors.
   * @return ScalarBuffer Packed `ggo` gradient.
   */
  static ScalarBuffer build_tensor_gradient(
      const PfKernelCache& cache,
      const std::vector<PfTensorTerm>& terms);

  /**
   * @brief Evaluates the full adjoint bundle for the current cache.
   *
   * @param cache Forward cache.
   * @param ggo Packed active-space two-electron tensor.
   * @param terms Optional tensor term descriptors for `ggo_grad`.
   * @return PfAdjointResult Kernel adjoint, sigma adjoint, 1-RDM primitive, and
   *     packed two-electron gradient.
   */
  static PfAdjointResult evaluate(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      const std::vector<PfTensorTerm>& terms = {},
      bool include_trace_path = true);

  /**
   * @brief Evaluates the closed-shell two-electron adjoint bundle without
   * building the intermediate tensor-term list.
   *
   * The returned `sigma_adjoint`, `spatial_density`, and `ggo_grad` contain
   * only the two-electron contribution.
   */
  static PfAdjointResult evaluate_closed_shell_two_electron(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo);

  /**
   * @brief Closed-shell exact two-electron adjoint with caller-owned workspaces.
   *
   * This avoids repeated heap allocations when the caller evaluates many pair
   * adjoints in sequence.
   */
  static void evaluate_closed_shell_two_electron_inplace(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      Matrix* full_workspace,
      Matrix* source_workspace,
      Matrix* kernel_adjoint,
      Matrix* sigma_adjoint,
      Matrix* spatial_density,
      ScalarBuffer* ggo_grad,
      PfClosedShellTwoElectronAdjointWorkspace* workspace = nullptr);

  /**
   * @brief Evaluates the closed-shell Hamiltonian adjoint bundle in one
   * reverse pass.
   *
   * This combines the one-electron `1-RDM` source adjoint and the exact
   * two-electron adjoint so the shared closed-shell trace/coefficient
   * backpropagation is performed only once.
   *
   * @param cache Forward cache.
   * @param one_rdm_adjoint Incoming spatial 1-RDM adjoint, typically `hho`.
   * @param ggo Packed active-space two-electron tensor.
   * @param spatial_density Output spatial overlap gradient contribution.
   * @param ggo_grad Output packed two-electron gradient contribution.
   * @param workspace Optional reusable closed-shell scratch storage.
   */
  static void evaluate_closed_shell_hamiltonian_inplace(
      const PfKernelCache& cache,
      const ConstMatrixRef& one_rdm_adjoint,
      const ScalarBuffer& ggo,
      Matrix* spatial_density,
      ScalarBuffer* ggo_grad,
      PfClosedShellTwoElectronAdjointWorkspace* workspace = nullptr);

};

}  // namespace xmvb::pfaffian_vbscf
