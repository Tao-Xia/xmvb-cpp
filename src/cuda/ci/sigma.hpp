#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "mcscf/ci/links.hpp"

namespace xmvb::mcscf { class DirectCiSigmaAction; }

namespace xmvb::cuda {

/** Exact packed-channel direct-CI action on a full current-RHS spin product.
 * A panel has n_alpha rows and n_beta*RHS columns, with beta varying fastest.
 * Input/output panels are caller-owned and charged to Context's device budget.
 * All-channel source-row slabs share a bounded workspace. No Hamiltonian
 * matrix or extra full-carrier channel cache is retained.
 */
class DirectCiSigma {
 public:
  /** channel_capacity bounds storage in full-carrier channel equivalents;
   * every contraction row still includes all channels. A complete row is the
   * minimum allocation, including in a one-product carrier.
   */
  DirectCiSigma(Context& context, const mcscf::DirectCiSigmaAction& source,
                int channel_capacity = 16);
  /** @brief Share immutable device links; update_integrals must precede use. */
  DirectCiSigma(Context& context, const DirectCiSigma& reference);
  DirectCiSigma(const DirectCiSigma&) = delete;
  DirectCiSigma& operator=(const DirectCiSigma&) = delete;

  /** Enqueues work on Context's stream; host consumers must synchronize. */
  void apply(linalg::ConstMatrixView coefficients,
             linalg::MatrixView image, bool add = false) const;

  /** Full, unsymmetrized transition cotangent of the packed channel kernel.
   * Computes (E left)^T (E right), summed over all RHS columns. Every view is
   * device-resident; the caller owns and receives an overwritten K cotangent.
   * Panels have the same layout as apply and must not alias an output.
   */
  void channel_adjoint(linalg::ConstMatrixView left,
                       linalg::ConstMatrixView right,
                       linalg::MatrixView channel_kernel_adjoint) const;

  /** Integral VJP of left^T H right, including the absorbed one-body and
   * normal-ordering terms. Outputs are symmetric full Frobenius cotangents.
   * The one-electron and pair-kernel matrices are overwritten on the stream.
   */
  void integral_adjoint(linalg::ConstMatrixView left,
                        linalg::ConstMatrixView right,
                        linalg::MatrixView one_electron_adjoint,
                        linalg::MatrixView pair_kernel_adjoint) const;

  /** Shares the left-channel scatter between H*response and the integral VJP
   * of response^T H state. All outputs are overwritten and device-resident.
   */
  void apply_with_adjoint(linalg::ConstMatrixView response,
                          linalg::ConstMatrixView state,
                          linalg::MatrixView image,
                          linalg::MatrixView one_electron_adjoint,
                          linalg::MatrixView pair_kernel_adjoint) const;

  /** One exact directional sweep on the shared spin graph. Overwrites
   * combined_image=H*response+dH*state and the integral cotangent of
   * response^T H state. The directional Hamiltonian is supplied by
   * direction.update_integrals(). Each source row is scattered once and both
   * products share its all-channel intermediate; no directional carrier or
   * full-carrier output slab is retained.
   */
  void apply_direction_with_adjoint(linalg::ConstMatrixView response,
      linalg::ConstMatrixView state,const DirectCiSigma& direction,
      linalg::MatrixView combined_image,
      linalg::MatrixView one_electron_adjoint,
      linalg::MatrixView pair_kernel_adjoint) const;

  /** Exact product-basis diagonal of the current Hamiltonian channel kernel.
   * Output has n_alpha rows and n_beta columns, in apply's product ordering.
   */
  void diagonal(linalg::MatrixView values) const;

  /** Rebuilds the resident Hamiltonian channel kernel from device integrals.
   * Uses the constructor's absorbed one-body and normal-ordering convention.
   * K is overwritten in place, so its address remains valid in captured graphs.
   */
  void update_integrals(linalg::ConstMatrixView one_electron,
                        linalg::ConstMatrixView pair_kernel);

  /** Applies sum_pq generator(p,q) a_p^+ a_q to a product panel.
   * The device generator may be nonsymmetric. Image is overwritten; no input
   * or output may alias another view. Each output product has one owner.
   */
  void apply_generator(linalg::ConstMatrixView coefficients,
                       linalg::ConstMatrixView orbital_generator,
                       linalg::MatrixView image) const;

  /** Independent full moments left[i]^T E(generator) right, sharing E(right).
   * Each output is overwritten, including both nonsymmetric orientations.
   * Batched left carriers are contiguous; a single left also accepts strides.
   * Only one bounded left tile is packed, never replicated full carriers.
   */
  void generator_moments(std::span<const linalg::ConstMatrixView> left,
      linalg::ConstMatrixView right,std::span<const linalg::MatrixView> outputs) const;

  /** Allocate the bounded channel slabs before stream capture. */
  void prepare(int block_width) const;
  /** Allocate the fused sigma and integral-adjoint workspace before capture. */
  void prepare_with_adjoint(int block_width) const {
    prepare_adjoint(block_width, true, true);
  }
  /** Allocate the independent generator-VJP row slabs before capture. */
  void prepare_generator_adjoint(int block_width = 1) const;

  int n_alpha_strings() const noexcept { return n_alpha_; }
  int n_beta_strings() const noexcept { return n_beta_; }
  int n_orbitals() const noexcept { return n_orbitals_; }
  int n_channels() const noexcept { return n_channels_; }
  std::size_t resident_bytes() const noexcept;
  /** Shared scratch changes invalidate graphs that captured its addresses.
   * All members of a clone family execute sequentially on Context's stream.
   */
  std::uint64_t workspace_generation() const noexcept { return workspace_->generation; }

 private:
  struct SpinGraph {
    /** Zero means absent; signed (source+1) encodes the exact fermionic sign. */
    Buffer<int> sources;
    SpinGraph() = default;
    SpinGraph(Context& context, const mcscf::SpinConnections& source, int channels);
    std::size_t resident_bytes() const noexcept {
      return sources.size()*sizeof(int);
    }
  };

  void prepare_adjoint(int block_width, bool with_sigma,
                       bool retain_channel_kernel) const;
  void contract_adjoint(linalg::ConstMatrixView left,
                        linalg::ConstMatrixView right,
                        linalg::MatrixView* image,
                        linalg::MatrixView channel_kernel_adjoint) const;
  void pullback_channel_kernel(linalg::ConstMatrixView channel_kernel_adjoint,
                               linalg::MatrixView one_electron_adjoint,
                               linalg::MatrixView pair_kernel_adjoint) const;
  Context* context_;
  int n_alpha_, n_beta_, n_orbitals_, n_electrons_, n_channels_, channel_capacity_;
  bool shared_spin_graph_;
  Buffer<double> kernel_;
  Buffer<double> effective_one_;
  std::shared_ptr<const SpinGraph> alpha_, beta_;
  struct Workspace {
    Matrix<double> scattered,transformed,right_scattered,channel_adjoint;
    std::uint64_t generation=0;
  };
  // One scratch owner per shared spin graph family, not per physical point.
  std::shared_ptr<Workspace> workspace_;
  Matrix<double>& scattered_;
  Matrix<double>& transformed_;
  Matrix<double>& right_scattered_;
  Matrix<double>& channel_adjoint_;
  // Generator VJPs borrow the channel panels (or sigma output slab) after
  // their preceding action has completed on the same stream. In particular,
  // a captured action's scattered_/transformed_ addresses are never changed.
  mutable Buffer<double> generator_moments_;
  mutable Matrix<double> generator_left_;
};

}  // namespace xmvb::cuda
