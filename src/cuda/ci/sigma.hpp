#pragma once

#include <cstddef>
#include <memory>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "mcscf/ci/links.hpp"

namespace xmvb::mcscf { class DirectCiSigmaAction; }

namespace xmvb::cuda {

/** Exact packed-channel direct-CI action on a full current-RHS spin product.
 * A panel has n_alpha rows and n_beta*RHS columns, with beta varying fastest.
 * Input/output panels are caller-owned and charged to Context's device budget.
 * Source-row/all-channel and full-row/output-channel slabs share a bounded
 * workspace. No Hamiltonian matrix or extra full-carrier channel cache is retained.
 */
class DirectCiSigma {
 public:
  DirectCiSigma(Context& context, const mcscf::DirectCiSigmaAction& source,
                int maximum_channel_tile = 16);
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

  /** Full nonsymmetric orbital cotangent of left^T E(generator) right.
   * The device output D satisfies sum_pq D(p,q) generator(p,q) for arbitrary
   * generator directions and is overwritten on the Context stream.
   */
  void one_body_generator_adjoint(linalg::ConstMatrixView left,
                                  linalg::ConstMatrixView right,
                                  linalg::MatrixView orbital_adjoint) const;

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
  int n_alpha_, n_beta_, n_orbitals_, n_electrons_, n_channels_, maximum_channel_tile_;
  bool shared_spin_graph_;
  Buffer<double> kernel_;
  Buffer<double> effective_one_;
  std::shared_ptr<const SpinGraph> alpha_, beta_;
  // Reused on the same stream, not allocated/freed at every sigma call.
  mutable Matrix<double> scattered_, transformed_, right_scattered_, channel_adjoint_;
  // Generator VJPs borrow the channel panels (or sigma output slab) after
  // their preceding action has completed on the same stream. In particular,
  // a captured action's scattered_/transformed_ addresses are never changed.
  mutable Buffer<double> generator_moments_;
  mutable Matrix<double> generator_left_;
};

}  // namespace xmvb::cuda
