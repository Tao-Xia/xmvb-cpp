#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "mcscf/ci/links.hpp"

namespace xmvb::mcscf { class DirectCiSigmaAction; class ProductSupport; }

namespace xmvb::cuda {

/** Device counterpart of the CPU column callbacks. Packing is
 * (alpha_rows, beta_local*width+rhs); callbacks borrow the current tile only.
 */
using CiTileReader=std::function<void(std::span<const int>,std::span<const int>,linalg::MatrixView)>;
/** One left carrier contributes to one caller-owned orbital output. */
struct CiStreamedGeneratorTerm {
  linalg::ConstMatrixView left;
  std::size_t output;
};
/** Terms sharing this reader consume each target and beta-neighbor tile once. */
struct CiStreamedGeneratorBatch {
  CiTileReader read;
  std::span<const CiStreamedGeneratorTerm> terms;
  double scale=1.0;
  bool transpose=false;
};
/** Borrowed compact output column. Source -1 denotes the whole halo panel;
 * otherwise source is one tile-local beta column. A null rows pointer uses
 * the tile halo; nonnull rows are the consumer's device-resident indices.
 * full_rows denotes the implicit identity row map, without allocating it.
 * Numerical columns can be views into the same packed allocation.
 */
struct CiTileColumn {
  int source;
  const int* rows;
  linalg::MatrixView values;
  bool full_rows=false;
};
/** Exact device-resident support of a compact column, shared by every lane. */
struct CiColumnMap {
  const int* rows;
  int size;
  std::int64_t offset;
};
/** Borrowed compact output panel. source_begin/count select consecutive
 * columns in the current, sorted tile beta list; maps start at first_beta.
 * lane is the first of width consecutive RHS slots in values. */
struct CiTilePanel {
  int first_beta,source_begin,count,lane,max_rows;
  const CiColumnMap* maps;
  linalg::MatrixView values;
};
/** Contract a batch into consumer-owned views with one launch. No view or
 * index pointer may outlive the writer callback. */
struct CiTileGather {
  std::function<void(std::span<const CiTileColumn>,bool)> columns;
  /** Reuse a compact panel's existing maps. Tile-local source columns are
   * obtained from the shared beta index; absent columns are not touched. */
  std::function<void(const CiTilePanel&)> panel;
};
using CiTileWriter=std::function<void(std::span<const int>,std::span<const int>,const CiTileGather&)>;
using CiColumnRelease=std::function<void(std::span<const int>)>;

class DirectCiSigma;
struct CiTileRequest {
  CiTileReader response;
  CiTileWriter image;
  const DirectCiSigma* direction=nullptr;
  linalg::MatrixView one_adjoint{},pair_adjoint{};
  /** Optional resident endpoint: consume signed links directly, without
   * gathering and copying one temporary halo column at a time. */
  linalg::MatrixView carrier_image{};
  /** First response reader yields exactly the common state on every tile.
   * Its channels occupy the state slab through all later requests; the caller
   * must keep that source live through the sweep. */
  bool response_is_state=false;
};

/** Borrowed resident endpoints for consumers not yet storing compact columns.
 * The executor is still the same support/halo tile sweep, not a second sigma.
 */
struct CiCarrierRequest {
  linalg::ConstMatrixView response;
  linalg::MatrixView image;
  const DirectCiSigma* direction=nullptr;
  linalg::MatrixView one_adjoint{},pair_adjoint{};
};

/** Exact packed-channel direct-CI action on a full current-RHS spin product.
 * A panel has n_alpha rows and n_beta*RHS columns, with beta varying fastest.
 * Input/output panels are caller-owned and charged to Context's device budget.
 * All-channel source-row slabs share a bounded workspace. No Hamiltonian
 * matrix or extra full-carrier channel cache is retained.
 * Nonzero exchange_parity requires identical spin graphs and certifies the
 * same transpose parity for every supplied carrier, not just equal dimensions.
 */
class DirectCiSigma {
 public:
  /** channel_capacity bounds each panel in representative-domain channel
   * equivalents (the full product domain when parity is zero). Every tiled
   * row still includes all channels; one complete row is the minimum.
   */
  DirectCiSigma(Context& context, const mcscf::DirectCiSigmaAction& source,
                int channel_capacity = 16,int exchange_parity = 0);
  /** @brief Share immutable device links; update_integrals must precede use. */
  DirectCiSigma(Context& context, const DirectCiSigma& reference);
  DirectCiSigma(const DirectCiSigma&) = delete;
  DirectCiSigma& operator=(const DirectCiSigma&) = delete;

  /** Enqueues work on Context's stream; host consumers must synchronize. */
  void apply(linalg::ConstMatrixView coefficients,
             linalg::MatrixView image, bool add = false) const;

  /** CPU SigmaTilePlan, exact channel envelopes and release schedule, executed
   * on device. Each distinct input and writer retains one bounded tile halo;
   * L2-sized numerical row panels reuse state channels across responses.
   * Source slabs follow ragged channel groups and images have full channels;
   * no product-sized carrier is allocated here.
   * All integral adjoints and H*response+dH*state images are device-resident.
   * workspace_bytes controls tiling only, not physical-space admission.
   */
  void apply_tiled_responses(int width,std::span<const CiTileRequest> requests,
      const CiTileReader& state,const mcscf::ProductSupport* support,
      std::size_t workspace_bytes,const CiColumnRelease& release={}) const;
  void apply_supported_responses(std::span<const CiCarrierRequest> requests,
      linalg::ConstMatrixView state,const mcscf::ProductSupport& support,
      std::size_t workspace_bytes,bool physical_image=true) const;

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
   * An optional component of response receives H*component and its integral
   * adjoint in the same sweep, sharing the state scatter and channel slabs.
   */
  void apply_direction_with_adjoint(linalg::ConstMatrixView response,
      linalg::ConstMatrixView state,const DirectCiSigma& direction,
      linalg::MatrixView combined_image,
      linalg::MatrixView one_electron_adjoint,
      linalg::MatrixView pair_kernel_adjoint,
      linalg::ConstMatrixView component = {},
      linalg::MatrixView component_image = {},
      linalg::MatrixView component_one_adjoint = {},
      linalg::MatrixView component_pair_adjoint = {}) const;

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

  /** Independent full moments left[i]^T E(generator) right from spin links.
   * Each output is overwritten, including both nonsymmetric orientations.
   * All carrier strides are respected. Only small reduction partials are
   * stored, never carrier-by-channel panels or expanded generator images.
   */
  void generator_moments(std::span<const linalg::ConstMatrixView> left,
      linalg::ConstMatrixView right,std::span<const linalg::MatrixView> outputs) const;

  /** CPU generator_adjoint_streamed counterpart with explicit left/output
   * pairs grouped by their common right reader. Every left is a device
   * n_alpha by panel_count matrix with one RHS. Readers fill requested beta
   * columns and all alpha rows. Outputs are overwritten unless add is true;
   * only the upper triangle is produced. symmetric_sum adds the transpose.
   * Exchange reduction requires full same-parity carriers across all panels.
   * One right tile and beta event schedule serve each batch's terms; dense
   * numerical storage is one carrier panel plus orbital-space reductions.
   */
  void generator_adjoint_streamed(int first,
      std::span<const CiStreamedGeneratorBatch> batches,
      std::span<const linalg::MatrixView> outputs,
      bool symmetric_sum=false,bool spin_exchange_reduced=false,bool add=false) const;

  /** One-body generator image for a beta panel, with a device reader for
   * target and spectator columns. The image is overwritten. */
  void apply_one_body_generator_columns(int first,int count,
      const CiTileReader& columns,linalg::ConstMatrixView generator,
      linalg::MatrixView image,bool alpha_only=false) const;

  /** Allocate the bounded channel slabs before stream capture. */
  void prepare(int block_width) const;
  /** Allocate the fused sigma and integral-adjoint workspace before capture. */
  void prepare_with_adjoint(int block_width) const {
    prepare_adjoint(block_width, true, true);
  }
  /** Return consumed channel panels to Context's stream-ordered pool.
   * Point-owned outputs remain valid. Invalidates this clone family's captured
   * workspace generation; graph owners must recapture before replay.
   */
  void reset_workspace() const;

  int n_alpha_strings() const noexcept { return n_alpha_; }
  int n_beta_strings() const noexcept { return n_beta_; }
  int n_orbitals() const noexcept { return n_orbitals_; }
  int n_channels() const noexcept { return n_channels_; }
  /** Same ordered E_P K_PQ E_Q kernel used by every sigma contraction. */
  linalg::ConstMatrixView channel_kernel() const noexcept {
    return {kernel_.data(),n_channels_,n_channels_,n_channels_};
  }
  /** Topology-certified common transpose parity, zero if not restricted. */
  int exchange_parity() const noexcept { return exchange_parity_; }
  std::size_t resident_bytes() const noexcept;
  /** Shared scratch changes invalidate graphs that captured its addresses.
   * All members of a clone family execute sequentially on Context's stream.
   */
  std::uint64_t workspace_generation() const noexcept { return workspace_->generation; }

 private:
  std::int64_t contraction_rows(int width) const noexcept {
    return (exchange_parity_ ? static_cast<std::int64_t>(n_alpha_)*(n_alpha_+1)/2:
        static_cast<std::int64_t>(n_alpha_)*n_beta_)*width;
  }
  void complete_exchange(linalg::MatrixView,bool average) const;
  std::size_t tile_resident_bytes() const noexcept;
  struct SpinGraph {
    mcscf::SpinConnections topology;
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
                               linalg::MatrixView pair_kernel_adjoint,double* scratch) const;
  Context* context_;
  int n_alpha_, n_beta_, n_orbitals_, n_electrons_, n_channels_, channel_capacity_;
  int exchange_parity_;
  bool shared_spin_graph_;
  Buffer<double> kernel_;
  Buffer<double> effective_one_;
  std::shared_ptr<const SpinGraph> alpha_, beta_;
  struct TileWorkspace;
  struct Workspace {
    Matrix<double> scattered,transformed,right_scattered,channel_adjoint;
    std::uint64_t generation=0;
    std::int64_t products=0;
    int moment_partitions=0;
    std::shared_ptr<TileWorkspace> tiles;
  };
  // One scratch owner per shared spin graph family, not per physical point.
  std::shared_ptr<Workspace> workspace_;
  Matrix<double>& scattered_;
  Matrix<double>& transformed_;
  Matrix<double>& right_scattered_;
  Matrix<double>& channel_adjoint_;
};

}  // namespace xmvb::cuda
