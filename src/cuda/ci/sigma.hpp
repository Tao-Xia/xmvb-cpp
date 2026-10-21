#pragma once

#include <cstddef>

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
  DirectCiSigma(const DirectCiSigma&) = delete;
  DirectCiSigma& operator=(const DirectCiSigma&) = delete;

  /** Enqueues work on Context's stream; host consumers must synchronize. */
  void apply(linalg::ConstMatrixView coefficients,
             linalg::MatrixView image, bool add = false) const;

  /** Allocate the bounded channel slabs before stream capture. */
  void prepare(int block_width) const;

  int n_alpha_strings() const noexcept { return n_alpha_; }
  int n_beta_strings() const noexcept { return n_beta_; }
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

  Context* context_;
  int n_alpha_, n_beta_, n_channels_, maximum_channel_tile_;
  bool shared_spin_graph_;
  Buffer<double> kernel_;
  SpinGraph alpha_, beta_;
  // Reused on the same stream, not allocated/freed at every sigma call.
  mutable Matrix<double> scattered_, transformed_;
};

}  // namespace xmvb::cuda
