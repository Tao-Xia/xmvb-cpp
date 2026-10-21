#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda/integrals/panel.hpp"
#include "cuda/memory/buffer.hpp"
#include "integrals/ao/input.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** Symmetric AO-pair matrix in lower block-triangular, dense B-by-B tiles.
 * Tiles are column-major and ordered by d=block_row-block_col, then block_col.
 * Missing CSR coefficients and padded lanes are zero. No numerical cutoff is
 * introduced by B: it is solely the structural tile edge.
 */
class PackedAoPairMatrix {
 public:
  struct DeviceView {
    const double* values = nullptr;
    std::int64_t pairs = 0;
    std::int64_t blocks = 0;
    int edge = 0;

#ifdef __CUDACC__
    /** Both indices must be valid physical pair indices. */
    __device__ double value(std::int64_t row, std::int64_t col) const {
      if (row < col) {
        const auto swap = row;
        row = col;
        col = swap;
      }
      // The production block edge is 512. Keep its address arithmetic exact
      // but avoid runtime 64-bit division for every integral read. Tiny graphs
      // retain their unpadded edge and use the general indexing expression.
      const std::int64_t p = edge==512 ? row>>9:row/edge;
      const std::int64_t q = edge==512 ? col>>9:col/edge;
      const std::int64_t d = p - q;
      const std::int64_t tile = d * blocks - d * (d - 1) / 2 + q;
      const auto i=edge==512 ? row&511:row%edge;
      const auto j=edge==512 ? col&511:col%edge;
      return values[tile * edge * edge + j * edge + i];
    }
#endif
  };

  /** Padded tile payload in bytes; SIZE_MAX denotes invalid/overflowing pairs. */
  [[nodiscard]] static std::size_t storage_bytes(std::int64_t pairs) noexcept;

  /** graph must be a validated symmetric CSR AO-pair graph. */
  PackedAoPairMatrix(Context& context, const integrals::AoPairGraph& graph, int n_bf);
  PackedAoPairMatrix(const PackedAoPairMatrix&) = delete;
  PackedAoPairMatrix& operator=(const PackedAoPairMatrix&) = delete;
  ~PackedAoPairMatrix() = default;

  [[nodiscard]] DeviceView view() const noexcept {
    return {values_.data(), n_pairs_, n_blocks_, edge_};
  }

  /** Peak temporary bytes for apply; one padded output tile only when the
   * requested interval has a partial edge block. No full pair panel is copied.
   */
  [[nodiscard]] std::size_t workspace_bytes(
      std::int64_t channels, std::int64_t first_row,
      std::int64_t row_count) const;

  /** target = G[first_row:first_row+target.rows, :] source, in FP64.
   * Both panels are device-resident [pair,channel] row-major with ld >= cols.
   * Source and target storage must not overlap. The target is overwritten.
   */
  void apply(ConstPairPanel source, PairPanel target,
             std::int64_t first_row = 0) const;

  /** plane(l, local_nu*n_bf+k) = (mu, first_nu+local_nu|k l).
   * The column count is a positive multiple of n_bf; the default fills all nu.
   */
  void ao_plane(int mu, linalg::MatrixView plane, int first_nu = 0) const;

 private:
  Context* context_ = nullptr;
  int n_bf_ = 0;
  int edge_ = 0;
  std::int64_t n_pairs_ = 0;
  std::int64_t n_blocks_ = 0;
  Buffer<double> values_;
};

}  // namespace xmvb::cuda
