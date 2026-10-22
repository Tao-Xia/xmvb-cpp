#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "cuda/ci/sigma.hpp"

namespace xmvb::cuda {

/** Exact symbolic rows of one spin-product column. Shared by all numerical
 * lanes, accepted states and response images; never amplitude-thresholded.
 */
struct ProductColumnPattern {
  bool full=false;
  int size=0;
  std::vector<int> rows;
  Buffer<int> device_rows;
};

/** Device counterpart of CPU ColumnTile. One packed allocation stores all
 * RHS lanes on exact column support. Dense views exist only for the current
 * bounded transform panel, never for the complete spin-product space.
 */
struct ProductColumns {
  ProductColumns()=default;
  ProductColumns(Context&,std::vector<std::shared_ptr<const ProductColumnPattern>>,
      int width);
  std::vector<std::shared_ptr<const ProductColumnPattern>> patterns;
  std::vector<std::int64_t> offsets;
  Buffer<CiColumnMap> maps;
  Matrix<double> values;
  int max_rows=0;

  std::size_t bytes() const noexcept;
  linalg::MatrixView column(int beta);
  void pack(Context&,linalg::MatrixView dense,bool unpack);
};

/** In-place parity projection of complete, resident beta panels. Panel p owns
 * beta columns [p*extent,(p+1)*extent); null panels may represent unreachable
 * columns, but every stored (alpha,beta) must have its transposed partner.
 * The last panel must be nonnull. The exact support is validated on the host;
 * all numerical values remain on device. */
void project_product_exchange(Context& context,
    std::span<ProductColumns* const> panels,int extent,int parity);

/** Read one numerical lane from compact panels into a requested dense device
 * tile. Missing columns or rows have exact zero value. Beta and alpha indices
 * are host metadata; all numerical gathering is on device. */
void read_product_columns(Context& context,
    std::span<const ProductColumns* const> panels,int extent,int lane,
    std::span<const int> betas,std::span<const int> rows,
    linalg::MatrixView destination);

/** Bounded metadata cache for repeated reads of borrowed compact panels.
 * Requests with identical beta/row indices share device descriptors across
 * numerical lanes. Source panels must outlive this reader; its pointer list
 * is copied, and changed value buffers or row maps are detected before reuse. No
 * numerical carrier is retained, and temporary host uploads remain synced.
 */
class ProductColumnReadCache {
 public:
  ProductColumnReadCache(Context&,std::span<const ProductColumns* const> panels,
      int extent,std::size_t metadata_budget);
  ~ProductColumnReadCache();
  ProductColumnReadCache(const ProductColumnReadCache&)=delete;
  ProductColumnReadCache& operator=(const ProductColumnReadCache&)=delete;
  void read(int lane,std::span<const int> betas,std::span<const int> rows,
      linalg::MatrixView destination);
  std::size_t bytes() const noexcept;
  std::size_t peak_bytes() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** CPU ProductImages retention rule: a complete accepted Y/(H-E)Y endpoint
 * must fit its payload allowance. Partial endpoints are not valid readers.
 * When it does not fit, callers keep their source representation and use the
 * same streamed sweep; this container never allocates a full carrier.
 */
class ProductImages {
 public:
  explicit ProductImages(std::size_t budget):budget_(budget) {}
  void begin(int panels);
  bool retain(int panel,ProductColumns columns);
  void finish();
  bool complete() const noexcept { return complete_; }
  int size() const noexcept { return static_cast<int>(panels_.size()); }
  std::size_t bytes() const noexcept { return bytes_; }
  std::size_t budget() const noexcept { return budget_; }
  const ProductColumns& at(int panel) const;
  ProductColumns& at(int panel);
 private:
  std::size_t budget_,bytes_=0;
  bool enabled_=false,complete_=false;
  int retained_=0;
  std::vector<ProductColumns> panels_;
  std::vector<bool> present_;
};

} // namespace xmvb::cuda
