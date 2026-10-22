#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cuda/memory/buffer.hpp"

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

struct ProductColumnMap {
  const int* rows;
  int size;
  std::int64_t offset;
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
  Buffer<ProductColumnMap> maps;
  Matrix<double> values;
  int max_rows=0;

  std::size_t bytes() const noexcept;
  linalg::MatrixView column(int beta);
  void pack(Context&,linalg::MatrixView dense,bool unpack);
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
  std::size_t bytes() const noexcept { return bytes_; }
  std::size_t budget() const noexcept { return budget_; }
  const ProductColumns& at(int panel) const;
 private:
  std::size_t budget_,bytes_=0;
  bool enabled_=false,complete_=false;
  int retained_=0;
  std::vector<ProductColumns> panels_;
  std::vector<bool> present_;
};

} // namespace xmvb::cuda
