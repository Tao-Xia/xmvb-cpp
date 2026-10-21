#pragma once

#include <span>
#include <vector>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

/** Borrowed matrix mapping one input segment to one disjoint output segment. */
struct BlockMapEntry {
  linalg::ConstMatrixView matrix;
  std::int64_t input_offset=0,output_offset=0;
};

/** Group independent GEMVs by shape. Matrices are borrowed, never duplicated.
 * Only pointer/offset metadata is resident. apply() allocates no workspace;
 * all entries overwrite their output segment. The caller owns the matrices
 * and guarantees disjoint outputs and nonaliasing input/output vectors.
 */
class BlockMap {
 public:
  BlockMap()=default;
  BlockMap(Context&,std::span<const BlockMapEntry>);
  BlockMap(BlockMap&&)=default;
  BlockMap& operator=(BlockMap&&)=default;

  /** Offsets are multiplied by the supplied strides. Transpose swaps the
   * entry's input/output offsets as well as transposing its matrix.
   */
  void apply(const double* input,double* output,bool transpose=false,
      std::int64_t input_stride=1,std::int64_t output_stride=1) const;

 private:
  struct Group {
    int rows=0,cols=0,ld=0,count=0;
    Buffer<const double*> matrices;
    Buffer<std::int64_t> input_offsets,output_offsets;
    mutable Buffer<const double*> inputs;
    mutable Buffer<double*> outputs;
  };
  Context* context_=nullptr;
  std::vector<Group> groups_;
};

} // namespace xmvb::cuda
