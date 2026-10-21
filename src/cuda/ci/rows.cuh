#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace xmvb::cuda {

/** One representative in beta-column order, with alpha contiguous. */
struct ProductRow {
  int alpha,beta;
  std::int64_t rhs,column;
  double weight;
};

/** Decode without a stored product-sized index map. Nonzero parity requires
 * identical spin spaces and a topology-certified transpose sector.
 */
__device__ inline ProductRow product_row(std::int64_t index,int na,int nb,int parity) {
  if (!parity) {
    const auto column=index/na;
    return {static_cast<int>(index%na),static_cast<int>(column%nb),column/nb,column,1.0};
  }
  const auto triangle=static_cast<std::int64_t>(na)*(na+1)/2;
  const auto rhs=index/triangle,local=index%triangle;
  const auto reverse=triangle-1-local;
  auto k=static_cast<std::int64_t>((sqrt(8.0*reverse+1.0)-1.0)*0.5);
  while (k*(k+1)/2>reverse) --k;
  while ((k+1)*(k+2)/2<=reverse) ++k;
  const int b=na-1-static_cast<int>(k);
  const auto prefix=static_cast<std::int64_t>(b)*na-static_cast<std::int64_t>(b)*(b-1)/2;
  const int a=b+static_cast<int>(local-prefix);
  return {a,b,rhs,rhs*nb+b,a==b ? 1.0:2.0};
}

} // namespace xmvb::cuda
