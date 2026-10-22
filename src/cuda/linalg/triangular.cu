#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {
__global__ void upper_axpy(linalg::MatrixView out,linalg::ConstMatrixView in,double scale) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=out.rows*out.cols) return;
  const auto row=i%out.rows,column=i/out.rows;
  if (row<=column) out.data[row+out.ld*column]+=scale*in.data[row+in.ld*column];
}
} // namespace

void add_upper(Context& context,linalg::MatrixView target,
    linalg::ConstMatrixView source,double scale) {
  if (target.rows!=source.rows || target.cols!=source.cols ||
      target.ld<target.rows || source.ld<source.rows)
    throw std::invalid_argument("CUDA upper cotangent dimensions");
  if (target.empty()) return;
  upper_axpy<<<blas_int((target.rows*target.cols+255)/256),256,0,context.stream()>>>(target,source,scale);
  check_cuda(cudaGetLastError(),"CUDA upper cotangent accumulation");
}
} // namespace xmvb::cuda
