#include "cuda/linalg/blas.hpp"
#include <type_traits>

namespace xmvb::cuda {
namespace {
template<class Scale>
__global__ void upper_axpy(linalg::MatrixView out,linalg::ConstMatrixView in,Scale scale) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=out.rows*out.cols) return;
  const auto row=i%out.rows,column=i/out.rows;
  double value;
  if constexpr (std::is_pointer_v<Scale>) value=*scale;
  else value=scale;
  if (row<=column) out.data[row+out.ld*column]+=value*in.data[row+in.ld*column];
}
template<class Scale>
void accumulate_upper(Context& context,linalg::MatrixView target,
    linalg::ConstMatrixView source,Scale scale) {
  if (target.rows!=source.rows || target.cols!=source.cols ||
      target.ld<target.rows || source.ld<source.rows)
    throw std::invalid_argument("CUDA upper cotangent dimensions");
  if (target.empty()) return;
  upper_axpy<<<blas_int((target.rows*target.cols+255)/256),256,0,context.stream()>>>(target,source,scale);
  check_cuda(cudaGetLastError(),"CUDA upper cotangent accumulation");
}
} // namespace

void add_upper(Context& context,linalg::MatrixView target,
    linalg::ConstMatrixView source,double scale) {
  accumulate_upper(context,target,source,scale);
}
void add_upper_device(Context& context,linalg::MatrixView target,
    linalg::ConstMatrixView source,const double* scale) {
  accumulate_upper(context,target,source,scale);
}
} // namespace xmvb::cuda
