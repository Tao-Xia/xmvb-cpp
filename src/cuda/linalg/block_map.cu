#include "cuda/linalg/block_map.hpp"

#include <map>
#include <tuple>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

__global__ void bind_vectors(const double* input,
                             double* output,
                             const std::int64_t* input_offsets,
                             const std::int64_t* output_offsets,
                             std::int64_t input_stride,
                             std::int64_t output_stride,
                             const double** inputs,
                             double** outputs,
                             int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        inputs[i] = input + input_stride * input_offsets[i];
        outputs[i] = output + output_stride * output_offsets[i];
    }
}

} // namespace

BlockMap::BlockMap(Context& context, std::span<const BlockMapEntry> entries) : context_(&context) {
    std::map<std::tuple<int, int, int>, std::vector<BlockMapEntry>> shapes;
    for (const auto& entry : entries) {
        const auto a = entry.matrix;
        if (!a.data || a.rows <= 0 || a.cols <= 0 || a.ld < a.rows || entry.input_offset < 0 ||
            entry.output_offset < 0)
            throw std::invalid_argument("CUDA block map entry dimensions");
        shapes[{blas_int(a.rows), blas_int(a.cols), blas_int(a.ld)}].push_back(entry);
    }
    for (const auto& [shape, blocks] : shapes) {
        Group group;
        std::tie(group.rows, group.cols, group.ld) = shape;
        group.count = blas_int(blocks.size());
        group.matrices = Buffer<const double*>(context, blocks.size());
        group.input_offsets = Buffer<std::int64_t>(context, blocks.size());
        group.output_offsets = Buffer<std::int64_t>(context, blocks.size());
        group.inputs = Buffer<const double*>(context, blocks.size());
        group.outputs = Buffer<double*>(context, blocks.size());
        std::vector<const double*> matrices;
        std::vector<std::int64_t> input_offsets, output_offsets;
        for (const auto& entry : blocks) {
            matrices.push_back(entry.matrix.data);
            input_offsets.push_back(entry.input_offset);
            output_offsets.push_back(entry.output_offset);
        }
        context.copy_to_device(
            group.matrices.data(), matrices.data(), matrices.size() * sizeof(double*));
        context.copy_to_device(group.input_offsets.data(),
                               input_offsets.data(),
                               input_offsets.size() * sizeof(std::int64_t));
        context.copy_to_device(group.output_offsets.data(),
                               output_offsets.data(),
                               output_offsets.size() * sizeof(std::int64_t));
        groups_.push_back(std::move(group));
    }
}

void BlockMap::apply(const double* input,
                     double* output,
                     bool transpose,
                     std::int64_t input_stride,
                     std::int64_t output_stride) const {
    if (groups_.empty())
        return;
    if (!input || !output || input == output || input_stride <= 0 || output_stride <= 0)
        throw std::invalid_argument("CUDA block map vector addresses/strides");
    const double one = 1.0, zero = 0.0;
    for (const auto& group : groups_) {
        bind_vectors<<<(group.count + 127) / 128, 128, 0, context_->stream()>>>(
            input,
            output,
            transpose ? group.output_offsets.data() : group.input_offsets.data(),
            transpose ? group.input_offsets.data() : group.output_offsets.data(),
            input_stride,
            output_stride,
            group.inputs.data(),
            group.outputs.data(),
            group.count);
        check_cuda(cudaGetLastError(), "CUDA block map vector binding");
        check_cublas(cublasDgemvBatched(context_->blas(),
                                        transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                                        group.rows,
                                        group.cols,
                                        &one,
                                        group.matrices.data(),
                                        group.ld,
                                        group.inputs.data(),
                                        1,
                                        &zero,
                                        group.outputs.data(),
                                        1,
                                        group.count),
                     "cuBLAS block map GEMV");
    }
}

} // namespace xmvb::cuda
