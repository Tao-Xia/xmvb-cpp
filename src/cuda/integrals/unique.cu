#include "cuda/integrals/exact.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xmvb::cuda {
namespace {

// Upload only a bounded batch, scatter into its final CSR row, and discard
// the COO staging. Never allocate a complete raw device integral stream.
template <class Index>
__global__ void upload_rows(const int* rows,
                            const int* columns,
                            const double* values,
                            std::int64_t count,
                            unsigned long long* cursor,
                            Index* column,
                            double* value,
                            double* diagonal) {
    for (auto k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < count;
         k += gridDim.x * static_cast<std::int64_t>(blockDim.x)) {
        const auto position = atomicAdd(cursor + rows[k], 1ULL);
        column[position] = columns[k];
        value[position] = values[k];
        if (rows[k] == columns[k])
            atomicAdd(diagonal + rows[k], values[k]);
    }
}

} // namespace

void ExactAoPairOperator::upload_unique(const integrals::AoPairGraph& graph) {
    auto& context = *context_;
    if (!n_integrals_)
        return;
    // Build row metadata only for this representation, never for a packed
    // operator that will not consume it. Values are not copied or transformed.
    std::vector<std::int64_t> prefix(n_pairs_ + 1, 0);
    std::size_t largest_chunk = 0;
    graph.for_each_chunk([&](auto rows, auto, auto values) {
        largest_chunk = std::max(largest_chunk, values.size());
        for (const int row : rows)
            ++prefix[row + 1];
    });
    for (int row = 0; row < n_pairs_; ++row)
        prefix[row + 1] += prefix[row];
    wide_ = n_integrals_ > std::numeric_limits<int>::max();
    values_ = Buffer<double>(context, n_integrals_);
    diagonal_ = Buffer<double>(context, n_pairs_);
    check_cuda(cudaMemsetAsync(diagonal_.data(), 0, n_pairs_ * sizeof(double), context.stream()),
               "clear unique CSR diagonal");
    const auto upload = [&]<class Index>(Buffer<Index>& offsets, Buffer<Index>& columns) {
        offsets = Buffer<Index>(context, n_pairs_ + 1);
        columns = Buffer<Index>(context, n_integrals_);
        const std::vector<Index> rows(prefix.begin(), prefix.end());
        context.copy_to_device(offsets.data(), rows.data(), rows.size() * sizeof(Index));
        Buffer<unsigned long long> cursor(context, n_pairs_);
        static_assert(sizeof(unsigned long long) == sizeof(std::int64_t));
        context.copy_to_device(cursor.data(), prefix.data(), n_pairs_ * sizeof(std::int64_t));
        int device = 0, multiprocessors = 0, active_blocks = 0;
        check_cuda(cudaGetDevice(&device), "unique CSR upload device");
        check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                   "unique CSR upload multiprocessors");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active_blocks, upload_rows<Index>, 256, 0),
                   "unique CSR upload occupancy");
        const auto population = static_cast<std::size_t>(multiprocessors) * active_blocks * 256;
        const auto capacity = std::min(
            {largest_chunk, population, response_capacity() / (2 * sizeof(int) + sizeof(double))});
        if (!capacity)
            throw std::runtime_error("unique CSR budget cannot hold an upload batch");
        Buffer<int> source_rows(context, capacity), source_columns(context, capacity);
        Buffer<double> source_values(context, capacity);
        auto* destination = columns.data();
        graph.for_each_chunk([&](auto rows, auto columns, auto values) {
            for (std::size_t begin = 0; begin < values.size(); begin += capacity) {
                const auto count = std::min(capacity, values.size() - begin);
                context.copy_to_device(
                    source_rows.data(), rows.data() + begin, count * sizeof(int));
                context.copy_to_device(
                    source_columns.data(), columns.data() + begin, count * sizeof(int));
                context.copy_to_device(
                    source_values.data(), values.data() + begin, count * sizeof(double));
                upload_rows<Index>
                    <<<static_cast<unsigned int>((count + 255) / 256), 256, 0, context.stream()>>>(
                        source_rows.data(),
                        source_columns.data(),
                        source_values.data(),
                        count,
                        cursor.data(),
                        destination,
                        values_.data(),
                        diagonal_.data());
                check_cuda(cudaGetLastError(), "upload unique CSR rows");
            }
        });
    };
    if (wide_)
        upload(wide_offsets_, wide_columns_);
    else
        upload(row_offsets_, columns_);
}

} // namespace xmvb::cuda
