#include "core/index.hpp"
#include "cuda/vb/transform.hpp"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/upload.hpp"
#include "cuda/vb/exterior.cuh"
#include "mcscf/ci/combinations.hpp"

namespace xmvb::cuda {
namespace {

using Edge = vb::ExteriorOrbitalTransform::ShearEdge;

template <bool Consecutive>
__global__ void selected_identity(const std::int64_t* rows,
                                  std::int64_t first,
                                  std::int64_t count,
                                  linalg::MatrixView output) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < count)
        output.data[(Consecutive ? first + row : rows[row]) + output.ld * row] = 1.0;
}

// The ordered elementary-factor coefficients are a small triangular problem,
// not a product-space contraction. Use exactly the CPU elimination ordering.
__global__ void factor_upper(linalg::ConstMatrixView upper,
                             linalg::MatrixView unit,
                             double* coefficients,
                             linalg::ConstMatrixView delta = {},
                             linalg::MatrixView delta_unit = {},
                             double* delta_coefficients = nullptr) {
    if (blockIdx.x || threadIdx.x)
        return;
    const int n = upper.rows;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (delta.data)
                delta_unit.data[i + delta_unit.ld * j] =
                    delta.data[i + delta.ld * j] / upper.data[j + upper.ld * j] -
                    upper.data[i + upper.ld * j] *
                        (delta.data[j + delta.ld * j] /
                         (upper.data[j + upper.ld * j] * upper.data[j + upper.ld * j]));
            unit.data[i + unit.ld * j] =
                upper.data[i + upper.ld * j] / upper.data[j + upper.ld * j];
        }
    int step = 0;
    for (int removed = 1; removed < n; ++removed)
        for (int inserted = removed - 1; inserted >= 0; --inserted) {
            const double coefficient = unit.data[inserted + unit.ld * removed];
            if (coefficients)
                coefficients[step] = coefficient;
            const double dc =
                delta.data ? delta_unit.data[inserted + delta_unit.ld * removed] : 0.0;
            if (delta.data)
                delta_coefficients[step] = dc;
            ++step;
            for (int i = 0; i < n; ++i) {
                if (delta.data)
                    delta_unit.data[i + delta_unit.ld * removed] -=
                        dc * unit.data[i + unit.ld * inserted] +
                        coefficient * delta_unit.data[i + delta_unit.ld * inserted];
                unit.data[i + unit.ld * removed] -= coefficient * unit.data[i + unit.ld * inserted];
            }
        }
}

__global__ void update_scales(linalg::ConstMatrixView upper,
                              const std::uint64_t* occupations,
                              double* scales,
                              double* inverse,
                              std::int64_t dimension,
                              linalg::ConstMatrixView delta = {},
                              double* delta_scales = nullptr) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= dimension)
        return;
    double value = 1.0, logarithmic = 0.0;
    for (int p = 0; p < upper.rows; ++p)
        if (occupations[i] & (std::uint64_t{1} << p)) {
            value *= upper.data[p + upper.ld * p];
            if (delta.data)
                logarithmic += delta.data[p + delta.ld * p] / upper.data[p + upper.ld * p];
        }
    if (scales)
        scales[i] = value;
    if (inverse)
        inverse[i] = 1.0 / value;
    if (delta.data)
        delta_scales[i] = value * logarithmic;
}

__global__ void inverse_scale(const double* scales, double* inverse, int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        inverse[i] = 1.0 / scales[i];
}

unsigned int grid_size(std::int64_t count) {
    constexpr std::int64_t threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("CUDA exterior kernel grid is too large");
    return static_cast<unsigned int>(blocks);
}

unsigned int tile_grid(std::int64_t tiles) {
    if (tiles > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("CUDA exterior tile grid is too large");
    return static_cast<unsigned int>(tiles);
}

int left_tile_lanes(std::int64_t rhs, std::int64_t dimension, int shared_values) {
    return static_cast<int>(
        std::min<std::int64_t>({32, rhs, std::max<std::int64_t>(1, shared_values / dimension)}));
}

int right_tile_lanes(std::int64_t dimension, int shared_values, int arrays = 1) {
    int lanes = 1;
    while (lanes < 32 && 2 * lanes <= shared_values / (arrays * dimension))
        lanes *= 2;
    return lanes;
}

// Each spin column/row tile is independent. Retain it in block-local memory
// for the complete ordered exterior product: sources and targets of one
// elementary shear are disjoint, and the barrier orders consecutive shears.
// This is not a new factorization, truncation, or full product-space cache.
template <bool Right, bool Tangent = false, bool Indirect = false>
__global__ void exterior_tile(linalg::MatrixView panel,
                              const Edge* edges,
                              const ExteriorTransform::Shear* schedule,
                              int steps,
                              const double* coefficients,
                              const double* scales,
                              std::int64_t dimension,
                              int lanes,
                              bool adjoint,
                              bool inverse,
                              const std::uint64_t* allowed = nullptr,
                              int support_columns = 0,
                              int words = 0,
                              linalg::MatrixView tangent = {},
                              const double* delta_scales = nullptr,
                              const double* delta_coefficients = nullptr,
                              double* const* columns = nullptr,
                              double* const* tangent_columns = nullptr) {
    static_assert(!Indirect || Right);
    extern __shared__ double values[];
    const auto row_tiles = (panel.rows + lanes - 1) / lanes;
    const auto column = Right ? (Indirect ? 0 : blockIdx.x / row_tiles * dimension)
                              : (support_columns ? blockIdx.x % support_columns : blockIdx.x);
    const auto row_begin = Right ? (Indirect ? static_cast<std::int64_t>(blockIdx.x) * lanes
                                             : blockIdx.x % row_tiles * lanes)
                                 : 0;
    // Supported left panels are RHS-major: lane l owns support column
    // `column` in a different RHS, so every lane uses the same edge mask.
    const auto left_begin =
        support_columns
            ? (static_cast<std::int64_t>(blockIdx.x) / support_columns * lanes) * support_columns +
                  column
            : static_cast<std::int64_t>(blockIdx.x) * lanes;
    const int left_stride = support_columns ? support_columns : 1;
    const int work = dimension * lanes;
    for (int k = threadIdx.x; k < work; k += blockDim.x) {
        const std::int64_t spin = k / lanes, lane = k % lanes;
        const auto row = Right ? row_begin + lane : spin;
        const auto col =
            Right ? column + spin : left_begin + static_cast<std::int64_t>(lane) * left_stride;
        const bool valid = row < panel.rows && col < panel.cols;
        double v = 0.0;
        if (valid) {
            if constexpr (Indirect)
                v = columns[spin][row];
            else
                v = panel.data[row + panel.ld * col];
        }
        if constexpr (Tangent) {
            double dv = 0.0;
            if (valid) {
                if constexpr (Indirect)
                    dv = tangent_columns[spin][row];
                else
                    dv = tangent.data[row + tangent.ld * col];
            }
            if (!adjoint)
                dv = scales[spin] * dv + delta_scales[spin] * v;
            values[work + k] = dv;
        }
        if (allowed && !(allowed[(col % support_columns) * words + spin / 64] &
                         (std::uint64_t{1} << (spin % 64))))
            v = 0;
        if (adjoint == inverse)
            v *= scales[spin];
        values[k] = v;
    }
    __syncthreads();
    for (int step = 0; step < steps; ++step) {
        const std::int64_t index = adjoint != inverse ? steps - 1 - step : step;
        const auto shear = schedule[index];
        for (int k = threadIdx.x; k < shear.count * lanes; k += blockDim.x) {
            const auto edge = edges[shear.offset + k / lanes];
            const int lane = k % lanes;
            const std::int64_t source = adjoint ? edge.target : edge.source;
            const std::int64_t target = adjoint ? edge.source : edge.target;
            if (allowed) {
                const auto* mask = allowed + (column % support_columns) * words;
                if (!(mask[source / 64] & (std::uint64_t{1} << (source % 64))) ||
                    !(mask[target / 64] & (std::uint64_t{1} << (target % 64))))
                    continue;
            }
            if constexpr (Tangent)
                values[work + target * lanes + lane] +=
                    edge.sign * (delta_coefficients[index] * values[source * lanes + lane] +
                                 coefficients[index] * values[work + source * lanes + lane]);
            values[target * lanes + lane] += (inverse ? -1.0 : 1.0) * coefficients[index] *
                                             edge.sign * values[source * lanes + lane];
        }
        __syncthreads();
    }
    for (int k = threadIdx.x; k < work; k += blockDim.x) {
        const std::int64_t spin = k / lanes, lane = k % lanes;
        const auto row = Right ? row_begin + lane : spin;
        const auto col =
            Right ? column + spin : left_begin + static_cast<std::int64_t>(lane) * left_stride;
        double v = values[k];
        if constexpr (Tangent) {
            double dv = values[work + k];
            if (adjoint)
                dv = scales[spin] * dv + delta_scales[spin] * v;
            if (row < panel.rows && col < panel.cols) {
                if constexpr (Indirect)
                    tangent_columns[spin][row] = dv;
                else
                    tangent.data[row + tangent.ld * col] = dv;
            }
        }
        if (adjoint != inverse)
            v *= scales[spin];
        if (row < panel.rows && col < panel.cols) {
            if constexpr (Indirect)
                columns[spin][row] = v;
            else
                panel.data[row + panel.ld * col] = v;
        }
    }
}

__global__ void scale_direction(linalg::MatrixView primal,
                                linalg::MatrixView tangent,
                                const double* scales,
                                const double* delta_scales) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= primal.rows * primal.cols)
        return;
    const auto a = i % primal.rows, b = i / primal.rows;
    auto& value = primal.data[a + primal.ld * b];
    auto& delta = tangent.data[a + tangent.ld * b];
    delta = scales[a] * delta + delta_scales[a] * value;
    value *= scales[a];
}

__global__ void
restrict_columns(linalg::MatrixView panel, const std::uint64_t* allowed, int columns, int words) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= panel.rows * panel.cols)
        return;
    const auto row = i % panel.rows, column = i / panel.rows;
    if (!(allowed[(column % columns) * words + row / 64] & (std::uint64_t{1} << (row % 64))))
        panel.data[row + panel.ld * column] = 0.0;
}

std::vector<double*> tile_columns(std::span<const linalg::MatrixView> tiles,
                                  std::int64_t dimension,
                                  std::int64_t* rows) {
    if (tiles.empty())
        throw std::invalid_argument("empty CUDA exterior right tiles");
    *rows = tiles.front().rows;
    if (*rows <= 0)
        throw std::invalid_argument("invalid CUDA exterior tile rows");
    std::vector<double*> columns;
    columns.reserve(dimension);
    for (const auto tile : tiles) {
        if (tile.rows != *rows || tile.cols <= 0 || tile.cols > dimension ||
            (dimension > 1 && tile.cols == dimension) || tile.ld < tile.rows || !tile.data ||
            tile.cols > dimension - core::checked_index<std::int64_t>(columns.size()))
            throw std::invalid_argument("invalid CUDA exterior right tile layout");
        for (std::int64_t beta = 0; beta < tile.cols; ++beta)
            columns.push_back(tile.data + beta * tile.ld);
    }
    if (columns.size() != static_cast<std::size_t>(dimension))
        throw std::invalid_argument("CUDA exterior right tiles do not cover beta dimension");
    return columns;
}

} // namespace

ExteriorTransform::ExteriorTransform(Context& context,
                                     const vb::ExteriorOrbitalTransform& source,
                                     std::size_t shared_memory_bytes)
    : context_(&context), dimension_(source.dimension()), n_orbitals_(source.n_orbitals()),
      scales_(context, static_cast<std::size_t>(source.dimension())),
      inverse_scales_(context, static_cast<std::size_t>(source.dimension())),
      shear_coefficients_(context, source.shear_count()),
      topology_(std::make_shared<Topology>(context, source.occupations().size())),
      unit_upper_(context, source.n_orbitals(), source.n_orbitals()) {
    if (dimension_ <= 0 || source.determinant_scales().size() != dimension_)
        throw std::invalid_argument("invalid CUDA exterior transform dimensions");
    int device = 0, shared_bytes = 0;
    check_cuda(cudaGetDevice(&device), "query exterior device");
    check_cuda(cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device),
               "query exterior block capacity");
    int optin_bytes = 0;
    check_cuda(
        cudaDeviceGetAttribute(&optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        "query exterior opt-in block capacity");
    const int capacity = std::max<std::int64_t>(shared_bytes, optin_bytes);
    shared_values_ = std::min<std::size_t>(shared_memory_bytes, capacity) / sizeof(double);
    tile_values_ = std::min<std::size_t>(shared_memory_bytes, shared_bytes) / sizeof(double);
    if (capacity > shared_bytes) {
        const auto enable = [capacity](const void* kernel) {
            check_cuda(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, capacity),
                "enable exterior shared capacity");
        };
        enable(reinterpret_cast<const void*>(exterior_tile<false>));
        enable(reinterpret_cast<const void*>(exterior_tile<false, true>));
        enable(reinterpret_cast<const void*>(exterior_tile<true>));
        enable(reinterpret_cast<const void*>(exterior_tile<true, false, true>));
        enable(reinterpret_cast<const void*>(exterior_tile<true, true, true>));
    }
    std::vector<std::uint64_t> choose((n_orbitals_ + 1) * (n_orbitals_ + 1));
    for (int n = 0; n <= n_orbitals_; ++n)
        for (int k = 0; k <= n_orbitals_; ++k)
            choose[n * (n_orbitals_ + 1) + k] = mcscf::detail::spin_choose(n, k);
    topology_->choose = Buffer<std::uint64_t>(context, choose.size());
    context.copy_to_device(
        topology_->choose.data(), choose.data(), choose.size() * sizeof(std::uint64_t));
    std::vector<std::int64_t> order(dimension_);
    bool canonical = true;
    for (int rank = 0; rank < dimension_; ++rank) {
        order[rank] = source.strings().index_from_rank(rank);
        canonical = canonical && order[rank] == rank;
    }
    if (!canonical) {
        topology_->order = Buffer<std::int64_t>(context, order.size());
        context.copy_to_device(
            topology_->order.data(), order.data(), order.size() * sizeof(std::int64_t));
    }
    context.copy_to_device(
        scales_.data(), source.determinant_scales().data(), scales_.size() * sizeof(double));
    inverse_scale<<<grid_size(dimension_), 256, 0, context.stream()>>>(
        scales_.data(), inverse_scales_.data(), dimension_);
    check_cuda(cudaGetLastError(), "inverse exterior scaling");
    std::vector<Edge> all_edges;
    std::vector<double> coefficients;
    coefficients.reserve(source.shear_count());
    topology_->shears.reserve(source.shear_count());
    for (std::size_t i = 0; i < source.shear_count(); ++i) {
        const auto edges = source.shear_edges(i);
        if (edges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("CUDA exterior shear is too large");
        topology_->shears.push_back(
            {all_edges.size(), core::checked_index<std::int64_t>(edges.size())});
        coefficients.push_back(source.shear_coefficient(i));
        all_edges.insert(all_edges.end(), edges.begin(), edges.end());
    }
    topology_->edges = Buffer<Edge>(context, all_edges.size());
    topology_->schedule = Buffer<Shear>(context, topology_->shears.size());
    context.copy_to_device(topology_->schedule.data(),
                           topology_->shears.data(),
                           topology_->shears.size() * sizeof(Shear));
    context.copy_to_device(
        topology_->edges.data(), all_edges.data(), all_edges.size() * sizeof(Edge));
    context.copy_to_device(
        shear_coefficients_.data(), coefficients.data(), coefficients.size() * sizeof(double));
    context.copy_to_device(topology_->occupations.data(),
                           source.occupations().data(),
                           topology_->occupations.size() * sizeof(std::uint64_t));
}

ExteriorTransform::ExteriorTransform(Context& context, const ExteriorTransform& reference)
    : context_(&context), dimension_(reference.dimension_), n_orbitals_(reference.n_orbitals_),
      shared_values_(reference.shared_values_), tile_values_(reference.tile_values_),
      scales_(context, reference.scales_.size()),
      inverse_scales_(context, reference.inverse_scales_.size()),
      shear_coefficients_(context, reference.shear_coefficients_.size()),
      topology_(reference.topology_),
      unit_upper_(context, reference.unit_upper_.rows(), reference.unit_upper_.cols()) {
    if (reference.context_ != &context)
        throw std::invalid_argument("CUDA exterior frame clone requires same context");
    const auto copy_numeric = [&](double* target, const double* source, std::size_t count) {
        if (count)
            check_cuda(cudaMemcpyAsync(target,
                                       source,
                                       count * sizeof(double),
                                       cudaMemcpyDeviceToDevice,
                                       context.stream()),
                       "clone CUDA exterior frame");
    };
    copy_numeric(scales_.data(), reference.scales_.data(), scales_.size());
    copy_numeric(inverse_scales_.data(), reference.inverse_scales_.data(), inverse_scales_.size());
    copy_numeric(shear_coefficients_.data(),
                 reference.shear_coefficients_.data(),
                 shear_coefficients_.size());
}

ExteriorTransform::~ExteriorTransform() = default;

std::size_t ExteriorTransform::right_pointer_upload_bytes() const noexcept {
    return right_pointer_upload_->size();
}

double* const* ExteriorTransform::upload_right_columns(const std::vector<double*>& columns) const {
    if (!right_pointer_upload_)
        right_pointer_upload_ = std::make_unique<PinnedUpload>(*context_);
    if (columns != right_columns_) {
        right_columns_.clear();
        right_pointer_upload_->assign(std::as_bytes(std::span{columns}));
        right_columns_ = columns;
    }
    return reinterpret_cast<double* const*>(right_pointer_upload_->data());
}

void ExteriorTransform::update_upper(linalg::ConstMatrixView upper) {
    if (upper.rows != n_orbitals_ || upper.cols != n_orbitals_ || upper.ld < upper.rows ||
        !upper.data)
        throw std::invalid_argument("CUDA exterior frame dimensions");
    factor_upper<<<1, 1, 0, context_->stream()>>>(
        upper, unit_upper_.view(), shear_coefficients_.data());
    update_scales<<<grid_size(dimension_), 256, 0, context_->stream()>>>(
        upper, topology_->occupations.data(), scales_.data(), inverse_scales_.data(), dimension_);
    check_cuda(cudaGetLastError(), "CUDA exterior frame update");
}

void ExteriorTransform::apply_left(linalg::MatrixView coefficients,
                                   bool adjoint,
                                   bool inverse) const {
    if (coefficients.rows != dimension_ || coefficients.cols <= 0 ||
        coefficients.ld < coefficients.rows || coefficients.data == nullptr)
        throw std::invalid_argument("invalid CUDA exterior left panel");
    if (!topology_->shears.empty() && dimension_ <= shared_values_) {
        exterior_tile<false>
            <<<tile_grid(coefficients.cols),
               256,
               static_cast<std::size_t>(dimension_) * sizeof(double),
               context_->stream()>>>(coefficients,
                                     topology_->edges.data(),
                                     topology_->schedule.data(),
                                     topology_->shears.size(),
                                     shear_coefficients_.data(),
                                     inverse ? inverse_scales_.data() : scales_.data(),
                                     dimension_,
                                     1,
                                     adjoint,
                                     inverse);
        check_cuda(cudaGetLastError(), "CUDA fused exterior left tile");
        return;
    }
    const auto scale = [&] {
        // cuBLAS permits in-place dgmm when lda == ldc, as here.
        check_cublas(cublasDdgmm_64(context_->blas(),
                                    CUBLAS_SIDE_LEFT,
                                    coefficients.rows,
                                    coefficients.cols,
                                    coefficients.data,
                                    coefficients.ld,
                                    inverse ? inverse_scales_.data() : scales_.data(),
                                    1,
                                    coefficients.data,
                                    coefficients.ld),
                     "cublasDdgmm exterior left scaling");
    };
    if (adjoint == inverse) {
        scale();
    }
    for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
        const int q = (adjoint != inverse) ? n_orbitals_ - 1 - step : step + 1;
        detail::exterior_column<false>
            <<<grid_size(coefficients.rows * coefficients.cols), 256, 0, context_->stream()>>>(
                coefficients,
                {},
                index_view(),
                dimension_,
                q,
                shear_coefficients_.data(),
                nullptr,
                adjoint,
                inverse);
    }
    if (adjoint != inverse) {
        scale();
    }
}

void ExteriorTransform::apply_supported_left(linalg::MatrixView panel,
                                             const std::uint64_t* allowed,
                                             int columns,
                                             bool adjoint) const {
    if (!allowed || columns <= 0 || panel.rows != dimension_ || panel.ld < panel.rows ||
        panel.cols <= 0 || panel.cols % columns)
        throw std::invalid_argument("invalid CUDA supported exterior pattern");
    const int words = (dimension_ + 63) / 64;
    if (dimension_ <= shared_values_) {
        const auto rhs = panel.cols / columns;
        const int lanes = left_tile_lanes(rhs, dimension_, tile_values_);
        const auto tiles = static_cast<std::int64_t>(columns) * ((rhs - 1) / lanes + 1);
        exterior_tile<false><<<tile_grid(tiles),
                               256,
                               static_cast<std::size_t>(dimension_) * lanes * sizeof(double),
                               context_->stream()>>>(panel,
                                                     topology_->edges.data(),
                                                     topology_->schedule.data(),
                                                     topology_->shears.size(),
                                                     shear_coefficients_.data(),
                                                     scales_.data(),
                                                     dimension_,
                                                     lanes,
                                                     adjoint,
                                                     false,
                                                     allowed,
                                                     columns,
                                                     words);
    } else {
        restrict_columns<<<grid_size(panel.rows * panel.cols), 256, 0, context_->stream()>>>(
            panel, allowed, columns, words);
        const auto scale = [&] {
            check_cublas(cublasDdgmm_64(context_->blas(),
                                        CUBLAS_SIDE_LEFT,
                                        panel.rows,
                                        panel.cols,
                                        panel.data,
                                        panel.ld,
                                        scales_.data(),
                                        1,
                                        panel.data,
                                        panel.ld),
                         "supported exterior scaling");
        };
        if (!adjoint)
            scale();
        for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
            const int q = (adjoint) ? n_orbitals_ - 1 - step : step + 1;
            detail::exterior_column<false>
                <<<grid_size(panel.rows * panel.cols), 256, 0, context_->stream()>>>(
                    panel,
                    {},
                    index_view(),
                    dimension_,
                    q,
                    shear_coefficients_.data(),
                    nullptr,
                    adjoint,
                    false,
                    allowed,
                    columns,
                    words);
        }
        if (adjoint)
            scale();
    }
    check_cuda(cudaGetLastError(), "supported exterior action");
}

ExteriorDirection ExteriorTransform::direction(linalg::ConstMatrixView upper,
                                               linalg::ConstMatrixView delta_upper) const {
    if (upper.rows != n_orbitals_ || upper.cols != n_orbitals_ || upper.ld < upper.rows ||
        delta_upper.rows != n_orbitals_ || delta_upper.cols != n_orbitals_ ||
        delta_upper.ld < delta_upper.rows || !upper.data || !delta_upper.data)
        throw std::invalid_argument("invalid CUDA exterior direction frame");
    auto& c = *context_;
    ExteriorDirection result{Buffer<double>(c, dimension_),
                             Buffer<double>(c, topology_->shears.size())};
    Matrix<double> unit(c, n_orbitals_, n_orbitals_), delta_unit(c, n_orbitals_, n_orbitals_);
    factor_upper<<<1, 1, 0, c.stream()>>>(
        upper, unit.view(), nullptr, delta_upper, delta_unit.view(), result.shears.data());
    update_scales<<<grid_size(dimension_), 256, 0, c.stream()>>>(upper,
                                                                 topology_->occupations.data(),
                                                                 nullptr,
                                                                 nullptr,
                                                                 dimension_,
                                                                 delta_upper,
                                                                 result.scales.data());
    check_cuda(cudaGetLastError(), "CUDA exterior direction factors");
    return result;
}

void ExteriorTransform::apply_directional_left(const ExteriorDirection& direction,
                                               linalg::MatrixView primal,
                                               linalg::MatrixView tangent,
                                               bool adjoint) const {
    if (primal.rows != dimension_ || primal.cols <= 0 || primal.ld < primal.rows ||
        tangent.rows != primal.rows || tangent.cols != primal.cols || tangent.ld < tangent.rows ||
        !primal.data || !tangent.data || primal.data == tangent.data ||
        direction.scales.size() != dimension_ ||
        direction.shears.size() != topology_->shears.size())
        throw std::invalid_argument("invalid CUDA directional exterior panel");
    auto& c = *context_;
    if (dimension_ <= shared_values_ / 2) {
        exterior_tile<false, true>
            <<<grid_size(primal.cols * 256), 256, 2ULL * dimension_ * sizeof(double), c.stream()>>>(
                primal,
                topology_->edges.data(),
                topology_->schedule.data(),
                topology_->shears.size(),
                shear_coefficients_.data(),
                scales_.data(),
                dimension_,
                1,
                adjoint,
                false,
                nullptr,
                0,
                0,
                tangent,
                direction.scales.data(),
                direction.shears.data());
    } else {
        const auto scale = [&] {
            scale_direction<<<grid_size(primal.rows * primal.cols), 256, 0, c.stream()>>>(
                primal, tangent, scales_.data(), direction.scales.data());
        };
        if (!adjoint)
            scale();
        for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
            const int q = (adjoint) ? n_orbitals_ - 1 - step : step + 1;
            detail::exterior_column<false, false, true>
                <<<grid_size(primal.rows * primal.cols), 256, 0, context_->stream()>>>(
                    primal,
                    tangent,
                    index_view(),
                    dimension_,
                    q,
                    shear_coefficients_.data(),
                    direction.shears.data(),
                    adjoint,
                    false);
        }
        if (adjoint)
            scale();
    }
    check_cuda(cudaGetLastError(), "CUDA directional exterior action");
}

void ExteriorTransform::apply_right(linalg::MatrixView coefficients,
                                    bool adjoint,
                                    bool inverse) const {
    if (coefficients.rows <= 0 || coefficients.cols <= 0 || coefficients.cols % dimension_ != 0 ||
        coefficients.ld < coefficients.rows || coefficients.data == nullptr)
        throw std::invalid_argument("invalid CUDA exterior right panel");
    const std::int64_t rhs = coefficients.cols / dimension_;
    if (!topology_->shears.empty() && dimension_ <= shared_values_) {
        const int lanes = right_tile_lanes(dimension_, tile_values_);
        const auto tiles = ((coefficients.rows + lanes - 1) / lanes) * rhs;
        exterior_tile<true>
            <<<grid_size(tiles * 256),
               256,
               static_cast<std::size_t>(dimension_) * lanes * sizeof(double),
               context_->stream()>>>(coefficients,
                                     topology_->edges.data(),
                                     topology_->schedule.data(),
                                     topology_->shears.size(),
                                     shear_coefficients_.data(),
                                     inverse ? inverse_scales_.data() : scales_.data(),
                                     dimension_,
                                     lanes,
                                     adjoint,
                                     inverse);
        check_cuda(cudaGetLastError(), "CUDA fused exterior right tile");
        return;
    }
    const auto scale = [&] {
        for (std::int64_t block = 0; block < rhs; ++block) {
            const auto panel = coefficients.columns(block * dimension_, dimension_);
            check_cublas(cublasDdgmm_64(context_->blas(),
                                        CUBLAS_SIDE_RIGHT,
                                        panel.rows,
                                        panel.cols,
                                        panel.data,
                                        panel.ld,
                                        inverse ? inverse_scales_.data() : scales_.data(),
                                        1,
                                        panel.data,
                                        panel.ld),
                         "cublasDdgmm exterior right scaling");
        }
    };
    if (adjoint == inverse) {
        scale();
    }
    for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
        const int q = (adjoint != inverse) ? n_orbitals_ - 1 - step : step + 1;
        detail::exterior_column<true>
            <<<grid_size(coefficients.rows * coefficients.cols), 256, 0, context_->stream()>>>(
                coefficients,
                {},
                index_view(),
                dimension_,
                q,
                shear_coefficients_.data(),
                nullptr,
                adjoint,
                inverse);
    }
    if (adjoint != inverse) {
        scale();
    }
}

void ExteriorTransform::apply_right_tiles(std::span<const linalg::MatrixView> tiles,
                                          bool adjoint,
                                          bool inverse) const {
    const nvtx3::scoped_range range{"exterior.right_tiles"};
    std::int64_t rows = 0;
    const auto host_columns = tile_columns(tiles, dimension_, &rows);
    auto* const columns = upload_right_columns(host_columns);
    apply_right_columns(columns, rows, adjoint, inverse);
}

void ExteriorTransform::apply_right_columns(double* const* columns,
                                            std::int64_t rows,
                                            bool adjoint,
                                            bool inverse) const {
    if (dimension_ <= shared_values_) {
        const int lanes = right_tile_lanes(dimension_, tile_values_);
        const auto row_tiles = (rows + lanes - 1) / lanes;
        exterior_tile<true, false, true>
            <<<tile_grid(row_tiles),
               256,
               static_cast<std::size_t>(dimension_) * lanes * sizeof(double),
               context_->stream()>>>({nullptr, rows, dimension_, 0},
                                     topology_->edges.data(),
                                     topology_->schedule.data(),
                                     topology_->shears.size(),
                                     shear_coefficients_.data(),
                                     inverse ? inverse_scales_.data() : scales_.data(),
                                     dimension_,
                                     lanes,
                                     adjoint,
                                     inverse,
                                     nullptr,
                                     0,
                                     0,
                                     {},
                                     nullptr,
                                     nullptr,
                                     columns);
        check_cuda(cudaGetLastError(), "CUDA fused exterior right tiles");
        return;
    }
    const auto scale = [&] {
        detail::scale_exterior_tiles<<<grid_size(rows * dimension_), 256, 0, context_->stream()>>>(
            columns, nullptr, rows, dimension_, inverse ? inverse_scales_.data() : scales_.data());
    };
    if (adjoint == inverse)
        scale();
    for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
        const int q = adjoint != inverse ? n_orbitals_ - 1 - step : step + 1;
        detail::exterior_column<true, true>
            <<<grid_size(rows * dimension_), 256, 0, context_->stream()>>>(
                {nullptr, rows, dimension_, 0},
                {},
                index_view(),
                dimension_,
                q,
                shear_coefficients_.data(),
                nullptr,
                adjoint,
                inverse,
                nullptr,
                0,
                0,
                columns);
    }
    if (adjoint != inverse)
        scale();
    check_cuda(cudaGetLastError(), "CUDA exterior right column groups");
}

void ExteriorTransform::apply_directional_right_tiles(const ExteriorDirection& direction,
                                                      std::span<const linalg::MatrixView> primal,
                                                      std::span<const linalg::MatrixView> tangent,
                                                      bool adjoint) const {
    const nvtx3::scoped_range range{"exterior.directional_right_tiles"};
    if (primal.size() != tangent.size() ||
        direction.scales.size() != static_cast<std::size_t>(dimension_) ||
        direction.shears.size() != topology_->shears.size())
        throw std::invalid_argument("invalid CUDA directional exterior right tiles");
    std::int64_t rows = 0, tangent_rows = 0;
    auto host_columns = tile_columns(primal, dimension_, &rows);
    auto tangent_columns = tile_columns(tangent, dimension_, &tangent_rows);
    if (rows != tangent_rows)
        throw std::invalid_argument("CUDA directional exterior tile rows disagree");
    for (std::size_t tile = 0; tile < primal.size(); ++tile)
        if (primal[tile].cols != tangent[tile].cols || primal[tile].data == tangent[tile].data)
            throw std::invalid_argument("CUDA directional exterior tile layouts disagree");
    host_columns.insert(host_columns.end(), tangent_columns.begin(), tangent_columns.end());
    auto* const columns = upload_right_columns(host_columns);
    if (!topology_->shears.empty() && dimension_ <= shared_values_ / 2) {
        const int lanes = right_tile_lanes(dimension_, tile_values_, 2);
        const auto row_tiles = (rows + lanes - 1) / lanes;
        exterior_tile<true, true, true><<<tile_grid(row_tiles),
                                          256,
                                          2ULL * dimension_ * lanes * sizeof(double),
                                          context_->stream()>>>({nullptr, rows, dimension_, 0},
                                                                topology_->edges.data(),
                                                                topology_->schedule.data(),
                                                                topology_->shears.size(),
                                                                shear_coefficients_.data(),
                                                                scales_.data(),
                                                                dimension_,
                                                                lanes,
                                                                adjoint,
                                                                false,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                {nullptr, rows, dimension_, 0},
                                                                direction.scales.data(),
                                                                direction.shears.data(),
                                                                columns,
                                                                columns + dimension_);
        check_cuda(cudaGetLastError(), "CUDA fused directional exterior right tiles");
        return;
    }
    auto* const derivatives = columns + dimension_;
    const auto scale = [&] {
        detail::scale_exterior_tiles<true>
            <<<grid_size(rows * dimension_), 256, 0, context_->stream()>>>(
                columns, derivatives, rows, dimension_, scales_.data(), direction.scales.data());
    };
    if (!adjoint)
        scale();
    for (int step = 0; dimension_ > 1 && step < n_orbitals_ - 1; ++step) {
        const int q = adjoint ? n_orbitals_ - 1 - step : step + 1;
        detail::exterior_column<true, true, true>
            <<<grid_size(rows * dimension_), 256, 0, context_->stream()>>>(
                {nullptr, rows, dimension_, 0},
                {nullptr, rows, dimension_, 0},
                index_view(),
                dimension_,
                q,
                shear_coefficients_.data(),
                direction.shears.data(),
                adjoint,
                false,
                nullptr,
                0,
                0,
                columns,
                derivatives);
    }
    if (adjoint)
        scale();
    check_cuda(cudaGetLastError(), "CUDA directional right column groups");
}

void ExteriorTransform::sample_rows(std::span<const std::int64_t> rows,
                                    linalg::MatrixView output) const {
    if (output.rows != dimension_ || output.cols != static_cast<std::int64_t>(rows.size()) ||
        output.ld < output.rows)
        throw std::invalid_argument("requested exterior-row dimensions");
    if (rows.empty())
        return;
    if (!output.data || std::any_of(rows.begin(), rows.end(), [&](std::int64_t row) {
            return row < 0 || row >= dimension_;
        }))
        throw std::invalid_argument("invalid requested exterior rows");
    auto& c = *context_;
    Buffer<std::int64_t> selected(c, rows.size());
    c.copy_to_device(selected.data(), rows.data(), rows.size_bytes());
    check_cuda(cudaMemset2DAsync(output.data,
                                 output.ld * sizeof(double),
                                 0,
                                 output.rows * sizeof(double),
                                 output.cols,
                                 c.stream()),
               "clear selected identity columns");
    selected_identity<false>
        <<<grid_size(rows.size()), 256, 0, c.stream()>>>(selected.data(), 0, rows.size(), output);
    check_cuda(cudaGetLastError(), "selected exterior identity columns");
    apply_left(output, true);
}

void ExteriorTransform::sample_rows(int first, linalg::MatrixView output) const {
    if (output.rows != dimension_ || output.cols < 0 || output.ld < output.rows || first < 0 ||
        first > dimension_ || output.cols > dimension_ - first)
        throw std::invalid_argument("requested exterior-row dimensions");
    if (!output.cols)
        return;
    if (!output.data)
        throw std::invalid_argument("invalid requested exterior rows");
    auto& c = *context_;
    check_cuda(cudaMemset2DAsync(output.data,
                                 output.ld * sizeof(double),
                                 0,
                                 output.rows * sizeof(double),
                                 output.cols,
                                 c.stream()),
               "clear selected identity columns");
    selected_identity<true>
        <<<grid_size(output.cols), 256, 0, c.stream()>>>(nullptr, first, output.cols, output);
    check_cuda(cudaGetLastError(), "selected exterior identity columns");
    apply_left(output, true);
}

void ExteriorTransform::sample_directional_rows(const ExteriorDirection& direction,
                                                std::span<const std::int64_t> rows,
                                                linalg::MatrixView primal,
                                                linalg::MatrixView tangent) const {
    if (primal.rows != dimension_ || primal.cols != static_cast<std::int64_t>(rows.size()) ||
        primal.ld < primal.rows || tangent.rows != primal.rows || tangent.cols != primal.cols ||
        tangent.ld < tangent.rows || primal.data == tangent.data)
        throw std::invalid_argument("requested directional exterior-row dimensions");
    if (rows.empty())
        return;
    if (!primal.data || !tangent.data ||
        std::any_of(rows.begin(), rows.end(), [&](std::int64_t row) {
            return row < 0 || row >= dimension_;
        }))
        throw std::invalid_argument("invalid requested directional exterior rows");
    auto& c = *context_;
    Buffer<std::int64_t> selected(c, rows.size());
    c.copy_to_device(selected.data(), rows.data(), rows.size_bytes());
    check_cuda(cudaMemset2DAsync(primal.data,
                                 primal.ld * sizeof(double),
                                 0,
                                 primal.rows * sizeof(double),
                                 primal.cols,
                                 c.stream()),
               "clear directional exterior primal rows");
    check_cuda(cudaMemset2DAsync(tangent.data,
                                 tangent.ld * sizeof(double),
                                 0,
                                 tangent.rows * sizeof(double),
                                 tangent.cols,
                                 c.stream()),
               "clear directional exterior tangent rows");
    selected_identity<false>
        <<<grid_size(rows.size()), 256, 0, c.stream()>>>(selected.data(), 0, rows.size(), primal);
    check_cuda(cudaGetLastError(), "directional exterior identity columns");
    apply_directional_left(direction, primal, tangent, true);
}

void ExteriorTransform::sample_directional_rows(const ExteriorDirection& direction,
                                                int first,
                                                linalg::MatrixView primal,
                                                linalg::MatrixView tangent) const {
    if (primal.rows != dimension_ || primal.cols < 0 || primal.ld < primal.rows ||
        tangent.rows != primal.rows || tangent.cols != primal.cols || tangent.ld < tangent.rows ||
        primal.data == tangent.data || first < 0 || first > dimension_ ||
        primal.cols > dimension_ - first)
        throw std::invalid_argument("requested directional exterior-row dimensions");
    if (!primal.cols)
        return;
    if (!primal.data || !tangent.data)
        throw std::invalid_argument("invalid requested directional exterior rows");
    auto& c = *context_;
    check_cuda(cudaMemset2DAsync(primal.data,
                                 primal.ld * sizeof(double),
                                 0,
                                 primal.rows * sizeof(double),
                                 primal.cols,
                                 c.stream()),
               "clear directional exterior primal rows");
    check_cuda(cudaMemset2DAsync(tangent.data,
                                 tangent.ld * sizeof(double),
                                 0,
                                 tangent.rows * sizeof(double),
                                 tangent.cols,
                                 c.stream()),
               "clear directional exterior tangent rows");
    selected_identity<true>
        <<<grid_size(primal.cols), 256, 0, c.stream()>>>(nullptr, first, primal.cols, primal);
    check_cuda(cudaGetLastError(), "directional exterior identity columns");
    apply_directional_left(direction, primal, tangent, true);
}

} // namespace xmvb::cuda
