#include "integrals/ao/tile_builder.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef EIGEN_USE_BLAS
#include <cblas.h>
#endif

#include "parallel/openmp.hpp"

namespace xmvb::integrals {
namespace {
// Local cell addresses and offsets fit in 16 bits. All global addresses are size_t.
constexpr int tile_extent = 128;
static_assert(tile_extent * tile_extent < 65536);
} // namespace

AoPairTiles::Builder::Builder(std::span<const std::size_t> shell_ends) : tiles_(new AoPairTiles) {
    if (shell_ends.empty() || shell_ends.front() == 0 ||
        std::adjacent_find(shell_ends.begin(), shell_ends.end(), std::greater_equal{}) !=
            shell_ends.end())
        throw std::invalid_argument("invalid AO shell pair boundaries");
    auto& tiles = *tiles_;
    tiles.pair_block_.resize(shell_ends.back());
    std::size_t begin = 0;
    for (auto end : shell_ends) {
        shell_first_.push_back(tiles.starts_.size());
        for (auto p = begin; p < end;) {
            const auto next = p + std::min<std::size_t>(tile_extent, end - p);
            std::fill(tiles.pair_block_.begin() + p,
                      tiles.pair_block_.begin() + next,
                      tiles.starts_.size());
            tiles.starts_.push_back(p);
            p = next;
        }
        begin = end;
    }
    shell_first_.push_back(tiles.starts_.size());
    tiles.rows_.resize(tiles.starts_.size());
    tiles.starts_.push_back(begin);
}

AoPairTiles::Builder::Shell::Shell(Builder& builder, std::size_t shell)
    : tiles_(*builder.tiles_), begin_(builder.shell_first_[shell]),
      rows_(builder.shell_first_[shell + 1] - begin_) {}

void AoPairTiles::Builder::Shell::add(std::size_t left, std::size_t right, double value) {
    if (value == 0.0)
        return;
    const auto block = tiles_.pair_block_[left], column = tiles_.pair_block_[right];
    rows_[block - begin_][column].add(left - tiles_.starts_[block],
                                      right - tiles_.starts_[column],
                                      tiles_.extent(block),
                                      tiles_.extent(column),
                                      block == column,
                                      value);
}

void AoPairTiles::Builder::Shell::Pending::add(
    int row, int col, int height, int width, bool diagonal, double value) {
    ++tile.count;
    tile.diagonal_count += diagonal && row == col;
    if (tile.dense) {
        tile.values[row * width + col] = value;
        if (diagonal)
            tile.values[col * width + row] = value;
        return;
    }
    tile.values.push_back(value);
    coordinates.push_back(row * tile_extent + col);
    const auto sparse_bytes =
        tile.count * (sizeof(double) + sizeof(std::uint16_t) + sizeof(Tile::Transposed)) +
        sizeof(std::uint16_t) * (height + width + 2);
    if (sizeof(double) * height * width > sparse_bytes)
        return;

    std::vector<double> dense(height * width);
    for (std::size_t k = 0; k < coordinates.size(); ++k) {
        const int i = coordinates[k] / tile_extent, j = coordinates[k] % tile_extent;
        dense[i * width + j] = tile.values[k];
        if (diagonal)
            dense[j * width + i] = tile.values[k];
    }
    tile.values = std::move(dense);
    std::vector<std::uint16_t>().swap(coordinates);
    tile.dense = true;
}

AoPairTiles::Tile
AoPairTiles::Builder::Shell::Pending::finish(int height, int width, bool diagonal) {
    if (tile.dense)
        return std::move(tile);

    Tile result;
    result.count = tile.count;
    result.diagonal_count = tile.diagonal_count;
    result.row_offsets.resize(height + 1);
    if (!diagonal)
        result.col_offsets.resize(width + 1);
    for (const auto cell : coordinates) {
        const int row = cell / tile_extent, col = cell % tile_extent;
        ++result.row_offsets[row + 1];
        if (diagonal && row != col)
            ++result.row_offsets[col + 1];
        if (!diagonal)
            ++result.col_offsets[col + 1];
    }
    std::partial_sum(
        result.row_offsets.begin(), result.row_offsets.end(), result.row_offsets.begin());
    std::partial_sum(
        result.col_offsets.begin(), result.col_offsets.end(), result.col_offsets.begin());
    result.values.resize(result.row_offsets.back());
    result.columns.resize(result.values.size());
    if (!diagonal)
        result.transpose.resize(result.values.size());
    auto row_offsets = result.row_offsets;
    auto col_offsets = result.col_offsets;
    for (std::size_t t = 0; t < coordinates.size(); ++t) {
        const std::uint16_t row = coordinates[t] / tile_extent, col = coordinates[t] % tile_extent;
        const auto k = row_offsets[row]++;
        result.values[k] = tile.values[t];
        result.columns[k] = col;
        if (!diagonal)
            result.transpose[col_offsets[col]++] = {row, k};
        else if (row != col) {
            const auto reflected = row_offsets[col]++;
            result.values[reflected] = tile.values[t];
            result.columns[reflected] = row;
        }
    }
    return result;
}

void AoPairTiles::Builder::Shell::finish() {
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const auto block = begin_ + i;
        auto& row = tiles_.rows_[block];
        auto& pending = rows_[i];
        row.reserve(pending.size());
        for (auto& [column, builder] : pending) {
            auto tile =
                builder.finish(tiles_.extent(block), tiles_.extent(column), block == column);
            tile.column = column;
            row.push_back(std::move(tile));
        }
        std::ranges::sort(row, {}, &Tile::column);
        pending.clear();
    }
}

std::shared_ptr<const AoPairTiles> AoPairTiles::Builder::finish() {
    auto& tiles = *tiles_;
    tiles.adjacency_.resize(tiles.rows_.size());
    tiles.offsets_.push_back(0);
    for (std::size_t block = 0; block < tiles.rows_.size(); ++block) {
        const auto& row = tiles.rows_[block];
        for (std::size_t slot = 0; slot < row.size(); ++slot) {
            const auto column = row[slot].column;
            tiles.adjacency_[block].push_back({column, slot});
            if (block != column)
                tiles.adjacency_[column].push_back({block, slot});
            tiles.canonical_.push_back({block, slot});
            tiles.diagonal_count_ += row[slot].diagonal_count;
            tiles.offsets_.push_back(tiles.offsets_.back() + row[slot].count);
        }
    }
    return std::move(tiles_);
}

std::size_t AoPairTiles::storage_bytes() const noexcept {
    std::size_t bytes =
        (starts_.capacity() + pair_block_.capacity() + offsets_.capacity()) * sizeof(std::size_t) +
        canonical_.capacity() * sizeof(Location) + rows_.capacity() * sizeof(rows_[0]) +
        adjacency_.capacity() * sizeof(adjacency_[0]);
    for (std::size_t block = 0; block < rows_.size(); ++block) {
        bytes += adjacency_[block].capacity() * sizeof(Location);
        bytes += rows_[block].capacity() * sizeof(Tile);
        for (const auto& tile : rows_[block])
            bytes += tile.values.capacity() * sizeof(double) +
                     (tile.row_offsets.capacity() + tile.col_offsets.capacity() +
                      tile.columns.capacity()) *
                         sizeof(std::uint16_t) +
                     tile.transpose.capacity() * sizeof(Tile::Transposed);
    }
    return bytes;
}

AoPairTiles::Cursor::Cursor(const AoPairTiles& tiles, std::size_t begin) : tiles_(&tiles) {
    const auto found = std::upper_bound(tiles.offsets_.begin(), tiles.offsets_.end(), begin);
    tile_ = static_cast<std::size_t>(found - tiles.offsets_.begin() - 1);
    seek();
    for (auto offset = tiles.offsets_[tile_]; offset < begin; ++offset)
        advance();
}

template <bool trans>
void AoPairTiles::apply_sparse(
    const Tile& tile, int rows, Eigen::Index width, const double* x, double* y) {
    const auto& offsets = trans ? tile.col_offsets : tile.row_offsets;
    for (int row = 0; row < rows; ++row) {
        auto* target = y + row * width;
        int edge = offsets[row];
        const int end = offsets[row + 1];
        const auto source = [&](int k) {
            if constexpr (trans)
                return x + tile.transpose[k].row * width;
            else
                return x + tile.columns[k] * width;
        };
        const auto value = [&](int k) {
            if constexpr (trans)
                return tile.values[tile.transpose[k].value];
            else
                return tile.values[k];
        };
        for (; end - edge >= 8; edge += 8) {
            const double *p0 = source(edge), *p1 = source(edge + 1), *p2 = source(edge + 2),
                         *p3 = source(edge + 3), *p4 = source(edge + 4), *p5 = source(edge + 5),
                         *p6 = source(edge + 6), *p7 = source(edge + 7);
            const double v0 = value(edge), v1 = value(edge + 1), v2 = value(edge + 2),
                         v3 = value(edge + 3), v4 = value(edge + 4), v5 = value(edge + 5),
                         v6 = value(edge + 6), v7 = value(edge + 7);
#pragma omp simd
            for (Eigen::Index k = 0; k < width; ++k) {
                double sum = target[k];
                sum += v0 * p0[k];
                sum += v1 * p1[k];
                sum += v2 * p2[k];
                sum += v3 * p3[k];
                sum += v4 * p4[k];
                sum += v5 * p5[k];
                sum += v6 * p6[k];
                sum += v7 * p7[k];
                target[k] = sum;
            }
        }
        for (; edge < end; ++edge) {
            const auto* input = source(edge);
            const double v = value(edge);
#pragma omp simd
            for (Eigen::Index k = 0; k < width; ++k)
                target[k] += v * input[k];
        }
    }
}

void AoPairTiles::apply(const Eigen::Ref<const Matrix>& input, Matrix& output) const {
    const Eigen::Index width = input.cols();
    output.resize(input.rows(), width);
    if (width == 0)
        return;
    const int threads = bounded_openmp_thread_count(effective_openmp_thread_count(), rows_.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (std::size_t block = 0; block < rows_.size(); ++block) {
        double* target = output.data() + starts_[block] * width;
        const int height = extent(block);
        std::fill_n(target, height * width, 0.0);
        for (const auto [other, slot] : adjacency_[block]) {
            const bool transpose = other > block;
            const auto& tile = transpose ? rows_[other][slot] : rows_[block][slot];
            const double* source = input.data() + starts_[other] * width;
            if (!tile.dense) {
                if (transpose)
                    apply_sparse<true>(tile, height, width, source, target);
                else
                    apply_sparse<false>(tile, height, width, source, target);
                continue;
            }
            const int depth = extent(other);
#ifdef EIGEN_USE_BLAS
            cblas_dgemm(CblasRowMajor,
                        transpose ? CblasTrans : CblasNoTrans,
                        CblasNoTrans,
                        height,
                        width,
                        depth,
                        1.0,
                        tile.values.data(),
                        transpose ? height : depth,
                        source,
                        width,
                        1.0,
                        target,
                        width);
#else
            Eigen::Map<Matrix> y(target, height, width);
            const Eigen::Map<const Matrix> x(source, depth, width);
            if (transpose)
                y.noalias() +=
                    Eigen::Map<const Matrix>(tile.values.data(), depth, height).transpose() * x;
            else
                y.noalias() += Eigen::Map<const Matrix>(tile.values.data(), height, depth) * x;
#endif
        }
    }
}

} // namespace xmvb::integrals
