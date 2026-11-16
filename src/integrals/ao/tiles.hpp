#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace xmvb::integrals {

/** One stored block triangle of the symmetric AO-pair matrix. */
class AoPairTiles {
  public:
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    class Builder;

    /** Traverse canonical nonzeros without expanding dense tiles or their transpose. */
    class Cursor {
      public:
        Cursor(const AoPairTiles& tiles, std::size_t begin);
        std::size_t left() const noexcept;
        std::size_t right() const noexcept;
        double value() const noexcept;
        void advance() noexcept;

      private:
        void seek() noexcept;
        const AoPairTiles* tiles_;
        std::size_t tile_ = 0;
        int row_ = 0, entry_ = 0;
    };

    std::size_t integral_count() const noexcept {
        return offsets_.back();
    }
    std::size_t directed_count() const noexcept {
        return integral_count() + (integral_count() - diagonal_count_);
    }
    std::size_t size() const noexcept {
        return starts_.back();
    }
    std::size_t storage_bytes() const noexcept;

    template <class Consumer> void visit_row(std::size_t pair, Consumer consume) const {
        const auto block = pair_block_[pair];
        const int row = pair - starts_[block];
        for (const auto [other, slot] : adjacency_[block]) {
            const bool transpose = other > block;
            const auto& tile = transpose ? rows_[other][slot] : rows_[block][slot];
            const auto base = starts_[other];
            if (tile.dense) {
                const int width = extent(transpose ? block : other);
                const int count = extent(other);
                for (int col = 0; col < count; ++col) {
                    const double value =
                        tile.values[transpose ? col * width + row : row * width + col];
                    if (value != 0.0)
                        consume(base + col, value);
                }
            } else if (transpose) {
                for (int k = tile.col_offsets[row]; k < tile.col_offsets[row + 1]; ++k)
                    consume(base + tile.transpose[k].row, tile.values[tile.transpose[k].value]);
            } else {
                for (int k = tile.row_offsets[row]; k < tile.row_offsets[row + 1]; ++k)
                    consume(base + tile.columns[k], tile.values[k]);
            }
        }
    }

    std::size_t unique_blocks() const noexcept {
        return canonical_.size();
    }
    std::size_t row_blocks() const noexcept {
        return rows_.size();
    }
    std::size_t row_begin(std::size_t block) const noexcept {
        return starts_[block];
    }
    int row_count(std::size_t block) const noexcept {
        return extent(block);
    }

    /** Consume rows of a canonical block, with each retained ERI visited once. */
    template <class Consumer> void visit_unique_block(std::size_t block, Consumer consume) const {
        const auto row_block = canonical_[block].block;
        const auto& tile = canonical(block);
        const auto left = starts_[row_block], right = starts_[tile.column];
        const bool diagonal = row_block == tile.column;
        const int height = extent(row_block), width = extent(tile.column);
        if (tile.dense) {
            for (int row = 0; row < height; ++row) {
                const auto* values = tile.values.data() + row * width;
                const int end = diagonal ? row + 1 : width;
                consume(left + row, [&](auto add) {
                    for (int col = 0; col < end; ++col)
                        if (values[col] != 0.0)
                            add(right + col, values[col]);
                });
            }
        } else {
            for (int row = 0; row < height; ++row)
                consume(left + row, [&](auto add) {
                    for (int k = tile.row_offsets[row]; k < tile.row_offsets[row + 1]; ++k) {
                        const int col = tile.columns[k];
                        if (!diagonal || col <= row)
                            add(right + col, tile.values[k]);
                    }
                });
        }
    }

    /** Visit a directed subpanel. Dense transposed blocks read contiguous runs
     * from the original storage and update several target rows together. */
    template <class Consumer>
    void visit_panel(std::size_t block, int begin, int count, Consumer consume) const {
        for (const auto [other, slot] : adjacency_[block]) {
            const bool transposed = other > block;
            const auto& tile = transposed ? rows_[other][slot] : rows_[block][slot];
            const auto source = starts_[other];
            if (transposed && tile.dense) {
                const int stride = extent(block), height = extent(other);
                for (int col = 0; col < height; ++col) {
                    const auto* values = tile.values.data() + col * stride + begin;
                    consume(source + col, [&](auto add) {
                        for (int row = 0; row < count; ++row)
                            if (values[row] != 0.0)
                                add(row, values[row]);
                    });
                }
            } else if (tile.dense) {
                const int width = extent(other);
                for (int row = 0; row < count; ++row) {
                    const auto* values = tile.values.data() + (begin + row) * width;
                    for (int col = 0; col < width; ++col)
                        if (values[col] != 0.0)
                            consume(source + col, [&](auto add) { add(row, values[col]); });
                }
            } else {
                for (int row = 0; row < count; ++row) {
                    const int index = begin + row;
                    if (transposed) {
                        for (int k = tile.col_offsets[index]; k < tile.col_offsets[index + 1];
                             ++k) {
                            const auto [col, value] = tile.transpose[k];
                            consume(source + col, [&](auto add) { add(row, tile.values[value]); });
                        }
                    } else {
                        for (int k = tile.row_offsets[index]; k < tile.row_offsets[index + 1]; ++k)
                            consume(source + tile.columns[k],
                                    [&](auto add) { add(row, tile.values[k]); });
                    }
                }
            }
        }
    }

    void apply(const Eigen::Ref<const Matrix>& input, Matrix& output) const;

  private:
    AoPairTiles() = default;
    struct Tile {
        struct Transposed {
            std::uint16_t row, value;
        };
        std::size_t column = 0, count = 0, diagonal_count = 0;
        bool dense = false;
        std::vector<double> values;
        std::vector<std::uint16_t> row_offsets, col_offsets, columns;
        std::vector<Transposed> transpose;
    };
    struct Location {
        std::size_t block, slot;
    };

    int extent(std::size_t block) const noexcept {
        return starts_[block + 1] - starts_[block];
    }
    const Tile& canonical(std::size_t index) const noexcept {
        const auto [block, slot] = canonical_[index];
        return rows_[block][slot];
    }

    template <bool Transpose>
    static void apply_sparse(
        const Tile& tile, int rows, Eigen::Index width, const double* input, double* output);

    std::vector<std::size_t> starts_, pair_block_, offsets_;
    std::vector<std::vector<Tile>> rows_;
    std::vector<std::vector<Location>> adjacency_;
    std::vector<Location> canonical_;
    std::size_t diagonal_count_ = 0;
};

/** Rows owned by one outer shell can be constructed by one independent worker. */
class AoPairTiles::Builder {
  public:
    class Shell;

    explicit Builder(std::span<const std::size_t> shell_ends);
    std::shared_ptr<const AoPairTiles> finish();

  private:
    friend class Shell;
    std::shared_ptr<AoPairTiles> tiles_;
    std::vector<std::size_t> shell_first_;
};

inline std::size_t AoPairTiles::Cursor::left() const noexcept {
    return tiles_->starts_[tiles_->canonical_[tile_].block] + row_;
}

inline std::size_t AoPairTiles::Cursor::right() const noexcept {
    const auto& tile = tiles_->canonical(tile_);
    return tiles_->starts_[tile.column] + (tile.dense ? entry_ : tile.columns[entry_]);
}

inline double AoPairTiles::Cursor::value() const noexcept {
    const auto& tile = tiles_->canonical(tile_);
    return tile.values[tile.dense ? row_ * tiles_->extent(tile.column) + entry_ : entry_];
}

inline void AoPairTiles::Cursor::advance() noexcept {
    ++entry_;
    seek();
}

inline void AoPairTiles::Cursor::seek() noexcept {
    while (tile_ < tiles_->canonical_.size()) {
        const auto block = tiles_->canonical_[tile_].block;
        const auto& tile = tiles_->canonical(tile_);
        const bool diagonal = block == tile.column;
        const int width = tiles_->extent(tile.column);
        while (row_ < tiles_->extent(block)) {
            if (tile.dense) {
                const int end = diagonal ? row_ + 1 : width;
                while (entry_ < end && tile.values[row_ * width + entry_] == 0.0)
                    ++entry_;
                if (entry_ < end)
                    return;
            } else {
                entry_ = std::max<int>(entry_, tile.row_offsets[row_]);
                while (entry_ < tile.row_offsets[row_ + 1] && diagonal &&
                       tile.columns[entry_] > row_)
                    ++entry_;
                if (entry_ < tile.row_offsets[row_ + 1])
                    return;
            }
            ++row_;
            entry_ = 0;
        }
        ++tile_;
        row_ = 0;
        entry_ = 0;
    }
}

} // namespace xmvb::integrals
