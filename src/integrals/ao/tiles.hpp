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
