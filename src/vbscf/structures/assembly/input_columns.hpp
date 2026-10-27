#pragma once

#include <cstdint>
#include <span>

#include "core/timing.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"

namespace xmvb::vb::detail {

/** @brief Bounded beta-column residency with batched, exact-demand generation.
 * Sigma requests neighboring columns, not contiguous transform panels.
 * Only missing requested columns are generated. Hits are consumed before
 * insertion, so an eviction can never invalidate a borrowed read pointer.
 * The single sigma coordinator owns the cache; worker tasks copy disjoint data.
 */
class InputColumns {
  public:
    using Build = std::function<std::vector<ColumnTile>(std::span<const Eigen::Index>)>;
    using Price = std::function<std::size_t(Eigen::Index)>;

    InputColumns(Eigen::Index n_alpha,
                 Eigen::Index width,
                 Eigen::Index extent,
                 std::size_t bytes,
                 const std::vector<std::uint8_t>& support,
                 Build build,
                 Price price)
        : width_(width), extent_(extent), support_(support), build_(std::move(build)),
          price_(std::move(price)), positions_(n_alpha, -1), columns_(support.size(), -1),
          cache_(
              bytes,
              [this](Eigen::Index beta) {
                  auto tiles = build_({&beta, 1});
                  return std::move(tiles.front());
              },
              {},
              [this](Eigen::Index beta) { return price_(beta); }) {}

    InputColumns(const InputColumns&) = delete;
    InputColumns& operator=(const InputColumns&) = delete;

    void prepare(const std::vector<Eigen::Index>& betas) {
        std::vector<Eigen::Index> missing;
        profiling::ScopedPhase phase(profiling::Phase::CiCacheSchedule);
        for (const Eigen::Index beta : betas)
            if (support_[beta] && !cache_.contains(beta))
                missing.push_back(beta);
        phase.stop();
        produce(missing, {});
    }

    /** @brief With zero_fill=false, only structural source entries are written.
   * The consumer must certify that all other halo slots are never read.
   */
    void read(const std::vector<Eigen::Index>& betas,
              const std::vector<Eigen::Index>& rows,
              Eigen::Ref<Eigen::MatrixXd> values,
              bool zero_fill) {
        std::vector<Eigen::Index> missing, ready;
        std::vector<const ColumnTile*> hits;
        profiling::ScopedPhase schedule(profiling::Phase::CiCacheSchedule);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(rows.size()); ++i)
            positions_[rows[i]] = i;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(betas.size()); ++i) {
            const Eigen::Index beta = betas[i];
            columns_[beta] = i;
            if (!support_[beta]) {
                if (zero_fill)
                    values.middleCols(i * width_, width_).setZero();
            } else if (cache_.contains(beta)) {
                ready.push_back(beta);
                hits.push_back(&cache_.get(beta));
            } else
                missing.push_back(beta);
        }
        schedule.stop();
        bool identity_rows = rows.size() == positions_.size();
        if (identity_rows)
            for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(rows.size()); ++row)
                if (rows[row] != row) {
                    identity_rows = false;
                    break;
                }
        const auto copy = [&](std::span<const Eigen::Index> keys, const auto& tile_at) {
            profiling::ScopedPhase phase(profiling::Phase::CiCacheCopy);
            const Eigen::Index workers = cooperative_openmp_thread_count();
            cooperative_openmp_for(keys.size() * width_, workers, [&](std::ptrdiff_t packed) {
                const Eigen::Index j = packed / width_, v = packed % width_;
                const auto& tile = tile_at(j);
                auto target = values.col(columns_[keys[j]] * width_ + v);
                if ((tile.pattern->full ||
                     tile.values.rows() == static_cast<Eigen::Index>(positions_.size())) &&
                    identity_rows) {
                    target = tile.values.col(v);
                } else if (tile.pattern->full) {
                    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(rows.size()); ++row)
                        target[row] = tile.values(rows[row], v);
                } else if (!tile.pattern->row_words.empty() &&
                           rows.size() < static_cast<std::size_t>(tile.values.rows())) {
                    // Gather only requested halo rows. The old zero/scatter read the
                    // entire cached support even when most rows were outside the halo.
                    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(rows.size());
                         ++row) {
                        const Eigen::Index source = tile.pattern->indexed_row(rows[row]);
                        if (source >= 0)
                            target[row] = tile.values(source, v);
                        else if (zero_fill)
                            target[row] = 0.0;
                    }
                } else {
                    if (zero_fill)
                        target.setZero();
                    for (Eigen::Index i = 0; i < tile.values.rows(); ++i) {
                        const Eigen::Index row = positions_[tile.pattern->rows[i]];
                        if (row >= 0)
                            target[row] = tile.values(i, v);
                    }
                }
            });
        };
        copy(ready, [&](Eigen::Index i) -> const ColumnTile& { return *hits[i]; });
        produce(missing,
                [&](std::span<const Eigen::Index> keys, const std::vector<ColumnTile>& tiles) {
                    copy(keys, [&](Eigen::Index i) -> const ColumnTile& { return tiles[i]; });
                });
        {
            profiling::ScopedPhase phase(profiling::Phase::CiCacheSchedule);
            for (const Eigen::Index row : rows)
                positions_[row] = -1;
            for (const Eigen::Index beta : betas)
                columns_[beta] = -1;
        }
    }

    const ColumnPanels& cache() const noexcept {
        return cache_;
    }
    /** Borrow a prepared column until the next cache mutation. */
    const ColumnTile& column(Eigen::Index beta) {
        return cache_.get(beta);
    }
    void release(Eigen::Index beta) {
        cache_.release(beta);
    }

  private:
    using Consume =
        std::function<void(std::span<const Eigen::Index>, const std::vector<ColumnTile>&)>;

    void produce(std::span<const Eigen::Index> missing, const Consume& consume) {
        for (std::size_t begin = 0; begin < missing.size(); begin += extent_) {
            const auto keys =
                missing.subspan(begin, std::min<std::size_t>(extent_, missing.size() - begin));
            std::size_t bytes = 0;
            for (Eigen::Index key : keys)
                bytes += price_(key);
            cache_.reserve(bytes);
            profiling::ScopedPhase generate(profiling::Phase::CiCacheGenerate);
            auto tiles = build_(keys);
            generate.stop();
            if (consume)
                consume(keys, tiles);
            profiling::ScopedPhase phase(profiling::Phase::CiCacheInsert);
            for (std::size_t j = 0; j < keys.size(); ++j)
                cache_.insert(keys[j], std::move(tiles[j]));
        }
    }

    Eigen::Index width_, extent_;
    const std::vector<std::uint8_t>& support_;
    Build build_;
    Price price_;
    std::vector<Eigen::Index> positions_, columns_;
    ColumnPanels cache_;
};

} // namespace xmvb::vb::detail
