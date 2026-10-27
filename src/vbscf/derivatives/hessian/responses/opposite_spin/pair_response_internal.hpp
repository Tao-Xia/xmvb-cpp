#pragma once

#include <cstdint>

#include "mcscf/ci/string_table.hpp"

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "integrals/active/two_electron_result.hpp"

namespace xmvb::vb::detail {

class DirectionalOppositeSpinPairTileView;

/** Directional cofactor data for one ordered unique-spin pair. */
struct DirectionalOppositeSpinPairData {
    Eigen::MatrixXd delta_overlap_submatrix;
    OppositeSpinPackedPairProjection delta_first_order_cofactor_projection;
};

/** Directional opposite-spin data local to one unique-spin pair tile. */
struct DirectionalOppositeSpinPairTile {
    std::int64_t left_begin = 0;
    std::int64_t right_begin = 0;
    Eigen::Index left_size = 0;
    Eigen::Index right_size = 0;
    std::vector<DirectionalOppositeSpinPairData> pairs;
    /** Pair-major contiguous storage, with one packed channel per column. */
    Eigen::MatrixXd raw_channel_values;
    /** Full derivative G delta_P + delta_G P, used by backward consumers. */
    Eigen::MatrixXd projected_channel_values;
    /** G delta_P alone: the second spin factor must not count delta_G twice. */
    Eigen::MatrixXd cofactor_projected_channel_values;

    const DirectionalOppositeSpinPairData& pair(Eigen::Index left_local,
                                                Eigen::Index right_local) const;

    DirectionalOppositeSpinPairTileView view(bool transposed = false) const;
};

/** Zero-copy orientation view of one canonical opposite-spin tile. */
class DirectionalOppositeSpinPairTileView {
  public:
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using ConstChannelMap = Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Stride>;

    DirectionalOppositeSpinPairTileView(const DirectionalOppositeSpinPairTile& storage,
                                        bool transposed) noexcept
        : storage_(&storage), transposed_(transposed) {}

    std::int64_t left_begin() const noexcept;
    std::int64_t right_begin() const noexcept;
    Eigen::Index left_size() const noexcept;
    Eigen::Index right_size() const noexcept;
    ConstChannelMap raw_channel(int packed_pair) const;
    ConstChannelMap projected_channel(int packed_pair) const;
    ConstChannelMap cofactor_projected_channel(int packed_pair) const;

    template <typename Consumer>
    void with_pair(Eigen::Index left_local, Eigen::Index right_local, Consumer&& consume) const {
        if (!transposed_) {
            consume(storage_->pair(left_local, right_local));
            return;
        }
        DirectionalOppositeSpinPairData pair = storage_->pair(right_local, left_local);
        pair.delta_overlap_submatrix.transposeInPlace();
        consume(pair);
    }

  private:
    ConstChannelMap channel_view(const Eigen::MatrixXd& channels, int packed_pair) const;

    const DirectionalOppositeSpinPairTile* storage_ = nullptr;
    bool transposed_ = false;
};

/**
 * @brief Builds one opposite-spin response tile from its same-spin cofactor tile.
 *
 * The RI path batches packed-pair projections and evaluates
 * `K delta_X + delta_K X` with matrix products. Its temporary column block is
 * chosen so that the dense GEMM workspace is no larger than one accepted RI
 * factor matrix.
 */
DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const xmvb::mcscf::StringTable& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    std::int64_t n_unique_determinants,
    int n_active_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const xmvb::mcscf::StringTable& unique_determinants,
    const AcceptedSpinPairTile& accepted_pair_tile,
    int n_active_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors = nullptr,
    const Eigen::MatrixXd* directional_ri_active_pair_factors = nullptr);

} // namespace xmvb::vb::detail
