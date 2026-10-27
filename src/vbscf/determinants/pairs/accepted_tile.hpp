#pragma once

#include "core/index.hpp"

#include "mcscf/ci/string_table.hpp"

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::vb {

struct AcceptedPairTileBuildOptions {
    bool materialize_projected_pair_values = false;
    bool populate_response_payload = false;
    bool populate_opposite_spin_projection = true;
};

/** Bounded accepted-point pair payload for one rectangular spin-string tile. */
struct AcceptedSpinPairTile {
    Eigen::Index left_begin = 0;
    Eigen::Index right_begin = 0;
    Eigen::Index left_size = 0;
    Eigen::Index right_size = 0;
    std::vector<SpinDeterminantPairEvaluation> pairs;

    const SpinDeterminantPairEvaluation& pair(Eigen::Index left_local,
                                              Eigen::Index right_local) const;
};

/**
 * @brief Generates accepted same-spin pairs without persistent quadratic storage.
 *
 * Every requested pair is evaluated independently with the canonical overlap
 * resolver.  The provider deliberately contains no low-rank update policy:
 * tiling controls memory independently of any future pair-generation
 * acceleration, and singular-pair handling remains identical to the exact
 * non-streamed evaluator.
 */
class AcceptedPairTileProvider {
  public:
    AcceptedPairTileProvider(mcscf::StringTable unique_spin_strings,
                             int n_active_orbitals,
                             double linear_dependence_threshold = 1.0e-12);

    Eigen::Index size() const noexcept {
        return core::checked_index<Eigen::Index>(unique_spin_strings_.size());
    }

    const mcscf::StringTable& unique_spin_strings() const noexcept {
        return unique_spin_strings_;
    }

    AcceptedSpinPairTile
    build(Eigen::Index left_begin,
          Eigen::Index left_end,
          Eigen::Index right_begin,
          Eigen::Index right_end,
          const std::vector<double>& active_overlap,
          const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
          const xmvb::integrals::ActiveSpaceTwoElectronResult& active_two_electron,
          AcceptedPairTileBuildOptions options = {}) const;

  private:
    mcscf::StringTable unique_spin_strings_;
    int n_active_orbitals_ = 0;
    DeterminantOverlapResolver overlap_resolver_;
    DeterminantPairEvaluator pair_evaluator_;
};

} // namespace xmvb::vb
