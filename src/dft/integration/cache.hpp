#pragma once

#include <array>
#include <span>
#include "core/workspace.hpp"
#include "dft/xc/functional.hpp"
#include "integrals/basis/values.hpp"

namespace xmvb::dft {
struct Grid;
using ConstMatrix = Eigen::Ref<const Eigen::MatrixXd>;

/** @brief Nonowning AO panel; resident and generated tiles use the same kernels. */
struct BasisTile {
    ConstMatrix value;
    std::array<ConstMatrix, 3> gradient;
    BasisTile(const integrals::BasisValues& values, Eigen::Index begin, Eigen::Index rows);
};

/** @brief Immutable-grid AO residency charged once to the shared workspace.
 * Retains a prefix sized together with the caller's live scratch, not a fixed
 * cache cap. Unresident panels are generated into the caller's one scratch tile.
 */
class GridBasis {
  public:
    GridBasis(const integrals::LibcintInput& input, const Grid& grid, bool gradient);
    int n_bf() const {
        return evaluator_.n_bf();
    }
    void prepare(std::size_t output_bytes, std::size_t point_bytes);
    void trim(std::size_t minimum_scratch);
    Eigen::Index rows(Eigen::Index begin, Eigen::Index requested) const;
    Eigen::Index resident_rows() const {
        return resident_.value.rows();
    }
    BasisTile tile(Eigen::Index begin, Eigen::Index rows, integrals::BasisValues* scratch) const;

  private:
    const Grid& grid_;
    integrals::BasisEvaluator evaluator_;
    bool gradient_;
    workspace::Reservation reservation_;
    integrals::BasisValues resident_;
};

/** @brief Reference-only weak-form data; no directional or AO-sized copies. */
struct ReferenceTile {
    std::array<Eigen::MatrixXd, 3> gradient;
    Eigen::MatrixXd vsigma;
    XcKernel kernel;
};

/** @brief One exact reference-density key and one budgeted prefix panel.
 * Physical trial evaluations do not invalidate the frozen-response reference.
 * This mutable workspace is used by a serialized SCF driver, not concurrently.
 */
class ResponseCache {
  public:
    void select(std::span<const ConstMatrix> density);
    void clear();
    void prepare(std::span<const ConstMatrix> density,
                 Eigen::Index max_rows,
                 std::size_t output_bytes,
                 std::size_t point_bytes,
                 bool gga);
    Eigen::Index rows(Eigen::Index begin, Eigen::Index requested) const;
    ReferenceTile* tile(Eigen::Index begin, Eigen::Index rows);

  private:
    workspace::Reservation reservation_;
    std::array<Eigen::MatrixXd, 2> density_;
    int spins_ = 0;
    Eigen::Index rows_ = 0;
    ReferenceTile data_;
};
} // namespace xmvb::dft
