#pragma once

#include <cstdint>

#include <cstddef>
#include <memory>
#include <limits>
#include <span>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::cuda {

class PinnedUpload;

/** CPU ExteriorTransformDirection, stored on the device. One instance is
 * reused by every primal/adjoint column panel at a fixed directional point.
 */
struct ExteriorDirection {
    Buffer<double> scales;
    Buffer<double> shears;
};

/** Device replay of the CPU transform's retained diagonal and ordered shears. */
class ExteriorTransform {
  public:
    /** Immutable combinatorial addressing; a null order denotes colex order. */
    struct Index {
        const std::uint64_t *masks, *choose;
        const std::int64_t* order;
        int orbitals;
    };
    struct Shear {
        std::size_t offset;
        std::int64_t count;
    };
    /** Resource ceiling used by the same exact fused/streamed column algorithm. */
    ExteriorTransform(Context& context,
                      const vb::ExteriorOrbitalTransform& source,
                      std::size_t shared_memory_bytes = std::numeric_limits<std::size_t>::max());
    /** Independent numerical frame with borrowed immutable spin topology. */
    ExteriorTransform(Context& context, const ExteriorTransform& reference);
    ~ExteriorTransform();
    ExteriorTransform(const ExteriorTransform&) = delete;
    ExteriorTransform& operator=(const ExteriorTransform&) = delete;

    /** @brief Update a positive upper orbital frame without rebuilding topology.
   * Numerical scales/shears remain device-resident and at fixed addresses so
   * captured actions observe subsequent accepted/trial orbital frames.
   */
    void update_upper(linalg::ConstMatrixView upper);

    /** Apply the exterior transform to spin rows, in place. */
    void
    apply_left(linalg::MatrixView coefficients, bool adjoint = false, bool inverse = false) const;
    /** CPU's exact invariant support, with borrowed immutable device bitsets.
   * Each of `columns` masks has ceil(dimension/64) consecutive words; RHS
   * lanes reuse those masks. The caller owns their budgeted lifetime.
   */
    void apply_supported_left(linalg::MatrixView coefficients,
                              const std::uint64_t* allowed,
                              int columns,
                              bool adjoint = false) const;
    /** Fixed device column table for graph replay.
   * The same right-transform kernels serve ordinary temporary tile callers.
   */
    void apply_right_columns(double* const* columns,
                             std::int64_t rows,
                             bool adjoint = false,
                             bool inverse = false) const;
    ExteriorDirection direction(linalg::ConstMatrixView upper,
                                linalg::ConstMatrixView delta_upper) const;
    /** Same ordered primal/tangent factors as CPU apply_directional_left and
   * apply_directional_adjoint_left. Only bounded single-spin panels are used.
   */
    void apply_directional_left(const ExteriorDirection&,
                                linalg::MatrixView primal,
                                linalg::MatrixView tangent,
                                bool adjoint = false) const;
    /** Apply it to spin columns in each consecutive RHS block, in place. */
    void
    apply_right(linalg::MatrixView coefficients, bool adjoint = false, bool inverse = false) const;
    /** Ordered right action across independent beta-column tiles of one lane.
   * Tiles have equal alpha row counts and together cover exactly dimension()
   * columns. Both fused and streaming updates operate directly on tile columns.
   */
    void apply_right_tiles(std::span<const linalg::MatrixView> tiles,
                           bool adjoint = false,
                           bool inverse = false) const;
    /** Directional counterpart with matching primal and tangent tile layouts. */
    void apply_directional_right_tiles(const ExteriorDirection& direction,
                                       std::span<const linalg::MatrixView> primal,
                                       std::span<const linalg::MatrixView> tangent,
                                       bool adjoint = false) const;

    /** Sample the prepared frame: column j contains T(rows[j],:)^T.
   * Replays the existing ordered adjoint on selected identity columns; no
   * second orbital factorization or per-thread exterior recursion is needed.
   * Storage is the caller's single-spin panel, never a product carrier.
   */
    void sample_rows(std::span<const std::int64_t> rows, linalg::MatrixView output) const;
    /** Sample consecutive rows [first, first + output.cols) without an index upload. */
    void sample_rows(int first, linalg::MatrixView output) const;
    /** Requested adjoint rows and their tangent, without a host identity panel. */
    void sample_directional_rows(const ExteriorDirection& direction,
                                 std::span<const std::int64_t> rows,
                                 linalg::MatrixView primal,
                                 linalg::MatrixView tangent) const;
    /** Sample consecutive primal/tangent rows without an index upload. */
    void sample_directional_rows(const ExteriorDirection& direction,
                                 int first,
                                 linalg::MatrixView primal,
                                 linalg::MatrixView tangent) const;

    std::int64_t dimension() const noexcept {
        return dimension_;
    }
    int n_orbitals() const noexcept {
        return n_orbitals_;
    }
    const std::uint64_t* occupations_data() const noexcept {
        return topology_->occupations.data();
    }
    const vb::ExteriorOrbitalTransform::ShearEdge* shear_edges_data() const noexcept {
        return topology_->edges.data();
    }
    const Shear* shear_schedule_data() const noexcept {
        return topology_->schedule.data();
    }
    std::size_t shear_count() const noexcept {
        return topology_->shears.size();
    }
    /** Current determinant diagonal scales, resident at a stable device address. */
    const double* scale_data() const noexcept {
        return scales_.data();
    }
    std::size_t resident_bytes() const noexcept {
        return (scales_.size() + inverse_scales_.size() + shear_coefficients_.size() +
                unit_upper_.rows() * unit_upper_.cols()) *
                   sizeof(double) +
               (topology_.use_count() == 1
                    ? topology_->occupations.size() * sizeof(std::uint64_t) +
                          topology_->schedule.size() * sizeof(Shear) +
                          topology_->choose.size() * sizeof(std::uint64_t) +
                          topology_->order.size() * sizeof(std::int64_t) +
                          topology_->edges.size() * sizeof(vb::ExteriorOrbitalTransform::ShearEdge)
                    : 0) +
               (right_pointer_upload_ ? right_pointer_upload_bytes() : 0);
    }

  private:
    Index index_view() const noexcept {
        return {topology_->occupations.data(),
                topology_->choose.data(),
                topology_->order.data(),
                n_orbitals_};
    }
    double* const* upload_right_columns(const std::vector<double*>& columns) const;
    std::size_t right_pointer_upload_bytes() const noexcept;
    struct Topology {
        explicit Topology(Context& context, std::size_t strings) : occupations(context, strings) {}
        Buffer<std::uint64_t> occupations, choose;
        Buffer<std::int64_t> order;
        Buffer<vb::ExteriorOrbitalTransform::ShearEdge> edges;
        std::vector<Shear> shears;
        Buffer<Shear> schedule;
    };
    Context* context_;
    std::int64_t dimension_;
    int n_orbitals_;
    int shared_values_ = 0, tile_values_ = 0;
    Buffer<double> scales_;
    Buffer<double> inverse_scales_;
    Buffer<double> shear_coefficients_;
    std::shared_ptr<Topology> topology_;
    Matrix<double> unit_upper_;
    mutable std::unique_ptr<PinnedUpload> right_pointer_upload_;
    mutable std::vector<double*> right_columns_;
};

} // namespace xmvb::cuda
