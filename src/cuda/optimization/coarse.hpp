#pragma once

#include <cstdint>

#include <functional>
#include <vector>

#include <Eigen/Core>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** Certified sampled balancing inverse at one frozen accepted point.
 * H images remain full-length; known-zero basis and metric columns use one
 * lazy contiguous panel per declared support. Only small projected matrices and
 * scalar reductions reach the host. Neither observe nor apply
 * evaluates H or M; supplied images must belong to the same accepted point.
 */
class DeviceCoarsePreconditioner {
  public:
    using BaseInverse = std::function<void(linalg::ConstMatrixView, linalg::MatrixView)>;
    struct SupportRange {
        std::int64_t begin, length;
        int capacity;
    };

    DeviceCoarsePreconditioner(Context& context,
                               std::int64_t dimension,
                               int capacity,
                               double lindep,
                               double action_accuracy = 0.0,
                               std::vector<SupportRange> supports = {});
    DeviceCoarsePreconditioner(const DeviceCoarsePreconditioner&) = delete;
    DeviceCoarsePreconditioner& operator=(const DeviceCoarsePreconditioner&) = delete;

    /** Admit a rank-resolved sample only after H and M reciprocity certification. */
    void observe(linalg::ConstMatrixView direction,
                 linalg::ConstMatrixView image,
                 linalg::ConstMatrixView metric_image);
    /** A sample supported on a declared range. H image is always full-length.
   */
    void observe_range(linalg::ConstMatrixView direction,
                       linalg::ConstMatrixView image,
                       linalg::ConstMatrixView metric_image,
                       std::int64_t begin);
    /** Signed symmetric balancing inverse with exactly one base inverse call. */
    void apply(linalg::ConstMatrixView residual,
               double eigenvalue,
               const BaseInverse& inverse,
               linalg::MatrixView output) const;

    int size() const noexcept {
        return size_;
    }
    std::int64_t dimension() const noexcept {
        return dimension_;
    }

  private:
    void validate_vector(linalg::ConstMatrixView vector) const;
    void ensure_finite(linalg::ConstMatrixView first,
                       linalg::ConstMatrixView second = {},
                       linalg::ConstMatrixView third = {}) const;
    double projection_roundoff(bool metric, const Eigen::MatrixXd& projected) const;
    void copy(linalg::MatrixView out, linalg::ConstMatrixView in) const;
    void scale(linalg::MatrixView out, double coefficient) const;
    Eigen::VectorXd project_basis(linalg::ConstMatrixView vector) const;
    void combine(linalg::MatrixView output,
                 linalg::ConstMatrixView panel,
                 const Eigen::VectorXd& coefficients,
                 double scale = 1.0) const;
    void combine_compact(linalg::MatrixView output,
                         const Eigen::VectorXd& coefficients,
                         bool metric,
                         double scale = 1.0) const;

    struct CompactColumn {
        std::int64_t begin, length;
        int panel, slot;
    };
    struct CompactPanel {
        std::int64_t begin, length;
        int used, capacity;
        Buffer<double> basis, metric;
    };
    const double* basis_data(const CompactColumn& column) const;
    const double* metric_data(const CompactColumn& column) const;
    linalg::ConstMatrixView panel_view(const CompactPanel& panel, bool metric, int used) const;

    Context* context_;
    std::int64_t dimension_;
    int capacity_, size_ = 0;
    double lindep_, action_accuracy_;
    std::vector<CompactColumn> columns_;
    std::vector<CompactPanel> panels_;
    Matrix<double> images_;
    mutable Matrix<double> sample_v_, sample_h_, sample_m_;
    mutable Matrix<double> coefficients_;
    // Small packed scalar results only; no full-length reduction temporaries.
    mutable Buffer<double> reductions_;
    mutable Buffer<int> finite_;
    Eigen::MatrixXd hessian_projection_, metric_projection_, sample_factor_;
};

} // namespace xmvb::cuda
