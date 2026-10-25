#pragma once

#include <memory>
#include <optional>
#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/preparation/overlap_pullback.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace xmvb::vb {
struct AcceptedPointContext;
struct VbScfCoupledKeyframeEvaluation;
class StructureMetricInverse;
class StructureTangentOperator;
struct SelectedMetric;

/** @brief One selected-space retraction and its immutable trial geometry. */
struct CiTransportPoint {
    OrbitalPreparationInput input;
    OrbitalPreparationResult prepared;
    Eigen::MatrixXd upper;
    Eigen::MatrixXd origin_direction;
    Eigen::MatrixXd base_coefficients;
    Eigen::MatrixXd coefficients;
    /** Exact Gram at this trial chart, shared with its keyframe when resident. */
    std::shared_ptr<const SelectedMetric> selected_metric;
};

/** @brief Consistent energy/gradient of the residual-balanced CIAH model.
 * Physical macro acceptance continues to use the objective, not this energy.
 */
struct CiTransportModel {
    double energy = 0.0;
    Eigen::VectorXd gradient;
};

/** @brief Projected finite CI transport; selected structures need not be closed. */
class CiTransport {
  public:
    CiTransport(const OrbitalPreparationInput& input,
                const SparseParameterLayout& layout,
                const OrbitalChart& chart,
                const AcceptedPointContext& origin,
                const StructureAction& action,
                const StructureTangentOperator& structure);

    CiTransportPoint retract(const Eigen::VectorXd& orbital_step,
                             const Eigen::VectorXd& structure_step) const;
    CiTransportModel evaluate_model(const CiTransportPoint& point,
                                    const VbScfCoupledKeyframeEvaluation& fixed) const;
    Eigen::VectorXd origin_structure_gradient() const;

  private:
    Eigen::MatrixXd solve(const Eigen::MatrixXd& upper,
                          const Eigen::MatrixXd& rhs,
                          const Eigen::MatrixXd* initial = nullptr) const;
    Eigen::VectorXd raw_direction(const Eigen::VectorXd& reduced) const;

    const OrbitalPreparationInput& input_;
    const SparseParameterLayout& layout_;
    const OrbitalChart& chart_;
    const AcceptedPointContext& origin_;
    const StructureAction& action_;
    const StructureTangentOperator& structure_;
    RepresentationOverlap overlap_;
    OrbitalOverlapPullback origin_pullback_;
    Eigen::MatrixXd upper_;
    Eigen::MatrixXd zero_;
    Eigen::MatrixXd c0_;
    Eigen::MatrixXd gc0_;
    double coordinate_scale_;
    Eigen::VectorXd diagonal_;
    const StructureMetricInverse* inverse_ = nullptr;
    /** One trial chart, shared by retraction and its gradient solve. */
    mutable Eigen::MatrixXd metric_upper_;
    mutable std::optional<RepresentationOverlap::PreparedMetric> prepared_metric_;
    mutable std::shared_ptr<const SelectedMetric> selected_metric_;
};

} // namespace xmvb::vb
